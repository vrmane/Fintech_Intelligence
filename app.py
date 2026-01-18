# ============================================================
# FINTECH EXECUTIVE VoC INTELLIGENCE – SINGLE PAGE
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client
import plotly.express as px

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Executive VoC Intelligence",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# SUPABASE CONNECTION
# ============================================================
@st.cache_resource
def init_supabase():
    return create_client(
        st.secrets["supabase"]["url"],
        st.secrets["supabase"]["key"]
    )

supabase = init_supabase()

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data(ttl=600)
def load_data():
    rows, start, batch = [], 0, 1000
    while True:
        res = supabase.table("reviews").select("*").range(start, start+batch-1).execute()
        if not res.data:
            break
        rows.extend(res.data)
        if len(res.data) < batch:
            break
        start += batch

    df = pd.DataFrame(rows)
    if df.empty:
        return df, []

    # ---- Normalisation ----
    df['Review_Date'] = pd.to_datetime(df['Review_Date'], errors='coerce')
    df = df.dropna(subset=['Review_Date'])

    df['Month'] = df['Review_Date'].dt.to_period("M").astype(str)
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df = df[df['Rating'].between(1,5)]

    df['Review_Text'] = df['Review_Text'].fillna("")
    df['char_count'] = df['Review_Text'].str.len()

    df['length_bucket'] = np.where(df['char_count'] <= 29, "Short (≤29)", "Detailed (30+)")

    df['sentiment'] = pd.cut(
        df['Rating'],
        bins=[0,2,3,5],
        labels=["Detractor", "Passive", "Promoter"]
    )

    net_cols = [c for c in df.columns if c.startswith("[NET]")]
    for c in net_cols:
        df[c] = df[c].fillna(0).astype(bool)

    return df, net_cols

df_raw, net_cols = load_data()
if df_raw.empty:
    st.stop()

# ============================================================
# SIDEBAR FILTERS (CRITICAL)
# ============================================================
with st.sidebar:
    st.header("🎛 Filters")

    brands = sorted(df_raw['App_Name'].unique())
    sel_brands = st.multiselect("Brand", brands, default=brands)

    min_d, max_d = df_raw['Review_Date'].min(), df_raw['Review_Date'].max()
    date_range = st.date_input("Date Range", [min_d, max_d])

    ratings = st.multiselect("Ratings", [1,2,3,4,5], default=[1,2,3,4,5])

    char_filter = st.radio(
        "Review Length",
        ["All", "Short (≤29)", "Detailed (30+)"],
        index=0
    )

# ============================================================
# APPLY FILTERS
# ============================================================
df = df_raw[
    (df_raw['App_Name'].isin(sel_brands)) &
    (df_raw['Rating'].isin(ratings)) &
    (df_raw['Review_Date'].dt.date >= date_range[0]) &
    (df_raw['Review_Date'].dt.date <= date_range[1])
]

if char_filter != "All":
    df = df[df['length_bucket'] == char_filter]

if df.empty:
    st.warning("No data for selected filters")
    st.stop()

# ============================================================
# EXECUTIVE HEADER
# ============================================================
st.title("📊 Executive Voice of Customer Intelligence")

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Total Reviews", f"{len(df):,}")
c2.metric("Avg Rating", f"{df['Rating'].mean():.2f}")
c3.metric("Median Rating", f"{df['Rating'].median():.1f}")
c4.metric("Detractors (1–2★)", f"{(df['Rating']<=2).mean()*100:.1f}%")
c5.metric("Promoters (4–5★)", f"{(df['Rating']>=4).mean()*100:.1f}%")

st.markdown("---")

# ============================================================
# MARKET VS BRAND POSITIONING
# ============================================================
st.subheader("🏆 Market & Brand Positioning")

rating_mat = (
    df.groupby(['App_Name','Rating'])
      .size()
      .unstack(fill_value=0)
      .reindex(columns=[1,2,3,4,5], fill_value=0)
)

rating_pct = rating_mat.div(rating_mat.sum(axis=1), axis=0)*100

st.plotly_chart(
    px.imshow(rating_pct, text_auto=".0f", aspect="auto",
              color_continuous_scale="RdYlGn"),
    use_container_width=True
)

# ============================================================
# REVIEW CREDIBILITY & TRUST
# ============================================================
st.subheader("📏 Review Credibility Signals")

cred = df.groupby('App_Name').agg(
    Avg_Chars=('char_count','mean'),
    Short_Positive_%=('length_bucket',
        lambda x: ((x=="Short (≤29)") &
                   (df.loc[x.index,'Rating']>=4)).mean()*100),
    Long_Negative_%=('length_bucket',
        lambda x: ((x=="Detailed (30+)") &
                   (df.loc[x.index,'Rating']<=2)).mean()*100)
).reset_index()

st.dataframe(cred.sort_values("Avg_Chars"))

# ============================================================
# DRIVERS & BARRIERS
# ============================================================
def net_heat(sub_df, title, scale):
    if sub_df.empty or not net_cols:
        st.info("Insufficient data")
        return
    base = sub_df.groupby('App_Name').size()
    mat = sub_df.groupby('App_Name')[net_cols].sum().T.div(base, axis=1)*100
    mat.index = mat.index.str.replace("[NET]","")
    st.plotly_chart(
        px.imshow(mat, text_auto=".1f", aspect="auto",
                  color_continuous_scale=scale),
        use_container_width=True
    )

st.subheader("🚀 What Users Love (Drivers)")
net_heat(df[df['Rating']>=4], "% of Promoters Mentioning Theme", "Greens")

st.subheader("🛑 What Hurts Experience (Barriers)")
net_heat(df[df['Rating']<=2], "% of Detractors Mentioning Theme", "Reds")

# ============================================================
# COMPETITIVE OVER-INDEX
# ============================================================
st.subheader("⚖️ Brand Over-Index vs Market (Promoters)")

market_base = df[df['Rating']>=4].shape[0]
market_net = df[df['Rating']>=4][net_cols].sum() / market_base * 100

brand = sel_brands[0] if len(sel_brands)==1 else sel_brands[0]
bdf = df[(df['App_Name']==brand) & (df['Rating']>=4)]
b_base = bdf.shape[0]
brand_net = bdf[net_cols].sum() / b_base * 100

over_index = (brand_net - market_net).sort_values(ascending=False).head(8)
over_index.index = over_index.index.str.replace("[NET]","")

st.bar_chart(over_index)

# ============================================================
# AUTO-GENERATED CEO SUMMARY
# ============================================================
st.markdown("---")
st.subheader("🧠 Auto-Generated CEO Summary")

market_avg = df['Rating'].mean()

summaries = []

for b in df['App_Name'].unique():
    bdf = df[df['App_Name']==b]

    avg = bdf['Rating'].mean()
    det = (bdf['Rating']<=2).mean()*100
    prom = (bdf['Rating']>=4).mean()*100

    driver = "N/A"
    barrier = "N/A"

    if net_cols:
        if not bdf[bdf['Rating']>=4].empty:
            driver = bdf[bdf['Rating']>=4][net_cols].sum().idxmax().replace("[NET]","")
        if not bdf[bdf['Rating']<=2].empty:
            barrier = bdf[bdf['Rating']<=2][net_cols].sum().idxmax().replace("[NET]","")

    summaries.append(
        f"**{b}** shows an average rating of **{avg:.2f}**, "
        f"{'above' if avg>market_avg else 'below'} the market average. "
        f"Promoters account for **{prom:.1f}%**, while detractors stand at **{det:.1f}%**. "
        f"Customers most frequently praise **{driver}**, whereas **{barrier}** emerges "
        f"as the primary friction point. Focus should be on amplifying strengths that drive "
        f"long, positive reviews while urgently addressing detractor-led themes to protect "
        f"brand momentum."
    )

st.markdown("\n\n".join(summaries))
