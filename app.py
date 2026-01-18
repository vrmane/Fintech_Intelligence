# =====================================================
# EXECUTIVE FINTECH INTELLIGENCE HUB (SUPABASE)
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client
import plotly.express as px

# =====================================================
# CONNECTION
# =====================================================
@st.cache_resource
def init_supabase():
    return create_client(
        st.secrets["supabase"]["url"],
        st.secrets["supabase"]["key"]
    )

supabase = init_supabase()

# =====================================================
# LOAD DATA
# =====================================================
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

    # ---------- Normalization ----------
    df['Review_Date'] = pd.to_datetime(df['Review_Date'], errors='coerce')
    df = df.dropna(subset=['Review_Date'])

    df['Month'] = df['Review_Date'].dt.to_period("M").astype(str)
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df = df[df['Rating'].between(1,5)]

    df['Review_Text'] = df['Review_Text'].fillna("")
    df['char_count'] = df['Review_Text'].str.len()
    df['length_bucket'] = np.where(df['char_count'] <= 29, "Short", "Detailed")

    # Sentiment structure
    df['sentiment'] = pd.cut(
        df['Rating'],
        bins=[0,2,3,5],
        labels=["Detractor", "Passive", "Promoter"]
    )

    # NET columns
    net_cols = [c for c in df.columns if c.startswith("[NET]")]
    for c in net_cols:
        df[c] = df[c].fillna(0).astype(bool)

    return df, net_cols

df_raw, net_cols = load_data()
if df_raw.empty:
    st.stop()

# =====================================================
# FILTERS
# =====================================================
with st.sidebar:
    st.title("🎛 Filters")

    brands = sorted(df_raw['App_Name'].unique())
    sel_brands = st.multiselect("Brand", brands, default=brands)

    min_d, max_d = df_raw['Review_Date'].min(), df_raw['Review_Date'].max()
    date_range = st.date_input("Date Range", [min_d, max_d])

    ratings = st.multiselect("Ratings", [1,2,3,4,5], default=[1,2,3,4,5])

df = df_raw[
    (df_raw['App_Name'].isin(sel_brands)) &
    (df_raw['Rating'].isin(ratings)) &
    (df_raw['Review_Date'].dt.date >= date_range[0]) &
    (df_raw['Review_Date'].dt.date <= date_range[1])
]

# =====================================================
# EXEC SUMMARY
# =====================================================
st.title("📊 Fintech Executive Intelligence")

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Reviews", f"{len(df):,}")
c2.metric("Avg Rating", f"{df['Rating'].mean():.2f}")
c3.metric("Median Rating", f"{df['Rating'].median():.1f}")
c4.metric("Detractors", f"{(df['Rating']<=2).mean()*100:.1f}%")
c5.metric("Promoters", f"{(df['Rating']>=4).mean()*100:.1f}%")

st.markdown("---")

# =====================================================
# TABS
# =====================================================
tabs = st.tabs([
    "🏆 Market & Brand",
    "📏 Credibility",
    "🚀 Drivers",
    "🛑 Barriers",
    "🧠 AI Executive Insights"
])

# =====================================================
# BRAND & MARKET
# =====================================================
with tabs[0]:
    mat = (
        df.groupby(['App_Name','Rating'])
          .size()
          .unstack(fill_value=0)
          .reindex(columns=[1,2,3,4,5], fill_value=0)
    )
    mat_pct = mat.div(mat.sum(axis=1), axis=0)*100
    st.plotly_chart(px.imshow(mat_pct, text_auto=".0f", aspect="auto",
                              color_continuous_scale="RdYlGn"), use_container_width=True)

# =====================================================
# CREDIBILITY
# =====================================================
with tabs[1]:
    cred = df.groupby('App_Name').agg(
        Avg_Chars=('char_count','mean'),
        Short_Positive_Pct=('length_bucket',
            lambda x: ((x=="Short") & (df.loc[x.index,'Rating']>=4)).mean()*100)
    ).reset_index()

    st.dataframe(cred.sort_values("Avg_Chars"))

# =====================================================
# DRIVERS / BARRIERS
# =====================================================
def net_heat(sub_df, title, scale):
    if sub_df.empty or not net_cols:
        st.info("Insufficient data")
        return
    base = sub_df.groupby('App_Name').size()
    mat = sub_df.groupby('App_Name')[net_cols].sum().T.div(base, axis=1)*100
    mat.index = mat.index.str.replace("[NET]","")
    st.plotly_chart(px.imshow(mat, text_auto=".1f", aspect="auto",
                              color_continuous_scale=scale),
                    use_container_width=True)

with tabs[2]:
    net_heat(df[df['Rating']>=4], "Positive Drivers (% of promoters)", "Greens")

with tabs[3]:
    net_heat(df[df['Rating']<=2], "Negative Barriers (% of detractors)", "Reds")

# =====================================================
# AI EXEC INSIGHTS
# =====================================================
with tabs[4]:
    market_avg = df['Rating'].mean()

    for brand in df['App_Name'].unique():
        b = df[df['App_Name']==brand]

        avg = b['Rating'].mean()
        det = (b['Rating']<=2).mean()*100
        prom = (b['Rating']>=4).mean()*100

        driver = "N/A"
        barrier = "N/A"
        if net_cols:
            if not b[b['Rating']>=4].empty:
                driver = b[b['Rating']>=4][net_cols].sum().idxmax().replace("[NET]","")
            if not b[b['Rating']<=2].empty:
                barrier = b[b['Rating']<=2][net_cols].sum().idxmax().replace("[NET]","")

        with st.expander(f"📌 {brand}", expanded=True):
            st.metric("Avg Rating", f"{avg:.2f}", f"{avg-market_avg:+.2f} vs market")
            st.markdown(f"""
**🔥 What to amplify:** {driver}  
**⚠️ What to fix:** {barrier}  

**Marketing-safe angle:**  
Themes driving *long, positive* reviews outperforming market.

**Executive risk:**  
High detractor density indicates experience gaps competitors can exploit.
""")
