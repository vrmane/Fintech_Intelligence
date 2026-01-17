import streamlit as st
import pandas as pd
from supabase import create_client
import plotly.express as px
import plotly.graph_objects as go
import requests
from streamlit_lottie import st_lottie

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Fintech Intelligence Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. GLOBAL CONSTANTS
# ==========================================
APP_MAP = {
    'moneyview': 'MoneyView',
    'kreditbee': 'KreditBee',
    'navi': 'Navi',
    'kissht': 'Kissht',
    'fibe': 'Fibe',
    'earlysalary': 'Fibe'
}

RATING_ORDER = [1, 2, 3, 4, 5]

COLOR_MAP = {
    'MoneyView': '#00d4ff',
    'KreditBee': '#ff9f00',
    'Navi': '#00ff9d',
    'Kissht': '#ff0055',
    'Fibe': '#bc13fe'
}

# ==========================================
# 3. DARK THEME CSS
# ==========================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: #e2e8f0;
}
#MainMenu, footer, header {visibility: hidden;}
section[data-testid="stSidebar"] {
    background-color: #0f172a;
    border-right: 1px solid #334155;
}
h1, h2, h3 {
    background: linear-gradient(45deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. HELPERS
# ==========================================
def load_lottieurl(url):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def dark_chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

# ==========================================
# 5. SUPABASE CONNECTION
# ==========================================
@st.cache_resource(show_spinner=False)
def init_connection():
    return create_client(
        st.secrets["supabase"]["url"],
        st.secrets["supabase"]["key"]
    )

supabase = init_connection()

# ==========================================
# 6. DATA LOADING & NORMALIZATION
# ==========================================
@st.cache_data(ttl=600, show_spinner=False)
def load_data():
    rows, start, batch = [], 0, 1000
    while True:
        resp = supabase.table("reviews").select("*").range(start, start+batch-1).execute()
        if not resp.data:
            break
        rows.extend(resp.data)
        if len(resp.data) < batch:
            break
        start += batch

    df = pd.DataFrame(rows)
    if df.empty:
        return df, []

    df.rename(columns={
        'App_Name': 'app_name',
        'Rating': 'score',
        'Review_Date': 'at',
        'Review_Text': 'content'
    }, inplace=True)

    # App normalization
    df['norm_app'] = (
        df['app_name']
        .astype(str)
        .str.lower()
        .apply(lambda x: next((v for k, v in APP_MAP.items() if k in x), None))
    )
    df.dropna(subset=['norm_app'], inplace=True)

    # Dates
    df['at'] = pd.to_datetime(df['at'], errors='coerce', utc=True)
    df.dropna(subset=['at'], inplace=True)
    df['at'] = df['at'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
    df['Month'] = df['at'].dt.to_period('M').astype(str)

    # Ratings
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df = df[df['score'].isin(RATING_ORDER)]

    # Content & length (FIXED BUG)
    df['content'] = df['content'].fillna("")
    df['char_count'] = df['content'].str.len()
    df['length_group'] = pd.cut(
        df['char_count'],
        bins=[-1, 29, 10_000],
        labels=['<=29 Chars', '>=30 Chars']
    )

    # NPS-style buckets
    df['sentiment_bucket'] = pd.cut(
        df['score'],
        bins=[0, 2, 3, 5],
        labels=['Detractor (1-2)', 'Passive (3)', 'Promoter (4-5)']
    )

    net_cols = [c for c in df.columns if str(c).startswith('[NET]')]
    for c in net_cols:
        df[c] = df[c].fillna(0).astype(bool)

    return df, net_cols

# ==========================================
# 7. LOAD WITH SPLASH
# ==========================================
if 'df' not in st.session_state:
    loader = st.empty()
    with loader:
        lottie = load_lottieurl("https://lottie.host/9e5c4644-841b-43c3-982d-19597143c690/w5h8o4t9zD.json")
        if lottie:
            st_lottie(lottie, height=250)
        st.markdown("### Syncing Intelligence Engine…")
    df_raw, net_cols = load_data()
    st.session_state['df'] = df_raw
    st.session_state['net_cols'] = net_cols
    loader.empty()

df_raw = st.session_state['df']
net_cols = st.session_state['net_cols']
if df_raw.empty:
    st.stop()

# ==========================================
# 8. SIDEBAR FILTERS
# ==========================================
with st.sidebar:
    st.markdown("## 🎛 Control Center")

    min_d, max_d = df_raw['at'].dt.date.min(), df_raw['at'].dt.date.max()
    date_range = st.date_input("Timeline", [min_d, max_d])

    sel_apps = st.multiselect(
        "Brands",
        sorted(df_raw['norm_app'].unique()),
        default=sorted(df_raw['norm_app'].unique())
    )

    sel_ratings = st.multiselect("Ratings", RATING_ORDER, default=RATING_ORDER)

    char_filter = st.radio(
        "Review Depth",
        ["All", "<=29 Chars", ">=30 Chars"]
    )

    search_query = st.text_input("Keyword search (min 3 chars)")

# ==========================================
# 9. FILTER LOGIC
# ==========================================
mask = (
    df_raw['norm_app'].isin(sel_apps) &
    df_raw['score'].isin(sel_ratings)
)

mask &= df_raw['at'].dt.date.between(date_range[0], date_range[1])

if char_filter != "All":
    mask &= df_raw['length_group'] == char_filter

if search_query and len(search_query) >= 3:
    mask &= df_raw['content'].str.contains(search_query, case=False)

df = df_raw[mask].copy()

if df.empty:
    st.warning("No data for selected filters.")
    st.stop()

# ==========================================
# 10. HEADER METRICS
# ==========================================
st.title("Fintech Intelligence Hub")
st.caption(f"{len(df):,} reviews analysed")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Total Reviews", f"{len(df):,}")
c2.metric("Avg Rating", f"{df['score'].mean():.2f}")
c3.metric("Median Rating", f"{df['score'].median():.1f}")

det_pct = (df['score'] <= 2).mean() * 100
c4.metric("Detractor Rate (1–2★)", f"{det_pct:.1f}%")

promo_pct = (df['score'] >= 4).mean() * 100
c5.metric("Promoter Rate (4–5★)", f"{promo_pct:.1f}%")

st.download_button(
    "⬇️ Download Filtered Data",
    df.to_csv(index=False),
    "filtered_reviews.csv",
    "text/csv"
)

st.markdown("---")

# ==========================================
# 11. TABS
# ==========================================
tabs = st.tabs([
    "📊 Ratings",
    "📏 Review Quality",
    "🚀 Drivers",
    "🛑 Barriers",
    "📅 Trends"
])

# ---- TAB 1: RATINGS ----
with tabs[0]:
    rating_dist = (
        df.groupby(['norm_app', 'score'])
          .size()
          .unstack(fill_value=0)
          .reindex(columns=RATING_ORDER, fill_value=0)
    )

    fig = px.imshow(
        rating_dist.div(rating_dist.sum(axis=1), axis=0) * 100,
        text_auto=".0f",
        aspect="auto",
        color_continuous_scale="RdYlGn"
    )
    st.plotly_chart(dark_chart(fig), use_container_width=True)

# ---- TAB 2: QUALITY ----
with tabs[1]:
    q = pd.crosstab(df['score'], df['length_group'], normalize='index') * 100
    fig = px.bar(q, barmode="stack")
    st.plotly_chart(dark_chart(fig), use_container_width=True)

# ---- TAB 3 & 4: DRIVERS / BARRIERS ----
def net_heatmap(sub_df, title, scale):
    if sub_df.empty or not net_cols:
        st.info("Insufficient data.")
        return

    base = sub_df.groupby('norm_app').size()
    mat = (
        sub_df.groupby('norm_app')[net_cols]
        .sum()
        .T.div(base, axis=1) * 100
    )
    mat.index = mat.index.str.replace('[NET]', '').str.strip()
    fig = px.imshow(mat, text_auto=".1f", aspect="auto", color_continuous_scale=scale)
    st.subheader(title)
    st.plotly_chart(dark_chart(fig), use_container_width=True)

with tabs[2]:
    net_heatmap(df[df['score'] >= 4], "Top Positive Drivers (% of positive reviews)", "Greens")

with tabs[3]:
    net_heatmap(df[df['score'] <= 2], "Top Negative Barriers (% of detractor reviews)", "Reds")

# ---- TAB 5: TRENDS ----
with tabs[4]:
    v = df.groupby(['Month', 'norm_app']).size().reset_index(name='Count')
    fig = px.line(v, x='Month', y='Count', color='norm_app', markers=True)
    st.plotly_chart(dark_chart(fig), use_container_width=True)
