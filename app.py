import streamlit as st
import pandas as pd
from supabase import create_client
import plotly.express as px
import plotly.graph_objects as go
import requests
from streamlit_lottie import st_lottie
from datetime import datetime, timedelta

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
# 3. ENHANCED DARK THEME CSS
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
    color: #e2e8f0;
}

#MainMenu, footer, header {visibility: hidden;}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 2px solid #334155;
}

h1 {
    background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 3rem !important;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}

h2, h3 {
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}

.metric-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.6), rgba(51, 65, 85, 0.4));
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 48px rgba(56, 189, 248, 0.2);
    border-color: rgba(56, 189, 248, 0.3);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(15, 23, 42, 0.6);
    padding: 8px;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(30, 41, 59, 0.4);
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 600;
    padding: 12px 24px;
    border: 1px solid transparent;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    color: white;
    border-color: rgba(59, 130, 246, 0.5);
}

.stDownloadButton button {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stDownloadButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(59, 130, 246, 0.4);
}

div[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.6), rgba(51, 65, 85, 0.4));
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 12px;
    padding: 1rem;
    backdrop-filter: blur(10px);
}

div[data-testid="stMetricValue"] {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.insight-box {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.1));
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. HELPERS
# ==========================================
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def dark_chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        margin=dict(t=40, b=40, l=40, r=40),
        hoverlabel=dict(
            bgcolor="rgba(30, 41, 59, 0.95)",
            font_size=13,
            font_family="Inter"
        )
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
    df['Week'] = df['at'].dt.to_period('W').astype(str)

    # Ratings
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df = df[df['score'].isin(RATING_ORDER)]

    # Content & length
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
    with loader.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            lottie = load_lottieurl("https://lottie.host/9e5c4644-841b-43c3-982d-19597143c690/w5h8o4t9zD.json")
            if lottie:
                st_lottie(lottie, height=250, key="loading")
            st.markdown("<h3 style='text-align: center;'>🚀 Syncing Intelligence Engine…</h3>", unsafe_allow_html=True)
    
    df_raw, net_cols = load_data()
    st.session_state['df'] = df_raw
    st.session_state['net_cols'] = net_cols
    loader.empty()

df_raw = st.session_state['df']
net_cols = st.session_state['net_cols']

if df_raw.empty:
    st.error("⚠️ No data available. Please check your database connection.")
    st.stop()

# ==========================================
# 8. SIDEBAR FILTERS
# ==========================================
with st.sidebar:
    st.markdown("## 🎛 Control Center")
    
    st.markdown("---")

    min_d, max_d = df_raw['at'].dt.date.min(), df_raw['at'].dt.date.max()
    
    # Quick date presets
    preset = st.selectbox(
        "⏱️ Quick Preset",
        ["Custom", "Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"]
    )
    
    if preset == "Last 7 Days":
        date_range = [max_d - timedelta(days=7), max_d]
    elif preset == "Last 30 Days":
        date_range = [max_d - timedelta(days=30), max_d]
    elif preset == "Last 90 Days":
        date_range = [max_d - timedelta(days=90), max_d]
    elif preset == "All Time":
        date_range = [min_d, max_d]
    else:
        date_range = st.date_input("📅 Timeline", [min_d, max_d], min_value=min_d, max_value=max_d)

    st.markdown("---")

    sel_apps = st.multiselect(
        "🏢 Brands",
        sorted(df_raw['norm_app'].unique()),
        default=sorted(df_raw['norm_app'].unique())
    )

    sel_ratings = st.multiselect(
        "⭐ Ratings",
        RATING_ORDER,
        default=RATING_ORDER,
        format_func=lambda x: f"{'⭐' * x} ({x})"
    )

    char_filter = st.radio(
        "📝 Review Depth",
        ["All", "<=29 Chars", ">=30 Chars"],
        help="Filter by review length"
    )

    st.markdown("---")

    search_query = st.text_input("🔍 Keyword Search", placeholder="Enter min 3 characters...")
    
    st.markdown("---")
    
    if st.button("🔄 Reset Filters", use_container_width=True):
        st.rerun()

# ==========================================
# 9. FILTER LOGIC
# ==========================================
mask = (
    df_raw['norm_app'].isin(sel_apps) &
    df_raw['score'].isin(sel_ratings)
)

if isinstance(date_range, list) and len(date_range) == 2:
    mask &= df_raw['at'].dt.date.between(date_range[0], date_range[1])

if char_filter != "All":
    mask &= df_raw['length_group'] == char_filter

if search_query and len(search_query) >= 3:
    mask &= df_raw['content'].str.contains(search_query, case=False, na=False)

df = df_raw[mask].copy()

if df.empty:
    st.warning("⚠️ No data matches your selected filters. Try adjusting the criteria.")
    st.stop()

# ==========================================
# 10. HEADER METRICS
# ==========================================
st.title("⚡ Fintech Intelligence Hub")
st.caption(f"Analyzing **{len(df):,}** reviews across **{df['norm_app'].nunique()}** brands")

st.markdown("---")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("📊 Total Reviews", f"{len(df):,}")

with c2:
    avg_rating = df['score'].mean()
    prev_avg = df_raw['score'].mean()
    delta = avg_rating - prev_avg
    st.metric("⭐ Avg Rating", f"{avg_rating:.2f}", delta=f"{delta:+.2f}" if abs(delta) > 0.01 else None)

with c3:
    st.metric("📍 Median Rating", f"{df['score'].median():.0f}★")

with c4:
    det_pct = (df['score'] <= 2).mean() * 100
    st.metric("🚨 Detractor Rate", f"{det_pct:.1f}%", delta=None, delta_color="inverse")

with c5:
    promo_pct = (df['score'] >= 4).mean() * 100
    st.metric("🎯 Promoter Rate", f"{promo_pct:.1f}%")

# Calculate NPS
nps = promo_pct - det_pct
st.markdown(f"""
<div class="insight-box">
    <strong>📈 Net Promoter Score (NPS):</strong> {nps:.1f}
    {'🟢 Excellent!' if nps > 50 else '🟡 Good' if nps > 0 else '🔴 Needs Improvement'}
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    st.download_button(
        "⬇️ Download Filtered Data",
        df.to_csv(index=False).encode('utf-8'),
        "filtered_reviews.csv",
        "text/csv",
        use_container_width=True
    )
with col2:
    st.metric("📦 Data Size", f"{len(df.to_csv()):,} bytes")

st.markdown("---")

# ==========================================
# 11. ENHANCED TABS
# ==========================================
tabs = st.tabs([
    "📊 Rating Distribution",
    "📏 Review Quality",
    "🚀 Success Drivers",
    "🛑 Pain Points",
    "📈 Trends & Patterns",
    "🔍 Deep Dive"
])

# ---- TAB 1: RATINGS ----
with tabs[0]:
    st.subheader("Rating Distribution Heatmap")
    
    rating_dist = (
        df.groupby(['norm_app', 'score'])
          .size()
          .unstack(fill_value=0)
          .reindex(columns=RATING_ORDER, fill_value=0)
    )
    
    rating_pct = rating_dist.div(rating_dist.sum(axis=1), axis=0) * 100

    fig = px.imshow(
        rating_pct,
        text_auto=".1f",
        aspect="auto",
        color_continuous_scale="RdYlGn",
        labels=dict(x="Rating", y="Brand", color="% of Reviews")
    )
    fig.update_xaxes(side="top")
    st.plotly_chart(dark_chart(fig), use_container_width=True)
    
    # Bar chart comparison
    st.subheader("Volume by Brand & Rating")
    fig2 = px.bar(
        rating_dist.reset_index(),
        x='norm_app',
        y=RATING_ORDER,
        barmode='group',
        color_discrete_sequence=px.colors.sequential.Plasma
    )
    st.plotly_chart(dark_chart(fig2), use_container_width=True)

# ---- TAB 2: QUALITY ----
with tabs[1]:
    st.subheader("Review Depth Analysis")
    
    q = pd.crosstab(df['score'], df['length_group'], normalize='index') * 100
    fig = px.bar(
        q,
        barmode="stack",
        labels={"value": "% of Reviews", "score": "Rating"},
        color_discrete_sequence=['#3b82f6', '#8b5cf6']
    )
    st.plotly_chart(dark_chart(fig), use_container_width=True)
    
    # Character count distribution
    st.subheader("Character Count Distribution")
    fig2 = px.histogram(
        df,
        x='char_count',
        nbins=50,
        color='norm_app',
        marginal='box',
        color_discrete_map=COLOR_MAP
    )
    st.plotly_chart(dark_chart(fig2), use_container_width=True)

# ---- TAB 3 & 4: DRIVERS / BARRIERS ----
def net_heatmap(sub_df, title, scale, emoji):
    if sub_df.empty or not net_cols:
        st.info("ℹ️ Insufficient data for this analysis.")
        return

    st.subheader(f"{emoji} {title}")
    
    base = sub_df.groupby('norm_app').size()
    mat = (
        sub_df.groupby('norm_app')[net_cols]
        .sum()
        .T.div(base, axis=1) * 100
    )
    
    mat.index = mat.index.str.replace('[NET]', '').str.strip()
    mat = mat.loc[mat.sum(axis=1).nlargest(15).index]  # Top 15
    
    fig = px.imshow(
        mat,
        text_auto=".1f",
        aspect="auto",
        color_continuous_scale=scale,
        labels=dict(x="Brand", y="Theme", color="% Mentioned")
    )
    st.plotly_chart(dark_chart(fig), use_container_width=True)
    
    # Top 5 themes overall
    top_themes = mat.mean(axis=1).nlargest(5)
    st.markdown("**Top 5 Themes Overall:**")
    for i, (theme, pct) in enumerate(top_themes.items(), 1):
        st.markdown(f"{i}. **{theme}**: {pct:.1f}%")

with tabs[2]:
    net_heatmap(df[df['score'] >= 4], "Success Drivers", "Greens", "🚀")

with tabs[3]:
    net_heatmap(df[df['score'] <= 2], "Pain Points & Barriers", "Reds", "🛑")

# ---- TAB 5: TRENDS ----
with tabs[4]:
    st.subheader("📈 Review Volume Over Time")
    
    time_gran = st.radio("Granularity", ["Monthly", "Weekly"], horizontal=True)
    
    if time_gran == "Monthly":
        v = df.groupby(['Month', 'norm_app']).size().reset_index(name='Count')
        x_col = 'Month'
    else:
        v = df.groupby(['Week', 'norm_app']).size().reset_index(name='Count')
        x_col = 'Week'
    
    fig = px.line(
        v,
        x=x_col,
        y='Count',
        color='norm_app',
        markers=True,
        color_discrete_map=COLOR_MAP
    )
    st.plotly_chart(dark_chart(fig), use_container_width=True)
    
    # Rating trend
    st.subheader("⭐ Average Rating Trend")
    rating_trend = df.groupby(['Month', 'norm_app'])['score'].mean().reset_index()
    fig2 = px.line(
        rating_trend,
        x='Month',
        y='score',
        color='norm_app',
        markers=True,
        color_discrete_map=COLOR_MAP
    )
    fig2.update_yaxes(range=[1, 5])
    st.plotly_chart(dark_chart(fig2), use_container_width=True)

# ---- TAB 6: DEEP DIVE ----
with tabs[5]:
    st.subheader("🔍 Sample Reviews")
    
    col1, col2 = st.columns(2)
    with col1:
        sample_app = st.selectbox("Select Brand", sorted(df['norm_app'].unique()))
    with col2:
        sample_rating = st.selectbox("Select Rating", sorted(df['score'].unique(), reverse=True))
    
    sample_df = df[(df['norm_app'] == sample_app) & (df['score'] == sample_rating)]
    
    if not sample_df.empty:
        st.write(f"Showing up to 10 reviews ({len(sample_df)} total)")
        for idx, row in sample_df.head(10).iterrows():
            with st.expander(f"⭐ {row['score']} - {row['at'].strftime('%Y-%m-%d')} ({row['char_count']} chars)"):
                st.write(row['content'] if row['content'] else "_No review text_")
    else:
        st.info("No reviews match this selection.")
    
    # Sentiment bucket breakdown
    st.subheader("🎭 Sentiment Distribution")
    sent_dist = df.groupby(['norm_app', 'sentiment_bucket']).size().reset_index(name='Count')
    fig = px.sunburst(
        sent_dist,
        path=['norm_app', 'sentiment_bucket'],
        values='Count',
        color='sentiment_bucket',
        color_discrete_map={
            'Detractor (1-2)': '#ef4444',
            'Passive (3)': '#f59e0b',
            'Promoter (4-5)': '#10b981'
        }
    )
    st.plotly_chart(dark_chart(fig), use_container_width=True)

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #64748b;'>Built with ❤️ using Streamlit • Data refreshes every 10 minutes</p>",
    unsafe_allow_html=True
)
