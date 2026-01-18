import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from supabase import create_client, Client

# =====================================================
# 0. GLOBAL GUARDRAILS (DO NOT TUNE LIGHTLY)
# =====================================================
MIN_VOLUME_BRAND = 50      # Minimum reviews per brand for insights
MIN_BASE_TREND = 30        # Minimum base for MoM / WoW
SCHEMA_VERSION = "v2.1"    # Bump when schema changes

# =====================================================
# 1. PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Strategic Intelligence Platform",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 2. EXECUTIVE-GRADE STYLING
# =====================================================
st.markdown("""
<style>
.stApp { background-color:#0b0f19; color:#e2e8f0; }
.ai-card {
    background: rgba(30,41,59,0.5);
    border-left:4px solid #38bdf8;
    padding:18px; border-radius:10px;
    margin-bottom:16px;
}
.ai-title { font-size:1.1rem; font-weight:700; }
.ai-bullet { font-size:0.95rem; color:#cbd5e1; margin-bottom:6px; }
.warn { color:#fbbf24; font-weight:600; }
.pos { color:#4ade80; font-weight:600; }
.neg { color:#f87171; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# 3. SUPABASE CONNECTION
# =====================================================
@st.cache_resource
def init_connection():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

supabase = init_connection()

# =====================================================
# 4. DATA LOADER (DEFENSIVE)
# =====================================================
@st.cache_data(ttl=600)
def load_data(schema_version=SCHEMA_VERSION):
    all_rows, start, batch = [], 0, 1000

    while True:
        res = supabase.table("reviews").select("*").range(start, start+batch-1).execute()
        rows = res.data
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < batch:
            break
        start += batch

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    # --- Date handling ---
    df['at'] = pd.to_datetime(df.get('Review_Date'), errors='coerce')
    df = df.dropna(subset=['at'])
    df['Month'] = df['at'].dt.strftime('%Y-%m')
    df['Week'] = df['at'].dt.to_period('W').astype(str)

    # --- Rating ---
    df['score'] = pd.to_numeric(df.get('Rating'), errors='coerce')

    # --- Review depth ---
    df['char_count'] = df.get('Review_Text', "").astype(str).str.len()
    df['length_bucket'] = np.where(df['char_count'] <= 29, "Brief (<=29)", "Detailed (>=30)")

    # --- Theme extraction ---
    theme_cols = []
    for c in df.columns:
        if str(c).startswith('[NET]'):
            clean = c.replace('[NET]', '').strip()
            df[clean] = (
                pd.to_numeric(df[c], errors='coerce')
                .fillna(0)
                .apply(lambda x: 1 if x > 0 else 0)
            )
            theme_cols.append(clean)

    st.session_state['theme_cols'] = theme_cols
    return df

# =====================================================
# 5. ANALYTICS (RESEARCH-SAFE)
# =====================================================
def safe_trend_change(df, theme, time_col):
    if theme not in df.columns:
        return None

    grp = df.groupby(time_col)[theme].agg(['sum', 'count'])
    if len(grp) < 2:
        return None

    last, prev = grp.iloc[-1], grp.iloc[-2]

    if last['count'] < MIN_BASE_TREND or prev['count'] < MIN_BASE_TREND:
        return None

    prev_share = prev['sum'] / prev['count']
    if prev_share == 0:
        return None

    last_share = last['sum'] / last['count']
    return ((last_share - prev_share) / prev_share) * 100

# =====================================================
# 6. CEO + RESEARCH INSIGHTS
# =====================================================
def generate_brand_insights(df, brand, themes):
    b = df[df['App_Name'] == brand]
    vol = len(b)

    if vol < MIN_VOLUME_BRAND:
        return [f"<span class='warn'>Insights suppressed — only {vol} reviews.</span>"]

    cohort_avg = df['score'].mean()
    brand_avg = b['score'].mean()
    delta = brand_avg - cohort_avg

    insights = [
        f"Market Position: CSAT is <b>{delta:+.2f}★</b> vs category average."
    ]

    pos = b[b['score'] >= 4]
    neg = b[b['score'] <= 3]

    if not pos.empty and themes:
        top_driver = pos[themes].sum().idxmax()
        pct = pos[top_driver].sum() / len(pos) * 100
        insights.append(
            f"Primary Driver: <span class='pos'>{top_driver}</span> "
            f"({pct:.1f}% of positive reviews)"
        )

    if not neg.empty and themes:
        top_barrier = neg[themes].sum().idxmax()
        pct = neg[top_barrier].sum() / len(neg) * 100
        insights.append(
            f"Primary Barrier: <span class='neg'>{top_barrier}</span> "
            f"({pct:.1f}% of complaints)"
        )

    insights.append("Implication: Removing the top barrier offers the fastest CSAT lift.")
    insights.append("Action: Assign owner, track weekly theme deltas.")

    return insights

# =====================================================
# 7. SIDEBAR FILTERS
# =====================================================
with st.sidebar:
    st.title("🎛️ Command Center")
    df_raw = load_data()

    if df_raw.empty:
        st.error("No data available.")
        st.stop()

    st.success(f"{len(df_raw):,} reviews loaded")

    # --- Core filters (CEO-safe) ---
    date_range = st.date_input(
        "Date Range",
        [df_raw['at'].min().date(), df_raw['at'].max().date()]
    )

    brands = sorted(df_raw['App_Name'].dropna().unique())
    sel_brands = st.multiselect("Brands", brands, default=brands)

    depth = st.radio("Review Depth", ["All", "Brief (<=29)", "Detailed (>=30)"])

    # ✅ NEW: Rating Filter
    rating_options = sorted(df_raw['score'].dropna().unique())
    sel_ratings = st.multiselect(
        "Rating",
        rating_options,
        default=rating_options,
        help="Filter reviews by star rating (1–5)"
    )

    st.markdown("---")

    # --- Research Mode ---
    advanced_mode = st.checkbox(
        "🔬 Advanced Analysis (Research Mode)",
        help="Enables granular filters. Insights may be suppressed if base size drops."
    )

    if advanced_mode:
        prod_cols = [c for c in ['Product_1', 'Product_2'] if c in df_raw.columns]
        all_products = set()
        for c in prod_cols:
            all_products.update(df_raw[c].dropna().unique())
        sel_products = st.multiselect("Product Type", sorted(all_products))

        if 'PL Status' in df_raw.columns:
            sel_pl = st.multiselect("PL Status", df_raw['PL Status'].dropna().unique())
        else:
            sel_pl = []

        if 'Sentiment' in df_raw.columns:
            sel_sent = st.multiselect("Sentiment", df_raw['Sentiment'].dropna().unique())
        else:
            sel_sent = []
    else:
        sel_products, sel_pl, sel_sent = [], [], []

# =====================================================
# 8. APPLY FILTERS (DEFENSIVE ORDER)
# =====================================================
df = df_raw.copy()

df = df[df['App_Name'].isin(sel_brands)]
df = df[(df['at'].dt.date >= date_range[0]) & (df['at'].dt.date <= date_range[1])]

if depth != "All":
    df = df[df['length_bucket'] == depth]

# ✅ Rating filter applied
if sel_ratings:
    df = df[df['score'].isin(sel_ratings)]

base_after_core = len(df)

if advanced_mode and sel_products:
    p_mask = pd.Series(False, index=df.index)
    for c in prod_cols:
        p_mask |= df[c].isin(sel_products)
    df = df[p_mask]

if advanced_mode and sel_pl:
    df = df[df['PL Status'].isin(sel_pl)]

if advanced_mode and sel_sent:
    df = df[df['Sentiment'].isin(sel_sent)]

if advanced_mode and len(df) < MIN_VOLUME_BRAND:
    st.warning(
        f"⚠ Filtered base reduced to {len(df)} reviews. "
        "AI insights and trends are confidence-suppressed."
    )

themes = [t for t in st.session_state.get('theme_cols', []) if t in df.columns]

# =====================================================
# 9. DASHBOARD
# =====================================================
st.title("🦅 Strategic Intelligence Platform")

tab_ai, tab_exec, tab_trends, tab_raw = st.tabs([
    "🤖 AI Analyst",
    "📊 Executive Summary",
    "📈 Trends",
    "🔍 Raw Data"
])

# ---------------- AI ANALYST ----------------
with tab_ai:
    st.info("Insights are volume-gated and confidence-aware.")

    cols = st.columns(2)
    for i, b in enumerate(sel_brands):
        with cols[i % 2]:
            insights = generate_brand_insights(df, b, themes)
            st.markdown(f"""
            <div class="ai-card">
                <div class="ai-title">{b}</div>
                {"".join([f"<div class='ai-bullet'>• {x}</div>" for x in insights])}
            </div>
            """, unsafe_allow_html=True)

# ---------------- EXEC SUMMARY ----------------
with tab_exec:
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Reviews", f"{len(df):,}")
    c2.metric("Avg CSAT", f"{df['score'].mean():.2f}★")

    advocacy = (
        (df['score'] == 5).sum() -
        (df['score'] <= 3).sum()
    ) / len(df) * 100 if len(df) else 0

    c3.metric("Advocacy Index (Proxy)", f"{advocacy:.0f}")

    st.warning(
        "⚠ Review-length and rating filters can skew sentiment distribution. "
        "Interpret trends directionally."
    )

# ---------------- TRENDS ----------------
with tab_trends:
    if themes:
        theme = st.selectbox("Theme", themes)
        mom = safe_trend_change(df, theme, 'Month')
        wow = safe_trend_change(df, theme, 'Week')

        st.metric("MoM Change", "—" if mom is None else f"{mom:+.1f}%")
        st.metric("WoW Change", "—" if wow is None else f"{wow:+.1f}%")
    else:
        st.info("No theme data available for current filters.")

# ---------------- RAW DATA ----------------
with tab_raw:
    st.dataframe(
        df[['at', 'App_Name', 'score', 'Review_Text', 'length_bucket']],
        use_container_width=True
    )
