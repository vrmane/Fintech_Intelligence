import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import requests
import time
from collections import Counter
import re
from supabase import create_client, Client
from streamlit_lottie import st_lottie

# ==========================================
# 1. PAGE CONFIG & STRATEGIC STYLING
# ==========================================
st.set_page_config(
    page_title="Strategic Intelligence Platform",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CEO-GRADE CSS ---
st.markdown("""
    <style>
        /* APP BACKGROUND */
        .stApp {
            background-color: #0b0f19; /* Obsidian Black */
            color: #e2e8f0;
        }
        
        /* CARD DESIGN */
        div[data-testid="metric-container"] {
            background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
            transition: transform 0.2s;
        }
        div[data-testid="metric-container"]:hover {
            border-color: #38bdf8;
            transform: translateY(-2px);
        }
        
        /* AI INSIGHT BOXES */
        .ai-card {
            background-color: rgba(30, 41, 59, 0.5);
            border: 1px solid #334155;
            border-left: 4px solid #38bdf8;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .ai-insight-box {
            background: rgba(16, 185, 129, 0.05);
            border: 1px solid #10b981;
            border-left: 5px solid #10b981;
            padding: 20px;
            border-radius: 8px;
            margin-top: 30px;
            margin-bottom: 25px;
        }
        .ai-header { font-weight: bold; color: #10b981; font-size: 1.1em; margin-bottom: 5px; }
        .ai-text { color: #d1fae5; font-size: 1em; line-height: 1.5; }
        
        /* TAB STYLING */
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: rgba(255, 255, 255, 0.02);
            border-radius: 8px;
            color: #94a3b8;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(56, 189, 248, 0.1);
            color: #38bdf8;
            border: 1px solid #38bdf8;
        }
        
        /* DATAFRAME */
        .stDataFrame { border: 1px solid #334155; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HELPER FUNCTIONS (LOADER & TEXT)
# ==========================================
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def get_top_words(text_series, top_n=20):
    # Simple stop words list
    stop_words = set(['the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'for', 'i', 'this', 'my', 'app', 'loan', 'money', 'very', 'good', 'bad', 'worst', 'best', 'application', 'not', 'but', 'on', 'with', 'are', 'was', 'have', 'be', 'so', 'me', 'you'])
    
    text = ' '.join(text_series.dropna().astype(str).tolist()).lower()
    # Remove punctuation/numbers
    text = re.sub(r'[^a-z\s]', '', text)
    words = [w for w in text.split() if w not in stop_words and len(w) > 2]
    
    counter = Counter(words)
    return pd.DataFrame(counter.most_common(top_n), columns=['Word', 'Count'])

# ==========================================
# 3. DATA ENGINE (SUPABASE)
# ==========================================
@st.cache_resource
def init_connection():
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except Exception as e:
        st.error("❌ Supabase Connection Failed. Check .streamlit/secrets.toml")
        st.stop()

supabase = init_connection()

@st.cache_data(ttl=600, show_spinner=False)
def load_data():
    all_rows = []
    start = 0
    batch_size = 1000
    
    while True:
        response = supabase.table("reviews").select("*").range(start, start + batch_size - 1).execute()
        rows = response.data
        if not rows: break
        all_rows.extend(rows)
        if len(rows) < batch_size: break
        start += batch_size

    if not all_rows: return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # --- CLEANING ---
    if 'Review_Date' in df.columns:
        df['at'] = pd.to_datetime(df['Review_Date'], errors='coerce')
        mask = df['at'].isna()
        if mask.any():
            df.loc[mask, 'at'] = pd.to_datetime(df.loc[mask, 'Review_Date'], format='%d-%m-%Y', errors='coerce')
        df = df.dropna(subset=['at'])
        df['Month'] = df['at'].dt.strftime('%Y-%m')
        df['Week'] = df['at'].dt.to_period('W').apply(lambda r: r.start_time)

    if 'Rating' in df.columns:
        df['score'] = pd.to_numeric(df['Rating'], errors='coerce')
    
    if 'Review_Text' in df.columns:
        df['char_count'] = df['Review_Text'].astype(str).str.len().fillna(0)
        df['length_bucket'] = df['char_count'].apply(lambda x: 'Brief (<=29)' if x <= 29 else 'Detailed (>=30)')

    # THEME EXTRACTION
    net_cols = [c for c in df.columns if str(c).startswith('[NET]')]
    clean_net_map = {}
    for col in net_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df[col] = df[col].apply(lambda x: 1 if x > 0 else 0)
        clean_name = col.replace('[NET]', '').replace('[NET] ', '').strip()
        clean_net_map[col] = clean_name
        df.rename(columns={col: clean_name}, inplace=True)
    
    st.session_state['theme_cols'] = list(clean_net_map.values())

    return df

# ==========================================
# 4. INITIAL LOADING SCREEN (INTERACTIVE)
# ==========================================
if 'data_loaded' not in st.session_state:
    loader = st.empty()
    with loader.container():
        # Sci-Fi Loader
        lottie_json = load_lottieurl("https://lottie.host/9e5c4644-841b-43c3-982d-19597143c690/w5h8o4t9zD.json") 
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if lottie_json: st_lottie(lottie_json, height=300, key="loader")
            st.markdown("<h3 style='text-align:center; color:#38bdf8;'>Establishing Secure Connection...</h3>", unsafe_allow_html=True)
            st.markdown("<p style='text-align:center; color:#94a3b8;'>Fetching encrypted intelligence from Supabase node.</p>", unsafe_allow_html=True)
    
    # Load Data
    df_raw = load_data()
    st.session_state['df_raw'] = df_raw
    st.session_state['data_loaded'] = True
    
    # Clear Loader
    time.sleep(1) # Visual polish
    loader.empty()
else:
    df_raw = st.session_state['df_raw']

if df_raw.empty:
    st.error("Connection Successful but No Data Found.")
    st.stop()

# ==========================================
# 5. ANALYTICS LOGIC
# ==========================================
def dark_chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="#94a3b8"),
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified"
    )
    return fig

def calculate_growth_matrix(df, theme_cols, time_col, brands):
    valid_themes = [t for t in theme_cols if t in df.columns]
    if not valid_themes: return pd.DataFrame()
    
    top_themes = df[valid_themes].sum().sort_values(ascending=False).head(10).index.tolist()
    matrix = {}
    
    for brand in brands:
        b_df = df[df['App_Name'] == brand].sort_values(time_col)
        periods = b_df[time_col].unique()
        if len(periods) < 2:
            matrix[brand] = [0.0] * len(top_themes)
            continue
            
        last = periods[-1]
        prev = periods[-2]
        
        last_df = b_df[b_df[time_col] == last]
        last_vol = len(last_df)
        
        prev_df = b_df[b_df[time_col] == prev]
        prev_vol = len(prev_df)
        
        growth = []
        for t in top_themes:
            if prev_vol == 0:
                g = 0.0
            else:
                share_last = last_df[t].sum() / last_vol if last_vol else 0
                share_prev = prev_df[t].sum() / prev_vol
                if share_prev == 0:
                    g = 0.0
                else:
                    g = ((share_last - share_prev) / share_prev) * 100
            growth.append(g)
        
        matrix[brand] = growth
        
    return pd.DataFrame(matrix, index=top_themes)

def get_brand_insights(df, brand, theme_cols):
    b_df = df[df['App_Name'] == brand]
    if b_df.empty: return ["No data available."]
    
    insights = []
    
    cohort_avg = df['score'].mean()
    brand_avg = b_df['score'].mean()
    diff = brand_avg - cohort_avg
    pos_text = "outperforming" if diff > 0 else "lagging behind"
    insights.append(f"**Market Position:** {brand} is <span class='highlight-neu'>{pos_text}</span> the cohort average by **{abs(diff):.2f} stars**.")
    
    pos_df = b_df[b_df['score'] >= 4]
    if not pos_df.empty and theme_cols:
        valid = [t for t in theme_cols if t in pos_df.columns]
        if valid:
            top_d = pos_df[valid].sum().idxmax()
            pct = (pos_df[top_d].sum() / len(pos_df)) * 100
            insights.append(f"**Growth Engine:** Positive sentiment is anchored on <span class='highlight-pos'>'{top_d}'</span>, appearing in **{pct:.1f}%** of 4-5★ reviews.")
            
    neg_df = b_df[b_df['score'] <= 3]
    if not neg_df.empty and theme_cols:
        valid = [t for t in theme_cols if t in neg_df.columns]
        if valid:
            top_b = neg_df[valid].sum().idxmax()
            pct = (neg_df[top_b].sum() / len(neg_df)) * 100
            insights.append(f"**Critical Risk:** The dominant friction point is <span class='highlight-neg'>'{top_b}'</span>, flagged in **{pct:.1f}%** of complaints.")
    
    share = (len(b_df) / len(df)) * 100
    insights.append(f"**Voice Share:** Commands **{share:.1f}%** of the filtered review volume.")
    
    return insights

def generate_global_summary(df, theme_cols, current_filters):
    if df.empty: return "No data available."
    
    avg_rating = df['score'].mean()
    vol = len(df)
    
    brand_perf = df.groupby('App_Name')['score'].mean().sort_values(ascending=False)
    leader = brand_perf.index[0]
    trailer = brand_perf.index[-1]
    
    pos_df = df[df['score'] >= 4]
    top_driver = "N/A"
    if not pos_df.empty and theme_cols:
        valid_themes = [t for t in theme_cols if t in pos_df.columns]
        if valid_themes:
            top_driver = pos_df[valid_themes].sum().idxmax()
            
    neg_df = df[df['score'] <= 3]
    top_barrier = "N/A"
    if not neg_df.empty and theme_cols:
        valid_themes = [t for t in theme_cols if t in neg_df.columns]
        if valid_themes:
            top_barrier = neg_df[valid_themes].sum().idxmax()
            
    filter_text = f"Filtered by: {current_filters}" if current_filters else "All Data"
    
    html = f"""
    <div class='ai-insight-box'>
        <div class='ai-header'>🤖 AI Analyst: Global Strategic Brief</div>
        <div class='ai-text'>
            • <b>Context ({filter_text}):</b> Analyzing <b>{vol:,}</b> reviews. The category sentiment is <b>{avg_rating:.2f} ⭐</b>.<br>
            • <b>Performance:</b> <b>{leader}</b> is currently outperforming, while <b>{trailer}</b> trails in satisfaction.<br>
            • <b>Key Drivers:</b> Positive sentiment is heavily driven by <b>'{top_driver}'</b>.<br>
            • <b>Critical Risks:</b> The most dominant complaint (Barrier) is <b>'{top_barrier}'</b>.<br>
            • <b>Recommendation:</b> If <b>{top_barrier}</b> relates to product flow, prioritize UI/UX fixes immediately.
        </div>
    </div>
    """
    return html

# ==========================================
# 6. SIDEBAR & FILTERS
# ==========================================
with st.sidebar:
    st.title("🎛️ Command Center")
    st.success(f"🟢 Live: {len(df_raw):,} Rows")
    st.markdown("---")
    
    min_d, max_d = df_raw['at'].min().date(), df_raw['at'].max().date()
    date_range = st.date_input("Period", [min_d, max_d], min_value=min_d, max_value=max_d)
    
    all_brands = sorted(df_raw['App_Name'].dropna().unique())
    sel_brands = st.multiselect("Brands", all_brands, default=all_brands)
    
    st.markdown("### 📊 Metrics")
    sel_ratings = st.multiselect("Ratings", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])

    st.markdown("### 📝 Review Depth")
    len_filter = st.radio(
        "Character Count", 
        ["All Reviews", "Brief (<=29)", "Detailed (>=30)"],
        index=0
    )
    
    prod_cols = [c for c in ['Product_1','Product_2'] if c in df_raw.columns]
    all_prods = set()
    for c in prod_cols: all_prods.update(df_raw[c].dropna().unique())
    sel_prods = st.multiselect("Product Type", sorted(list(all_prods)))
    
    if 'PL Status' in df_raw.columns:
        sel_pl = st.multiselect("PL Status", df_raw['PL Status'].dropna().unique())
    else: sel_pl = []
    
    if 'Sentiment' in df_raw.columns:
        sel_sent = st.multiselect("Sentiment", df_raw['Sentiment'].unique())
    else: sel_sent = []

# --- APPLY FILTERING ---
if len(date_range) == 2:
    mask = (df_raw['at'].dt.date >= date_range[0]) & (df_raw['at'].dt.date <= date_range[1])
else:
    mask = [True] * len(df_raw)

mask &= df_raw['App_Name'].isin(sel_brands)
mask &= df_raw['score'].isin(sel_ratings)

if len_filter == "Brief (<=29)": mask &= (df_raw['length_bucket'] == 'Brief (<=29 chars)')
elif len_filter == "Detailed (>=30)": mask &= (df_raw['length_bucket'] == 'Detailed (>=30 chars)')

if sel_prods:
    p_mask = pd.Series(False, index=df_raw.index)
    for c in prod_cols: p_mask |= df_raw[c].isin(sel_prods)
    mask &= p_mask
    
if sel_pl: mask &= (df_raw['PL Status'].isin(sel_pl))
if sel_sent: mask &= (df_raw['Sentiment'].isin(sel_sent))

df = df_raw[mask].copy()
theme_cols = st.session_state.get('theme_cols', [])

# ==========================================
# 7. DASHBOARD LAYOUT
# ==========================================

st.title("🦅 Strategic Intelligence Platform")

tab_ai, tab_exec, tab_drivers, tab_compare, tab_trends, tab_text, tab_raw = st.tabs([
    "🤖 AI Analyst", "📊 Boardroom", "🚀 Drivers & Barriers", "⚔️ Head-to-Head", "📈 Trends", "🔡 Text Analytics", "🔍 Data"
])

# === TAB 1: AI ANALYST ===
with tab_ai:
    st.markdown("### 🤖 Brand Strategic Briefs")
    cols = st.columns(2)
    for i, brand in enumerate(sel_brands):
        with cols[i % 2]:
            insights = get_brand_insights(df, brand, theme_cols)
            st.markdown(f"""
            <div class="ai-card">
                <div style="font-size:1.2em; font-weight:bold; margin-bottom:10px;">🔹 {brand}</div>
                {"".join([f'<div style="margin-bottom:5px;">✦ {txt}</div>' for txt in insights])}
            </div>
            """, unsafe_allow_html=True)
            
    # GLOBAL SUMMARY AT BOTTOM
    filter_desc = f"{len_filter}, {len(sel_brands)} Brands"
    st.markdown(generate_global_summary(df, theme_cols, filter_desc), unsafe_allow_html=True)

# === TAB 2: BOARDROOM ===
with tab_exec:
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Total Volume", f"{len(df):,}")
    with k2: st.metric("Avg Rating", f"{df['score'].mean():.2f} ⭐")
    
    prom = len(df[df['score']==5])
    det = len(df[df['score']<=3])
    nps = ((prom - det) / len(df) * 100) if len(df) > 0 else 0
    with k3: st.metric("NPS Proxy", f"{nps:.0f}")
    
    risk = (len(df[df['score']==1]) / len(df) * 100) if len(df) > 0 else 0
    with k4: st.metric("Critical Risk", f"{risk:.1f}%", delta="1-Star %", delta_color="inverse")
    
    st.markdown("---")
    
    bs = df.groupby('App_Name').agg(
        Vol=('score','count'), CSAT=('score','mean'),
        One=('score', lambda x: (x==1).sum()), Five=('score', lambda x: (x==5).sum())
    ).reset_index()
    bs['NPS'] = ((bs['Five'] - bs['One'])/bs['Vol']*100).round(1)
    
    fig = px.scatter(bs, x="CSAT", y="NPS", size="Vol", color="App_Name", text="App_Name", 
                     title="Brand Matrix (Size=Volume)", height=500)
    fig.update_traces(textposition='top center')
    st.plotly_chart(dark_chart(fig), use_container_width=True)

# === TAB 3: DRIVERS & BARRIERS ===
with tab_drivers:
    st.markdown("### 🚦 Strategic Drivers & Barriers")
    
    pos_bases = df[df['score']>=4].groupby('App_Name').size()
    neg_bases = df[df['score']<=3].groupby('App_Name').size()
    
    c_h1, c_h2 = st.columns(2)
    with c_h1:
        st.markdown("**🚀 Drivers (Positive Theme Intensity)**")
        if not pos_bases.empty and theme_cols:
            valid = [t for t in theme_cols if t in df.columns]
            p_data = df[df['score']>=4].groupby('App_Name')[valid].sum().T
            p_pct = p_data.div(pos_bases, axis=1) * 100
            p_pct['Avg'] = p_pct.mean(axis=1)
            p_pct = p_pct.sort_values('Avg', ascending=False).head(10).drop(columns=['Avg'])
            st.plotly_chart(px.imshow(p_pct, text_auto='.1f', aspect="auto", color_continuous_scale='Greens'), use_container_width=True)
            
    with c_h2:
        st.markdown("**🛑 Barriers (Negative Theme Intensity)**")
        if not neg_bases.empty and theme_cols:
            valid = [t for t in theme_cols if t in df.columns]
            n_data = df[df['score']<=3].groupby('App_Name')[valid].sum().T
            n_pct = n_data.div(neg_bases, axis=1) * 100
            n_pct['Avg'] = n_pct.mean(axis=1)
            n_pct = n_pct.sort_values('Avg', ascending=False).head(10).drop(columns=['Avg'])
            st.plotly_chart(px.imshow(n_pct, text_auto='.1f', aspect="auto", color_continuous_scale='Reds'), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🧬 Theme Evolution (Multi-Theme Comparison)")
    
    evo_type = st.radio("Trend Category", ["Drivers (Positive)", "Barriers (Negative)"], horizontal=True, key="evo_type")
    trend_src = df[df['score'] >= 4] if "Positive" in evo_type else df[df['score'] <= 3]

    if not trend_src.empty and theme_cols:
        top_opts = trend_src[theme_cols].sum().sort_values(ascending=False).head(20).index.tolist()
        sel_themes = st.multiselect("Compare Themes (Aggregated)", top_opts, default=top_opts[:3])
        
        if sel_themes:
            t_view = st.radio("Time Grain", ["Monthly", "Weekly"], horizontal=True, key="evo_time")
            t_col = 'Month' if t_view == "Monthly" else 'Week'
            
            trend_data = []
            grouped = trend_src.groupby(t_col)
            for t_val, group in grouped:
                base_vol = len(group)
                if base_vol == 0: continue
                for theme in sel_themes:
                    if theme in group.columns:
                        count = group[theme].sum()
                        pct = (count / base_vol) * 100
                        trend_data.append({t_col: str(t_val), "Theme": theme, "Prevalence": pct})
            
            if trend_data:
                plot_df = pd.DataFrame(trend_data).sort_values(t_col)
                fig_evo = px.line(plot_df, x=t_col, y="Prevalence", color="Theme", markers=True, title=f"Evolution of Selected {evo_type}", labels={"Prevalence": "% of Reviews"})
                st.plotly_chart(dark_chart(fig_evo), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📈 Theme Momentum (Growth Matrix)")
    c_t1, c_t2 = st.columns(2)
    
    with c_t1:
        st.markdown("**📅 MoM Growth %**")
        mom_matrix = calculate_growth_matrix(trend_src, theme_cols, 'Month', sel_brands)
        if not mom_matrix.empty:
            st.plotly_chart(px.imshow(mom_matrix, text_auto='.1f', aspect="auto", color_continuous_scale='RdBu', color_continuous_midpoint=0), use_container_width=True)
            
    with c_t2:
        st.markdown("**⚡ WoW Growth %**")
        wow_matrix = calculate_growth_matrix(trend_src, theme_cols, 'Week', sel_brands)
        if not wow_matrix.empty:
            st.plotly_chart(px.imshow(wow_matrix, text_auto='.1f', aspect="auto", color_continuous_scale='RdBu', color_continuous_midpoint=0), use_container_width=True)

    # 4. DEEP DIVE TABLE
    st.markdown("#### 🔍 Brand Deep Dive Table")
    target_brand = st.selectbox("Select Brand", sel_brands)
    
    if target_brand and theme_cols:
        b_df = df[df['App_Name'] == target_brand]
        
        def calculate_trend_val(sub_df, group_col, time_col):
            d_sorted = sub_df.sort_values(time_col)
            periods = d_sorted[time_col].unique()
            if len(periods) < 2: return 0.0
            last, prev = periods[-1], periods[-2]
            v_last = d_sorted[d_sorted[time_col]==last][group_col].sum()
            v_prev = d_sorted[d_sorted[time_col]==prev][group_col].sum()
            return ((v_last - v_prev) / v_prev * 100) if v_prev > 0 else 0

        def build_table(sub_df, base, themes):
            if sub_df.empty or base == 0: return pd.DataFrame()
            valid = [t for t in themes if t in sub_df.columns]
            counts = sub_df[valid].sum().sort_values(ascending=False).head(10)
            data = []
            for theme, count in counts.items():
                if count == 0: continue
                mom = calculate_trend_val(sub_df, theme, 'Month')
                wow = calculate_trend_val(sub_df, theme, 'Week')
                data.append({
                    "Theme": theme, "Count": int(count), "% of Base": f"{(count/base)*100:.1f}%",
                    "MoM Vol": f"{mom:+.1f}%", "WoW Vol": f"{wow:+.1f}%"
                })
            return pd.DataFrame(data)

        c1, c2 = st.columns(2)
        with c1:
            pos_df = b_df[b_df['score'] >= 4]
            st.markdown(f"**🚀 Drivers (Base: {len(pos_df):,})**")
            dt = build_table(pos_df, len(pos_df), theme_cols)
            if not dt.empty: st.dataframe(dt, hide_index=True, use_container_width=True)
            else: st.warning("No Data")

        with c2:
            neg_df = b_df[b_df['score'] <= 3]
            st.markdown(f"**🛑 Barriers (Base: {len(neg_df):,})**")
            bt = build_table(neg_df, len(neg_df), theme_cols)
            if not bt.empty: st.dataframe(bt, hide_index=True, use_container_width=True)
            else: st.warning("No Data")

# === TAB 4: HEAD TO HEAD ===
with tab_compare:
    c1, c2 = st.columns(2)
    with c1: b1 = st.selectbox("Brand A", sel_brands, index=0 if sel_brands else None)
    with c2: b2 = st.selectbox("Brand B", [b for b in sel_brands if b!=b1], index=0 if len(sel_brands)>1 else None)
    
    if b1 and b2:
        def get_stats(b):
            d = df[df['App_Name']==b]
            if d.empty: return ["0", "0", "0"]
            v = len(d)
            s = d['score'].mean()
            n = ((len(d[d['score']==5]) - len(d[d['score']<=3]))/v)*100
            return [f"{s:.2f}", f"{n:.0f}", f"{v:,}"]
            
        comp = pd.DataFrame({"Metric": ["CSAT", "NPS", "Vol"], b1: get_stats(b1), b2: get_stats(b2)}).set_index("Metric")
        st.dataframe(comp, use_container_width=True)
        
        if theme_cols:
            d1 = df[(df['App_Name']==b1) & (df['score']>=4)][theme_cols].sum().nlargest(5).index.tolist()
            d2 = df[(df['App_Name']==b2) & (df['score']>=4)][theme_cols].sum().nlargest(5).index.tolist()
            common = list(set(d1 + d2))
            if common:
                def gp(b):
                    d = df[df['App_Name']==b]
                    return [(d[t].sum()/len(d)*100) if not d.empty else 0 for t in common]
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=gp(b1), theta=common, fill='toself', name=b1))
                fig.add_trace(go.Scatterpolar(r=gp(b2), theta=common, fill='toself', name=b2))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True)), template="plotly_dark", title="Overlap")
                st.plotly_chart(fig, use_container_width=True)

# === TAB 5: TRENDS ===
with tab_trends:
    view = st.radio("Time View", ["Monthly", "Weekly"], horizontal=True)
    t_col = 'Month' if view == "Monthly" else 'Week'
    if 'at' in df.columns:
        trend = df.groupby([t_col, 'App_Name'])['score'].agg(['mean', 'count']).reset_index()
        if view == "Weekly": trend['Week'] = trend['Week'].astype(str)
        
        st.plotly_chart(dark_chart(px.line(trend, x=t_col, y='mean', color='App_Name', markers=True, title="CSAT")), use_container_width=True)
        st.plotly_chart(dark_chart(px.bar(trend, x=t_col, y='count', color='App_Name', title="Volume")), use_container_width=True)

# === TAB 6: TEXT ANALYTICS (NEW) ===
with tab_text:
    st.markdown("### 🔡 Deep Text Analytics")
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("**Word Frequency (Top 20)**")
        txt_type = st.radio("Sentiment Source", ["Positive (4-5★)", "Negative (1-3★)"], horizontal=True)
        if "Positive" in txt_type:
            txt_df = df[df['score']>=4]
        else:
            txt_df = df[df['score']<=3]
            
        if not txt_df.empty:
            words_df = get_top_words(txt_df['Review_Text'])
            st.dataframe(words_df, height=300, use_container_width=True)
            
            # Simple Bar
            fig = px.bar(words_df, x='Count', y='Word', orientation='h', title=f"Top Words in {txt_type}")
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(dark_chart(fig), use_container_width=True)
    
    with c2:
        st.markdown("**Review Length Distribution**")
        fig = px.histogram(df, x='char_count', color='App_Name', nbins=50, title="Character Count Distribution")
        st.plotly_chart(dark_chart(fig), use_container_width=True)

# === TAB 7: DATA ===
with tab_raw:
    st.dataframe(df[['at', 'App_Name', 'score', 'Review_Text', 'length_bucket']], use_container_width=True)
