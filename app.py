import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from supabase import create_client, Client

# ==========================================
# 1. PAGE CONFIG & STRATEGIC STYLING
# ==========================================
st.set_page_config(
    page_title="Strategic Intelligence Platform",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CEO-GRADE CSS ---
st.markdown("""
    <style>
        .stApp { background-color: #0b0f19; color: #e2e8f0; }
        
        /* METRIC CARDS */
        div[data-testid="metric-container"] {
            background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
        }
        
        /* AI INSIGHT BOX */
        .ai-insight-box {
            background: rgba(16, 185, 129, 0.05);
            border: 1px solid #10b981;
            border-left: 5px solid #10b981;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 25px;
        }
        .ai-header { font-weight: bold; color: #10b981; font-size: 1.1em; margin-bottom: 5px; }
        .ai-text { color: #d1fae5; font-size: 1em; line-height: 1.5; }

        /* TABS */
        .stTabs [data-baseweb="tab-list"] { gap: 20px; }
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(255, 255, 255, 0.02);
            border-radius: 8px;
            color: #94a3b8;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(56, 189, 248, 0.1);
            color: #38bdf8;
            border: 1px solid #38bdf8;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LIVE SUPABASE DATA ENGINE
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

@st.cache_data(ttl=600)
def load_data():
    all_rows = []
    start, batch_size = 0, 1000
    
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
        # Create Cut Points for Filtering
        df['length_bucket'] = df['char_count'].apply(lambda x: 'Brief (<=29)' if x <= 29 else 'Detailed (>=30)')

    # THEME EXTRACTION (Drivers/Barriers)
    net_cols = [c for c in df.columns if str(c).startswith('[NET]')]
    clean_net_map = {}
    for col in net_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df[col] = df[col].apply(lambda x: 1 if x > 0 else 0) # Binary
        clean_name = col.replace('[NET]', '').replace('[NET] ', '').strip()
        clean_net_map[col] = clean_name
        df.rename(columns={col: clean_name}, inplace=True)
    
    st.session_state['theme_cols'] = list(clean_net_map.values())

    return df

# ==========================================
# 3. AI INSIGHT ENGINE
# ==========================================
def generate_sharp_insights(df, theme_cols):
    if df.empty: return "No data available for analysis."
    
    avg_rating = df['score'].mean()
    total_reviews = len(df)
    
    # 1. Sentiment Leader
    brand_stats = df.groupby('App_Name')['score'].mean()
    leader = brand_stats.idxmax()
    laggard = brand_stats.idxmin()
    
    # 2. Top Driver (Overall)
    pos_df = df[df['score'] >= 4]
    top_driver = "N/A"
    if not pos_df.empty and theme_cols:
        valid_themes = [t for t in theme_cols if t in pos_df.columns]
        top_driver = pos_df[valid_themes].sum().idxmax()
        
    # 3. Top Barrier (Overall)
    neg_df = df[df['score'] <= 3]
    top_barrier = "N/A"
    if not neg_df.empty and theme_cols:
        valid_themes = [t for t in theme_cols if t in neg_df.columns]
        top_barrier = neg_df[valid_themes].sum().idxmax()

    insight_text = f"""
    <div class='ai-insight-box'>
        <div class='ai-header'>🤖 AI Analyst: Strategic Brief</div>
        <div class='ai-text'>
            • <b>Market Pulse:</b> Analyzing <b>{total_reviews:,}</b> reviews. The category average is <b>{avg_rating:.2f} ⭐</b>.<br>
            • <b>Leaderboard:</b> <b>{leader}</b> is currently leading satisfaction ({brand_stats[leader]:.2f} ⭐), while <b>{laggard}</b> represents the highest churn risk.<br>
            • <b>The "Why":</b> The primary engine of growth is <b>'{top_driver}'</b> (Top Driver), whereas <b>'{top_barrier}'</b> is the #1 friction point across the board.<br>
            • <b>Action:</b> Focus marketing on <i>{top_driver}</i> and prioritize product fixes for <i>{top_barrier}</i>.
        </div>
    </div>
    """
    return insight_text

# ==========================================
# 4. SIDEBAR & FILTERS
# ==========================================
with st.sidebar:
    st.title("🎛️ Command Center")
    with st.spinner("Connecting..."):
        df_raw = load_data()
    
    if df_raw.empty:
        st.error("Database Empty.")
        st.stop()

    st.success(f"🟢 Live: {len(df_raw):,} Rows")
    st.markdown("---")
    
    # --- FILTERS ---
    min_d, max_d = df_raw['at'].min().date(), df_raw['at'].max().date()
    date_range = st.date_input("Period", [min_d, max_d], min_value=min_d, max_value=max_d)
    
    all_brands = sorted(df_raw['App_Name'].dropna().unique())
    sel_brands = st.multiselect("Brands", all_brands, default=all_brands)
    
    # NEW: Review Length Filter
    st.markdown("### 📝 Review Depth")
    len_filter = st.radio(
        "Character Count", 
        ["All Reviews", "Brief (<=29 chars)", "Detailed (>=30 chars)"],
        index=0
    )
    
    # Product Filter
    prod_cols = [c for c in ['Product_1','Product_2'] if c in df_raw.columns]
    all_prods = set()
    for c in prod_cols: all_prods.update(df_raw[c].dropna().unique())
    sel_prods = st.multiselect("Product Type", sorted(list(all_prods)))

# --- APPLY FILTERING ---
if len(date_range) == 2:
    mask = (df_raw['at'].dt.date >= date_range[0]) & (df_raw['at'].dt.date <= date_range[1])
else:
    mask = [True] * len(df_raw)

mask &= df_raw['App_Name'].isin(sel_brands)

if len_filter == "Brief (<=29 chars)":
    mask &= (df_raw['length_bucket'] == 'Brief (<=29)')
elif len_filter == "Detailed (>=30 chars)":
    mask &= (df_raw['length_bucket'] == 'Detailed (>=30)')

if sel_prods:
    p_mask = pd.Series(False, index=df_raw.index)
    for c in prod_cols: p_mask |= df_raw[c].isin(sel_prods)
    mask &= p_mask

df = df_raw[mask].copy()
theme_cols = st.session_state.get('theme_cols', [])

# ==========================================
# 5. DASHBOARD LAYOUT
# ==========================================

st.title("🦅 Strategic Intelligence Platform")

# --- AI INSIGHT SECTION ---
if not df.empty:
    st.markdown(generate_sharp_insights(df, theme_cols), unsafe_allow_html=True)

# --- TABS ---
tab_exec, tab_drivers, tab_compare, tab_trends, tab_raw = st.tabs([
    "📊 Boardroom Summary", 
    "🚀 Top 10 Drivers & Barriers", 
    "⚔️ Head-to-Head", 
    "📈 Trends (MoM/WoW)", 
    "🔍 Data Explorer"
])

# === TAB 1: SUMMARY ===
with tab_exec:
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Total Volume", f"{len(df):,}")
    with k2: st.metric("Avg Rating", f"{df['score'].mean():.2f} ⭐")
    with k3:
        prom = len(df[df['score']==5])
        det = len(df[df['score']<=3])
        nps = ((prom - det) / len(df)) * 100 if len(df)>0 else 0
        st.metric("NPS Proxy", f"{nps:.0f}")
    with k4:
        risk = (len(df[df['score']==1]) / len(df)) * 100 if len(df)>0 else 0
        st.metric("Critical Risk", f"{risk:.1f}%", delta="1-Star %", delta_color="inverse")

    st.markdown("---")
    
    # Scatter Matrix
    brand_stats = df.groupby('App_Name').agg(
        Volume=('score', 'count'),
        CSAT=('score', 'mean'),
        One_Star=('score', lambda x: (x==1).sum()),
        Five_Star=('score', lambda x: (x==5).sum())
    ).reset_index()
    brand_stats['NPS'] = ((brand_stats['Five_Star'] - brand_stats['One_Star']) / brand_stats['Volume'] * 100).round(1)
    
    fig = px.scatter(
        brand_stats, x="CSAT", y="NPS", size="Volume", color="App_Name",
        hover_name="App_Name", text="App_Name", title="Brand Performance Matrix",
        color_discrete_sequence=px.colors.qualitative.Bold, height=500
    )
    fig.update_traces(textposition='top center')
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

# === TAB 2: DRIVERS & BARRIERS (CORE REQUEST) ===
with tab_drivers:
    st.markdown("### 🚦 The Strategic Landscape: Drivers vs. Barriers")
    st.info("Aggregated View at **NET x Brand Level**. Top 10 themes driving 4-5★ (Drivers) and 1-3★ (Barriers).")
    
    c1, c2 = st.columns(2)
    
    # 1. DRIVERS (4-5 Stars)
    with c1:
        st.markdown("#### 🚀 Top 10 Drivers (Positive)")
        df_pos = df[df['score'].isin([4, 5])]
        
        if not df_pos.empty and theme_cols:
            # Group by Brand, sum all theme columns
            valid_themes = [t for t in theme_cols if t in df_pos.columns]
            driver_data = df_pos.groupby('App_Name')[valid_themes].sum().T
            
            # Identify Top 10 Themes globally across selected brands
            driver_data['Total'] = driver_data.sum(axis=1)
            driver_data = driver_data.sort_values('Total', ascending=False).head(10).drop(columns=['Total'])
            
            fig_d = px.imshow(driver_data, text_auto=True, aspect="auto", 
                              color_continuous_scale='Greens', title="Top 10 Drivers (Volume)")
            fig_d.update_layout(template="plotly_dark")
            st.plotly_chart(fig_d, use_container_width=True)
        else:
            st.warning("Insufficient Positive Data")

    # 2. BARRIERS (1-3 Stars)
    with c2:
        st.markdown("#### 🛑 Top 10 Barriers (Negative)")
        df_neg = df[df['score'].isin([1, 2, 3])]
        
        if not df_neg.empty and theme_cols:
            valid_themes = [t for t in theme_cols if t in df_neg.columns]
            barrier_data = df_neg.groupby('App_Name')[valid_themes].sum().T
            
            # Identify Top 10 Barriers globally
            barrier_data['Total'] = barrier_data.sum(axis=1)
            barrier_data = barrier_data.sort_values('Total', ascending=False).head(10).drop(columns=['Total'])
            
            fig_b = px.imshow(barrier_data, text_auto=True, aspect="auto", 
                              color_continuous_scale='Reds', title="Top 10 Barriers (Volume)")
            fig_b.update_layout(template="plotly_dark")
            st.plotly_chart(fig_b, use_container_width=True)
        else:
            st.warning("Insufficient Negative Data")

# === TAB 3: HEAD TO HEAD ===
with tab_compare:
    st.markdown("### ⚔️ Competitive Battleground")
    c1, c2 = st.columns(2)
    with c1: b1 = st.selectbox("Brand A", sel_brands, index=0 if sel_brands else None)
    with c2: 
        opts = [b for b in sel_brands if b != b1]
        b2 = st.selectbox("Brand B", opts, index=0 if opts else None)
    
    if b1 and b2:
        # Comparison Data
        def get_stats(b):
            d = df[df['App_Name']==b]
            v = len(d)
            s = d['score'].mean() if v else 0
            n = (((len(d[d['score']==5]) - len(d[d['score']<=3])) / v) * 100) if v else 0
            return [f"{s:.2f}", f"{n:.0f}", f"{v:,}"]
            
        comp_df = pd.DataFrame({
            "Metric": ["CSAT", "NPS Proxy", "Volume"],
            b1: get_stats(b1),
            b2: get_stats(b2)
        }).set_index("Metric")
        
        st.dataframe(comp_df, use_container_width=True)
        
        # Radar Chart
        if theme_cols:
            # Find common top drivers
            d1 = df[(df['App_Name']==b1) & (df['score']>=4)][theme_cols].sum().sort_values(ascending=False).head(5).index.tolist()
            d2 = df[(df['App_Name']==b2) & (df['score']>=4)][theme_cols].sum().sort_values(ascending=False).head(5).index.tolist()
            common = list(set(d1 + d2))
            
            def get_pct(b, themes):
                d = df[df['App_Name']==b]
                if d.empty: return [0]*len(themes)
                return [(d[t].sum()/len(d)*100) for t in themes]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=get_pct(b1, common), theta=common, fill='toself', name=b1))
            fig.add_trace(go.Scatterpolar(r=get_pct(b2, common), theta=common, fill='toself', name=b2))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True)), template="plotly_dark", title="Theme Overlap (%)")
            st.plotly_chart(fig, use_container_width=True)

# === TAB 4: TRENDS ===
with tab_trends:
    st.markdown("### 📈 Time Series Intelligence")
    view = st.radio("View", ["Monthly (MoM)", "Weekly (WoW)"], horizontal=True)
    time_col = 'Month' if "Monthly" in view else 'Week'
    
    if 'at' in df.columns:
        trend = df.groupby([time_col, 'App_Name'])['score'].agg(['mean', 'count']).reset_index()
        if view == "Weekly (WoW)": trend['Week'] = trend['Week'].astype(str)
        
        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(trend, x=time_col, y='mean', color='App_Name', markers=True, title="CSAT Trend")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(trend, x=time_col, y='count', color='App_Name', title="Volume Trend")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

# === TAB 5: EXPLORER ===
with tab_raw:
    st.markdown("### 🔍 Raw Data Stream")
    st.dataframe(df[['at', 'App_Name', 'score', 'Review_Text', 'length_bucket']], use_container_width=True)
