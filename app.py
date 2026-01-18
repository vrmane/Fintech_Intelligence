import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from supabase import create_client, Client

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
        
        /* TEXT HIERARCHY */
        h1 {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(to right, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        h2, h3 {
            color: #f1f5f9;
            font-weight: 600;
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

@st.cache_data(ttl=600)  # Refresh cache every 10 mins
def load_data():
    all_rows = []
    start = 0
    batch_size = 1000
    
    # Pagination Loop
    while True:
        response = supabase.table("reviews").select("*").range(start, start + batch_size - 1).execute()
        rows = response.data
        if not rows: break
        all_rows.extend(rows)
        if len(rows) < batch_size: break
        start += batch_size

    if not all_rows: return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # --- ROBUST DATA CLEANING ---
    if 'Review_Date' in df.columns:
        df['at'] = pd.to_datetime(df['Review_Date'], errors='coerce')
        # Fallback format
        mask = df['at'].isna()
        if mask.any():
            df.loc[mask, 'at'] = pd.to_datetime(df.loc[mask, 'Review_Date'], format='%d-%m-%Y', errors='coerce')
        
        df = df.dropna(subset=['at'])
        df['Month'] = df['at'].dt.strftime('%Y-%m')
        # Weekly Period (Start of Week)
        df['Week'] = df['at'].dt.to_period('W').apply(lambda r: r.start_time)

    if 'Rating' in df.columns:
        df['score'] = pd.to_numeric(df['Rating'], errors='coerce')
    
    if 'Review_Text' in df.columns:
        df['char_count'] = df['Review_Text'].astype(str).str.len().fillna(0)
        # 2. Add Character Filter Logic
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
# 3. ANALYTICS LOGIC (TRENDS & INSIGHTS)
# ==========================================
def calculate_trend_change(df, metric_col, group_col, time_col, target_group):
    # Helper to calculate MoM/WoW change for a specific group (e.g., Theme)
    # Sort by time
    df_sorted = df.sort_values(time_col)
    
    # Get last two periods
    periods = df_sorted[time_col].unique()
    if len(periods) < 2: return 0.0
    
    last_period = periods[-1]
    prev_period = periods[-2]
    
    # Get metric value for target group in last vs prev
    def get_val(p):
        d = df_sorted[df_sorted[time_col] == p]
        return d[metric_col].sum() if metric_col in d.columns else len(d)

    # For Themes, we need to filter first? No, df passed here is already filtered or aggregated?
    # Let's assume df is the raw data with the theme column (0/1)
    
    val_last = df_sorted[df_sorted[time_col] == last_period][group_col].sum()
    val_prev = df_sorted[df_sorted[time_col] == prev_period][group_col].sum()
    
    # Normalize by total volume in that period to get "Share" change (more accurate than raw volume change)
    base_last = len(df_sorted[df_sorted[time_col] == last_period])
    base_prev = len(df_sorted[df_sorted[time_col] == prev_period])
    
    if base_prev == 0: return 0.0
    
    share_last = val_last / base_last
    share_prev = val_prev / base_prev
    
    # Return percentage point change or relative growth? 
    # Relative growth of share is standard.
    if share_prev == 0: return 0.0
    return ((share_last - share_prev) / share_prev) * 100

def generate_ai_insight(df, theme_cols, current_filters):
    if df.empty: return "No data available."
    
    avg_rating = df['score'].mean()
    vol = len(df)
    
    # Sentiment Leader
    brand_perf = df.groupby('App_Name')['score'].mean().sort_values(ascending=False)
    leader = brand_perf.index[0]
    trailer = brand_perf.index[-1]
    
    # Top Driver (Global)
    pos_df = df[df['score'] >= 4]
    top_driver = "N/A"
    if not pos_df.empty and theme_cols:
        # Sum themes present in columns
        valid_themes = [t for t in theme_cols if t in pos_df.columns]
        if valid_themes:
            top_driver = pos_df[valid_themes].sum().idxmax()
            
    # Top Barrier (Global)
    neg_df = df[df['score'] <= 3]
    top_barrier = "N/A"
    if not neg_df.empty and theme_cols:
        valid_themes = [t for t in theme_cols if t in neg_df.columns]
        if valid_themes:
            top_barrier = neg_df[valid_themes].sum().idxmax()
            
    filter_text = f"Filtered by: {current_filters}" if current_filters else "All Data"
    
    html = f"""
    <div class='ai-insight-box'>
        <div class='ai-header'>🤖 AI Analyst: Live Strategic Brief</div>
        <div class='ai-text'>
            • <b>Context ({filter_text}):</b> Analyzing <b>{vol:,}</b> reviews. The average sentiment is <b>{avg_rating:.2f} ⭐</b>.<br>
            • <b>Performance:</b> <b>{leader}</b> is currently outperforming the cohort, while <b>{trailer}</b> trails in satisfaction.<br>
            • <b>Key Drivers:</b> Positive sentiment is heavily driven by <b>'{top_driver}'</b>.<br>
            • <b>Critical Risks:</b> The most dominant complaint (Barrier) identified is <b>'{top_barrier}'</b>.<br>
            • <b>Recommendation:</b> If <b>{top_barrier}</b> relates to product flow, prioritize UI/UX fixes immediately.
        </div>
    </div>
    """
    return html

def run_strategic_analysis(df, brand, theme_cols):
    b_df = df[df['App_Name'] == brand]
    if b_df.empty: return None
    
    vol = len(b_df)
    score = b_df['score'].mean()
    prom = len(b_df[b_df['score']==5])
    det = len(b_df[b_df['score']<=3])
    nps = ((prom - det) / vol * 100) if vol else 0
    
    pos_df = b_df[b_df['score']>=4]
    drivers = []
    if not pos_df.empty and theme_cols:
        v = [t for t in theme_cols if t in pos_df.columns]
        s = pos_df[v].sum().sort_values(ascending=False).head(3)
        drivers = [(k,v,v/len(pos_df)*100) for k,v in s.items()]
        
    neg_df = b_df[b_df['score']<=3]
    barriers = []
    if not neg_df.empty and theme_cols:
        v = [t for t in theme_cols if t in neg_df.columns]
        s = neg_df[v].sum().sort_values(ascending=False).head(3)
        barriers = [(k,v,v/len(neg_df)*100) for k,v in s.items()]
        
    return {"score":score, "nps":nps, "drivers":drivers, "barriers":barriers, "vol":vol}

# ==========================================
# 4. SIDEBAR & INITIALIZATION
# ==========================================
with st.sidebar:
    st.title("🎛️ Command Center")
    with st.spinner("Syncing..."):
        df_raw = load_data()
    
    if df_raw.empty:
        st.error("No Data.")
        st.stop()

    st.success(f"🟢 Live: {len(df_raw):,} Rows")
    st.markdown("---")
    
    # FILTERS
    min_d, max_d = df_raw['at'].min().date(), df_raw['at'].max().date()
    date_range = st.date_input("Period", [min_d, max_d], min_value=min_d, max_value=max_d)
    
    all_brands = sorted(df_raw['App_Name'].dropna().unique())
    sel_brands = st.multiselect("Brands", all_brands, default=all_brands)
    
    # 2. Add Character Filter (Requested)
    st.markdown("### 📝 Review Depth")
    len_filter = st.radio(
        "Character Count", 
        ["All Reviews", "Brief (<=29 chars)", "Detailed (>=30 chars)"],
        index=0
    )
    
    prod_cols = [c for c in ['Product_1','Product_2'] if c in df_raw.columns]
    all_prods = set()
    for c in prod_cols: all_prods.update(df_raw[c].dropna().unique())
    sel_prods = st.multiselect("Product Type", sorted(list(all_prods)))
    
    if 'PL Status' in df_raw.columns:
        sel_pl = st.multiselect("PL Status", df_raw['PL Status'].dropna().unique())
    else: sel_pl = []

# --- FILTERING ---
if len(date_range) == 2:
    mask = (
        (df_raw['at'].dt.date >= date_range[0]) & 
        (df_raw['at'].dt.date <= date_range[1]) & 
        (df_raw['App_Name'].isin(sel_brands))
    )
else:
    mask = df_raw['App_Name'].isin(sel_brands)

# Apply Char Filter
if len_filter == "Brief (<=29 chars)":
    mask &= (df_raw['length_bucket'] == 'Brief (<=29)')
elif len_filter == "Detailed (>=30 chars)":
    mask &= (df_raw['length_bucket'] == 'Detailed (>=30)')

if sel_prods:
    p_mask = pd.Series(False, index=df_raw.index)
    for c in prod_cols: p_mask |= df_raw[c].isin(sel_prods)
    mask &= p_mask
if sel_pl: mask &= (df_raw['PL Status'].isin(sel_pl))

df = df_raw[mask].copy()
theme_cols = st.session_state.get('theme_cols', [])

# ==========================================
# 5. DASHBOARD LAYOUT
# ==========================================

st.title("🦅 Strategic Intelligence Platform")

# 3. AI Insights (Dynamic based on filters)
filter_desc = f"{len_filter}, {len(sel_brands)} Brands"
st.markdown(generate_ai_insight(df, theme_cols, filter_desc), unsafe_allow_html=True)

tab_exec, tab_drivers, tab_compare, tab_trends, tab_raw = st.tabs([
    "📊 Boardroom Summary", 
    "🚀 Drivers & Barriers (Deep Dive)", 
    "⚔️ Head-to-Head", 
    "📈 Trends (MoM/WoW)", 
    "🔍 Data Explorer"
])

# === TAB 1: SUMMARY ===
with tab_exec:
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Total Volume", f"{len(df):,}")
    with k2: st.metric("Market CSAT", f"{df['score'].mean():.2f} ⭐")
    with k3:
        prom = len(df[df['score']==5])
        det = len(df[df['score']<=3])
        nps = ((prom - det) / len(df)) * 100 if len(df)>0 else 0
        st.metric("NPS Proxy", f"{nps:.0f}")
    with k4:
        risk = (len(df[df['score']==1]) / len(df)) * 100 if len(df)>0 else 0
        st.metric("Critical Risk", f"{risk:.1f}%", delta="1-Star %", delta_color="inverse")

    st.markdown("---")
    
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

# === TAB 2: DRIVERS & BARRIERS (UPDATED AS PER REQUEST) ===
with tab_drivers:
    st.markdown("### 🚦 Strategic Drivers & Barriers")
    st.info("Select a Brand to view its Top 10 Drivers (from 4-5★ reviews) and Barriers (from 1-3★ reviews) with trends.")
    
    target_brand = st.selectbox("Select Brand to Analyze", sel_brands)
    
    if target_brand and theme_cols:
        b_df = df[df['App_Name'] == target_brand]
        
        # --- DRIVERS (4-5 Stars) ---
        pos_df = b_df[b_df['score'] >= 4]
        pos_base = len(pos_df)
        
        # --- BARRIERS (1-3 Stars) ---
        neg_df = b_df[b_df['score'] <= 3]
        neg_base = len(neg_df)
        
        c1, c2 = st.columns(2)
        
        # Helper to build table
        def build_theme_table(sub_df, base, themes):
            if sub_df.empty or base == 0: return pd.DataFrame()
            
            # 1. Sum counts
            valid = [t for t in themes if t in sub_df.columns]
            counts = sub_df[valid].sum().sort_values(ascending=False).head(10)
            
            # 2. Build Data
            data = []
            for theme, count in counts.items():
                if count == 0: continue
                # Calc MoM/WoW
                mom = calculate_trend_change(sub_df, theme, theme, 'Month', theme)
                wow = calculate_trend_change(sub_df, theme, theme, 'Week', theme)
                
                data.append({
                    "Theme": theme,
                    "Count": int(count),
                    "% of Base": f"{(count/base)*100:.1f}%",
                    "MoM Trend": f"{mom:+.1f}%",
                    "WoW Trend": f"{wow:+.1f}%"
                })
            return pd.DataFrame(data)

        with c1:
            st.markdown(f"#### 🚀 Top 10 Drivers (Positive)")
            st.markdown(f"**Base:** {pos_base:,} Positive Reviews")
            driver_table = build_theme_table(pos_df, pos_base, theme_cols)
            if not driver_table.empty:
                st.dataframe(driver_table, hide_index=True, use_container_width=True)
            else:
                st.warning("No Positive Data Found")

        with c2:
            st.markdown(f"#### 🛑 Top 10 Barriers (Negative)")
            st.markdown(f"**Base:** {neg_base:,} Negative Reviews")
            barrier_table = build_theme_table(neg_df, neg_base, theme_cols)
            if not barrier_table.empty:
                st.dataframe(barrier_table, hide_index=True, use_container_width=True)
            else:
                st.warning("No Negative Data Found")

# === TAB 3: HEAD TO HEAD ===
with tab_compare:
    c1, c2 = st.columns(2)
    with c1: b1 = st.selectbox("Brand A", sel_brands, index=0 if sel_brands else None)
    with c2: 
        opts = [b for b in sel_brands if b != b1]
        b2 = st.selectbox("Brand B", opts, index=0 if opts else None)
    
    if b1 and b2:
        d1 = run_strategic_analysis(df, b1, theme_cols)
        d2 = run_strategic_analysis(df, b2, theme_cols)
        
        cdf = pd.DataFrame({
            "Metric": ["CSAT", "NPS Proxy", "Volume"],
            b1: [f"{d1['score']:.2f}", f"{d1['nps']:.0f}", f"{d1['vol']:,}"],
            b2: [f"{d2['score']:.2f}", f"{d2['nps']:.0f}", f"{d2['vol']:,}"]
        }).set_index("Metric")
        st.dataframe(cdf, use_container_width=True)
        
        # Radar
        if theme_cols:
            t1 = [x[0] for x in d1['drivers']] if d1['drivers'] else []
            t2 = [x[0] for x in d2['drivers']] if d2['drivers'] else []
            common = list(set(t1 + t2))[:6]
            if common:
                def gp(b, tlist):
                    d = df[df['App_Name']==b]
                    return [(d[t].sum()/len(d)*100) if not d.empty else 0 for t in tlist]
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=gp(b1, common), theta=common, fill='toself', name=b1))
                fig.add_trace(go.Scatterpolar(r=gp(b2, common), theta=common, fill='toself', name=b2))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True)), template="plotly_dark", title="Theme Overlap")
                st.plotly_chart(fig, use_container_width=True)

# === TAB 4: TRENDS ===
with tab_trends:
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
    q = st.text_input("Deep Search", placeholder="Type keywords...")
    f_df = df.copy()
    if q: f_df = f_df[f_df['Review_Text'].astype(str).str.contains(q, case=False, na=False)]
    st.dataframe(f_df[['at', 'App_Name', 'score', 'Review_Text', 'length_bucket']], use_container_width=True)
