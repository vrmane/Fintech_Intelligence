import streamlit as st
import pandas as pd
from supabase import create_client, Client
import plotly.express as px
import plotly.graph_objects as go
import requests
from streamlit_lottie import st_lottie

# ==========================================
# 1. PAGE CONFIG & DARK GLASS CSS
# ==========================================
st.set_page_config(
    page_title="Fintech Intelligence Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS: DARK GLASSMORPHISM THEME ---
st.markdown("""
    <style>
        /* MAIN BACKGROUND: Deep Dark Gradient */
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #e2e8f0;
        }
        
        /* HIDE DEFAULT STREAMLIT ELEMENTS */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* SIDEBAR STYLING */
        section[data-testid="stSidebar"] {
            background-color: #0f172a;
            border-right: 1px solid #334155;
        }
        
        /* GLASS CARDS (Metrics & Containers) */
        div[data-testid="metric-container"] {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-5px);
            border-color: #38bdf8; /* Light Blue Glow */
            box-shadow: 0 10px 15px rgba(56, 189, 248, 0.2);
        }
        
        /* CUSTOM TITLES WITH GRADIENT */
        h1, h2, h3 {
            background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }
        
        /* TABS STYLING */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            color: #94a3b8;
            border: 1px solid transparent;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(56, 189, 248, 0.1) !important;
            color: #38bdf8 !important;
            border: 1px solid #38bdf8 !important;
        }
        
        /* DATAFRAME STYLING */
        .stDataFrame {
            border: 1px solid #334155;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# --- HELPER: LOTTIE ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except:
        return None

# --- CONFIGURATION ---
APP_MAP = {
    'moneyview': 'MoneyView', 'kreditbee': 'KreditBee',
    'navi': 'Navi', 'kissht': 'Kissht',
    'fibe': 'Fibe', 'earlysalary': 'Fibe'
}
# Neon Color Palette for Dark Mode
COLOR_MAP = {
    'MoneyView': '#00d4ff',   # Cyan
    'KreditBee': '#ff9f00',   # Orange
    'Navi': '#00ff9d',        # Neon Green
    'Kissht': '#ff0055',      # Neon Red
    'Fibe': '#bc13fe'         # Neon Purple
}

# ==========================================
# 2. DATA LOADING
# ==========================================
@st.cache_resource(show_spinner=False)
def init_connection():
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except:
        st.error("Supabase Connection Failed. Check Secrets.")
        st.stop()

supabase = init_connection()

@st.cache_data(ttl=600, show_spinner=False)
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

    if not all_rows: return pd.DataFrame(), []

    df = pd.DataFrame(all_rows)
    rename_map = {
        'App_Name': 'app_name', 'Rating': 'score',
        'Review_Date': 'at', 'Review_Text': 'content',
        'Product_1': 'product_1', 'Product_2': 'product_2',
        'Product_3': 'product_3', 'Product_4': 'product_4',
        'Sentiment': 'sentiment'
    }
    df.rename(columns=rename_map, inplace=True)
    
    if 'app_name' in df.columns:
        df['norm_app'] = df['app_name'].str.lower().apply(
            lambda x: next((v for k, v in APP_MAP.items() if k in str(x)), None)
        )
        df = df.dropna(subset=['norm_app'])
    
    if 'at' in df.columns:
        df['at'] = pd.to_datetime(df['at'], errors='coerce', utc=True)
        df['at'] = df['at'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None) 
        df = df.dropna(subset=['at'])
        df['Month'] = df['at'].dt.strftime('%Y-%m')

    if 'score' in df.columns:
        df['score'] = pd.to_numeric(df['score'], errors='coerce')

    if 'content' in df.columns:
        df['char_count'] = df['content'].astype(str).str.len().fillna(0)
        df['length_group'] = df['char_count'].apply(lambda x: '<=29 Chars' if x <= 29 else '>=30 Chars')

    net_cols = [c for c in df.columns if str(c).startswith('[NET]')]
    for col in net_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df, net_cols

# ==========================================
# 3. LOADING SCREEN (DARK MODE)
# ==========================================
if 'df' not in st.session_state:
    loader = st.empty()
    with loader.container():
        # Dark Mode Tech Loader
        lottie_json = load_lottieurl("https://lottie.host/9e5c4644-841b-43c3-982d-19597143c690/w5h8o4t9zD.json") 
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if lottie_json: st_lottie(lottie_json, height=300)
            st.markdown("<h3 style='text-align:center; color:#38bdf8;'>Syncing Neural Network...</h3>", unsafe_allow_html=True)
    
    df_raw, net_cols = load_data()
    st.session_state['df'] = df_raw
    st.session_state['net_cols'] = net_cols
    loader.empty()
else:
    df_raw = st.session_state['df']
    net_cols = st.session_state['net_cols']

if df_raw.empty: st.stop()

# ==========================================
# 4. SIDEBAR (CONTROLS)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2111/2111615.png", width=50)
    st.markdown("### 🎛️ Control Center")
    st.markdown("---")
    
    # Date
    min_date, max_date = df_raw['at'].min().date(), df_raw['at'].max().date()
    date_range = st.date_input("📅 Timeline", [min_date, max_date], min_value=min_date, max_value=max_date)

    st.markdown("---")
    
    # Filters
    all_apps = sorted(df_raw['norm_app'].unique())
    sel_apps = st.multiselect("Select Brands", all_apps, default=all_apps)
    sel_ratings = st.multiselect("Star Ratings", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])

    # NEW: Length Filter
    st.markdown("---")
    st.markdown("📝 **Feedback Depth**")
    char_filter = st.radio(
        "Filter by Length",
        ["All Reviews", "Brief (<=29 Chars)", "Detailed (>=30 Chars)"],
        index=0
    )
    
    # Products
    st.markdown("---")
    prod_set = set()
    for c in ['product_1', 'product_2', 'product_3', 'product_4']:
        if c in df_raw.columns: prod_set.update(df_raw[c].dropna().unique())
    all_products = sorted([str(p) for p in prod_set if p and str(p).lower() != 'nan'])
    sel_products = st.multiselect("Product Lines", all_products)
    
    search_query = st.text_input("🔍 Keyword Search", placeholder="Search...")

# --- FILTER LOGIC ---
mask = (df_raw['norm_app'].isin(sel_apps)) & (df_raw['score'].isin(sel_ratings))
if len(date_range) == 2:
    mask &= (df_raw['at'].dt.date >= date_range[0]) & (df_raw['at'].dt.date <= date_range[1])

if char_filter == "Brief (<=29 Chars)":
    mask &= (df_raw['length_group'] == '<=29 Chars')
elif char_filter == "Detailed (>=30 Chars)":
    mask &= (df_raw['length_group'] == '>=30 Chars')

if sel_products:
    p_mask = pd.Series(False, index=df_raw.index)
    for c in ['product_1', 'product_2', 'product_3', 'product_4']:
        if c in df_raw.columns: p_mask |= df_raw[c].isin(sel_products)
    mask &= p_mask

if search_query:
    mask &= df_raw['content'].str.contains(search_query, case=False, na=False)

df = df_raw[mask].copy()

# ==========================================
# 5. DASHBOARD LAYOUT
# ==========================================

# HEADER
c_head1, c_head2 = st.columns([3, 1])
with c_head1:
    st.title("Fintech Intelligence Hub")
    st.markdown(f"Real-time strategic analysis | **{len(df):,} Reviews** Analyzed")
with c_head2:
    # Just a visual placeholder for "Live Status"
    st.markdown("#### 🟢 System Live")

st.markdown("---")

# METRIC CARDS (GLASS STYLE)
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Total Volume", f"{len(df):,}", delta="Live Update")
with m2:
    avg_score = df['score'].mean()
    st.metric("Avg Score", f"{avg_score:.2f} ⭐", delta="vs Market Benchmark", delta_color="off")
with m3:
    pos_pct = (len(df[df['score']>=4]) / len(df) * 100) if not df.empty else 0
    st.metric("Positive Sentiment", f"{pos_pct:.1f}%", delta="Satisfaction")
with m4:
    neg_pct = (len(df[df['score']<=3]) / len(df) * 100) if not df.empty else 0
    st.metric("Risk Level", f"{neg_pct:.1f}%", delta="-High" if neg_pct > 40 else "Stable", delta_color="inverse")

st.markdown("<br>", unsafe_allow_html=True)

# TABS
tab_overview, tab_prod, tab_drive, tab_barr, tab_qual, tab_insights, tab_monthly = st.tabs([
    "📊 Overview", "📦 Products", "🚀 Drivers", "🛑 Barriers", "📏 Quality", "🧠 AI Strategy", "📅 Trends"
])

# HELPER: DARK CHART
def dark_chart(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

# === TAB 1: OVERVIEW ===
with tab_overview:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("### 📊 Rating Distribution")
        rating_counts = df['score'].value_counts().sort_index().reset_index()
        rating_counts.columns = ['Rating', 'Count']
        fig = px.bar(rating_counts, x='Rating', y='Count', color='Rating', text_auto=True, 
                     color_discrete_sequence=['#ef4444', '#f97316', '#eab308', '#84cc16', '#22c55e'])
        st.plotly_chart(dark_chart(fig), use_container_width=True)
    with c2:
        st.markdown("### 🏆 Competitive Matrix")
        brand_score = pd.crosstab(df['norm_app'], df['score'], normalize='index') * 100
        fig = px.imshow(brand_score, text_auto='.0f', aspect="auto", color_continuous_scale='RdBu')
        st.plotly_chart(dark_chart(fig), use_container_width=True)

# === TAB 2: PRODUCTS ===
with tab_prod:
    def get_product_data(d):
        cols = [c for c in ['norm_app','product_1','product_2','product_3','product_4'] if c in d.columns]
        return d[cols].melt(id_vars=['norm_app'], value_name='Product').dropna()
    
    prod_df = get_product_data(df)
    prod_df = prod_df[prod_df['Product'].str.len() > 1] 

    if not prod_df.empty:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("### 📦 Product Mix")
            brand_base = prod_df.groupby('norm_app').size().reset_index(name='Total')
            prod_counts = prod_df.groupby(['Product', 'norm_app']).size().reset_index(name='Count')
            prod_stats = pd.merge(prod_counts, brand_base, on='norm_app')
            prod_stats['%'] = (prod_stats['Count'] / prod_stats['Total']) * 100
            
            fig = px.bar(prod_stats, x='norm_app', y='Count', color='Product', 
                         text=prod_stats['%'].apply(lambda x: f"{x:.0f}%"))
            st.plotly_chart(dark_chart(fig), use_container_width=True)
        
        with c2:
            st.markdown("### 🔢 Volume Table")
            display_df = prod_stats.pivot(index='Product', columns='norm_app', values='Count').fillna(0).astype(int)
            display_df['Total'] = display_df.sum(axis=1)
            st.dataframe(display_df.sort_values('Total', ascending=False).style.background_gradient(cmap='Blues'), height=400)

# === TAB 3: DRIVERS ===
with tab_drive:
    df_pos = df[df['score'].isin([4, 5])]
    if not df_pos.empty and net_cols:
        base = df_pos.groupby('norm_app').size()
        valid_cols = [c for c in net_cols if c in df_pos.columns]
        sums = df_pos.groupby('norm_app')[valid_cols].sum().T
        sums.index = sums.index.str.replace('[NET]', '', regex=False).str.strip()
        pct = sums.div(base, axis=1).fillna(0) * 100
        pct['Avg'] = pct.mean(axis=1)
        top_drivers = pct.sort_values('Avg', ascending=False).head(10).drop(columns=['Avg'])
        
        st.markdown("### 🚀 Top Growth Drivers")
        st.plotly_chart(dark_chart(px.imshow(top_drivers, text_auto='.1f', aspect="auto", color_continuous_scale='Greens')), use_container_width=True)
    else:
        st.info("Insufficient data.")

# === TAB 4: BARRIERS ===
with tab_barr:
    df_neg = df[df['score'].isin([1, 2, 3])]
    if not df_neg.empty and net_cols:
        base = df_neg.groupby('norm_app').size()
        valid_cols = [c for c in net_cols if c in df_neg.columns]
        sums = df_neg.groupby('norm_app')[valid_cols].sum().T
        sums.index = sums.index.str.replace('[NET]', '', regex=False).str.strip()
        pct = sums.div(base, axis=1).fillna(0) * 100
        pct['Avg'] = pct.mean(axis=1)
        top_barr = pct.sort_values('Avg', ascending=False).head(10).drop(columns=['Avg'])
        
        st.markdown("### 🛑 Top Churn Risks")
        st.plotly_chart(dark_chart(px.imshow(top_barr, text_auto='.1f', aspect="auto", color_continuous_scale='Reds')), use_container_width=True)
    else:
        st.info("Insufficient data.")

# === TAB 5: QUALITY ===
with tab_qual:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📏 Review Depth vs Score")
        len_rating = pd.crosstab(df['score'], df['length_group'], normalize='index') * 100
        fig = px.bar(len_rating, x=len_rating.index, y=len_rating.columns, 
                     color_discrete_map={'<=29 Chars':'#fbbf24', '>=30 Chars':'#38bdf8'})
        st.plotly_chart(dark_chart(fig), use_container_width=True)
    with c2:
        st.markdown("### 🖊️ Avg Characters per Brand")
        avg_len = df.groupby('norm_app')['char_count'].mean().reset_index().sort_values('char_count')
        fig = px.bar(avg_len, y='norm_app', x='char_count', orientation='h', text_auto='.0f', color='norm_app', color_discrete_map=COLOR_MAP)
        st.plotly_chart(dark_chart(fig), use_container_width=True)

# === TAB 6: AI INSIGHTS ===
with tab_insights:
    st.markdown("### 🧠 Neural Strategic Profiling")
    sel_brands = df['norm_app'].unique()
    cat_avg = df['score'].mean()
    
    for brand in sel_brands:
        b_df = df[df['norm_app'] == brand]
        b_score = b_df['score'].mean()
        
        # Get Drivers/Barriers
        driver = "N/A"
        barrier = "N/A"
        if net_cols:
            b_pos = b_df[b_df['score']>=4]
            if not b_pos.empty: 
                driver = b_pos[[c for c in net_cols if c in b_pos.columns]].sum().idxmax().replace('[NET]','').strip()
            b_neg = b_df[b_df['score']<=3]
            if not b_neg.empty: 
                barrier = b_neg[[c for c in net_cols if c in b_neg.columns]].sum().idxmax().replace('[NET]','').strip()

        with st.expander(f"📌 {brand} | Strategy Card", expanded=True):
            c1, c2 = st.columns([1, 4])
            with c1:
                st.metric("Health Score", f"{b_score:.2f}", f"{((b_score-cat_avg)/cat_avg)*100:.1f}%")
            with c2:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:5px;">
                <ul style="list-style-type:none; padding:0;">
                    <li>🔥 <b>Growth Engine:</b> {driver}</li>
                    <li>⚠️ <b>Risk Factor:</b> {barrier}</li>
                    <li>💬 <b>Engagement:</b> {b_df['char_count'].mean():.0f} avg characters.</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

# === TAB 7: TRENDS ===
with tab_monthly:
    if 'Month' in df.columns:
        df_trend = df.sort_values('at')
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 📈 Volume Trend")
            v = df_trend.groupby(['Month','norm_app']).size().reset_index(name='Count')
            fig = px.line(v, x='Month', y='Count', color='norm_app', markers=True, color_discrete_map=COLOR_MAP)
            st.plotly_chart(dark_chart(fig), use_container_width=True)
        with c2:
            st.markdown("### 📉 Satisfaction Trend")
            s = df_trend.groupby(['Month','norm_app'])['score'].mean().reset_index()
            fig = px.line(s, x='Month', y='score', color='norm_app', markers=True, color_discrete_map=COLOR_MAP)
            fig.update_yaxes(range=[1, 5])
            st.plotly_chart(dark_chart(fig), use_container_width=True)
