import streamlit as st
import pandas as pd
from supabase import create_client, Client
import plotly.express as px
import requests
from streamlit_lottie import st_lottie

# ==========================================
# 1. PAGE CONFIG & DARK COMMAND CENTER CSS
# ==========================================
st.set_page_config(
    page_title="Fintech Command Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        /* BACKGROUND */
        .stApp {
            background: #0e1117; /* Deep Matte Black/Blue */
            color: #e0e0e0;
        }
        
        /* HIDE DEFAULTS */
        #MainMenu, footer, header {visibility: hidden;}
        
        /* CARDS (Glass Effect) */
        .css-1r6slb0, .css-12w0qpk { 
            background-color: #1a1c24;
            border: 1px solid #333;
            border-radius: 10px;
            padding: 20px;
        }
        
        /* METRIC CARDS */
        div[data-testid="metric-container"] {
            background: #1f2937;
            border: 1px solid #374151;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        div[data-testid="metric-container"] label {
            color: #9ca3af; /* Muted label */
        }
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
            color: #f3f4f6; /* Bright Value */
        }
        
        /* SECTION HEADERS */
        .section-header {
            font-size: 24px;
            font-weight: 700;
            color: #38bdf8; /* Light Blue */
            margin-top: 40px;
            margin-bottom: 20px;
            border-bottom: 1px solid #374151;
            padding-bottom: 10px;
        }
        
        /* DATAFRAME */
        .stDataFrame { border: 1px solid #374151; }
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
COLOR_MAP = {
    'MoneyView': '#00d4ff', 'KreditBee': '#ff9f00',
    'Navi': '#00ff9d', 'Kissht': '#ff0055', 'Fibe': '#bc13fe'
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
        st.error("Supabase Connection Failed.")
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
# 3. LOADING STATE
# ==========================================
if 'df' not in st.session_state:
    loader = st.empty()
    with loader.container():
        # Sci-Fi Loader
        lottie_json = load_lottieurl("https://lottie.host/9e5c4644-841b-43c3-982d-19597143c690/w5h8o4t9zD.json") 
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if lottie_json: st_lottie(lottie_json, height=300)
            st.markdown("<h3 style='text-align:center; color:#38bdf8;'>Initializing Command Center...</h3>", unsafe_allow_html=True)
    
    df_raw, net_cols = load_data()
    st.session_state['df'] = df_raw
    st.session_state['net_cols'] = net_cols
    loader.empty()
else:
    df_raw = st.session_state['df']
    net_cols = st.session_state['net_cols']

if df_raw.empty: st.stop()

# ==========================================
# 4. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("### 🎛️ Parameters")
    
    min_date, max_date = df_raw['at'].min().date(), df_raw['at'].max().date()
    date_range = st.date_input("Timeline", [min_date, max_date], min_value=min_date, max_value=max_date)

    st.markdown("---")
    
    all_apps = sorted(df_raw['norm_app'].unique())
    sel_apps = st.multiselect("Brands", all_apps, default=all_apps)
    sel_ratings = st.multiselect("Ratings", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])

    st.markdown("---")
    char_filter = st.radio("Review Depth", ["All", "Brief (<=29)", "Detailed (>=30)"], index=0)
    
    st.markdown("---")
    prod_set = set()
    for c in ['product_1', 'product_2', 'product_3', 'product_4']:
        if c in df_raw.columns: prod_set.update(df_raw[c].dropna().unique())
    all_products = sorted([str(p) for p in prod_set if p and str(p).lower() != 'nan'])
    sel_products = st.multiselect("Products", all_products)
    
    search_query = st.text_input("Search", placeholder="Keywords...")

# --- APPLY FILTERS ---
mask = (df_raw['norm_app'].isin(sel_apps)) & (df_raw['score'].isin(sel_ratings))
if len(date_range) == 2:
    mask &= (df_raw['at'].dt.date >= date_range[0]) & (df_raw['at'].dt.date <= date_range[1])

if char_filter == "Brief (<=29)": mask &= (df_raw['length_group'] == '<=29 Chars')
elif char_filter == "Detailed (>=30)": mask &= (df_raw['length_group'] == '>=30 Chars')

if sel_products:
    p_mask = pd.Series(False, index=df_raw.index)
    for c in ['product_1', 'product_2', 'product_3', 'product_4']:
        if c in df_raw.columns: p_mask |= df_raw[c].isin(sel_products)
    mask &= p_mask

if search_query:
    mask &= df_raw['content'].str.contains(search_query, case=False, na=False)

df = df_raw[mask].copy()

# HELPER: DARK CHART THEME
def dark_chart(fig):
    fig.update_layout(
        template="plotly_dark", 
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="#e0e0e0")
    )
    return fig

# ==========================================
# 5. SINGLE PAGE DASHBOARD LAYOUT
# ==========================================

# --- HERO SECTION ---
st.title("⚡ Fintech Command Center")
st.markdown(f"**Live Analysis:** {len(df):,} Reviews Loaded | **Status:** Active")

# --- HUD (HEADS UP DISPLAY) ---
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Total Volume", f"{len(df):,}")
with m2:
    st.metric("Avg Rating", f"{df['score'].mean():.2f} ⭐")
with m3:
    pos = (len(df[df['score']>=4]) / len(df) * 100) if not df.empty else 0
    st.metric("Positive %", f"{pos:.1f}%")
with m4:
    neg = (len(df[df['score']<=3]) / len(df) * 100) if not df.empty else 0
    st.metric("Negative %", f"{neg:.1f}%", delta="-Risk" if neg > 40 else "Stable", delta_color="inverse")

# --- SECTION 1: MARKET LANDSCAPE ---
st.markdown('<div class="section-header">1. Market Landscape</div>', unsafe_allow_html=True)
c1, c2 = st.columns([3, 2])
with c1:
    st.markdown("**Ratings Distribution**")
    rc = df['score'].value_counts().sort_index().reset_index()
    rc.columns = ['Rating', 'Count']
    fig = px.bar(rc, x='Rating', y='Count', color='Rating', text_auto=True, color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(dark_chart(fig), use_container_width=True)
with c2:
    st.markdown("**Competitive Heatmap**")
    bs = pd.crosstab(df['norm_app'], df['score'], normalize='index') * 100
    fig = px.imshow(bs, text_auto='.0f', aspect="auto", color_continuous_scale='RdBu')
    st.plotly_chart(dark_chart(fig), use_container_width=True)

# --- SECTION 2: PRODUCT MIX ---
st.markdown('<div class="section-header">2. Product & Volume</div>', unsafe_allow_html=True)
cols = [c for c in ['norm_app','product_1','product_2','product_3','product_4'] if c in df.columns]
prod_df = df[cols].melt(id_vars=['norm_app'], value_name='Product').dropna()
prod_df = prod_df[prod_df['Product'].str.len() > 1]

if not prod_df.empty:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("**Product Mix by Brand**")
        pb = prod_df.groupby('norm_app').size().reset_index(name='Total')
        pc = prod_df.groupby(['Product', 'norm_app']).size().reset_index(name='Count')
        ps = pd.merge(pc, pb, on='norm_app')
        ps['%'] = (ps['Count'] / ps['Total']) * 100
        fig = px.bar(ps, x='norm_app', y='Count', color='Product', text=ps['%'].apply(lambda x: f"{x:.0f}%"))
        st.plotly_chart(dark_chart(fig), use_container_width=True)
    with c2:
        st.markdown("**Volume Table**")
        dd = ps.pivot(index='Product', columns='norm_app', values='Count').fillna(0).astype(int)
        dd['Total'] = dd.sum(axis=1)
        st.dataframe(dd.sort_values('Total', ascending=False).style.background_gradient(cmap='Blues'), height=400)

# --- SECTION 3: STRATEGIC DRIVERS ---
st.markdown('<div class="section-header">3. Strategic Drivers vs Barriers</div>', unsafe_allow_html=True)
c_drv, c_bar = st.columns(2)

with c_drv:
    st.markdown("🚀 **Top Growth Drivers (4-5 Star)**")
    df_pos = df[df['score'].isin([4, 5])]
    if not df_pos.empty and net_cols:
        base = df_pos.groupby('norm_app').size()
        v_cols = [c for c in net_cols if c in df_pos.columns]
        sums = df_pos.groupby('norm_app')[v_cols].sum().T
        sums.index = sums.index.str.replace('[NET]', '', regex=False).str.strip()
        pct = sums.div(base, axis=1).fillna(0) * 100
        pct['Avg'] = pct.mean(axis=1)
        top_d = pct.sort_values('Avg', ascending=False).head(10).drop(columns=['Avg'])
        st.plotly_chart(dark_chart(px.imshow(top_d, text_auto='.1f', aspect="auto", color_continuous_scale='Greens')), use_container_width=True)
    else:
        st.info("No Data")

with c_bar:
    st.markdown("🛑 **Top Churn Risks (1-3 Star)**")
    df_neg = df[df['score'].isin([1, 2, 3])]
    if not df_neg.empty and net_cols:
        base = df_neg.groupby('norm_app').size()
        v_cols = [c for c in net_cols if c in df_neg.columns]
        sums = df_neg.groupby('norm_app')[v_cols].sum().T
        sums.index = sums.index.str.replace('[NET]', '', regex=False).str.strip()
        pct = sums.div(base, axis=1).fillna(0) * 100
        pct['Avg'] = pct.mean(axis=1)
        top_b = pct.sort_values('Avg', ascending=False).head(10).drop(columns=['Avg'])
        st.plotly_chart(dark_chart(px.imshow(top_b, text_auto='.1f', aspect="auto", color_continuous_scale='Reds')), use_container_width=True)
    else:
        st.info("No Data")

# --- SECTION 4: TRENDS ---
st.markdown('<div class="section-header">4. Performance Trends</div>', unsafe_allow_html=True)
if 'Month' in df.columns:
    df_trend = df.sort_values('at')
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Volume Trend**")
        v = df_trend.groupby(['Month','norm_app']).size().reset_index(name='Count')
        fig = px.line(v, x='Month', y='Count', color='norm_app', markers=True, color_discrete_map=COLOR_MAP)
        st.plotly_chart(dark_chart(fig), use_container_width=True)
    with c2:
        st.markdown("**Rating Trend**")
        s = df_trend.groupby(['Month','norm_app'])['score'].mean().reset_index()
        fig = px.line(s, x='Month', y='score', color='norm_app', markers=True, color_discrete_map=COLOR_MAP)
        fig.update_yaxes(range=[1, 5])
        st.plotly_chart(dark_chart(fig), use_container_width=True)

# --- SECTION 5: AI INSIGHTS FOOTER ---
st.markdown('<div class="section-header">5. Automated Strategic Profiling</div>', unsafe_allow_html=True)

brands = df['norm_app'].unique()
cat_avg = df['score'].mean()

# Grid Layout for Brand Cards
cols = st.columns(3) # 3 Cards per row

for i, brand in enumerate(brands):
    col = cols[i % 3] # Distribute across 3 columns
    b_df = df[df['norm_app'] == brand]
    b_score = b_df['score'].mean()
    
    # Logic
    driver = "N/A"
    barrier = "N/A"
    if net_cols:
        p = b_df[b_df['score']>=4]
        if not p.empty: driver = p[[c for c in net_cols if c in p.columns]].sum().idxmax().replace('[NET]','').strip()
        n = b_df[b_df['score']<=3]
        if not n.empty: barrier = n[[c for c in net_cols if c in n.columns]].sum().idxmax().replace('[NET]','').strip()

    with col:
        st.markdown(f"""
        <div style="background:#1a1c24; padding:15px; border-radius:10px; border:1px solid #333; margin-bottom:20px;">
            <h4 style="color:{COLOR_MAP.get(brand, '#fff')}; margin:0;">{brand}</h4>
            <h2 style="margin:5px 0;">{b_score:.2f} ⭐</h2>
            <p style="color:#888; font-size:12px;">Vs Category: {((b_score-cat_avg)/cat_avg)*100:+.1f}%</p>
            <hr style="border-color:#333;">
            <p><b>🔥 Top Driver:</b><br>{driver}</p>
            <p><b>⚠️ Top Risk:</b><br>{barrier}</p>
        </div>
        """, unsafe_allow_html=True)
