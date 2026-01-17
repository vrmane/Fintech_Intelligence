import streamlit as st
import pandas as pd
from supabase import create_client, Client
import plotly.express as px
import plotly.graph_objects as go
import requests
from streamlit_lottie import st_lottie

# ==========================================
# 1. PAGE CONFIG & CUSTOM CSS (WEBSITE LOOK)
# ==========================================
st.set_page_config(
    page_title="Fintech Intelligence Hub",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR "WEBSITE" FEEL ---
st.markdown("""
    <style>
        /* Main Background */
        .stApp {
            background-color: #f8f9fa;
        }
        /* Hide Default Menu/Footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Custom Cards for Metrics */
        div[data-testid="metric-container"] {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
            transition: transform 0.2s;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            box-shadow: 4px 4px 15px rgba(0,0,0,0.1);
        }
        
        /* Headers */
        h1, h2, h3 {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #1f2937;
        }
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e0e0e0;
        }
        
        /* Tab Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #ffffff;
            border-radius: 5px;
            color: #4b5563;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .stTabs [aria-selected="true"] {
            background-color: #eff6ff;
            color: #1d4ed8;
            font-weight: bold;
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
COLOR_MAP = {
    'MoneyView': '#1f77b4', 'KreditBee': '#ff7f0e',
    'Navi': '#2ca02c', 'Kissht': '#d62728', 'Fibe': '#9467bd'
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
# 3. LOADING SCREEN
# ==========================================
if 'df' not in st.session_state:
    loader = st.empty()
    with loader.container():
        lottie_json = load_lottieurl("https://lottie.host/67705423-745b-4303-9d8a-662551406e22/94bZp6i0k8.json")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if lottie_json: st_lottie(lottie_json, height=300)
            st.markdown("<h3 style='text-align:center; color:#555;'>Initializing Analytics Engine...</h3>", unsafe_allow_html=True)
    
    df_raw, net_cols = load_data()
    st.session_state['df'] = df_raw
    st.session_state['net_cols'] = net_cols
    loader.empty()
else:
    df_raw = st.session_state['df']
    net_cols = st.session_state['net_cols']

if df_raw.empty: st.stop()

# ==========================================
# 4. ADVANCED SIDEBAR FILTERS
# ==========================================
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    
    # Date
    min_date, max_date = df_raw['at'].min().date(), df_raw['at'].max().date()
    date_range = st.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

    st.markdown("---")
    
    # Standard Filters
    all_apps = sorted(df_raw['norm_app'].unique())
    sel_apps = st.multiselect("Select Brands", all_apps, default=all_apps)
    
    sel_ratings = st.multiselect("Select Ratings", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])

    # NEW: Character Length Filter
    st.markdown("---")
    st.markdown("📝 **Review Depth**")
    char_filter = st.radio(
        "Filter by Length",
        ["All Reviews", "Short (<=29 Chars)", "Long (>=30 Chars)"],
        index=0
    )
    
    # Products
    st.markdown("---")
    prod_set = set()
    for c in ['product_1', 'product_2', 'product_3', 'product_4']:
        if c in df_raw.columns: prod_set.update(df_raw[c].dropna().unique())
    all_products = sorted([str(p) for p in prod_set if p and str(p).lower() != 'nan'])
    sel_products = st.multiselect("Product Focus", all_products)
    
    search_query = st.text_input("Search Keywords", placeholder="e.g. 'fraud', 'fast'")

# --- APPLYING FILTERS ---
mask = (df_raw['norm_app'].isin(sel_apps)) & (df_raw['score'].isin(sel_ratings))
if len(date_range) == 2:
    mask &= (df_raw['at'].dt.date >= date_range[0]) & (df_raw['at'].dt.date <= date_range[1])

# Apply New Char Length Filter
if char_filter == "Short (<=29 Chars)":
    mask &= (df_raw['length_group'] == '<=29 Chars')
elif char_filter == "Long (>=30 Chars)":
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
# 5. MAIN DASHBOARD UI
# ==========================================

# --- HERO SECTION (WEBSITE HEADER) ---
st.markdown(f"# 🚀 Fintech Intelligence Hub")
st.markdown(f"**Analysis Period:** {date_range[0]} to {date_range[1]}")

# --- HERO METRICS (CARDS) ---
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Total Volume", f"{len(df):,}", delta=f"{len(df)/len(df_raw)*100:.1f}% of Total")
with m2:
    avg_score = df['score'].mean()
    st.metric("Avg Rating", f"{avg_score:.2f} ⭐", delta_color="off")
with m3:
    pos_pct = (len(df[df['score']>=4]) / len(df) * 100) if not df.empty else 0
    st.metric("Positive Sentiment", f"{pos_pct:.1f}%")
with m4:
    neg_pct = (len(df[df['score']<=3]) / len(df) * 100) if not df.empty else 0
    st.metric("Negative Sentiment", f"{neg_pct:.1f}%", delta="-Action Required" if neg_pct > 40 else "Normal", delta_color="inverse")

st.markdown("---")

# --- NAVIGATION TABS ---
tab_overview, tab_prod, tab_drive, tab_barr, tab_qual, tab_insights, tab_monthly = st.tabs([
    "📊 Overview", "📦 Products", "🚀 Drivers (Pos)", "🛑 Barriers (Neg)", "📏 Quality", "🧠 AI Insights", "📅 Trends"
])

# === TAB 1: OVERVIEW ===
with tab_overview:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📊 Ratings Distribution")
        rating_counts = df['score'].value_counts().sort_index().reset_index()
        rating_counts.columns = ['Rating', 'Count']
        fig = px.bar(rating_counts, x='Rating', y='Count', color='Rating', text_auto=True, color_discrete_sequence=px.colors.sequential.RdBu)
        fig.update_layout(xaxis_title=None, yaxis_title=None, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("#### 🏆 Brand Performance Matrix")
        brand_score = pd.crosstab(df['norm_app'], df['score'], normalize='index') * 100
        fig = px.imshow(brand_score, text_auto='.0f', aspect="auto", color_continuous_scale='RdBu', labels=dict(color="%"))
        st.plotly_chart(fig, use_container_width=True)

# === TAB 2: PRODUCTS ===
with tab_prod:
    def get_product_data(d):
        cols = [c for c in ['norm_app','product_1','product_2','product_3','product_4'] if c in d.columns]
        return d[cols].melt(id_vars=['norm_app'], value_name='Product').dropna()
    
    prod_df = get_product_data(df)
    prod_df = prod_df[prod_df['Product'].str.len() > 1] # Remove empty strings

    if not prod_df.empty:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("#### 📦 Product Mix by Brand")
            brand_base = prod_df.groupby('norm_app').size().reset_index(name='Total')
            prod_counts = prod_df.groupby(['Product', 'norm_app']).size().reset_index(name='Count')
            prod_stats = pd.merge(prod_counts, brand_base, on='norm_app')
            prod_stats['%'] = (prod_stats['Count'] / prod_stats['Total']) * 100
            
            fig = px.bar(prod_stats, x='norm_app', y='Count', color='Product', text=prod_stats['%'].apply(lambda x: f"{x:.0f}%"))
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            st.markdown("#### 🔢 Volume Data")
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
        
        st.markdown("#### 🚀 Top Drivers of Satisfaction (4-5 Stars)")
        st.plotly_chart(px.imshow(top_drivers, text_auto='.1f', aspect="auto", color_continuous_scale='Greens'), use_container_width=True)
    else:
        st.info("Insufficient positive data.")

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
        
        st.markdown("#### 🛑 Top Barriers / Complaints (1-3 Stars)")
        st.plotly_chart(px.imshow(top_barr, text_auto='.1f', aspect="auto", color_continuous_scale='Reds'), use_container_width=True)
    else:
        st.info("Insufficient negative data.")

# === TAB 5: QUALITY ===
with tab_qual:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📏 Review Length vs Rating")
        len_rating = pd.crosstab(df['score'], df['length_group'], normalize='index') * 100
        st.plotly_chart(px.bar(len_rating, x=len_rating.index, y=len_rating.columns, color_discrete_map={'<=29 Chars':'#ff7f0e', '>=30 Chars':'#1f77b4'}), use_container_width=True)
    with c2:
        st.markdown("#### 🖊️ Avg Characters per Brand")
        avg_len = df.groupby('norm_app')['char_count'].mean().reset_index().sort_values('char_count')
        st.plotly_chart(px.bar(avg_len, y='norm_app', x='char_count', orientation='h', text_auto='.0f', color='norm_app', color_discrete_map=COLOR_MAP), use_container_width=True)

# === TAB 6: AI INSIGHTS ===
with tab_insights:
    st.markdown("#### 🧠 Automated Strategic Profiling")
    sel_brands = df['norm_app'].unique()
    cat_avg = df['score'].mean()
    
    for brand in sel_brands:
        b_df = df[df['norm_app'] == brand]
        b_score = b_df['score'].mean()
        diff = ((b_score - cat_avg)/cat_avg)*100
        
        # Top Driver
        b_pos = b_df[b_df['score']>=4]
        driver = "N/A"
        if not b_pos.empty and net_cols:
             driver = b_pos[[c for c in net_cols if c in b_pos.columns]].sum().idxmax().replace('[NET]','').strip()
        
        # Top Barrier
        b_neg = b_df[b_df['score']<=3]
        barrier = "N/A"
        if not b_neg.empty and net_cols:
             barrier = b_neg[[c for c in net_cols if c in b_neg.columns]].sum().idxmax().replace('[NET]','').strip()

        with st.expander(f"📌 {brand} Analysis ({b_score:.2f} ⭐)", expanded=True):
            c1, c2 = st.columns([1, 4])
            with c1:
                st.metric("Score", f"{b_score:.2f}", f"{diff:.1f}%")
            with c2:
                st.markdown(f"""
                - **Superpower:** {driver}
                - **Achilles Heel:** {barrier}
                - **User Engagement:** Avg {b_df['char_count'].mean():.0f} chars per review.
                """)

# === TAB 7: TRENDS ===
with tab_monthly:
    if 'Month' in df.columns:
        df_trend = df.sort_values('at')
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 📈 Volume Trend")
            v = df_trend.groupby(['Month','norm_app']).size().reset_index(name='Count')
            st.plotly_chart(px.line(v, x='Month', y='Count', color='norm_app', markers=True, color_discrete_map=COLOR_MAP), use_container_width=True)
        with c2:
            st.markdown("#### 📉 Rating Trend")
            s = df_trend.groupby(['Month','norm_app'])['score'].mean().reset_index()
            st.plotly_chart(px.line(s, x='Month', y='score', color='norm_app', markers=True, color_discrete_map=COLOR_MAP), use_container_width=True)
