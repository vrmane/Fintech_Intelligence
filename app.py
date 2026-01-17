import streamlit as st
import pandas as pd
from supabase import create_client, Client
import plotly.express as px
import plotly.graph_objects as go
import requests
from streamlit_lottie import st_lottie

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Fintech Strategic Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HELPER: LOAD LOTTIE ANIMATION ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# --- CONFIGURATION (APP MAPPING) ---
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
# 2. SUPABASE CONNECTION & DATA LOADING
# ==========================================
@st.cache_resource
def init_connection():
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except Exception as e:
        st.error("❌ Supabase credentials missing in .streamlit/secrets.toml")
        st.stop()

supabase = init_connection()

@st.cache_data(ttl=600)
def load_data():
    # --- A. FETCH ALL DATA (Pagination Loop) ---
    all_rows = []
    start = 0
    batch_size = 1000  
    
    while True:
        response = supabase.table("reviews").select("*").range(start, start + batch_size - 1).execute()
        rows = response.data
        
        if not rows:
            break
            
        all_rows.extend(rows)
        
        if len(rows) < batch_size:
            break
            
        start += batch_size

    if not all_rows:
        return pd.DataFrame(), []

    # --- B. CREATE DATAFRAME ---
    df = pd.DataFrame(all_rows)

    # --- C. RENAME COLUMNS ---
    rename_map = {
        'App_Name': 'app_name', 
        'Rating': 'score',
        'Review_Date': 'at', 
        'Review_Text': 'content',
        'Product_1': 'product_1', 
        'Product_2': 'product_2',
        'Product_3': 'product_3', 
        'Product_4': 'product_4',
        'Sentiment': 'sentiment'
    }
    df.rename(columns=rename_map, inplace=True)

    # --- D. DATA PROCESSING & CLEANING ---
    
    # 1. Normalize App Names
    if 'app_name' in df.columns:
        df['norm_app'] = df['app_name'].str.lower().apply(
            lambda x: next((v for k, v in APP_MAP.items() if k in str(x)), None)
        )
        df = df.dropna(subset=['norm_app'])
    else:
        st.error("Column 'App_Name' missing from Supabase data.")
        st.stop()

    # 2. Date Parsing (FIXED FOR IST)
    if 'at' in df.columns:
        df['at'] = pd.to_datetime(df['at'], errors='coerce', utc=True)
        df['at'] = df['at'].dt.tz_convert('Asia/Kolkata')
        df['at'] = df['at'].dt.tz_localize(None) 
        df = df.dropna(subset=['at'])
        df['Month'] = df['at'].dt.strftime('%Y-%m')

    # 3. Numeric Score
    if 'score' in df.columns:
        df['score'] = pd.to_numeric(df['score'], errors='coerce')

    # 4. Character Count Analysis Helper
    if 'content' in df.columns:
        df['char_count'] = df['content'].astype(str).str.len().fillna(0)
        df['length_group'] = df['char_count'].apply(lambda x: '<=29 Chars' if x <= 29 else '>=30 Chars')

    # 5. Handle [NET] Columns
    net_cols = [c for c in df.columns if str(c).startswith('[NET]')]
    for col in net_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df, net_cols

# ==========================================
# 3. INITIAL LOADING ANIMATION (MOTION GRAPHICS)
# ==========================================

# Placeholder for the loader
loader_placeholder = st.empty()

# Check if data is already in session state (optional, for even faster reloads)
if 'df' not in st.session_state:
    
    # -- SHOW LOADER --
    with loader_placeholder.container():
        # URL for a "Data Dashboard Analysis" Lottie Animation
        # You can swap this URL with any JSON from lottiefiles.com
        lottie_url = "https://lottie.host/67705423-745b-4303-9d8a-662551406e22/94bZp6i0k8.json"
        lottie_json = load_lottieurl(lottie_url)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if lottie_json:
                st_lottie(lottie_json, height=300, key="loader_anim")
            else:
                st.spinner("Connecting to Supabase...") # Fallback
            
            st.markdown(
                """
                <h3 style='text-align: center; color: #666;'>
                🚀 Connecting to Supabase & Analyzing 45,000+ Reviews...
                </h3>
                """, 
                unsafe_allow_html=True
            )

    # -- FETCH DATA --
    df_raw, net_cols = load_data()
    
    # Save to session state to avoid reloading on minor interactions if needed
    st.session_state['df'] = df_raw
    st.session_state['net_cols'] = net_cols

    # -- CLEAR LOADER --
    loader_placeholder.empty()

else:
    # Retrieve from session state
    df_raw = st.session_state['df']
    net_cols = st.session_state['net_cols']


if df_raw.empty:
    st.warning("No data found in Supabase.")
    st.stop()

# ==========================================
# 4. SIDEBAR FILTERS
# ==========================================
st.sidebar.title("🎛️ Slice & Dice")

# A. Date Filter
if not df_raw.empty:
    min_date = df_raw['at'].min().date()
    max_date = df_raw['at'].max().date()
    date_range = st.sidebar.date_input("📅 Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
else:
    date_range = [pd.Timestamp.now().date(), pd.Timestamp.now().date()]

# B. App Filter
all_apps = sorted(df_raw['norm_app'].unique())
sel_apps = st.sidebar.multiselect("📱 Apps", all_apps, default=all_apps)

# C. Product Filter
prod_set = set()
for c in ['product_1', 'product_2', 'product_3', 'product_4']:
    if c in df_raw.columns:
        prod_set.update(df_raw[c].dropna().unique())
all_products = sorted([str(p) for p in prod_set if p and str(p).lower() != 'nan'])
sel_products = st.sidebar.multiselect("🏷️ Products", all_products)

# D. Rating Filter
sel_ratings = st.sidebar.multiselect("⭐ Ratings", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])

# E. Search
search_query = st.sidebar.text_input("🔍 Search Reviews", "")

# --- APPLY FILTERS ---
if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (
        df_raw['norm_app'].isin(sel_apps) &
        (df_raw['at'].dt.date >= start_date) &
        (df_raw['at'].dt.date <= end_date) &
        (df_raw['score'].isin(sel_ratings))
    )
else:
    mask = (df_raw['norm_app'].isin(sel_apps))

if sel_products:
    p_mask = pd.Series(False, index=df_raw.index)
    for c in ['product_1', 'product_2', 'product_3', 'product_4']:
        if c in df_raw.columns:
            p_mask |= df_raw[c].isin(sel_products)
    mask &= p_mask

if search_query:
    mask &= df_raw['content'].str.contains(search_query, case=False, na=False)

df = df_raw[mask].copy()

# ==========================================
# 5. MAIN DASHBOARD UI
# ==========================================
st.title("🚀 Fintech Strategic Analysis Dashboard")
if len(date_range) == 2:
    st.markdown(f"**Data Period:** {date_range[0]} to {date_range[1]} | **Reviews:** {len(df):,}")

# TABS
tab_overview, tab_prod, tab_drive, tab_barr, tab_qual, tab_insights, tab_monthly = st.tabs([
    "📊 Overview & Ratings",
    "📦 Products",
    "🚀 Drivers (4-5★)",
    "🛑 Barriers (1-3★)",
    "📏 Quality & Sentiment",
    "🧠 AI & Strategic Insights",
    "📅 Monthly Trends"
])

# === TAB 1: OVERVIEW & RATINGS ===
with tab_overview:
    st.subheader("1. Ratings Distribution & Review Ratios")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Overall Ratings Distribution")
        rating_counts = df['score'].value_counts().sort_index().reset_index()
        rating_counts.columns = ['Rating', 'Count']
        fig_rating = px.bar(rating_counts, x='Rating', y='Count', color='Rating', text_auto=True,
                            title="Overall Rating Count")
        st.plotly_chart(fig_rating, use_container_width=True)
    with c2:
        st.markdown("### Ratings Ratio Across Brands")
        brand_score = pd.crosstab(df['norm_app'], df['score'], normalize='index') * 100
        fig_heatmap = px.imshow(brand_score, text_auto='.1f', aspect="auto",
                                labels=dict(x="Rating", y="Brand", color="%"),
                                title="Rating % Share per Brand",
                                color_continuous_scale='RdBu') 
        st.plotly_chart(fig_heatmap, use_container_width=True)

# === TAB 2: PRODUCTS ===
with tab_prod:
    st.subheader("2. Product Volume Analysis")
    def get_product_data(dataframe):
        cols_to_use = [c for c in ['norm_app', 'product_1', 'product_2', 'product_3', 'product_4'] if c in dataframe.columns]
        pm = dataframe[cols_to_use].melt(id_vars=['norm_app'], value_name='Product')
        return pm[pm['Product'].notna() & (pm['Product'] != '') & (pm['Product'] != 'None')]

    prod_df = get_product_data(df)
    if prod_df.empty:
        st.warning("No product data found.")
    else:
        st.markdown("### Product Volume Matrix (Base, Count, %)")
        brand_base = prod_df.groupby('norm_app').size().reset_index(name='Brand_Base')
        prod_counts = prod_df.groupby(['Product', 'norm_app']).size().reset_index(name='Count')
        prod_stats = pd.merge(prod_counts, brand_base, on='norm_app')
        prod_stats['% of Brand'] = (prod_stats['Count'] / prod_stats['Brand_Base']) * 100
        
        pivot_count = prod_stats.pivot(index='Product', columns='norm_app', values='Count').fillna(0).astype(int)
        pivot_pct = prod_stats.pivot(index='Product', columns='norm_app', values='% of Brand').fillna(0).round(1)
        
        display_df = pivot_count.copy().astype(str)
        for col in display_df.columns:
            display_df[col] = pivot_count[col].astype(str) + " (" + pivot_pct[col].astype(str) + "%)"
            
        display_df['Total Count'] = pivot_count.sum(axis=1)
        display_df = display_df.sort_values('Total Count', ascending=False)
        st.dataframe(display_df.style.highlight_max(axis=0, color='lightgreen'))

        st.markdown("### Product Mix by Brand")
        fig_prod_bar = px.bar(prod_stats, x='norm_app', y='Count', color='Product', 
                              text=prod_stats['% of Brand'].apply(lambda x: f"{x:.1f}%"),
                              title="Product Composition per Brand")
        st.plotly_chart(fig_prod_bar, use_container_width=True)

# === TAB 3: TOP DRIVERS (Positive) ===
with tab_drive:
    st.subheader("3. Top Drivers (Positive Themes from 4-5★)")
    df_pos = df[df['score'].isin([4, 5])]
    if df_pos.empty or not net_cols:
        st.warning("No positive data available.")
    else:
        base_counts = df_pos.groupby('norm_app').size()
        st.markdown(f"**Base (Total 4-5★ Reviews):** {base_counts.sum()}")
        valid_net_cols = [c for c in net_cols if c in df_pos.columns]
        
        driver_sums = df_pos.groupby('norm_app')[valid_net_cols].sum().T
        driver_sums.index = driver_sums.index.str.replace('[NET]', '', regex=False).str.strip()
        driver_pct = driver_sums.div(base_counts, axis=1).fillna(0) * 100
        
        driver_pct['Average'] = driver_pct.mean(axis=1)
        top_10_drivers = driver_pct.sort_values('Average', ascending=False).head(10).drop(columns=['Average'])
        
        st.markdown("#### Top 10 Drivers (% of 4-5★ Reviews)")
        st.dataframe(top_10_drivers.style.background_gradient(cmap='Greens', axis=None).format("{:.1f}%"))
        
        fig_drive_heat = px.imshow(top_10_drivers, text_auto='.1f', aspect="auto",
                                   color_continuous_scale='Greens', title="Driver Intensity Heatmap")
        st.plotly_chart(fig_drive_heat, use_container_width=True)

# === TAB 4: TOP BARRIERS (Negative) ===
with tab_barr:
    st.subheader("4. Top Barriers (Negative Themes from 1-3★)")
    df_neg = df[df['score'].isin([1, 2, 3])]
    if df_neg.empty or not net_cols:
        st.warning("No negative data available.")
    else:
        base_counts_neg = df_neg.groupby('norm_app').size()
        st.markdown(f"**Base (Total 1-3★ Reviews):** {base_counts_neg.sum()}")
        valid_net_cols = [c for c in net_cols if c in df_neg.columns]
        
        barrier_sums = df_neg.groupby('norm_app')[valid_net_cols].sum().T
        barrier_sums.index = barrier_sums.index.str.replace('[NET]', '', regex=False).str.strip()
        barrier_pct = barrier_sums.div(base_counts_neg, axis=1).fillna(0) * 100
        
        barrier_pct['Average'] = barrier_pct.mean(axis=1)
        top_10_barriers = barrier_pct.sort_values('Average', ascending=False).head(10).drop(columns=['Average'])
        
        st.markdown("#### Top 10 Barriers (% of 1-3★ Reviews)")
        st.dataframe(top_10_barriers.style.background_gradient(cmap='Reds', axis=None).format("{:.1f}%"))
        
        fig_barr_heat = px.imshow(top_10_barriers, text_auto='.1f', aspect="auto",
                                  color_continuous_scale='Reds', title="Barrier Intensity Heatmap")
        st.plotly_chart(fig_barr_heat, use_container_width=True)

# === TAB 5: QUALITY & SENTIMENT ===
with tab_qual:
    st.subheader("5. Review Quality & Deep Dive")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Sentiment by Rating")
        if 'sentiment' in df.columns:
            sent_rating = pd.crosstab(df['score'], df['sentiment'], normalize='index') * 100
            fig_sent_stack = px.bar(sent_rating, x=sent_rating.index, y=sent_rating.columns,
                                    title="Sentiment Distribution per Rating Level",
                                    color_discrete_map={'Positive': '#2ca02c', 'Negative': '#d62728', 'Neutral': '#ff7f0e'})
            st.plotly_chart(fig_sent_stack, use_container_width=True)
    with c2:
        st.markdown("### Review Length by Rating")
        len_rating = pd.crosstab(df['score'], df['length_group'], normalize='index') * 100
        fig_len_stack = px.bar(len_rating, x=len_rating.index, y=len_rating.columns,
                               title="Short vs Long Reviews per Rating Level",
                               color_discrete_map={'<=29 Chars': '#ff7f0e', '>=30 Chars': '#1f77b4'})
        st.plotly_chart(fig_len_stack, use_container_width=True)
    st.markdown("---")
    st.markdown("### Average Review Length (Characters)")
    avg_len_overall = df['char_count'].mean()
    avg_len_brand = df.groupby('norm_app')['char_count'].mean().reset_index().sort_values('char_count')
    col_a, col_b = st.columns([1, 3])
    col_a.metric("Overall Avg Length", f"{avg_len_overall:.0f} chars")
    fig_avg_len = px.bar(avg_len_brand, x='char_count', y='norm_app', orientation='h',
                         text_auto='.0f', title="Avg Character Count by Brand",
                         color='norm_app', color_discrete_map=COLOR_MAP)
    col_b.plotly_chart(fig_avg_len, use_container_width=True)

# === TAB 6: AI & STRATEGIC INSIGHTS ===
with tab_insights:
    st.subheader("🧠 Sharp AI Strategic Insights")
    st.markdown("Automated strategic profiling per brand based on current data patterns.")

    selected_brands = df['norm_app'].unique()
    cat_avg_rating = df['score'].mean()
    cat_avg_len = df['char_count'].mean()

    for brand in selected_brands:
        with st.expander(f"📌 Strategic Insights for: {brand}", expanded=True):
            b_df = df[df['norm_app'] == brand]
            
            b_rating = b_df['score'].mean()
            rating_diff = ((b_rating - cat_avg_rating) / cat_avg_rating) * 100
            rating_status = "Outperforming" if rating_diff > 0 else "Underperforming"
            
            b_pos = b_df[b_df['score'].isin([4,5])]
            if not b_pos.empty and net_cols:
                valid_drivers = [c for c in net_cols if c in b_pos.columns]
                driver_sums = b_pos[valid_drivers].sum()
                top_driver_name = driver_sums.idxmax().replace('[NET]', '').strip()
                top_driver_pct = (driver_sums.max() / len(b_pos) * 100)
            else:
                top_driver_name = "N/A"
                top_driver_pct = 0
                
            b_neg = b_df[b_df['score'].isin([1,2,3])]
            if not b_neg.empty and net_cols:
                valid_barriers = [c for c in net_cols if c in b_neg.columns]
                barrier_sums = b_neg[valid_barriers].sum()
                top_barrier_name = barrier_sums.idxmax().replace('[NET]', '').strip()
                top_barrier_pct = (barrier_sums.max() / len(b_neg) * 100)
            else:
                top_barrier_name = "N/A"
                top_barrier_pct = 0
            
            if 'product_1' in b_df.columns:
                top_prod = b_df['product_1'].mode()[0] if not b_df['product_1'].empty else "N/A"
                prod_share = (len(b_df[b_df['product_1'] == top_prod]) / len(b_df) * 100)
            else:
                top_prod = "N/A"
                prod_share = 0

            b_len = b_df['char_count'].mean()
            len_diff = "Detailed" if b_len > cat_avg_len else "Brief"

            col_i1, col_i2 = st.columns([1, 4])
            with col_i1:
                st.metric("Avg Score", f"{b_rating:.2f}", delta=f"{rating_diff:.1f}% vs Cat. Avg")
            with col_i2:
                st.markdown(f"""
                * **Competitive Position:** Currently **{rating_status}** the category average ({cat_avg_rating:.2f}).
                * **The Superpower:** Strongest driver is **'{top_driver_name}'**, mentioned in **{top_driver_pct:.1f}%** of positive reviews.
                * **The Achilles Heel:** Primary friction point is **'{top_barrier_name}'**, appearing in **{top_barrier_pct:.1f}%** of complaints.
                * **Product Anchor:** Heavily reliant on **{top_prod}**, which accounts for **{prod_share:.1f}%** of total volume.
                * **User Voice:** Users tend to leave **{len_diff}** feedback (Avg {b_len:.0f} chars), suggesting { 'high engagement/frustration' if b_len > cat_avg_len else 'transactional usage' }.
                """)

# === TAB 7: MONTHLY TRENDS ===
with tab_monthly:
    st.subheader("7. Monthly Brand Comparison Trends")
    st.info("Month-over-month performance comparison (Volume, Ratings & Drivers).")

    if 'Month' not in df.columns:
        st.error("Month column is missing.")
    else:
        df_trend = df.sort_values('at')
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### Monthly Review Volume")
            trend_vol = df_trend.groupby(['Month', 'norm_app']).size().reset_index(name='Count')
            fig_vol = px.line(trend_vol, x='Month', y='Count', color='norm_app', markers=True, 
                              title="Monthly Volume by Brand", color_discrete_map=COLOR_MAP)
            st.plotly_chart(fig_vol, use_container_width=True)

        with c2:
            st.markdown("### Monthly Average Rating")
            trend_score = df_trend.groupby(['Month', 'norm_app'])['score'].mean().reset_index()
            fig_score = px.line(trend_score, x='Month', y='score', color='norm_app', markers=True, 
                                title="Monthly Avg Rating by Brand", color_discrete_map=COLOR_MAP)
            fig_score.update_yaxes(range=[1, 5])
            st.plotly_chart(fig_score, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🌡️ Heatmap: Rating Performance over Time")
        pivot_rating = trend_score.pivot(index='norm_app', columns='Month', values='score')
        pivot_rating = pivot_rating[sorted(pivot_rating.columns)]
        fig_heat = px.imshow(pivot_rating, text_auto='.2f', aspect="auto",
                             color_continuous_scale='RdBu', title="Monthly Average Rating (Red=Low, Blue=High)")
        st.plotly_chart(fig_heat, use_container_width=True)
