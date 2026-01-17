import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Fintech Strategic Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
DB_NAME = r'C:\Users\vidya\PycharmProjects\GRSQL1\fintech_reviews.db'

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
# 2. DATA LOADING (Robust & Cached)
# ==========================================
@st.cache_data
def load_data():
    engine = create_engine(f'sqlite:///{DB_NAME}')

    try:
        df = pd.read_sql("SELECT * FROM reviews", engine)
    except Exception as e:
        st.error(f"Could not read database. Error: {e}")
        return pd.DataFrame(), []

    # Standardize Column Names
    rename_map = {
        'App_Name': 'app_name', 'Rating': 'score',
        'Review_Date': 'at', 'Review_Text': 'content',
        'Product_1': 'product_1', 'Product_2': 'product_2',
        'Product_3': 'product_3', 'Product_4': 'product_4',
        'Sentiment': 'sentiment'
    }
    df.rename(columns=rename_map, inplace=True)

    # Normalize App Names
    df['norm_app'] = df['app_name'].str.lower().apply(
        lambda x: next((v for k, v in APP_MAP.items() if k in str(x)), None)
    )
    df = df.dropna(subset=['norm_app'])

    # Date Parsing
    df['at'] = df['at'].astype(str)
    df['at'] = pd.to_datetime(df['at'], errors='coerce', dayfirst=False, format='mixed')
    df = df.dropna(subset=['at'])
    df['Month'] = df['at'].dt.strftime('%Y-%m')

    # Character Count Analysis Helper
    df['char_count'] = df['content'].str.len().fillna(0)
    df['length_group'] = df['char_count'].apply(lambda x: '<=29 Chars' if x <= 29 else '>=30 Chars')

    # Identify [NET] Columns dynamically
    net_cols = [c for c in df.columns if c.startswith('[NET]')]

    return df, net_cols


# Load Data
df_raw, net_cols = load_data()

if df_raw.empty:
    st.stop()

# ==========================================
# 3. SIDEBAR FILTERS (Slice & Dice) - KEPT AS IS
# ==========================================
st.sidebar.title("🎛️ Slice & Dice")

# A. Date Filter
min_date, max_date = df_raw['at'].min().date(), df_raw['at'].max().date()
date_range = st.sidebar.date_input("📅 Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# B. App Filter
all_apps = sorted(df_raw['norm_app'].unique())
sel_apps = st.sidebar.multiselect("📱 Apps", all_apps, default=all_apps)

# C. Product Filter
prod_set = set()
for c in ['product_1', 'product_2', 'product_3', 'product_4']:
    if c in df_raw.columns:
        prod_set.update(df_raw[c].dropna().unique())
all_products = sorted([str(p) for p in prod_set if p])
sel_products = st.sidebar.multiselect("🏷️ Products", all_products)

# D. Rating Filter
sel_ratings = st.sidebar.multiselect("⭐ Ratings", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])

# E. Search
search_query = st.sidebar.text_input("🔍 Search Reviews", "")

# --- APPLY FILTERS ---
mask = (
        df_raw['norm_app'].isin(sel_apps) &
        (df_raw['at'].dt.date >= date_range[0]) &
        (df_raw['at'].dt.date <= date_range[1]) &
        (df_raw['score'].isin(sel_ratings))
)

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
# 4. MAIN DASHBOARD
# ==========================================
st.title("🚀 Fintech Strategic Analysis Dashboard")
st.markdown(f"**Data Period:** {date_range[0]} to {date_range[1]} | **Reviews:** {len(df):,}")

# TABS
tab_sent, tab_prod, tab_drive, tab_barr, tab_qual = st.tabs([
    "😊 Sentiment (Overall vs MoM)",
    "📦 Products (Top 5 + Others)",
    "🚀 Top Drivers (Positive)",
    "🛑 Top Barriers (Negative)",
    "📏 Review Quality (Length)"
])

# === TAB 1: SENTIMENT (Overall vs MoM) ===
with tab_sent:
    st.subheader("1. Sentiment Analysis")

    if 'sentiment' not in df.columns:
        st.error("Sentiment column missing in database.")
    else:
        c1, c2 = st.columns(2)

        # A. Overall Sentiment
        with c1:
            st.markdown("### Overall Sentiment Distribution")
            sent_overall = df['sentiment'].value_counts().reset_index()
            sent_overall.columns = ['Sentiment', 'Count']
            fig_sent_pie = px.pie(sent_overall, names='Sentiment', values='Count', hole=0.4, color='Sentiment',
                                  color_discrete_map={'Positive': '#2ca02c', 'Negative': '#d62728',
                                                      'Neutral': '#ff7f0e'})
            st.plotly_chart(fig_sent_pie, use_container_width=True)

        # B. MoM Sentiment Trend
        with c2:
            st.markdown("### MoM Sentiment Trend")
            # Group by Month and Sentiment
            sent_mom = df.groupby(['Month', 'sentiment']).size().reset_index(name='Count')
            # Calculate percentages to normalize
            sent_mom['Total'] = sent_mom.groupby('Month')['Count'].transform('sum')
            sent_mom['Pct'] = (sent_mom['Count'] / sent_mom['Total']) * 100

            fig_sent_mom = px.bar(sent_mom, x='Month', y='Pct', color='sentiment',
                                  title="Sentiment Share % MoM",
                                  color_discrete_map={'Positive': '#2ca02c', 'Negative': '#d62728',
                                                      'Neutral': '#ff7f0e'})
            st.plotly_chart(fig_sent_mom, use_container_width=True)

# === TAB 2: PRODUCTS (Top 5 + Others) ===
with tab_prod:
    st.subheader("2. Product Volume Analysis")


    # helper to melt products
    def get_product_data(dataframe):
        pm = dataframe[['Month', 'norm_app', 'product_1', 'product_2', 'product_3', 'product_4']].melt(
            id_vars=['Month', 'norm_app'], value_name='Product'
        )
        return pm[pm['Product'].notna() & (pm['Product'] != '')]


    prod_df = get_product_data(df)

    if prod_df.empty:
        st.warning("No product data found.")
    else:
        # 1. Identify Top 5 Products Overall
        top_products = prod_df['Product'].value_counts().head(5).index.tolist()

        # 2. Relabel others
        prod_df['Product_Grouped'] = prod_df['Product'].apply(lambda x: x if x in top_products else 'Others')

        # Display Base
        st.metric("BASE (Total Product Mentions)", f"{len(prod_df):,}")

        c1, c2 = st.columns(2)

        # A. Overall Volume
        with c1:
            st.markdown("### Overall Product Volume (Top 5 + Others)")
            overall_counts = prod_df['Product_Grouped'].value_counts().reset_index()
            overall_counts.columns = ['Product', 'Count']
            fig_prod_over = px.bar(overall_counts, x='Product', y='Count', color='Product', text_auto=True)
            st.plotly_chart(fig_prod_over, use_container_width=True)

        # B. MoM Volume
        with c2:
            st.markdown("### MoM Product Trends")
            mom_counts = prod_df.groupby(['Month', 'Product_Grouped']).size().reset_index(name='Count')
            fig_prod_mom = px.line(mom_counts, x='Month', y='Count', color='Product_Grouped', markers=True)
            st.plotly_chart(fig_prod_mom, use_container_width=True)

# === TAB 3: TOP DRIVERS (Positive NETs) ===
with tab_drive:
    st.subheader("3. Top Drivers (Positive Themes from 4-5★)")

    # Filter 4-5 Stars
    df_pos = df[df['score'].isin([4, 5])]
    base_counts = df_pos.groupby('norm_app').size()  # Base is total positive reviews

    if df_pos.empty or not net_cols:
        st.warning("No positive data or NET columns available.")
    else:
        # Aggregate NETs
        net_sums = df_pos.groupby('norm_app')[net_cols].sum().T
        net_sums.index = net_sums.index.str.replace('[NET]', '')

        # Calculate % of Base
        net_pct = net_sums.div(base_counts, axis=1).fillna(0) * 100

        # Filter Top 10 Drivers (by average across apps or max)
        net_pct['Average'] = net_pct.mean(axis=1)
        top_drivers = net_pct.sort_values('Average', ascending=False).head(10).drop(columns=['Average'])

        st.markdown("#### Top Drivers (% of Positive Reviews)")
        st.dataframe(top_drivers.style.background_gradient(cmap='Greens', axis=None).format("{:.1f}%"))

        # Chart
        top_drivers_melt = top_drivers.reset_index().melt(id_vars='index')
        fig_drive = px.bar(top_drivers_melt, y='index', x='value', color='norm_app', barmode='group',
                           orientation='h', labels={'value': '% of Base', 'index': 'Driver'},
                           color_discrete_map=COLOR_MAP, height=600)
        fig_drive.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_drive, use_container_width=True)

# === TAB 4: TOP BARRIERS (Negative NETs) ===
with tab_barr:
    st.subheader("4. Top Barriers (Negative Themes from 1-3★)")

    # Filter 1-3 Stars
    df_neg = df[df['score'].isin([1, 2, 3])]
    base_counts_neg = df_neg.groupby('norm_app').size()

    if df_neg.empty or not net_cols:
        st.warning("No negative data or NET columns available.")
    else:
        # Aggregate NETs
        net_sums_neg = df_neg.groupby('norm_app')[net_cols].sum().T
        net_sums_neg.index = net_sums_neg.index.str.replace('[NET]', '')

        # Calculate % of Base
        net_pct_neg = net_sums_neg.div(base_counts_neg, axis=1).fillna(0) * 100

        # Filter Top 10 Barriers
        net_pct_neg['Average'] = net_pct_neg.mean(axis=1)
        top_barriers = net_pct_neg.sort_values('Average', ascending=False).head(10).drop(columns=['Average'])

        st.markdown("#### Top Barriers (% of Negative Reviews)")
        st.dataframe(top_barriers.style.background_gradient(cmap='Reds', axis=None).format("{:.1f}%"))

        # Chart
        top_barr_melt = top_barriers.reset_index().melt(id_vars='index')
        fig_barr = px.bar(top_barr_melt, y='index', x='value', color='norm_app', barmode='group',
                          orientation='h', labels={'value': '% of Base', 'index': 'Barrier'},
                          color_discrete_map=COLOR_MAP, height=600)
        fig_barr.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_barr, use_container_width=True)

# === TAB 5: REVIEW QUALITY (Char Count) ===
with tab_qual:
    st.subheader("5. Review Quality (Character Count)")
    st.info("Comparison of Short (<=29 chars) vs Long (>=30 chars) reviews.")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Overall Split")
        len_counts = df['length_group'].value_counts().reset_index()
        len_counts.columns = ['Length Group', 'Count']
        fig_len_pie = px.pie(len_counts, names='Length Group', values='Count', color='Length Group',
                             color_discrete_map={'<=29 Chars': '#ff7f0e', '>=30 Chars': '#1f77b4'})
        st.plotly_chart(fig_len_pie, use_container_width=True)

    with c2:
        st.markdown("### Split by Brand")
        len_brand = df.groupby(['norm_app', 'length_group']).size().reset_index(name='Count')
        # Calculate % for stacked bar
        len_brand['Total'] = len_brand.groupby('norm_app')['Count'].transform('sum')
        len_brand['Pct'] = (len_brand['Count'] / len_brand['Total']) * 100

        fig_len_bar = px.bar(len_brand, x='norm_app', y='Pct', color='length_group',
                             title="Review Length Distribution by Brand",
                             color_discrete_map={'<=29 Chars': '#ff7f0e', '>=30 Chars': '#1f77b4'})
        st.plotly_chart(fig_len_bar, use_container_width=True)

    # Show Raw Data sample for Short Reviews
    with st.expander("View Sample Short Reviews (<=29 chars)"):
        st.dataframe(df[df['length_group'] == '<=29 Chars'][['at', 'norm_app', 'score', 'content']].head(100))