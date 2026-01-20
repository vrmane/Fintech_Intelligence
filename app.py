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
from datetime import timedelta, datetime
import pytz
import gc

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
        .stApp { background-color: #0b0f19; color: #e2e8f0; }
        div[data-testid="metric-container"] {
            background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
        }
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
            margin-bottom: 25px;
        }
        .timestamp-box {
            font-size: 0.85em;
            color: #94a3b8;
            margin-bottom: 15px;
            border-bottom: 1px solid #334155;
            padding-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .live-dot {
            height: 8px;
            width: 8px;
            background-color: #22c55e;
            border-radius: 50%;
            display: inline-block;
        }
        hr { margin: 2em 0; border-color: #334155; }
        h1, h2, h3, h4 { color: #f8fafc; font-family: 'Inter', sans-serif; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except:
        return None

@st.cache_data(show_spinner=False)
def get_top_words_pct(text_series, total_count, top_n=20):
    stop_words = set(['the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'for', 'i', 'this', 'my', 'app', 'loan', 'money', 'very', 'good', 'bad', 'worst', 'best', 'application', 'not', 'but', 'on', 'with', 'are', 'was', 'have', 'be', 'so', 'me', 'you', 'please', 'give'])
    counter = Counter()
    for text in text_series.dropna():
        clean_text = re.sub(r'[^a-z\s]', '', str(text).lower())
        words = [w for w in clean_text.split() if w not in stop_words and len(w) > 2]
        counter.update(words)
    data = []
    for word, count in counter.most_common(top_n):
        pct = (count / total_count) * 100
        data.append({'Word': word, 'Pct': pct, 'Count': count})
    return pd.DataFrame(data)

def calculate_delta(df, col, agg_func='count'):
    if 'Month' not in df.columns or df.empty: return 0, 0
    months = sorted(df['Month'].unique())
    if len(months) < 2: return 0, 0
    curr, prev = months[-1], months[-2]
    
    if agg_func == 'count':
        curr_val = len(df[df['Month'] == curr])
        prev_val = len(df[df['Month'] == prev])
    elif agg_func == 'mean':
        curr_val = df[df['Month'] == curr][col].mean()
        prev_val = df[df['Month'] == prev][col].mean()
    
    if prev_val == 0: return curr_val, 0
    delta = ((curr_val - prev_val) / prev_val) * 100
    return curr_val, delta

# ==========================================
# 3. DATA ENGINE
# ==========================================
@st.cache_resource
def init_connection():
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except: return None

supabase = init_connection()

@st.cache_data(ttl=900, show_spinner=False)
def load_data():
    all_rows = []
    start, batch = 0, 2000
    while True:
        res = supabase.table("reviews").select("*").range(start, start + batch - 1).execute()
        if not res.data: break
        all_rows.extend(res.data)
        if len(res.data) < batch: break
        start += batch

    if not all_rows: return pd.DataFrame()
    df = pd.DataFrame(all_rows)

    ist = pytz.timezone('Asia/Kolkata')
    
    if 'Review_Date' in df.columns:
        df['at'] = pd.to_datetime(df['Review_Date'], errors='coerce')
        if df['at'].dt.tz is None:
            df['at'] = df['at'].dt.tz_localize('UTC').dt.tz_convert(ist)
        else:
            df['at'] = df['at'].dt.tz_convert(ist)
        df.dropna(subset=['at'], inplace=True)
        df['Month'] = df['at'].dt.strftime('%Y-%m')
        df['Week'] = df['at'].dt.strftime('%Y-W%V')

    if 'Rating' in df.columns:
        df['score'] = pd.to_numeric(df['Rating'], errors='coerce')
        df['Sentiment_Label'] = pd.cut(df['score'], bins=[0, 2, 3, 5], labels=['Negative', 'Neutral', 'Positive'])
    
    if 'Review_Text' in df.columns:
        df['char_count'] = df['Review_Text'].astype(str).str.len().fillna(0).astype(np.int32)
        df['length_bucket'] = df['char_count'].apply(lambda x: 'Brief (<=29)' if x <= 29 else 'Detailed (>=30)').astype('category')

    for col in ['App_Name', 'Product_1', 'Sentiment']:
        if col in df.columns: df[col] = df[col].astype('category')

    net_cols = [c for c in df.columns if str(c).startswith('[NET]')]
    clean_net_map = {}
    for col in net_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.int8)
        clean_name = col.replace('[NET]', '').replace('[NET] ', '').strip()
        clean_net_map[col] = clean_name
        df.rename(columns={col: clean_name}, inplace=True)
    
    st.session_state['theme_cols'] = list(clean_net_map.values())
    st.session_state['last_fetched'] = datetime.now(ist).strftime("%d %b %Y, %I:%M %p IST")
    gc.collect()
    return df

# ==========================================
# 4. INITIAL LOADER
# ==========================================
if 'data_loaded' not in st.session_state:
    loader = st.empty()
    with loader.container():
        lottie_json = load_lottieurl("https://lottie.host/9e5c4644-841b-43c3-982d-19597143c690/w5h8o4t9zD.json") 
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if lottie_json: st_lottie(lottie_json, height=300, key="loader")
            st.markdown("<h3 style='text-align:center; color:#38bdf8;'>Establishing Secure Connection...</h3>", unsafe_allow_html=True)
    df_raw = load_data()
    st.session_state['df_raw'] = df_raw
    st.session_state['data_loaded'] = True
    time.sleep(1)
    loader.empty()
else:
    df_raw = st.session_state['df_raw']

if df_raw.empty:
    st.error("No Data Found.")
    st.stop()

# ==========================================
# 5. VISUALIZATION FUNCTIONS
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
        last, prev = periods[-1], periods[-2]
        last_df = b_df[b_df[time_col] == last]
        prev_df = b_df[b_df[time_col] == prev]
        last_vol, prev_vol = len(last_df), len(prev_df)
        growth = []
        for t in top_themes:
            if prev_vol == 0: g = 0.0
            else:
                s_last = last_df[t].sum() / last_vol if last_vol else 0
                s_prev = prev_df[t].sum() / prev_vol
                g = ((s_last - s_prev) / s_prev * 100) if s_prev > 0 else 0
            growth.append(g)
        matrix[brand] = growth
    return pd.DataFrame(matrix, index=top_themes)

def get_brand_insights(df, brand, theme_cols):
    b_df = df[df['App_Name'] == brand]
    if b_df.empty: return ["No data."]
    insights = []
    cohort_avg = df['score'].mean()
    brand_avg = b_df['score'].mean()
    diff = brand_avg - cohort_avg
    pos_text = "outperforming" if diff > 0 else "lagging behind"
    insights.append(f"**Position:** {pos_text} market by **{abs(diff):.2f} ⭐**")
    
    pos_df = b_df[b_df['score'] >= 4]
    if not pos_df.empty and theme_cols:
        valid = [t for t in theme_cols if t in pos_df.columns]
        if valid:
            top_d = pos_df[valid].sum().idxmax()
            pct = (pos_df[top_d].sum() / len(pos_df)) * 100
            insights.append(f"**Driver:** <span style='color:#4ade80'>{top_d}</span> (**{pct:.0f}%** of positive)")
            
    neg_df = b_df[b_df['score'] <= 3]
    if not neg_df.empty and theme_cols:
        valid = [t for t in theme_cols if t in neg_df.columns]
        if valid:
            top_b = neg_df[valid].sum().idxmax()
            pct = (neg_df[top_b].sum() / len(neg_df)) * 100
            insights.append(f"**Risk:** <span style='color:#f87171'>{top_b}</span> (**{pct:.0f}%** of negative)")
    return insights

def build_period_matrix(sub_df, theme_cols, sel_brands):
    if sub_df.empty or not theme_cols: return None, None
    periods = sorted(sub_df['Period'].unique())
    base_matrix = sub_df.groupby(['Period', 'App_Name']).size().unstack(fill_value=0)
    valid = [t for t in theme_cols if t in sub_df.columns]
    top_themes = sub_df[valid].sum().sort_values(ascending=False).head(20).index.tolist()
    
    data = []
    base_row_data = {}
    for p in periods:
        for b in sel_brands:
            if b not in base_matrix.columns: continue
            base_row_data[(p, b)] = base_matrix.loc[p, b]

    for theme in top_themes:
        row = {}
        for p in periods:
            for b in sel_brands:
                if b not in base_matrix.columns: continue
                mask = (sub_df['Period'] == p) & (sub_df['App_Name'] == b)
                if mask.any():
                    count = sub_df.loc[mask, theme].sum()
                    base = base_matrix.loc[p, b]
                    val = (count / base * 100) if base > 0 else 0
                else: val = 0
                row[(p, b)] = val
        data.append(row)
        
    final_df = pd.DataFrame(data, index=top_themes)
    final_df.columns = pd.MultiIndex.from_tuples(final_df.columns)
    final_df = final_df.sort_index(axis=1)
    base_row_df = pd.DataFrame([base_row_data], index=["Base (N)"])
    base_row_df.columns = pd.MultiIndex.from_tuples(base_row_df.columns)
    base_row_df = base_row_df.reindex(columns=final_df.columns)
    return pd.concat([base_row_df, final_df]), top_themes

# === NEW: Brand Matrix Builder ===
def build_brand_matrix(sub_df, theme_cols, sel_brands):
    if sub_df.empty or not theme_cols: return None, None
    
    # 1. Base Counts per Brand
    # Use reindex to ensure all selected brands exist even if count is 0
    base_counts = sub_df['App_Name'].value_counts().reindex(sel_brands, fill_value=0)
    
    # 2. Top Themes
    valid = [t for t in theme_cols if t in sub_df.columns]
    top_themes = sub_df[valid].sum().sort_values(ascending=False).head(20).index.tolist()
    
    # 3. Build Data
    data = []
    for theme in top_themes:
        row = {}
        for brand in sel_brands:
            base = base_counts[brand]
            if base > 0:
                count = sub_df[sub_df['App_Name'] == brand][theme].sum()
                row[brand] = (count / base) * 100
            else:
                row[brand] = 0
        data.append(row)
        
    final_df = pd.DataFrame(data, index=top_themes)
    
    # 4. Base Row
    base_row_df = pd.DataFrame([base_counts.to_dict()], index=["Base (N)"])
    
    # Combine
    combined_df = pd.concat([base_row_df, final_df])
    return combined_df, top_themes

# ==========================================
# 6. SIDEBAR & FILTERS
# ==========================================
with st.sidebar:
    st.title("🎛️ Command Center")
    st.success(f"🟢 Live: {len(df_raw):,} Rows")
    st.markdown("---")
    
    min_d, max_d = df_raw['at'].min().date(), df_raw['at'].max().date()
    date_range = st.date_input("Period", [min_d, max_d], min_value=min_d, max_value=max_d)
    
    all_brands = sorted(df_raw['App_Name'].unique().tolist())
    sel_brands = st.multiselect("Brands", all_brands, default=all_brands)
    
    st.markdown("### 📊 Metrics")
    sel_ratings = st.multiselect("Ratings", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
    
    sel_sent = []
    if 'Sentiment' in df_raw.columns:
        sent_opts = sorted(df_raw['Sentiment'].dropna().unique().tolist())
        sel_sent = st.multiselect("Sentiment", sent_opts)
    
    sel_prods = []
    if 'Product_1' in df_raw.columns:
        all_prods = sorted(df_raw['Product_1'].dropna().unique().tolist())
        sel_prods = st.multiselect("Product Type", all_prods)

# GLOBAL FILTER APPLICATION
if len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0]).tz_localize('Asia/Kolkata')
    end_date = pd.to_datetime(date_range[1]).tz_localize('Asia/Kolkata') + timedelta(days=1) - timedelta(seconds=1)
    mask = (df_raw['at'] >= start_date) & (df_raw['at'] <= end_date)
else:
    mask = pd.Series([True] * len(df_raw))

mask &= df_raw['App_Name'].isin(sel_brands)
mask &= df_raw['score'].isin(sel_ratings)
if sel_prods: mask &= df_raw['Product_1'].isin(sel_prods)
if sel_sent: mask &= df_raw['Sentiment'].isin(sel_sent)

df = df_raw[mask].copy()
theme_cols = st.session_state.get('theme_cols', [])

# ==========================================
# 7. DASHBOARD
# ==========================================
st.title("🦅 Strategic Intelligence Platform")

# Timestamp
last_time = st.session_state.get('last_fetched', 'Just now')
st.markdown(f"<div class='timestamp-box'><span class='live-dot'></span>Data Last Fetched: {last_time}</div>", unsafe_allow_html=True)

tab_exec, tab_drivers, tab_compare, tab_monthly, tab_trends, tab_text, tab_ai = st.tabs([
    "📊 Boardroom Summary", "🚀 Drivers & Barriers", "⚔️ Head-to-Head", "📅 Period-Over-Period Matrix", "📈 Trends", "🔡 Text Analytics", "🤖 AI Analyst"
])

# === TAB 1: BOARDROOM ===
with tab_exec:
    curr_vol, delta_vol = calculate_delta(df, 'score', 'count')
    curr_csat, delta_csat = calculate_delta(df, 'score', 'mean')
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Total Volume", f"{len(df):,}", delta=f"{delta_vol:.1f}% MoM")
    with k2: st.metric("Avg Rating", f"{df['score'].mean():.2f} ⭐", delta=f"{delta_csat:.2f} pts MoM")
    prom = len(df[df['score']==5])
    det = len(df[df['score']<=3])
    vol = len(df)
    nps = ((prom - det) / vol * 100) if vol > 0 else 0
    with k3: st.metric("NPS Proxy", f"{nps:.0f}")
    risk = (len(df[df['score']==1]) / vol * 100) if vol > 0 else 0
    with k4: st.metric("Critical Risk", f"{risk:.1f}%", delta="1-Star %", delta_color="inverse")
    
    st.markdown("---")
    
    st.markdown("#### 🏥 Brand Pulse (Live Breakdown)")
    kpi_df = df.groupby('App_Name', observed=True).agg(
        Vol=('score', 'count'),
        CSAT=('score', 'mean'),
        Promoters=('score', lambda x: (x==5).sum()),
        Detractors=('score', lambda x: (x<=3).sum())
    ).reset_index()
    kpi_df = kpi_df[kpi_df['Vol'] > 0]
    kpi_df['NPS Proxy'] = ((kpi_df['Promoters'] - kpi_df['Detractors']) / kpi_df['Vol'] * 100).round(0)
    kpi_df['CSAT'] = kpi_df['CSAT'].round(2)
    deltas = []
    for brand in kpi_df['App_Name']:
        b_df = df[df['App_Name'] == brand]
        _, d_vol = calculate_delta(b_df, 'score', 'count')
        _, d_csat = calculate_delta(b_df, 'score', 'mean')
        deltas.append({'App_Name': brand, 'Vol Delta': f"{d_vol:+.1f}%", 'CSAT Delta': f"{d_csat:+.2f}"})
    if deltas:
        delta_df = pd.DataFrame(deltas)
        final_kpi = kpi_df.merge(delta_df, on='App_Name')[['App_Name', 'Vol', 'Vol Delta', 'CSAT', 'CSAT Delta', 'NPS Proxy']]
        st.dataframe(final_kpi.style.background_gradient(subset=['CSAT'], cmap='Greens'), use_container_width=True, hide_index=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        fig_don = px.pie(kpi_df, values='Vol', names='App_Name', hole=0.4, title="Volume Share by Brand")
        fig_don.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(dark_chart(fig_don), use_container_width=True, key="exec_don")
    with c2:
        fig_scat = px.scatter(kpi_df, x="CSAT", y="NPS Proxy", size="Vol", color="App_Name", text="App_Name", title="Strategic Positioning")
        fig_scat.update_traces(textposition='top center')
        st.plotly_chart(dark_chart(fig_scat), use_container_width=True, key="exec_scat")

    c3, c4 = st.columns(2)
    with c3:
        sent_counts = df.groupby(['App_Name', 'Sentiment_Label'], observed=True).size().reset_index(name='Count')
        tot = sent_counts.groupby('App_Name')['Count'].transform('sum')
        sent_counts['Pct'] = (sent_counts['Count'] / tot) * 100
        fig_stack = px.bar(sent_counts, x="App_Name", y="Pct", color="Sentiment_Label", 
                           color_discrete_map={'Positive': '#10b981', 'Neutral': '#64748b', 'Negative': '#ef4444'},
                           title="Sentiment Ratio (%)", text_auto='.0f',
                           hover_data={'Count': True, 'Pct': ':.1f'})
        st.plotly_chart(dark_chart(fig_stack), use_container_width=True, key="exec_stack")
    with c4:
        len_counts = df.groupby(['App_Name', 'length_bucket'], observed=True).size().reset_index(name='Count')
        l_tot = len_counts.groupby('App_Name')['Count'].transform('sum')
        len_counts['Pct'] = (len_counts['Count'] / l_tot) * 100
        fig_len = px.bar(len_counts, x="App_Name", y="Pct", color="length_bucket", barmode='group',
                         title="Brief vs Detailed (%)", text_auto='.0f',
                         hover_data={'Count': True, 'Pct': ':.1f'})
        st.plotly_chart(dark_chart(fig_len), use_container_width=True, key="exec_len")

# === TAB 2: DRIVERS & BARRIERS (TABLES WITH BASE) ===
with tab_drivers:
    st.markdown("### 🚦 Strategic Drivers & Barriers")
    
    # 1. DRIVERS TABLE
    st.markdown("#### 🚀 Drivers (4-5★)")
    drivers_df = df[df['score'] >= 4]
    df_d, top_d = build_brand_matrix(drivers_df, theme_cols, sel_brands)
    if df_d is not None:
        st.dataframe(
            df_d.style
            .background_gradient(cmap='Greens', subset=pd.IndexSlice[top_d, :], axis=None)
            .format("{:.1f}", subset=pd.IndexSlice[top_d, :])
            .format("{:.0f}", subset=pd.IndexSlice[['Base (N)'], :])
            .set_properties(subset=pd.IndexSlice[['Base (N)'], :], **{'background-color': '#fff2cc', 'color': 'black', 'font-weight': 'bold'}),
            use_container_width=True
        )
    else: st.info("No data.")
    
    st.markdown("---")
    
    # 2. BARRIERS TABLE
    st.markdown("#### 🛑 Barriers (1-3★)")
    barriers_df = df[df['score'] <= 3]
    df_b, top_b = build_brand_matrix(barriers_df, theme_cols, sel_brands)
    if df_b is not None:
        st.dataframe(
            df_b.style
            .background_gradient(cmap='Reds', subset=pd.IndexSlice[top_b, :], axis=None)
            .format("{:.1f}", subset=pd.IndexSlice[top_b, :])
            .format("{:.0f}", subset=pd.IndexSlice[['Base (N)'], :])
            .set_properties(subset=pd.IndexSlice[['Base (N)'], :], **{'background-color': '#fff2cc', 'color': 'black', 'font-weight': 'bold'}),
            use_container_width=True
        )
    else: st.info("No data.")

    st.markdown("---")
    st.markdown("### 🧬 Theme Evolution (Brand Comparison)")
    evo_type = st.radio("Category", ["Drivers (Positive)", "Barriers (Negative)"], horizontal=True, key="db_evo_type")
    trend_src = df[df['score'] >= 4] if "Positive" in evo_type else df[df['score'] <= 3]

    if not trend_src.empty and theme_cols:
        top_opts = trend_src[theme_cols].sum().sort_values(ascending=False).head(20).index.tolist()
        sel_theme = st.selectbox("Select One Theme to Compare Across Brands", top_opts, index=0, key="db_theme_sel")
        if sel_theme:
            t_view = st.radio("View", ["Monthly", "Weekly"], horizontal=True, key="db_time_view")
            t_col = 'Month' if t_view == "Monthly" else 'Week'
            trend_data = []
            grouped = trend_src.groupby([t_col, 'App_Name'], observed=True)
            for (t_val, brand), group in grouped:
                base_vol = len(group)
                if base_vol == 0: continue
                if sel_theme in group.columns:
                    count = group[sel_theme].sum()
                    pct = (count / base_vol) * 100
                    trend_data.append({t_col: str(t_val), "App_Name": brand, "Prevalence": pct, "Base": base_vol})
            if trend_data:
                plot_df = pd.DataFrame(trend_data).sort_values(t_col)
                fig_evo = px.line(plot_df, x=t_col, y="Prevalence", color="App_Name", markers=True, 
                                  title=f"Evolution of '{sel_theme}' (%)", text="Prevalence", 
                                  hover_data={"Base": True, "Prevalence": ":.1f"})
                fig_evo.update_traces(textposition="top center", texttemplate='%{text:.1f}')
                st.plotly_chart(dark_chart(fig_evo), use_container_width=True, key="db_evo_chart")

    c_t1, c_t2 = st.columns(2)
    with c_t1:
        st.markdown("#### 📅 MoM Growth %")
        mom_matrix = calculate_growth_matrix(trend_src, theme_cols, 'Month', sel_brands)
        if not mom_matrix.empty:
            st.plotly_chart(px.imshow(mom_matrix, text_auto='.1f', aspect="auto", color_continuous_scale='RdBu', color_continuous_midpoint=0), use_container_width=True, key="mom_matrix")
    with c_t2:
        st.markdown("#### ⚡ WoW Growth %")
        wow_matrix = calculate_growth_matrix(trend_src, theme_cols, 'Week', sel_brands)
        if not wow_matrix.empty:
            st.plotly_chart(px.imshow(wow_matrix, text_auto='.1f', aspect="auto", color_continuous_scale='RdBu', color_continuous_midpoint=0), use_container_width=True, key="wow_matrix")

# === TAB 3: HEAD TO HEAD ===
with tab_compare:
    c1, c2 = st.columns(2)
    with c1: b1 = st.selectbox("Brand A", sel_brands, index=0 if sel_brands else None, key="h2h_b1")
    with c2: b2 = st.selectbox("Brand B", [b for b in sel_brands if b!=b1], index=0 if len(sel_brands)>1 else None, key="h2h_b2")
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
                fig.update_layout(polar=dict(radialaxis=dict(visible=True)), template="plotly_dark", title="Overlap (%)")
                st.plotly_chart(fig, use_container_width=True, key="h2h_radar")

# === TAB 4: PERIOD MATRIX ===
with tab_monthly:
    st.markdown("### 📅 Period-Over-Period Matrix (Percentage Only)")
    c_m1, c_m2 = st.columns(2)
    time_grain = c_m1.selectbox("Time Grain", ["Week", "Month", "Quarter", "Year"], index=1, key="m_time_grain")
    time_lookback = c_m2.selectbox("Time Range", ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Last 6 Months", "Last 12 Months", "All Time"], index=3, key="m_lookback")
    
    # Apply Lookback
    max_date = df['at'].max()
    if time_lookback == "Last 7 Days": start_date = max_date - timedelta(days=7)
    elif time_lookback == "Last 30 Days": start_date = max_date - timedelta(days=30)
    elif time_lookback == "Last 90 Days": start_date = max_date - timedelta(days=90)
    elif time_lookback == "Last 6 Months": start_date = max_date - timedelta(days=180)
    elif time_lookback == "Last 12 Months": start_date = max_date - timedelta(days=365)
    else: start_date = df['at'].min()
    
    m_base = df[df['at'] >= start_date].copy()
    
    if time_grain == "Week":
        m_base['Period'] = m_base['at'].dt.strftime('%Y-W%V')
    elif time_grain == "Month":
        m_base['Period'] = m_base['at'].dt.to_period('M').astype(str)
    elif time_grain == "Quarter":
        m_base['Period'] = m_base['at'].dt.to_period('Q').astype(str)
    else:
        m_base['Period'] = m_base['at'].dt.to_period('Y').astype(str)
        
    st.markdown("#### 🚀 Drivers (4-5★)")
    drivers_df = m_base[m_base['score'] >= 4]
    df_d, top_d = build_period_matrix(drivers_df, theme_cols, sel_brands)
    if df_d is not None:
        st.dataframe(
            df_d.style
            .background_gradient(cmap='Greens', subset=pd.IndexSlice[top_d, :], axis=None)
            .format("{:.1f}", subset=pd.IndexSlice[top_d, :])
            .format("{:.0f}", subset=pd.IndexSlice[['Base (N)'], :])
            .set_properties(subset=pd.IndexSlice[['Base (N)'], :], **{'background-color': '#fff2cc', 'color': 'black', 'font-weight': 'bold'}),
            use_container_width=True
        )
    else: st.info("No Driver data.")
        
    st.markdown("---")
    st.markdown("#### 🛑 Barriers (1-3★)")
    barriers_df = m_base[m_base['score'] <= 3]
    df_b, top_b = build_period_matrix(barriers_df, theme_cols, sel_brands)
    if df_b is not None:
        st.dataframe(
            df_b.style
            .background_gradient(cmap='Reds', subset=pd.IndexSlice[top_b, :], axis=None)
            .format("{:.1f}", subset=pd.IndexSlice[top_b, :])
            .format("{:.0f}", subset=pd.IndexSlice[['Base (N)'], :])
            .set_properties(subset=pd.IndexSlice[['Base (N)'], :], **{'background-color': '#fff2cc', 'color': 'black', 'font-weight': 'bold'}),
            use_container_width=True
        )
    else: st.info("No Barrier data.")

# === TAB 5: TRENDS ===
with tab_trends:
    view = st.radio("Time View", ["Monthly", "Weekly"], horizontal=True, key="tr_view")
    t_col = 'Month' if view == "Monthly" else 'Week'
    if 'at' in df.columns:
        trend = df.groupby([t_col, 'App_Name'], observed=True).size().reset_index(name='Count')
        tot = df.groupby(t_col, observed=True).size().reset_index(name='Total')
        trend = trend.merge(tot, on=t_col)
        trend['SoV'] = (trend['Count'] / trend['Total']) * 100
        csat = df.groupby([t_col, 'App_Name'], observed=True)['score'].agg(['mean','count']).reset_index()
        csat.columns = [t_col, 'App_Name', 'CSAT', 'Base']
        if view == "Weekly": 
            trend['Week'] = trend['Week'].astype(str)
            csat['Week'] = csat['Week'].astype(str)
        c_tr1, c_tr2 = st.columns(2)
        with c_tr1:
            fig_csat = px.line(csat, x=t_col, y='CSAT', color='App_Name', markers=True, title="CSAT Trend", 
                               text='CSAT', hover_data={'Base':True, 'CSAT':':.2f'})
            fig_csat.update_traces(textposition="top center", texttemplate='%{text:.2f}')
            st.plotly_chart(dark_chart(fig_csat), use_container_width=True, key="tr_csat")
        with c_tr2:
            fig_sov = px.area(trend, x=t_col, y='SoV', color='App_Name', title="Share of Voice (%)", 
                              hover_data={'Count':True, 'SoV':':.1f'})
            st.plotly_chart(dark_chart(fig_sov), use_container_width=True, key="tr_vol")

# === TAB 6: TEXT ANALYTICS ===
with tab_text:
    st.markdown("### 🔡 Deep Text Analytics")
    c1, c2 = st.columns(2)
    with c1:
        txt_brand = st.selectbox("Select Brand", ["All"] + sel_brands, key="txt_brand")
        txt_type = st.radio("Sentiment", ["Positive (4-5★)", "Negative (1-3★)"], horizontal=True, key="txt_radio")
        t_df = df if txt_brand == "All" else df[df['App_Name'] == txt_brand]
        t_df = t_df[t_df['score']>=4] if "Positive" in txt_type else t_df[t_df['score']<=3]
        if not t_df.empty:
            base = len(t_df)
            words_df = get_top_words_pct(t_df['Review_Text'], base)
            fig = px.bar(words_df, x='Pct', y='Word', orientation='h', 
                         title=f"Top Words (Base: {base})", labels={'Pct':'% of Reviews'},
                         text='Pct', hover_data={'Count':True, 'Pct':':.1f'})
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(dark_chart(fig), use_container_width=True, key="txt_bar")
    with c2:
        fig = px.histogram(df, x='char_count', color='App_Name', nbins=50, title="Review Length Dist.")
        st.plotly_chart(dark_chart(fig), use_container_width=True, key="txt_hist")

# === TAB 7: AI ANALYST ===
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
    
    st.markdown("---")
    # AI Summary Only Here
    vol = len(df)
    avg_rating = df['score'].mean()
    html = f"""
    <div class='ai-insight-box'>
        <div class='ai-header'>🤖 AI Analyst: Global Summary</div>
        <div class='ai-text'>
            • <b>Analysis Scope:</b> {vol:,} reviews across {len(sel_brands)} brands.<br>
            • <b>Overall Sentiment:</b> {avg_rating:.2f} ⭐ average rating.<br>
            • <b>Observation:</b> Review the Period Matrix to identify if recent drops in CSAT correlate with specific barriers like 'Hidden Charges' or 'Tech Issues'.
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
