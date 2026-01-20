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
from datetime import timedelta

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

def get_top_words_pct(text_series, total_count, top_n=20):
    stop_words = set(['the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'for', 'i', 'this', 'my', 'app', 'loan', 'money', 'very', 'good', 'bad', 'worst', 'best', 'application', 'not', 'but', 'on', 'with', 'are', 'was', 'have', 'be', 'so', 'me', 'you', 'please', 'give'])
    text = ' '.join(text_series.dropna().astype(str).tolist()).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [w for w in text.split() if w not in stop_words and len(w) > 2]
    counter = Counter(words)
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

@st.cache_data(ttl=600, show_spinner=False)
def load_data():
    all_rows = []
    start, batch = 0, 1000
    while True:
        res = supabase.table("reviews").select("*").range(start, start + batch - 1).execute()
        if not res.data: break
        all_rows.extend(res.data)
        if len(res.data) < batch: break
        start += batch

    if not all_rows: return pd.DataFrame()
    df = pd.DataFrame(all_rows)

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
        df['Sentiment_Label'] = pd.cut(df['score'], bins=[0, 2, 3, 5], labels=['Negative', 'Neutral', 'Positive'])
    
    if 'Review_Text' in df.columns:
        df['char_count'] = df['Review_Text'].astype(str).str.len().fillna(0)
        df['length_bucket'] = df['char_count'].apply(lambda x: 'Brief (<=29)' if x <= 29 else 'Detailed (>=30)')

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
# 5. VISUALIZATION FUNCTIONS (UPDATED FOR LABELS)
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

def generate_global_summary(df, theme_cols, current_filters):
    if df.empty: return "No data."
    avg_rating = df['score'].mean()
    vol = len(df)
    leader = df.groupby('App_Name')['score'].mean().idxmax()
    
    pos_df = df[df['score'] >= 4]
    top_driver = "N/A"
    if not pos_df.empty and theme_cols:
        valid = [t for t in theme_cols if t in pos_df.columns]
        if valid: top_driver = pos_df[valid].sum().idxmax()
            
    neg_df = df[df['score'] <= 3]
    top_barrier = "N/A"
    if not neg_df.empty and theme_cols:
        valid = [t for t in theme_cols if t in neg_df.columns]
        if valid: top_barrier = neg_df[valid].sum().idxmax()
            
    html = f"""
    <div class='ai-insight-box'>
        <div class='ai-header'>🤖 AI Analyst: Global Strategic Brief</div>
        <div class='ai-text'>
            • <b>Context:</b> Analyzing <b>{vol:,}</b> reviews ({current_filters}). Category CSAT: <b>{avg_rating:.2f} ⭐</b>.<br>
            • <b>Leader:</b> <b>{leader}</b> is currently setting the benchmark for satisfaction.<br>
            • <b>Market Driver:</b> <b>'{top_driver}'</b> is the key sentiment engine.<br>
            • <b>Market Barrier:</b> <b>'{top_barrier}'</b> is the top friction point.
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
    
    # Range is applied globally, but Tab 4 has its own override if needed
    min_d, max_d = df_raw['at'].min().date(), df_raw['at'].max().date()
    date_range = st.date_input("Period", [min_d, max_d], min_value=min_d, max_value=max_d)
    
    all_brands = sorted(df_raw['App_Name'].dropna().unique())
    sel_brands = st.multiselect("Brands", all_brands, default=all_brands)
    
    st.markdown("### 📊 Metrics")
    sel_ratings = st.multiselect("Ratings", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
    
    if 'Sentiment' in df_raw.columns:
        sent_opts = sorted(df_raw['Sentiment'].dropna().unique())
        sel_sent = st.multiselect("Sentiment", sent_opts)
    else: sel_sent = []
    
    if 'Product_1' in df_raw.columns:
        all_prods = sorted(df_raw['Product_1'].dropna().unique())
        sel_prods = st.multiselect("Product Type", all_prods)
    else: sel_prods = []

if len(date_range) == 2:
    mask = (df_raw['at'].dt.date >= date_range[0]) & (df_raw['at'].dt.date <= date_range[1])
else:
    mask = [True] * len(df_raw)

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

filter_desc = f"{len(sel_brands)} Brands"
st.markdown(generate_global_summary(df, theme_cols, filter_desc), unsafe_allow_html=True)

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
    nps = ((prom - det) / len(df) * 100) if len(df) > 0 else 0
    with k3: st.metric("NPS Proxy", f"{nps:.0f}")
    risk = (len(df[df['score']==1]) / len(df) * 100) if len(df) > 0 else 0
    with k4: st.metric("Critical Risk", f"{risk:.1f}%", delta="1-Star %", delta_color="inverse")
    
    st.markdown("---")
    
    st.markdown("#### 🏥 Brand Pulse (Live Breakdown)")
    kpi_df = df.groupby('App_Name').agg(
        Vol=('score', 'count'),
        CSAT=('score', 'mean'),
        Promoters=('score', lambda x: (x==5).sum()),
        Detractors=('score', lambda x: (x<=3).sum())
    ).reset_index()
    kpi_df['NPS Proxy'] = ((kpi_df['Promoters'] - kpi_df['Detractors']) / kpi_df['Vol'] * 100).round(0)
    kpi_df['CSAT'] = kpi_df['CSAT'].round(2)
    deltas = []
    for brand in kpi_df['App_Name']:
        b_df = df[df['App_Name'] == brand]
        _, d_vol = calculate_delta(b_df, 'score', 'count')
        _, d_csat = calculate_delta(b_df, 'score', 'mean')
        deltas.append({'App_Name': brand, 'Vol Delta': f"{d_vol:+.1f}%", 'CSAT Delta': f"{d_csat:+.2f}"})
    delta_df = pd.DataFrame(deltas)
    final_kpi = kpi_df.merge(delta_df, on='App_Name')[['App_Name', 'Vol', 'Vol Delta', 'CSAT', 'CSAT Delta', 'NPS Proxy']]
    st.dataframe(final_kpi.style.background_gradient(subset=['CSAT'], cmap='Greens'), use_container_width=True, hide_index=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🏆 Market Share")
        fig_don = px.pie(kpi_df, values='Vol', names='App_Name', hole=0.4, title="Volume Share by Brand")
        fig_don.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(dark_chart(fig_don), use_container_width=True, key="exec_don")
    with c2:
        st.markdown("#### 🎯 Quality vs. Quantity")
        fig_scat = px.scatter(kpi_df, x="CSAT", y="NPS Proxy", size="Vol", color="App_Name", text="App_Name", title="Strategic Positioning")
        fig_scat.update_traces(textposition='top center')
        st.plotly_chart(dark_chart(fig_scat), use_container_width=True, key="exec_scat")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### 📊 Sentiment Mix")
        sent_counts = df.groupby(['App_Name', 'Sentiment_Label']).size().reset_index(name='Count')
        tot = sent_counts.groupby('App_Name')['Count'].transform('sum')
        sent_counts['Pct'] = (sent_counts['Count'] / tot) * 100
        fig_stack = px.bar(sent_counts, x="App_Name", y="Pct", color="Sentiment_Label", 
                           color_discrete_map={'Positive': '#10b981', 'Neutral': '#64748b', 'Negative': '#ef4444'},
                           title="Sentiment Ratio (%)", text_auto='.0f',
                           hover_data={'Count': True, 'Pct': ':.1f'})
        st.plotly_chart(dark_chart(fig_stack), use_container_width=True, key="exec_stack")
    with c4:
        st.markdown("#### 📝 Engagement Depth")
        len_counts = df.groupby(['App_Name', 'length_bucket']).size().reset_index(name='Count')
        l_tot = len_counts.groupby('App_Name')['Count'].transform('sum')
        len_counts['Pct'] = (len_counts['Count'] / l_tot) * 100
        fig_len = px.bar(len_counts, x="App_Name", y="Pct", color="length_bucket", barmode='group',
                         title="Brief vs Detailed (%)", text_auto='.0f',
                         hover_data={'Count': True, 'Pct': ':.1f'})
        st.plotly_chart(dark_chart(fig_len), use_container_width=True, key="exec_len")

# === TAB 2: DRIVERS & BARRIERS ===
with tab_drivers:
    st.markdown("### 🚦 Strategic Drivers & Barriers")
    c_h1, c_h2 = st.columns(2)
    pos_bases = df[df['score']>=4].groupby('App_Name').size()
    neg_bases = df[df['score']<=3].groupby('App_Name').size()
    
    with c_h1:
        st.markdown("**🚀 Drivers (%)**")
        if not pos_bases.empty and theme_cols:
            valid = [t for t in theme_cols if t in df.columns]
            p_data = df[df['score']>=4].groupby('App_Name')[valid].sum().T
            p_pct = p_data.div(pos_bases, axis=1) * 100
            p_pct['Avg'] = p_pct.mean(axis=1)
            p_pct = p_pct.sort_values('Avg', ascending=False).head(10).drop(columns=['Avg'])
            st.plotly_chart(px.imshow(p_pct, text_auto='.1f', aspect="auto", color_continuous_scale='Greens', title=f"Base: {pos_bases.sum():,}"), use_container_width=True, key="db_pos")
    with c_h2:
        st.markdown("**🛑 Barriers (%)**")
        if not neg_bases.empty and theme_cols:
            valid = [t for t in theme_cols if t in df.columns]
            n_data = df[df['score']<=3].groupby('App_Name')[valid].sum().T
            n_pct = n_data.div(neg_bases, axis=1) * 100
            n_pct['Avg'] = n_pct.mean(axis=1)
            n_pct = n_pct.sort_values('Avg', ascending=False).head(10).drop(columns=['Avg'])
            st.plotly_chart(px.imshow(n_pct, text_auto='.1f', aspect="auto", color_continuous_scale='Reds', title=f"Base: {neg_bases.sum():,}"), use_container_width=True, key="db_neg")

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
            grouped = trend_src.groupby([t_col, 'App_Name'])
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

# === TAB 4: MONTHLY REPORT (NEW DYNAMIC) ===
with tab_monthly:
    st.markdown("### 📅 Period-Over-Period Matrix (Percentage Only)")
    
    # 1. Filters
    c_m1, c_m2, c_m3 = st.columns(3)
    rep_type = c_m1.radio("Focus", ["Drivers (4-5★)", "Barriers (1-3★)"], horizontal=True, key="m_rep_type")
    
    time_grain = c_m2.selectbox("Time Grain", ["Week", "Month", "Quarter", "Year"], index=1, key="m_time_grain")
    
    time_lookback = c_m3.selectbox("Time Range", ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Last 6 Months", "Last 12 Months", "All Time"], index=3, key="m_lookback")
    
    # 2. Filter Logic
    if rep_type == "Drivers (4-5★)":
        m_df = df[df['score'] >= 4]
        color_scale = 'Greens'
    else:
        m_df = df[df['score'] <= 3]
        color_scale = 'Reds'
        
    # Apply Lookback
    max_date = df['at'].max()
    if time_lookback == "Last 7 Days":
        start_date = max_date - timedelta(days=7)
    elif time_lookback == "Last 30 Days":
        start_date = max_date - timedelta(days=30)
    elif time_lookback == "Last 90 Days":
        start_date = max_date - timedelta(days=90)
    elif time_lookback == "Last 6 Months":
        start_date = max_date - timedelta(days=180)
    elif time_lookback == "Last 12 Months":
        start_date = max_date - timedelta(days=365)
    else:
        start_date = df['at'].min()
        
    m_df = m_df[m_df['at'] >= start_date]
    
    # 3. Group by Time Grain
    if time_grain == "Week":
        m_df['Period'] = m_df['at'].dt.to_period('W').apply(lambda r: r.start_time.strftime('%Y-W%U'))
    elif time_grain == "Month":
        m_df['Period'] = m_df['at'].dt.to_period('M').astype(str)
    elif time_grain == "Quarter":
        m_df['Period'] = m_df['at'].dt.to_period('Q').astype(str)
    else:
        m_df['Period'] = m_df['at'].dt.to_period('Y').astype(str)
        
    # 4. Build Matrix
    if not m_df.empty and theme_cols:
        # Get Sorted Periods
        periods = sorted(m_df['Period'].unique())
        
        # Calculate Base per Cell (Brand x Period)
        base_matrix = m_df.groupby(['Period', 'App_Name']).size().unstack(fill_value=0)
        
        # Get Top Themes
        valid = [t for t in theme_cols if t in m_df.columns]
        top_m_themes = m_df[valid].sum().sort_values(ascending=False).head(20).index.tolist()
        
        # Construct DataFrame
        matrix_rows = []
        for theme in top_m_themes:
            row = {'Theme': theme}
            for p in periods:
                for b in sel_brands:
                    if b not in base_matrix.columns: continue
                    # Count matches
                    mask = (m_df['Period'] == p) & (m_df['App_Name'] == b)
                    count = m_df[mask][theme].sum()
                    base = base_matrix.loc[p, b]
                    val = (count / base * 100) if base > 0 else 0
                    
                    col_name = f"{p} | {b}"
                    row[col_name] = val
            matrix_rows.append(row)
            
        final_m_df = pd.DataFrame(matrix_rows).set_index('Theme')
        
        st.markdown(f"**Top 20 {rep_type} by {time_grain}** (Values are %)")
        st.dataframe(final_m_df.style.background_gradient(cmap=color_scale, axis=None).format("{:.1f}"), use_container_width=True)
        
        with st.expander("Show Base Sizes (N)"):
            st.dataframe(base_matrix.T, use_container_width=True)
    else:
        st.info("No data for selected range.")

# === TAB 5: TRENDS ===
with tab_trends:
    view = st.radio("Time View", ["Monthly", "Weekly"], horizontal=True, key="tr_view")
    t_col = 'Month' if view == "Monthly" else 'Week'
    if 'at' in df.columns:
        trend = df.groupby([t_col, 'App_Name']).size().reset_index(name='Count')
        tot = df.groupby(t_col).size().reset_index(name='Total')
        trend = trend.merge(tot, on=t_col)
        trend['SoV'] = (trend['Count'] / trend['Total']) * 100
        csat = df.groupby([t_col, 'App_Name'])['score'].agg(['mean','count']).reset_index()
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
    filter_desc = f"{len(sel_brands)} Brands"
    st.markdown(generate_global_summary(df, theme_cols, filter_desc), unsafe_allow_html=True)
