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
            margin-top: 30px;
            margin-bottom: 25px;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 20px; }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
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
            • <b>Leader:</b> <b>{leader}</b> is setting the benchmark.<br>
            • <b>Market Driver:</b> <b>'{top_driver}'</b> is the key sentiment engine.<br>
            • <b>Market Barrier:</b> <b>'{top_barrier}'</b> is the top friction point.
        </div>
    </div>
    """
    return html

# ==========================================
# 6. FILTERS
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

tab_exec, tab_drivers, tab_compare, tab_trends, tab_text, tab_raw, tab_ai = st.tabs([
    "📊 Boardroom Summary", "🚀 Drivers & Barriers", "⚔️ Head-to-Head", "📈 Trends", "🔡 Text Analytics", "🔍 Data", "🤖 AI Analyst"
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
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🏥 Brand Matrix (Health Check)")
        bs = df.groupby('App_Name').agg(
            Vol=('score','count'), CSAT=('score','mean'),
            One=('score', lambda x: (x==1).sum()), Five=('score', lambda x: (x==5).sum())
        ).reset_index()
        bs['NPS'] = ((bs['Five'] - bs['One'])/bs['Vol']*100).round(1)
        fig_mtx = px.scatter(bs, x="CSAT", y="NPS", size="Vol", color="App_Name", text="App_Name", 
                         title="Brand Matrix (Size=Volume)", height=400,
                         hover_data={"Vol": True, "CSAT": ":.2f", "NPS": ":.1f"})
        fig_mtx.update_traces(textposition='top center')
        st.plotly_chart(dark_chart(fig_mtx), use_container_width=True, key="exec_matrix")
        
    with c2:
        st.markdown("#### 📊 Sentiment Composition")
        sent_counts = df.groupby(['App_Name', 'Sentiment_Label']).size().reset_index(name='Count')
        total_counts = sent_counts.groupby('App_Name')['Count'].transform('sum')
        sent_counts['Pct'] = (sent_counts['Count'] / total_counts) * 100
        fig_stack = px.bar(sent_counts, x="App_Name", y="Pct", color="Sentiment_Label", 
                           color_discrete_map={'Positive': '#10b981', 'Neutral': '#64748b', 'Negative': '#ef4444'},
                           title="Sentiment Ratio (Base in Hover)", height=400,
                           hover_data={'Count': True, 'Pct': ':.1f'})
        st.plotly_chart(dark_chart(fig_stack), use_container_width=True, key="exec_stack")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### 📝 Engagement Quality")
        len_counts = df.groupby(['App_Name', 'length_bucket']).size().reset_index(name='Count')
        tot_len = len_counts.groupby('App_Name')['Count'].transform('sum')
        len_counts['Pct'] = (len_counts['Count'] / tot_len) * 100
        fig_len = px.bar(len_counts, x="App_Name", y="Pct", color="length_bucket", barmode='group',
                         title="Brief vs Detailed Reviews (%)", height=350,
                         hover_data={'Count': True, 'Pct': ':.1f'})
        st.plotly_chart(dark_chart(fig_len), use_container_width=True, key="exec_len")
    
    with c4:
        st.markdown("#### 🍰 Product Mix")
        if 'Product_1' in df.columns:
            prod_counts = df['Product_1'].value_counts().reset_index()
            prod_counts.columns = ['Product', 'Count']
            fig_pie = px.pie(prod_counts, values='Count', names='Product', hole=0.4, 
                             title="Product Category Share", hover_data=['Count'])
            st.plotly_chart(dark_chart(fig_pie), use_container_width=True, key="exec_pie")

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
    
    # AGGREGATE INTENSITY
    st.markdown("#### 📈 Aggregate Intensity Trends (Across Brands)")
    c_agg1, c_agg2 = st.columns(2)
    
    def get_agg_trend(sub_df, themes, time_col):
        if sub_df.empty: return pd.DataFrame()
        valid = [t for t in themes if t in sub_df.columns]
        # Calculate row-wise "Any Theme" hit
        sub_df['Has_Theme'] = sub_df[valid].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
        # Group
        trend = sub_df.groupby([time_col, 'App_Name']).agg(
            Pct=('Has_Theme', 'mean'),
            Base=('Has_Theme', 'count')
        ).reset_index()
        trend['Pct'] = trend['Pct'] * 100
        if time_col == 'Week': trend['Week'] = trend['Week'].astype(str)
        return trend

    agg_view = st.radio("Time View", ["Monthly", "Weekly"], horizontal=True, key="agg_view")
    agg_col = 'Month' if agg_view == "Monthly" else 'Week'

    with c_agg1:
        st.markdown("**Driver Intensity (Positive)**")
        pos_trend = get_agg_trend(df[df['score']>=4], theme_cols, agg_col)
        if not pos_trend.empty:
            fig_pt = px.line(pos_trend, x=agg_col, y='Pct', color='App_Name', markers=True, 
                             title="% of Reviews with Drivers", hover_data={'Base':True, 'Pct':':.1f'})
            st.plotly_chart(dark_chart(fig_pt), use_container_width=True, key="agg_pos")
    
    with c_agg2:
        st.markdown("**Barrier Intensity (Negative)**")
        neg_trend = get_agg_trend(df[df['score']<=3], theme_cols, agg_col)
        if not neg_trend.empty:
            fig_nt = px.line(neg_trend, x=agg_col, y='Pct', color='App_Name', markers=True, 
                             title="% of Reviews with Barriers", hover_data={'Base':True, 'Pct':':.1f'})
            st.plotly_chart(dark_chart(fig_nt), use_container_width=True, key="agg_neg")

    st.markdown("---")
    
    # SPECIFIC THEME COMPARE
    st.markdown("#### 🧬 Deep Dive: Specific Theme Trend")
    evo_src = st.radio("Source", ["Positive (4-5★)", "Negative (1-3★)"], horizontal=True, key="evo_src")
    src_df = df[df['score']>=4] if "Positive" in evo_src else df[df['score']<=3]
    
    if not src_df.empty and theme_cols:
        top_t = src_df[theme_cols].sum().sort_values(ascending=False).head(20).index.tolist()
        sel_t = st.selectbox("Select Theme to Compare Across Brands", top_t, key="sel_t")
        
        if sel_t:
            # Calculate prevalence per brand per month
            theme_trend = src_df.groupby([agg_col, 'App_Name']).agg(
                Count=(sel_t, 'sum'),
                Base=(sel_t, 'count')
            ).reset_index()
            theme_trend['Pct'] = (theme_trend['Count'] / theme_trend['Base']) * 100
            if agg_col == 'Week': theme_trend['Week'] = theme_trend['Week'].astype(str)
            
            fig_tt = px.line(theme_trend, x=agg_col, y='Pct', color='App_Name', markers=True,
                             title=f"Trend: '{sel_t}' across Brands", hover_data={'Base':True, 'Pct':':.1f'})
            st.plotly_chart(dark_chart(fig_tt), use_container_width=True, key="evo_tt")

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

# === TAB 4: TRENDS ===
with tab_trends:
    view = st.radio("Time View", ["Monthly", "Weekly"], horizontal=True, key="tr_view")
    t_col = 'Month' if view == "Monthly" else 'Week'
    if 'at' in df.columns:
        # SoV
        trend = df.groupby([t_col, 'App_Name']).size().reset_index(name='Count')
        tot = df.groupby(t_col).size().reset_index(name='Total')
        trend = trend.merge(tot, on=t_col)
        trend['SoV'] = (trend['Count'] / trend['Total']) * 100
        
        csat = df.groupby([t_col, 'App_Name'])['score'].agg(['mean','count']).reset_index()
        csat.columns = [t_col, 'App_Name', 'CSAT', 'Base']
        
        if view == "Weekly": 
            trend['Week'] = trend['Week'].astype(str)
            csat['Week'] = csat['Week'].astype(str)
        
        st.plotly_chart(dark_chart(px.line(csat, x=t_col, y='CSAT', color='App_Name', markers=True, title="CSAT Trend", hover_data={'Base':True, 'CSAT':':.2f'})), use_container_width=True, key="tr_csat")
        st.plotly_chart(dark_chart(px.line(trend, x=t_col, y='SoV', color='App_Name', markers=True, title="Share of Voice (%)", hover_data={'Count':True, 'SoV':':.1f'})), use_container_width=True, key="tr_vol")

# === TAB 5: TEXT ANALYTICS ===
with tab_text:
    st.markdown("### 🔡 Deep Text Analytics")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Word Frequency (% of Reviews)**")
        txt_type = st.radio("Sentiment", ["Positive (4-5★)", "Negative (1-3★)"], horizontal=True, key="txt_radio")
        txt_df = df[df['score']>=4] if "Positive" in txt_type else df[df['score']<=3]
        if not txt_df.empty:
            base = len(txt_df)
            words_df = get_top_words_pct(txt_df['Review_Text'], base)
            fig = px.bar(words_df, x='Pct', y='Word', orientation='h', 
                         title=f"Top Words (Base: {base})", labels={'Pct':'% of Reviews'},
                         hover_data={'Count':True, 'Pct':':.1f'})
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(dark_chart(fig), use_container_width=True, key="txt_bar")
            
    with c2:
        st.markdown("**Review Length Distribution**")
        fig = px.histogram(df, x='char_count', color='App_Name', nbins=50, title="Character Count Dist.")
        st.plotly_chart(dark_chart(fig), use_container_width=True, key="txt_hist")

# === TAB 6: DATA ===
with tab_raw:
    st.dataframe(df[['at', 'App_Name', 'score', 'Review_Text', 'length_bucket']], use_container_width=True)

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
