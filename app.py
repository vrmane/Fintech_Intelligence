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
# 3. ANALYTICS LOGIC
# ==========================================
def dark_chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="#94a3b8"),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    return fig

def calculate_growth_matrix(df, theme_cols, time_col, brands):
    # Generates a matrix of growth % for Top 10 themes vs Brands
    
    # 1. Identify Top 10 Themes globally for this slice
    valid_themes = [t for t in theme_cols if t in df.columns]
    if not valid_themes: return pd.DataFrame()
    
    top_themes = df[valid_themes].sum().sort_values(ascending=False).head(10).index.tolist()
    
    # 2. Calculate Growth for each (Theme, Brand) pair
    matrix = {}
    
    for brand in brands:
        b_df = df[df['App_Name'] == brand].sort_values(time_col)
        # Get periods
        periods = b_df[time_col].unique()
        if len(periods) < 2:
            matrix[brand] = [0.0] * len(top_themes)
            continue
            
        last = periods[-1]
        prev = periods[-2]
        
        current_counts = b_df[b_df[time_col] == last][top_themes].sum()
        prev_counts = b_df[b_df[time_col] == prev][top_themes].sum()
        
        # Normalize by volume? Or raw growth? 
        # Metric: % Growth in Volume Share (to handle volume spikes)
        curr_vol = len(b_df[b_df[time_col] == last])
        prev_vol = len(b_df[b_df[time_col] == prev])
        
        if prev_vol == 0: 
            growth = [0.0] * len(top_themes)
        else:
            growth = []
            for t in top_themes:
                s_curr = current_counts[t] / curr_vol if curr_vol else 0
                s_prev = prev_counts[t] / prev_vol
                g = ((s_curr - s_prev) / s_prev * 100) if s_prev > 0 else 0
                growth.append(g)
        
        matrix[brand] = growth
        
    return pd.DataFrame(matrix, index=top_themes)

def get_brand_insights(df, brand, theme_cols):
    b_df = df[df['App_Name'] == brand]
    if b_df.empty: return ["No data available."]
    
    insights = []
    
    cohort_avg = df['score'].mean()
    brand_avg = b_df['score'].mean()
    diff = brand_avg - cohort_avg
    pos_text = "outperforming" if diff > 0 else "lagging behind"
    insights.append(f"**Market Position:** {brand} is <span class='highlight-neu'>{pos_text}</span> the cohort average by **{abs(diff):.2f} stars**.")
    
    pos_df = b_df[b_df['score'] >= 4]
    if not pos_df.empty and theme_cols:
        valid = [t for t in theme_cols if t in pos_df.columns]
        if valid:
            top_d = pos_df[valid].sum().idxmax()
            pct = (pos_df[top_d].sum() / len(pos_df)) * 100
            insights.append(f"**Growth Engine:** Positive sentiment is anchored on <span class='highlight-pos'>'{top_d}'</span>, appearing in **{pct:.1f}%** of 4-5★ reviews.")
            
    neg_df = b_df[b_df['score'] <= 3]
    if not neg_df.empty and theme_cols:
        valid = [t for t in theme_cols if t in neg_df.columns]
        if valid:
            top_b = neg_df[valid].sum().idxmax()
            pct = (neg_df[top_b].sum() / len(neg_df)) * 100
            insights.append(f"**Critical Risk:** The dominant friction point is <span class='highlight-neg'>'{top_b}'</span>, flagged in **{pct:.1f}%** of complaints.")
    
    share = (len(b_df) / len(df)) * 100
    insights.append(f"**Voice Share:** Commands **{share:.1f}%** of the filtered review volume.")
    
    return insights

# ==========================================
# 4. SIDEBAR & FILTERS
# ==========================================
with st.sidebar:
    st.title("🎛️ Command Center")
    with st.spinner("Connecting..."):
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
    
    # NEW: Rating Filter
    st.markdown("### 📊 Metrics")
    sel_ratings = st.multiselect("Rating", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])

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
    
    if 'Sentiment' in df_raw.columns:
        sel_sent = st.multiselect("Sentiment", df_raw['Sentiment'].unique())
    else: sel_sent = []

# --- APPLY FILTERING ---
if len(date_range) == 2:
    mask = (df_raw['at'].dt.date >= date_range[0]) & (df_raw['at'].dt.date <= date_range[1])
else:
    mask = [True] * len(df_raw)

mask &= df_raw['App_Name'].isin(sel_brands)
mask &= df_raw['score'].isin(sel_ratings) # Apply Rating Filter

if len_filter == "Brief (<=29 chars)":
    mask &= (df_raw['length_bucket'] == 'Brief (<=29 chars)')
elif len_filter == "Detailed (>=30 chars)":
    mask &= (df_raw['length_bucket'] == 'Detailed (>=30 chars)')

if sel_prods:
    p_mask = pd.Series(False, index=df_raw.index)
    for c in prod_cols: p_mask |= df_raw[c].isin(sel_prods)
    mask &= p_mask
    
if sel_pl: mask &= (df_raw['PL Status'].isin(sel_pl))
if sel_sent: mask &= (df_raw['Sentiment'].isin(sel_sent))

df = df_raw[mask].copy()
theme_cols = st.session_state.get('theme_cols', [])

# ==========================================
# 5. DASHBOARD LAYOUT
# ==========================================

st.title("🦅 Strategic Intelligence Platform")

tab_ai, tab_exec, tab_drivers, tab_compare, tab_trends, tab_raw = st.tabs([
    "🤖 AI Analyst",
    "📊 Boardroom Summary", 
    "🚀 Drivers & Barriers", 
    "⚔️ Head-to-Head", 
    "📈 Trends", 
    "🔍 Data"
])

# === TAB 1: AI ANALYST ===
with tab_ai:
    st.markdown("### 🤖 Live Strategic Briefing")
    st.info(f"Context: {len_filter} | {len(sel_brands)} Brands | {len(sel_ratings)} Rating Levels")
    cols = st.columns(2)
    for i, brand in enumerate(sel_brands):
        with cols[i % 2]:
            insights = get_brand_insights(df, brand, theme_cols)
            html_content = f"""
            <div class="ai-card">
                <div class="ai-title">🔹 {brand}</div>
                {"".join([f'<div class="ai-bullet">✦ {txt}</div>' for txt in insights])}
            </div>
            """
            st.markdown(html_content, unsafe_allow_html=True)

# === TAB 2: SUMMARY ===
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
    brand_stats = df.groupby('App_Name').agg(
        Volume=('score', 'count'),
        CSAT=('score', 'mean'),
        One_Star=('score', lambda x: (x==1).sum()),
        Five_Star=('score', lambda x: (x==5).sum())
    ).reset_index()
    brand_stats['NPS'] = ((brand_stats['Five_Star'] - brand_stats['One_Star']) / brand_stats['Volume'] * 100).round(1)
    
    fig = px.scatter(brand_stats, x="CSAT", y="NPS", size="Volume", color="App_Name", hover_name="App_Name", text="App_Name", title="Brand Performance Matrix", color_discrete_sequence=px.colors.qualitative.Bold, height=500)
    fig.update_traces(textposition='top center')
    st.plotly_chart(dark_chart(fig), use_container_width=True)

# === TAB 3: DRIVERS & BARRIERS (UPDATED) ===
with tab_drivers:
    st.markdown("### 🚦 Strategic Drivers & Barriers")
    
    # 1. CROSS-BRAND HEATMAPS
    st.markdown("#### ⚔️ Market Landscape (Theme Share)")
    pos_df_global = df[df['score']>=4]
    neg_df_global = df[df['score']<=3]
    pos_bases = pos_df_global.groupby('App_Name').size()
    neg_bases = neg_df_global.groupby('App_Name').size()
    
    c_h1, c_h2 = st.columns(2)
    with c_h1:
        st.markdown("**🚀 Drivers (Positive)**")
        if not pos_bases.empty and theme_cols:
            valid = [t for t in theme_cols if t in df.columns]
            p_data = pos_df_global.groupby('App_Name')[valid].sum().T
            p_pct = p_data.div(pos_bases, axis=1) * 100
            p_pct['Avg'] = p_pct.mean(axis=1)
            p_pct = p_pct.sort_values('Avg', ascending=False).head(10).drop(columns=['Avg'])
            st.plotly_chart(px.imshow(p_pct, text_auto='.1f', aspect="auto", color_continuous_scale='Greens'), use_container_width=True)
            
    with c_h2:
        st.markdown("**🛑 Barriers (Negative)**")
        if not neg_bases.empty and theme_cols:
            valid = [t for t in theme_cols if t in df.columns]
            n_data = neg_df_global.groupby('App_Name')[valid].sum().T
            n_pct = n_data.div(neg_bases, axis=1) * 100
            n_pct['Avg'] = n_pct.mean(axis=1)
            n_pct = n_pct.sort_values('Avg', ascending=False).head(10).drop(columns=['Avg'])
            st.plotly_chart(px.imshow(n_pct, text_auto='.1f', aspect="auto", color_continuous_scale='Reds'), use_container_width=True)

    st.markdown("---")
    
    # 2. CROSS-BRAND TRENDS (NEW REQUEST)
    st.markdown("#### 📈 Theme Momentum (Cross-Brand Trends)")
    trend_mode = st.radio("Show Trends For:", ["Drivers (Positive)", "Barriers (Negative)"], horizontal=True)
    
    if trend_mode == "Drivers (Positive)":
        trend_df = pos_df_global
        scale = 'Greens'
    else:
        trend_df = neg_df_global
        scale = 'Reds'
        
    c_t1, c_t2 = st.columns(2)
    
    with c_t1:
        st.markdown("**📅 MoM Growth %**")
        mom_matrix = calculate_growth_matrix(trend_df, theme_cols, 'Month', sel_brands)
        if not mom_matrix.empty:
            st.plotly_chart(px.imshow(mom_matrix, text_auto='.1f', aspect="auto", color_continuous_scale='RdBu', color_continuous_midpoint=0), use_container_width=True)
        else:
            st.info("Insufficient Monthly Data")
            
    with c_t2:
        st.markdown("**⚡ WoW Growth %**")
        wow_matrix = calculate_growth_matrix(trend_df, theme_cols, 'Week', sel_brands)
        if not wow_matrix.empty:
            st.plotly_chart(px.imshow(wow_matrix, text_auto='.1f', aspect="auto", color_continuous_scale='RdBu', color_continuous_midpoint=0), use_container_width=True)
        else:
            st.info("Insufficient Weekly Data")

    st.markdown("---")

    # 3. SINGLE BRAND DEEP DIVE
    st.markdown("#### 🔍 Brand Deep Dive Table")
    target_brand = st.selectbox("Select Brand", sel_brands)
    
    if target_brand and theme_cols:
        b_df = df[df['App_Name'] == target_brand]
        
        # Calculate trend helper
        def get_trend_val(sub_df, metric_col, group_col, time_col, target):
            d_sorted = sub_df.sort_values(time_col)
            periods = d_sorted[time_col].unique()
            if len(periods) < 2: return 0.0
            last, prev = periods[-1], periods[-2]
            
            # Share of volume
            v_last = d_sorted[d_sorted[time_col]==last][group_col].sum()
            b_last = len(d_sorted[d_sorted[time_col]==last])
            s_last = v_last/b_last if b_last else 0
            
            v_prev = d_sorted[d_sorted[time_col]==prev][group_col].sum()
            b_prev = len(d_sorted[d_sorted[time_col]==prev])
            s_prev = v_prev/b_prev if b_prev else 0
            
            if s_prev == 0: return 0.0
            return ((s_last - s_prev)/s_prev)*100

        def build_table(sub_df, base, themes):
            if sub_df.empty or base == 0: return pd.DataFrame()
            valid = [t for t in themes if t in sub_df.columns]
            counts = sub_df[valid].sum().sort_values(ascending=False).head(10)
            data = []
            for theme, count in counts.items():
                if count == 0: continue
                mom = get_trend_val(sub_df, theme, theme, 'Month', theme)
                wow = get_trend_val(sub_df, theme, theme, 'Week', theme)
                data.append({
                    "Theme": theme, "Count": int(count), "% of Base": f"{(count/base)*100:.1f}%",
                    "MoM": f"{mom:+.1f}%", "WoW": f"{wow:+.1f}%"
                })
            return pd.DataFrame(data)

        c1, c2 = st.columns(2)
        with c1:
            pos_df = b_df[b_df['score'] >= 4]
            st.markdown(f"**🚀 Drivers (Base: {len(pos_df):,})**")
            dt = build_table(pos_df, len(pos_df), theme_cols)
            if not dt.empty: st.dataframe(dt, hide_index=True, use_container_width=True)
            else: st.warning("No Data")

        with c2:
            neg_df = b_df[b_df['score'] <= 3]
            st.markdown(f"**🛑 Barriers (Base: {len(neg_df):,})**")
            bt = build_table(neg_df, len(neg_df), theme_cols)
            if not bt.empty: st.dataframe(bt, hide_index=True, use_container_width=True)
            else: st.warning("No Data")

# === TAB 4: HEAD TO HEAD ===
with tab_compare:
    c1, c2 = st.columns(2)
    with c1: b1 = st.selectbox("Brand A", sel_brands, index=0 if sel_brands else None)
    with c2: b2 = st.selectbox("Brand B", [b for b in sel_brands if b!=b1], index=0 if len(sel_brands)>1 else None)
    
    if b1 and b2:
        def get_stats(b):
            d = df[df['App_Name']==b]
            v = len(d)
            s = d['score'].mean() if v else 0
            n = (((len(d[d['score']==5]) - len(d[d['score']<=3])) / v) * 100) if v else 0
            return [f"{s:.2f}", f"{n:.0f}", f"{v:,}"]
        
        comp_df = pd.DataFrame({"Metric": ["CSAT", "NPS Proxy", "Volume"], b1: get_stats(b1), b2: get_stats(b2)}).set_index("Metric")
        st.dataframe(comp_df, use_container_width=True)
        
        if theme_cols:
            d1 = df[(df['App_Name']==b1) & (df['score']>=4)][theme_cols].sum().sort_values(ascending=False).head(5).index.tolist()
            d2 = df[(df['App_Name']==b2) & (df['score']>=4)][theme_cols].sum().sort_values(ascending=False).head(5).index.tolist()
            common = list(set(d1 + d2))
            
            def get_pct(b, themes):
                d = df[df['App_Name']==b]
                return [(d[t].sum()/len(d)*100) if not d.empty else 0 for t in themes]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=get_pct(b1, common), theta=common, fill='toself', name=b1))
            fig.add_trace(go.Scatterpolar(r=get_pct(b2, common), theta=common, fill='toself', name=b2))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True)), template="plotly_dark", title="Theme Overlap")
            st.plotly_chart(fig, use_container_width=True)

# === TAB 5: TRENDS ===
with tab_trends:
    view = st.radio("View", ["Monthly (MoM)", "Weekly (WoW)"], horizontal=True)
    time_col = 'Month' if "Monthly" in view else 'Week'
    if 'at' in df.columns:
        trend = df.groupby([time_col, 'App_Name'])['score'].agg(['mean', 'count']).reset_index()
        if view == "Weekly (WoW)": trend['Week'] = trend['Week'].astype(str)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(trend, x=time_col, y='mean', color='App_Name', markers=True, title="CSAT Trend")
            st.plotly_chart(dark_chart(fig), use_container_width=True)
        with c2:
            fig = px.bar(trend, x=time_col, y='count', color='App_Name', title="Volume Trend")
            st.plotly_chart(dark_chart(fig), use_container_width=True)

# === TAB 6: DATA ===
with tab_raw:
    st.dataframe(df[['at', 'App_Name', 'score', 'Review_Text', 'length_bucket']], use_container_width=True)
