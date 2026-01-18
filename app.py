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
    initial_sidebar_state="collapsed"
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
        df['review_depth'] = df['char_count'].apply(lambda x: 'Detailed (>30 chars)' if x >= 30 else 'Brief')

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
# 3. "ANALYST AI" LOGIC
# ==========================================
def run_strategic_analysis(df, brand, theme_cols):
    b_df = df[df['App_Name'] == brand]
    if b_df.empty: return None
    
    vol = len(b_df)
    score = b_df['score'].mean()
    
    promoters = len(b_df[b_df['score']==5])
    detractors = len(b_df[b_df['score']<=3])
    nps_proxy = ((promoters - detractors) / vol) * 100 if vol > 0 else 0
    
    pos_df = b_df[b_df['score'] >= 4]
    if not pos_df.empty and theme_cols:
        valid_themes = [t for t in theme_cols if t in pos_df.columns]
        drivers = pos_df[valid_themes].sum().sort_values(ascending=False).head(3)
        top_drivers = [(k, v, (v/len(pos_df)*100)) for k,v in drivers.items() if v > 0]
    else:
        top_drivers = []

    neg_df = b_df[b_df['score'] <= 3]
    if not neg_df.empty and theme_cols:
        valid_themes = [t for t in theme_cols if t in neg_df.columns]
        barriers = neg_df[valid_themes].sum().sort_values(ascending=False).head(3)
        top_barriers = [(k, v, (v/len(neg_df)*100)) for k,v in barriers.items() if v > 0]
    else:
        top_barriers = []
        
    return {
        "brand": brand, "vol": vol, "score": score, "nps": nps_proxy,
        "drivers": top_drivers, "barriers": top_barriers
    }

# ==========================================
# 4. SIDEBAR & INITIALIZATION
# ==========================================
with st.sidebar:
    st.title("🎛️ Command Center")
    
    with st.spinner("Connecting to Live Database..."):
        df_raw = load_data()
    
    if df_raw.empty:
        st.error("⚠️ No Data Found.")
        st.stop()

    st.success(f"🟢 Live: {len(df_raw):,} Rows")
    st.markdown("---")
    
    # --- GLOBAL FILTERS ---
    min_d, max_d = df_raw['at'].min().date(), df_raw['at'].max().date()
    date_range = st.date_input("Period", [min_d, max_d], min_value=min_d, max_value=max_d)
    
    all_brands = sorted(df_raw['App_Name'].dropna().unique())
    sel_brands = st.multiselect("Brands", all_brands, default=all_brands)
    
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

# --- FILTERING ---
if len(date_range) == 2:
    mask = (
        (df_raw['at'].dt.date >= date_range[0]) & 
        (df_raw['at'].dt.date <= date_range[1]) & 
        (df_raw['App_Name'].isin(sel_brands))
    )
else:
    mask = df_raw['App_Name'].isin(sel_brands)

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
if len(date_range) == 2:
    st.markdown(f"**Analysis Scope:** {len(df):,} Reviews | {len(sel_brands)} Brands | {date_range[0]} to {date_range[1]}")

tab_exec, tab_trends, tab_compare, tab_themes, tab_raw = st.tabs([
    "📊 Boardroom Summary", 
    "📈 Market Trends (MoM/WoW)", 
    "⚔️ Head-to-Head", 
    "🧠 Thematic Deep Dive", 
    "🔍 Data Explorer"
])

# === TAB 1: BOARDROOM SUMMARY ===
with tab_exec:
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Total Volume", f"{len(df):,}")
    with k2: st.metric("Market CSAT", f"{df['score'].mean():.2f} ⭐")
    with k3:
        prom = len(df[df['score']==5])
        det = len(df[df['score']<=3])
        nps = ((prom - det) / len(df)) * 100 if len(df)>0 else 0
        st.metric("Aggregated NPS", f"{nps:.0f}", delta="Proxy Score")
    with k4:
        risk = (len(df[df['score']==1]) / len(df)) * 100 if len(df)>0 else 0
        st.metric("Critical Risk Ratio", f"{risk:.1f}%", delta="1-Star Vol", delta_color="inverse")

    st.markdown("---")
    st.markdown("### 🏥 Brand Health Matrix")
    
    brand_stats = df.groupby('App_Name').agg(
        Volume=('score', 'count'),
        CSAT=('score', 'mean'),
        One_Star=('score', lambda x: (x==1).sum()),
        Five_Star=('score', lambda x: (x==5).sum())
    ).reset_index()
    
    brand_stats['NPS_Proxy'] = ((brand_stats['Five_Star'] - brand_stats['One_Star']) / brand_stats['Volume'] * 100).round(1)
    brand_stats['CSAT'] = brand_stats['CSAT'].round(2)
    
    fig = px.scatter(
        brand_stats, x="CSAT", y="NPS_Proxy", size="Volume", color="App_Name",
        hover_name="App_Name", text="App_Name", title="Brand Positioning (Size = Volume)",
        color_discrete_sequence=px.colors.qualitative.Bold, height=500
    )
    fig.update_traces(textposition='top center')
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", xaxis_title="Customer Satisfaction (CSAT)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🤖 Analyst AI: Strategic Briefs")
    cols = st.columns(3)
    for i, brand in enumerate(sel_brands):
        with cols[i % 3]:
            data = run_strategic_analysis(df, brand, theme_cols)
            if not data: continue
            with st.container():
                st.markdown(f"#### {brand}")
                color = "#10b981" if data['nps'] > 20 else "#f59e0b" if data['nps'] > 0 else "#ef4444"
                st.markdown(f"<h2 style='color:{color}; margin:0;'>{data['score']:.2f} ⭐</h2>", unsafe_allow_html=True)
                st.caption(f"NPS Proxy: {data['nps']:.0f}")
                
                st.markdown("**🚀 Drivers:**")
                if data['drivers']:
                    for t, c, p in data['drivers']: st.markdown(f"- {t} **({p:.0f}%)**")
                else: st.markdown("- *N/A*")
                
                st.markdown("**⚠️ Killers:**")
                if data['barriers']:
                    for t, c, p in data['barriers']: st.markdown(f"- {t} **({p:.0f}%)**")
                else: st.markdown("- *N/A*")
                st.markdown("---")

# === TAB 2: MARKET TRENDS (MoM / WoW) ===
with tab_trends:
    st.markdown("### 📅 Monthly (MoM) & Weekly (WoW) Dynamics")
    
    # Toggle for View
    view_mode = st.radio("Time Aggregation", ["Monthly", "Weekly"], horizontal=True, label_visibility="collapsed")
    
    if 'at' in df.columns:
        if view_mode == "Monthly":
            # MONTHLY LOGIC
            trend_df = df.groupby(['Month', 'App_Name']).agg(
                Volume=('score', 'count'),
                Rating=('score', 'mean')
            ).reset_index()
            time_col = 'Month'
            
        else:
            # WEEKLY LOGIC
            trend_df = df.groupby(['Week', 'App_Name']).agg(
                Volume=('score', 'count'),
                Rating=('score', 'mean')
            ).reset_index()
            trend_df['Week'] = trend_df['Week'].astype(str) # Convert for plotting
            time_col = 'Week'

        # 1. CHARTS
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{view_mode} Volume Velocity**")
            fig_v = px.line(trend_df, x=time_col, y='Volume', color='App_Name', markers=True)
            fig_v.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_v, use_container_width=True)
        with c2:
            st.markdown(f"**{view_mode} CSAT Trajectory**")
            fig_r = px.line(trend_df, x=time_col, y='Rating', color='App_Name', markers=True)
            fig_r.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", yaxis_range=[1,5])
            st.plotly_chart(fig_r, use_container_width=True)

        st.markdown("---")
        
        # 2. GROWTH HEATMAPS (MoM / WoW)
        c3, c4 = st.columns(2)
        
        # Volume Growth
        pivot_vol = trend_df.pivot(index=time_col, columns='App_Name', values='Volume').fillna(0)
        growth_vol = pivot_vol.pct_change() * 100
        growth_vol = growth_vol.iloc[1:] # Drop first row (NaNs)
        
        with c3:
            st.markdown(f"**🔥 Volume Growth % ({'MoM' if view_mode=='Monthly' else 'WoW'})**")
            if not growth_vol.empty:
                # FIX: use color_continuous_midpoint=0 for centering
                fig_h1 = px.imshow(growth_vol.T, text_auto='.1f', aspect="auto", 
                                   color_continuous_scale='RdBu', color_continuous_midpoint=0)
                fig_h1.update_layout(template="plotly_dark")
                st.plotly_chart(fig_h1, use_container_width=True)
            else:
                st.info("Not enough data history to calculate growth.")

        # Rating Change
        pivot_rat = trend_df.pivot(index=time_col, columns='App_Name', values='Rating')
        diff_rat = pivot_rat.diff()
        diff_rat = diff_rat.iloc[1:]
        
        with c4:
            st.markdown(f"**⭐ Rating Change ({'MoM' if view_mode=='Monthly' else 'WoW'})**")
            if not diff_rat.empty:
                fig_h2 = px.imshow(diff_rat.T, text_auto='.2f', aspect="auto", 
                                   color_continuous_scale='RdBu', color_continuous_midpoint=0)
                fig_h2.update_layout(template="plotly_dark")
                st.plotly_chart(fig_h2, use_container_width=True)
            else:
                st.info("Not enough data history to calculate rating change.")

# === TAB 3: HEAD TO HEAD ===
with tab_compare:
    st.markdown("### ⚔️ Competitive Battleground")
    c1, c2 = st.columns(2)
    with c1: brand_a = st.selectbox("Brand A", sel_brands, index=0)
    with c2: 
        opts = [b for b in sel_brands if b != brand_a]
        brand_b = st.selectbox("Brand B", opts, index=0 if opts else 0)
    
    if brand_a and brand_b:
        data_a = run_strategic_analysis(df, brand_a, theme_cols)
        data_b = run_strategic_analysis(df, brand_b, theme_cols)
        
        comp_data = {
            "Metric": ["Avg Rating", "NPS Proxy", "Volume Share", "1-Star %"],
            brand_a: [f"{data_a['score']:.2f}", f"{data_a['nps']:.0f}", f"{(data_a['vol']/len(df)*100):.1f}%", f"{(len(df[(df['App_Name']==brand_a) & (df['score']==1)])/data_a['vol']*100):.1f}%"],
            brand_b: [f"{data_b['score']:.2f}", f"{data_b['nps']:.0f}", f"{(data_b['vol']/len(df)*100):.1f}%", f"{(len(df[(df['App_Name']==brand_b) & (df['score']==1)])/data_b['vol']*100):.1f}%"]
        }
        st.dataframe(pd.DataFrame(comp_data).set_index("Metric"), use_container_width=True)
        
        if theme_cols:
            drivers_a = [x[0] for x in data_a['drivers']] if data_a['drivers'] else []
            drivers_b = [x[0] for x in data_b['drivers']] if data_b['drivers'] else []
            common_themes = list(set(drivers_a + drivers_b))[:6]
            
            if common_themes:
                def get_theme_pct(b_name, t_list):
                    d = df[df['App_Name'] == b_name]
                    if d.empty: return [0]*len(t_list)
                    return [(d[t].sum() / len(d) * 100) for t in t_list]

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=get_theme_pct(brand_a, common_themes), theta=common_themes, fill='toself', name=brand_a))
                fig.add_trace(go.Scatterpolar(r=get_theme_pct(brand_b, common_themes), theta=common_themes, fill='toself', name=brand_b))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True)), template="plotly_dark", title="Thematic Footprint Comparison")
                st.plotly_chart(fig, use_container_width=True)

# === TAB 4: THEMATIC DEEP DIVE ===
with tab_themes:
    st.markdown("### 🧠 Thematic Heatmap (Marketer's View)")
    if theme_cols:
        heatmap_data = df.groupby('App_Name')[theme_cols].sum()
        heatmap_pct = heatmap_data.div(df.groupby('App_Name').size(), axis=0) * 100
        heatmap_pct = heatmap_pct.T
        heatmap_pct['Total'] = heatmap_pct.sum(axis=1)
        heatmap_pct = heatmap_pct.sort_values('Total', ascending=False).drop(columns=['Total']).head(20)
        
        fig = px.imshow(heatmap_pct, text_auto='.1f', aspect="auto", color_continuous_scale='Magma', title="Theme Intensity (%)", height=800)
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🛑 1-Star Correlation Analysis")
        df['is_one_star'] = (df['score'] == 1).astype(int)
        corrs = {}
        for t in theme_cols:
            if df[t].sum() > 5: corrs[t] = df['is_one_star'].corr(df[t])
        
        if corrs:
            corr_df = pd.DataFrame(list(corrs.items()), columns=['Theme', 'Correlation']).sort_values('Correlation', ascending=False).head(10)
            fig_corr = px.bar(corr_df, x='Correlation', y='Theme', orientation='h', title="Themes driving 1-Star Ratings", color='Correlation', color_continuous_scale='Reds')
            fig_corr.update_layout(template="plotly_dark", yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_corr, use_container_width=True)

# === TAB 5: RAW DATA ===
with tab_raw:
    st.markdown("### 🔍 Data Explorer")
    q = st.text_input("Deep Search (Review Text)", placeholder="e.g. 'fraud', 'hidden charges'...")
    f_df = df.copy()
    if q: f_df = f_df[f_df['Review_Text'].astype(str).str.contains(q, case=False, na=False)]
    
    st.dataframe(
        f_df[['at', 'App_Name', 'score', 'Review_Text', 'Sentiment']],
        use_container_width=True, height=600,
        column_config={"score": st.column_config.NumberColumn("Rating", format="%d ⭐"), "Review_Text": st.column_config.TextColumn("Feedback", width="large")}
    )
