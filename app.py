import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import requests
import time
import re
import gc
from collections import Counter
from datetime import timedelta, datetime
import pytz
from supabase import create_client
from streamlit_lottie import st_lottie

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Strategic Intelligence Platform",
    page_icon="🦅",
    layout="wide",
)

# ======================================================
# CLEAN EXECUTIVE CSS
# ======================================================

st.markdown("""
<style>
.stApp { background:#0b0f19; color:#e2e8f0; }
.block { border:1px solid #334155; padding:18px; border-radius:14px; background:#0f172a; }
.kpi { background:#111827; border-radius:12px; padding:18px; border:1px solid #334155; }
hr { border-color:#334155; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# HELPERS
# ======================================================

def dark_chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.04)",
        font=dict(color="#e5e7eb"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


@st.cache_resource
def init_connection():
    try:
        return create_client(
            st.secrets["supabase"]["url"],
            st.secrets["supabase"]["key"]
        )
    except:
        return None


supabase = init_connection()


@st.cache_data(ttl=900)
def load_data():
    rows = []
    start = 0
    while True:
        res = supabase.table("reviews").select("*").range(start, start + 2000).execute()
        if not res.data:
            break
        rows.extend(res.data)
        if len(res.data) < 2000:
            break
        start += 2000

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    ist = pytz.timezone("Asia/Kolkata")

    df["at"] = pd.to_datetime(df["Review_Date"], errors="coerce")
    df["at"] = df["at"].dt.tz_localize("UTC").dt.tz_convert(ist)
    df.dropna(subset=["at"], inplace=True)

    df["Month"] = df["at"].dt.strftime("%Y-%m")
    df["Week"] = df["at"].dt.strftime("%Y-W%V")

    df["score"] = pd.to_numeric(df["Rating"], errors="coerce")

    df["Sentiment_Label"] = pd.cut(
        df["score"], bins=[0,2,3,5], labels=["Negative","Neutral","Positive"]
    )

    df["char_count"] = df["Review_Text"].astype(str).str.len()
    df["length_bucket"] = np.where(df["char_count"]<=29,"Brief","Detailed")

    theme_cols = []
    for col in df.columns:
        try:
            s = pd.to_numeric(df[col], errors="coerce").fillna(0)
            if set(s.unique()).issubset({0,1}):
                theme_cols.append(col)
        except:
            pass

    st.session_state["theme_cols"] = theme_cols
    st.session_state["last_fetched"] = datetime.now(ist).strftime("%d %b %Y %I:%M %p")

    gc.collect()
    return df


# ======================================================
# LOAD DATA
# ======================================================

if "df" not in st.session_state:
    with st.spinner("Loading strategic data..."):
        st.session_state["df"] = load_data()

df_raw = st.session_state["df"]

if df_raw.empty:
    st.error("No data found.")
    st.stop()

# ======================================================
# COMMAND PANEL (UPI STYLE)
# ======================================================

with st.sidebar.container(border=True):

    st.subheader("🎛 Command Center")
    st.metric("Live Reviews", f"{len(df_raw):,}")

    st.divider()

    min_d = df_raw["at"].min().date()
    max_d = df_raw["at"].max().date()

    date_range = st.date_input("Period", [min_d, max_d])

    st.divider()

    brands = sorted(df_raw["App_Name"].dropna().unique())
    sel_brands = st.multiselect("Brands", brands, default=brands)

    st.divider()

    sel_ratings = st.multiselect("Ratings", [1,2,3,4,5], default=[1,2,3,4,5])


# ======================================================
# FILTER
# ======================================================

start = pd.to_datetime(date_range[0]).tz_localize("Asia/Kolkata")
end = pd.to_datetime(date_range[1]).tz_localize("Asia/Kolkata") + timedelta(days=1)

df = df_raw[
    (df_raw["at"]>=start) &
    (df_raw["at"]<=end) &
    (df_raw["App_Name"].isin(sel_brands)) &
    (df_raw["score"].isin(sel_ratings))
].copy()

theme_cols = st.session_state.get("theme_cols", [])

# ======================================================
# NAV
# ======================================================

tabs = [
    "📊 Boardroom",
    "🚀 Drivers & Barriers",
    "⚔️ Head to Head",
    "📈 Trends",
    "🔡 Text",
    "🤖 AI Brief"
]

active = st.radio("", tabs, horizontal=True)

st.caption(f"Last updated: {st.session_state.get('last_fetched')}")

# ======================================================
# BOARDROOM TAB
# ======================================================

if active == "📊 Boardroom":

    kpi = st.container(border=True)
    with kpi:
        c1,c2,c3,c4 = st.columns(4)

        vol = len(df)
        avg = df["score"].mean()

        nps = ((len(df[df["score"]==5]) - len(df[df["score"]<=3]))/vol*100) if vol else 0
        risk = (len(df[df["score"]==1])/vol*100) if vol else 0

        c1.metric("Volume", f"{vol:,}")
        c2.metric("Avg Rating", f"{avg:.2f}")
        c3.metric("NPS Proxy", f"{nps:.0f}")
        c4.metric("1★ Risk %", f"{risk:.1f}")

    st.divider()

    brand_kpi = df.groupby("App_Name")["score"].agg(["count","mean"]).reset_index()
    brand_kpi.columns = ["Brand","Volume","CSAT"]

    left,right = st.columns(2)

    with left.container(border=True):
        fig = px.pie(brand_kpi, values="Volume", names="Brand", hole=0.45)
        st.plotly_chart(dark_chart(fig), use_container_width=True)

    with right.container(border=True):
        fig = px.scatter(
            brand_kpi,
            x="CSAT",
            y="Volume",
            size="Volume",
            color="Brand",
            text="Brand"
        )
        st.plotly_chart(dark_chart(fig), use_container_width=True)
# ======================================================
# DRIVERS & BARRIERS
# ======================================================

elif active == "🚀 Drivers & Barriers":

    pos = df[df["score"] >= 4]
    neg = df[df["score"] <= 3]

    def aggregate(sub):
        if sub.empty or not theme_cols:
            return pd.DataFrame()
        counts = sub[theme_cols].sum().sort_values(ascending=False).head(12)
        out = pd.DataFrame({
            "Theme": counts.index,
            "Pct": counts.values / len(sub) * 100
        })
        return out

    left,right = st.columns(2)

    with left.container(border=True):
        st.subheader("Top Drivers")
        agg = aggregate(pos)
        if not agg.empty:
            fig = px.bar(agg, x="Pct", y="Theme", orientation="h", text="Pct")
            st.plotly_chart(dark_chart(fig), use_container_width=True)
        else:
            st.caption("No driver data")

    with right.container(border=True):
        st.subheader("Top Barriers")
        agg = aggregate(neg)
        if not agg.empty:
            fig = px.bar(agg, x="Pct", y="Theme", orientation="h", text="Pct")
            st.plotly_chart(dark_chart(fig), use_container_width=True)
        else:
            st.caption("No barrier data")

# ======================================================
# HEAD TO HEAD
# ======================================================

elif active == "⚔️ Head to Head":

    c1,c2 = st.columns(2)
    b1 = c1.selectbox("Brand A", sel_brands)
    b2 = c2.selectbox("Brand B", [b for b in sel_brands if b!=b1])

    d1 = df[df["App_Name"]==b1]
    d2 = df[df["App_Name"]==b2]

    k = st.container(border=True)
    with k:
        c1,c2,c3 = st.columns(3)
        c1.metric(b1+" Avg", f"{d1['score'].mean():.2f}")
        c2.metric(b2+" Avg", f"{d2['score'].mean():.2f}")
        c3.metric("Volume Gap", f"{len(d1)-len(d2):,}")

    trend = df[df["App_Name"].isin([b1,b2])].groupby(
        ["Month","App_Name"]
    )["score"].mean().reset_index()

    with st.container(border=True):
        fig = px.line(trend, x="Month", y="score", color="App_Name", markers=True)
        st.plotly_chart(dark_chart(fig), use_container_width=True)

# ======================================================
# TRENDS
# ======================================================

elif active == "📈 Trends":

    sent = df.groupby(["Month","Sentiment_Label"]).size().reset_index(name="Count")
    sent["Pct"] = sent["Count"] / sent.groupby("Month")["Count"].transform("sum") * 100

    with st.container(border=True):
        fig = px.bar(
            sent, x="Month", y="Pct", color="Sentiment_Label",
            title="Sentiment trend"
        )
        st.plotly_chart(dark_chart(fig), use_container_width=True)

    csat = df.groupby(["Month","App_Name"])["score"].mean().reset_index()

    with st.container(border=True):
        fig = px.line(csat, x="Month", y="score", color="App_Name", markers=True)
        st.plotly_chart(dark_chart(fig), use_container_width=True)

# ======================================================
# TEXT ANALYTICS
# ======================================================

elif active == "🔡 Text":

    def top_words(series, base, n=20):
        stop = {"the","and","to","of","is","for","in","it","this","that","app","loan","money"}
        counter = Counter()
        for t in series.dropna():
            t = re.sub(r"[^a-z\s]","",str(t).lower())
            for w in t.split():
                if w not in stop and len(w)>2:
                    counter[w]+=1
        data = [
            {"Word":k,"Pct":v/base*100}
            for k,v in counter.most_common(n)
        ]
        return pd.DataFrame(data)

    brand = st.selectbox("Brand", ["All"]+sel_brands)
    sent = st.radio("Sentiment", ["Positive","Negative"], horizontal=True)

    sub = df if brand=="All" else df[df["App_Name"]==brand]
    sub = sub[sub["score"]>=4] if sent=="Positive" else sub[sub["score"]<=3]

    if not sub.empty:
        words = top_words(sub["Review_Text"], len(sub))

        with st.container(border=True):
            fig = px.bar(words, x="Pct", y="Word", orientation="h", text="Pct")
            st.plotly_chart(dark_chart(fig), use_container_width=True)

# ======================================================
# AI BRIEF
# ======================================================

elif active == "🤖 AI Brief":

    for brand in sel_brands:
        b = df[df["App_Name"]==brand]
        if b.empty:
            continue

        avg = b["score"].mean()
        vol = len(b)

        with st.container(border=True):
            st.subheader(brand)
            st.write(f"Average Rating: **{avg:.2f}**")
            st.write(f"Volume: **{vol:,} reviews**")

            if theme_cols:
                pos = b[b["score"]>=4][theme_cols].sum()
                neg = b[b["score"]<=3][theme_cols].sum()

                if not pos.empty:
                    st.write("Top Driver:", pos.idxmax())

                if not neg.empty:
                    st.write("Top Barrier:", neg.idxmax())
