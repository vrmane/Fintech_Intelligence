import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
import gc
from collections import Counter
from datetime import datetime, timedelta
from supabase import create_client

# =====================================================
# CONFIG
# =====================================================

st.set_page_config("Strategic Intelligence Platform", "🦅", layout="wide")

st.markdown("""
<style>
.stApp{background:#0b0f19;color:#e5e7eb}
.card{border:1px solid #334155;padding:18px;border-radius:14px;background:#0f172a}
</style>
""", unsafe_allow_html=True)

# =====================================================
# DB
# =====================================================

@st.cache_resource
def db():
    return create_client(
        st.secrets["supabase"]["url"],
        st.secrets["supabase"]["key"]
    )

supabase = db()

# =====================================================
# DATA LOADER (FULLY TZ SAFE)
# =====================================================

@st.cache_data(ttl=900)
def load_data():

    all_rows = []
    start = 0

    while True:
        res = supabase.table("reviews").select("*").range(start, start + 2000).execute()
        if not res.data:
            break
        all_rows.extend(res.data)
        if len(res.data) < 2000:
            break
        start += 2000

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    # ---- SAFE datetime handling ----
    df["at"] = pd.to_datetime(df["Review_Date"], utc=True, errors="coerce")
    df.dropna(subset=["at"], inplace=True)

    df["at"] = df["at"].dt.tz_convert("Asia/Kolkata")

    df["Month"] = df["at"].dt.to_period("M").astype(str)
    df["Week"] = df["at"].dt.strftime("%Y-W%V")

    df["score"] = pd.to_numeric(df["Rating"], errors="coerce")

    df["Sentiment_Label"] = pd.cut(
        df["score"], [0,2,3,5], labels=["Negative","Neutral","Positive"]
    )

    df["char_count"] = df["Review_Text"].astype(str).str.len()
    df["length_bucket"] = np.where(df["char_count"]<=29,"Brief","Detailed")

    # ---- Theme detection ----
    theme_cols = []
    for c in df.columns:
        try:
            s = pd.to_numeric(df[c], errors="coerce").fillna(0)
            if set(s.unique()).issubset({0,1}):
                theme_cols.append(c)
        except:
            pass

    st.session_state["theme_cols"] = theme_cols
    st.session_state["last_fetched"] = datetime.now().strftime("%d %b %Y %I:%M %p")

    gc.collect()
    return df

# =====================================================
# LOAD
# =====================================================

if "df" not in st.session_state:
    with st.spinner("Loading strategic data..."):
        st.session_state["df"] = load_data()

df_raw = st.session_state["df"]

if df_raw.empty:
    st.error("No data available")
    st.stop()

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar.container(border=True):
    st.subheader("🎛 Command Center")

    st.metric("Live Reviews", f"{len(df_raw):,}")

    min_d = df_raw["at"].min().date()
    max_d = df_raw["at"].max().date()

    date_range = st.date_input("Period", [min_d, max_d])

    brands = sorted(df_raw["App_Name"].dropna().unique())
    sel_brands = st.multiselect("Brands", brands, default=brands)

    sel_ratings = st.multiselect("Ratings", [1,2,3,4,5], default=[1,2,3,4,5])

# =====================================================
# FILTER (CRASH SAFE)
# =====================================================

start = pd.Timestamp(date_range[0], tz="Asia/Kolkata")
end = pd.Timestamp(date_range[1], tz="Asia/Kolkata") + pd.Timedelta(days=1)

df = df_raw[
    (df_raw["at"] >= start) &
    (df_raw["at"] <= end) &
    (df_raw["App_Name"].isin(sel_brands)) &
    (df_raw["score"].isin(sel_ratings))
].copy()

theme_cols = st.session_state.get("theme_cols", [])

# =====================================================
# NAV
# =====================================================

tabs = ["📊 Boardroom","🚀 Drivers & Barriers","⚔️ Head-to-Head","📈 Trends","🔡 Text","🤖 AI Brief"]
active = st.radio("", tabs, horizontal=True)

st.caption(f"Last fetched: {st.session_state.get('last_fetched')}")

# =====================================================
# BOARDROOM
# =====================================================

if active == "📊 Boardroom":

    k1,k2,k3,k4 = st.columns(4)

    vol = len(df)
    avg = df["score"].mean()

    nps = ((len(df[df["score"]==5]) - len(df[df["score"]<=3])) / vol * 100) if vol else 0
    risk = len(df[df["score"]==1]) / vol * 100 if vol else 0

    k1.metric("Volume", f"{vol:,}")
    k2.metric("Avg Rating", f"{avg:.2f}")
    k3.metric("NPS Proxy", f"{nps:.0f}")
    k4.metric("1★ Risk %", f"{risk:.1f}")

    brand = df.groupby("App_Name")["score"].agg(["count","mean"]).reset_index()

    c1,c2 = st.columns(2)

    with c1:
        st.plotly_chart(px.pie(brand, values="count", names="App_Name", hole=0.45),
                        use_container_width=True)

    with c2:
        st.plotly_chart(
            px.scatter(brand, x="mean", y="count", size="count", color="App_Name"),
            use_container_width=True
        )

# =====================================================
# DRIVERS & BARRIERS
# =====================================================

elif active == "🚀 Drivers & Barriers":

    def agg(sub):
        if sub.empty or not theme_cols:
            return pd.DataFrame()
        c = sub[theme_cols].sum().sort_values(ascending=False).head(12)
        return pd.DataFrame({"Theme": c.index, "Pct": c.values/len(sub)*100})

    pos = df[df["score"]>=4]
    neg = df[df["score"]<=3]

    l,r = st.columns(2)

    with l:
        st.subheader("Top Drivers")
        a = agg(pos)
        if not a.empty:
            st.plotly_chart(px.bar(a, x="Pct", y="Theme", orientation="h"),
                            use_container_width=True)

    with r:
        st.subheader("Top Barriers")
        a = agg(neg)
        if not a.empty:
            st.plotly_chart(px.bar(a, x="Pct", y="Theme", orientation="h"),
                            use_container_width=True)

# =====================================================
# HEAD TO HEAD
# =====================================================

elif active == "⚔️ Head-to-Head":

    b1,b2 = st.columns(2)
    brand1 = b1.selectbox("Brand A", sel_brands)
    brand2 = b2.selectbox("Brand B", [b for b in sel_brands if b!=brand1])

    trend = df[df["App_Name"].isin([brand1,brand2])] \
                .groupby(["Month","App_Name"])["score"].mean().reset_index()

    st.plotly_chart(
        px.line(trend, x="Month", y="score", color="App_Name", markers=True),
        use_container_width=True
    )

# =====================================================
# TRENDS
# =====================================================

elif active == "📈 Trends":

    sent = df.groupby(["Month","Sentiment_Label"]).size().reset_index(name="Count")
    sent["Pct"] = sent["Count"] / sent.groupby("Month")["Count"].transform("sum") * 100

    st.plotly_chart(
        px.bar(sent, x="Month", y="Pct", color="Sentiment_Label"),
        use_container_width=True
    )

# =====================================================
# TEXT
# =====================================================

elif active == "🔡 Text":

    def top_words(s, base):
        stop = {"the","and","to","of","is","for","in","app","loan"}
        c = Counter()
        for t in s.dropna():
            t = re.sub(r"[^a-z\s]","",str(t).lower())
            for w in t.split():
                if w not in stop and len(w)>2:
                    c[w]+=1
        return pd.DataFrame(
            [{"Word":k,"Pct":v/base*100} for k,v in c.most_common(20)]
        )

    brand = st.selectbox("Brand", ["All"] + sel_brands)
    sent = st.radio("Sentiment", ["Positive","Negative"], horizontal=True)

    sub = df if brand=="All" else df[df["App_Name"]==brand]
    sub = sub[sub["score"]>=4] if sent=="Positive" else sub[sub["score"]<=3]

    if not sub.empty:
        words = top_words(sub["Review_Text"], len(sub))
        st.plotly_chart(px.bar(words, x="Pct", y="Word", orientation="h"),
                        use_container_width=True)

# =====================================================
# AI BRIEF
# =====================================================

elif active == "🤖 AI Brief":

    for b in sel_brands:
        sub = df[df["App_Name"]==b]
        if sub.empty:
            continue

        st.subheader(b)
        st.write("Avg Rating:", round(sub["score"].mean(),2))
        st.write("Volume:", len(sub))

        if theme_cols:
            pos = sub[sub["score"]>=4][theme_cols].sum()
            neg = sub[sub["score"]<=3][theme_cols].sum()

            if not pos.empty:
                st.write("Top Driver:", pos.idxmax())
            if not neg.empty:
                st.write("Top Barrier:", neg.idxmax())
