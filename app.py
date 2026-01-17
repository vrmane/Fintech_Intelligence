import streamlit as st
from supabase import create_client, Client
import pandas as pd
import plotly.express as px

# 1. Setup Page Config
st.set_page_config(page_title="Reviews Dashboard", layout="wide")


# 2. Initialize Connection (Cached to avoid reconnecting on every click)
@st.cache_resource
def init_connection():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)


supabase = init_connection()


# 3. Fetch Data (Cached to improve speed)
@st.cache_data(ttl=600)  # Cache clears every 10 mins automatically
def get_data():
    # Fetch all rows from the 'reviews' table
    response = supabase.table("reviews").select("*").execute()

    # Convert to Pandas DataFrame
    df = pd.DataFrame(response.data)

    # Ensure Date columns are actual Date Objects (not strings)
    if not df.empty:
        df['Review_Date'] = pd.to_datetime(df['Review_Date'])
        df['Inserted_On'] = pd.to_datetime(df['Inserted_On'])

        # Ensure Rating is numeric
        df['Rating'] = pd.to_numeric(df['Rating'])

    return df


# --- MAIN APP UI ---

st.title("📊 App Reviews Dashboard")

# Load Data
with st.spinner('Fetching data from Supabase...'):
    df = get_data()

if df.empty:
    st.warning("No data found in Supabase table 'reviews'.")
else:
    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Filters")

    # Filter by App Name
    app_list = ["All"] + list(df['App_Name'].unique())
    selected_app = st.sidebar.selectbox("Select App", app_list)

    # Filter by Rating
    rating_filter = st.sidebar.slider("Filter by Rating", 1, 5, (1, 5))

    # Apply Filters
    filtered_df = df.copy()
    if selected_app != "All":
        filtered_df = filtered_df[filtered_df['App_Name'] == selected_app]

    filtered_df = filtered_df[
        (filtered_df['Rating'] >= rating_filter[0]) &
        (filtered_df['Rating'] <= rating_filter[1])
        ]

    # --- TOP METRICS ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", f"{len(filtered_df):,}")
    col2.metric("Average Rating", f"{filtered_df['Rating'].mean():.2f} ⭐")

    # Calculate latest review date
    latest_date = filtered_df['Review_Date'].max().strftime('%Y-%m-%d')
    col3.metric("Latest Review", latest_date)

    st.markdown("---")

    # --- CHARTS ---
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Ratings Distribution")
        rating_counts = filtered_df['Rating'].value_counts().reset_index()
        rating_counts.columns = ['Rating', 'Count']
        fig_bar = px.bar(rating_counts, x='Rating', y='Count', color='Rating', text='Count')
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_chart2:
        st.subheader("Reviews Over Time")
        # Group by Month for cleaner trend line
        trend_df = filtered_df.set_index('Review_Date').resample('M').size().reset_index(name='Count')
        fig_line = px.line(trend_df, x='Review_Date', y='Count', markers=True)
        st.plotly_chart(fig_line, use_container_width=True)

    # --- RAW DATA TABLE ---
    st.subheader("📝 Raw Review Data")
    st.dataframe(
        filtered_df[['Review_Date', 'App_Name', 'Rating', 'Review_Text', 'Gemini_Audit_Output']],
        use_container_width=True,
        hide_index=True
    )