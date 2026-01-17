import streamlit as st
from supabase import create_client, Client
import pandas as pd
import plotly.express as px

# 1. Setup Page Config
st.set_page_config(page_title="Reviews Dashboard", layout="wide")

# 2. Initialize Connection
@st.cache_resource
def init_connection():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

supabase = init_connection()

# 3. Fetch ALL Data (Fixed Pagination)
@st.cache_data(ttl=600) 
def get_data():
    all_rows = []
    start = 0
    # We set this to 1000 because Supabase's default API limit is 1000.
    # If we ask for more, the API cuts us off at 1000, and our loop logic breaks.
    batch_size = 1000 
    
    # Progress placeholder
    progress_text = st.empty()
    
    while True:
        progress_text.text(f"Fetching rows {start} to {start + batch_size}...")
        
        # Fetch batch
        response = supabase.table("reviews").select("*").range(start, start + batch_size - 1).execute()
        rows = response.data
        
        if not rows:
            break
            
        all_rows.extend(rows)
        
        # If we get fewer than batch_size, we've reached the end
        if len(rows) < batch_size:
            break
            
        start += batch_size

    progress_text.empty() # Clear the progress text
    
    # Convert to DataFrame
    df = pd.DataFrame(all_rows)
    
    if not df.empty:
        if 'Review_Date' in df.columns:
            df['Review_Date'] = pd.to_datetime(df['Review_Date'])
        if 'Inserted_On' in df.columns:
            df['Inserted_On'] = pd.to_datetime(df['Inserted_On'])
        if 'Rating' in df.columns:
            df['Rating'] = pd.to_numeric(df['Rating'])
            
    return df

# --- MAIN APP UI ---

st.title("📊 App Reviews Dashboard")

# Load Data
with st.spinner('Loading data... (This may take a moment for 40k+ rows)'):
    df = get_data()

if df.empty:
    st.warning("No data found in Supabase table 'reviews'.")
else:
    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Filters")
    
    app_list = ["All"] + sorted(list(df['App_Name'].dropna().unique()))
    selected_app = st.sidebar.selectbox("Select App", app_list)
    
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
    
    avg_rating = filtered_df['Rating'].mean()
    col2.metric("Average Rating", f"{avg_rating:.2f} ⭐" if not pd.isna(avg_rating) else "N/A")
    
    if not filtered_df['Review_Date'].isnull().all():
        latest_date = filtered_df['Review_Date'].max().strftime('%Y-%m-%d')
    else:
        latest_date = "N/A"
    col3.metric("Latest Review", latest_date)

    st.markdown("---")

    # --- CHARTS ---
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Ratings Distribution")
        if not filtered_df.empty:
            rating_counts = filtered_df['Rating'].value_counts().reset_index()
            rating_counts.columns = ['Rating', 'Count']
            fig_bar = px.bar(rating_counts, x='Rating', y='Count', color='Rating', text='Count')
            st.plotly_chart(fig_bar, use_container_width=True)

    with col_chart2:
        st.subheader("Reviews Over Time")
        if not filtered_df.empty:
            trend_df = filtered_df.set_index('Review_Date').resample('M').size().reset_index(name='Count')
            fig_line = px.line(trend_df, x='Review_Date', y='Count', markers=True)
            st.plotly_chart(fig_line, use_container_width=True)

    # --- RAW DATA TABLE ---
    st.subheader("📝 Raw Review Data")
    display_cols = ['Review_Date', 'App_Name', 'Rating', 'Review_Text', 'Gemini_Audit_Output']
    actual_cols = [c for c in display_cols if c in filtered_df.columns]
    
    st.dataframe(
        filtered_df[actual_cols],
        use_container_width=True,
        hide_index=True
    )
