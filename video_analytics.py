import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from api_client import get_vehicle_details
import os
import json
import time
import io
import base64
import tempfile
from PIL import Image
import cv2

# Set page configuration
st.set_page_config(
    page_title="Vehicle Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for styling
st.markdown("""
<style>
    .header-style {
        font-size: 28px;
        font-weight: bold;
        color: #2E86C1;
        padding: 10px;
        border-bottom: 2px solid #2E86C1;
        margin-bottom: 20px;
    }
    .subheader-style {
        font-size: 22px;
        font-weight: bold;
        color: #2E86C1;
        padding: 5px;
        margin: 15px 0;
    }
    .metric-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #F8F9F9;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin: 10px 0;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #2E86C1;
    }
    .metric-label {
        font-size: 14px;
        color: #555;
    }
    .dataframe th {
        background-color: #2E86C1 !important;
        color: white !important;
    }
    .dataframe td {
        text-align: center !important;
    }
    .stButton button {
        background-color: #2E86C1 !important;
        color: white !important;
        border-radius: 5px;
    }
    .filter-section {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .chart-container {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin: 15px 0;
    }
    .export-button {
        text-align: right;
        margin: 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2E86C1;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def get_csv_download_link(df, filename="data.csv", text="Download CSV"):
    """Generate a download link for a DataFrame CSV."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-link">{text}</a>'
    return href

def get_excel_download_link(df, filename="data.xlsx", text="Download Excel"):
    """Generate a download link for a DataFrame Excel file."""
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}" class="download-link">{text}</a>'
    return href

def calculate_metrics(df):
    """Calculate key metrics from vehicle data."""
    metrics = {
        "total_vehicles": len(df),
        "average_filling_time": df["duration"].mean() if "duration" in df.columns else 0,
        "max_filling_time": df["duration"].max() if "duration" in df.columns and len(df) > 0 else 0,
        "min_filling_time": df["duration"].min() if "duration" in df.columns and len(df) > 0 else 0,
        "active_vehicles": len(df[df["in_roi"] == True]) if "in_roi" in df.columns else 0
    }
    
    # Get vehicle types if available
    if "vehicle_type" in df.columns:
        metrics["vehicle_types"] = df["vehicle_type"].value_counts().to_dict()
    else:
        metrics["vehicle_types"] = {"Unknown": len(df)}
        
    return metrics

def format_duration(seconds):
    """Format seconds into a readable time format."""
    if pd.isna(seconds) or seconds == 0:
        return "N/A"
    
    minutes, seconds = divmod(int(seconds), 60)
    if minutes > 60:
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes}m {seconds}s"
    return f"{minutes}m {seconds}s"

def parse_time(time_str):
    """Parse time string to datetime object."""
    if pd.isna(time_str) or time_str == "":
        return None
    try:
        return datetime.strptime(time_str, "%H:%M:%S")
    except:
        return None

def create_hourly_distribution(df):
    """Create hourly distribution of vehicle entries."""
    if "entry_time" not in df.columns or len(df) == 0:
        return pd.DataFrame({"hour": range(24), "count": [0] * 24})
    
    # Parse entry times
    df_copy = df.copy()
    df_copy["entry_hour"] = df_copy["entry_time"].apply(
        lambda x: parse_time(x).hour if parse_time(x) else None
    )
    
    # Count entries per hour
    hourly_counts = df_copy["entry_hour"].value_counts().reset_index()
    hourly_counts.columns = ["hour", "count"]
    
    # Make sure all hours are represented
    all_hours = pd.DataFrame({"hour": range(24)})
    hourly_counts = pd.merge(all_hours, hourly_counts, on="hour", how="left").fillna(0)
    
    return hourly_counts

def create_vehicle_type_chart(vehicle_types):
    """Create chart for vehicle type distribution."""
    if not vehicle_types:
        return None
    
    # Create DataFrame from vehicle types
    df = pd.DataFrame(list(vehicle_types.items()), columns=["type", "count"])
    
    # Create chart
    fig = px.pie(df, values="count", names="type", title="Vehicle Type Distribution", 
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_duration_histogram(df):
    """Create histogram of filling durations."""
    if "duration" not in df.columns or len(df) == 0:
        return None
    
    # Filter out rows with duration
    df_filtered = df[df["duration"] > 0].copy()
    if len(df_filtered) == 0:
        return None
    
    # Create histogram
    fig = px.histogram(df_filtered, x="duration", 
                       title="Distribution of Filling Duration (seconds)",
                       labels={"duration": "Duration (seconds)", "count": "Number of Vehicles"},
                       nbins=20)
    fig.update_layout(bargap=0.1)
    return fig

def create_hourly_trend_chart(hourly_df):
    """Create hourly trend chart."""
    fig = px.bar(hourly_df, x="hour", y="count", 
                 title="Hourly Distribution of Vehicle Entries",
                 labels={"hour": "Hour of Day", "count": "Number of Vehicles"})
    fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1))
    return fig

def load_data(petrol_pump_id, vehicle_id=None, date_filter=None, use_sample_data=False, state_filter=None):
    """Load data from API or sample data."""
    if use_sample_data:
        # Create sample data for testing
        return create_sample_data()
    
    # Call the API to get data
    data = get_vehicle_details(petrol_pump_id, vehicle_id)
    
    if not data or len(data) == 0:
        st.warning("No data returned from API.")
        return pd.DataFrame()
    
    try:
        # Process the data
        processed_data = []
        for item in data:
            try:
                # Extract duration
                duration = 0
                filling_time = item.get('FillingTime', '')
                if filling_time and isinstance(filling_time, str):
                    try:
                        parts = filling_time.split()
                        if len(parts) > 0:
                            duration = float(parts[0])
                    except:
                        pass
                
                # Create entry
                processed_data.append({
                    'vehicle_id': item.get('VehicleID', 'unknown'),
                    'entry_time': item.get('EnteringTime', ''),
                    'exit_time': item.get('ExitTime', ''),
                    'duration': duration,
                    'in_roi': False,  # Default to False
                    'date': item.get('Date', datetime.now().strftime('%Y-%m-%d')),
                    'vehicle_type': item.get('VehicleType', 'Unknown'),
                    'status': 'Completed' if item.get('ExitTime') else 'Active'
                })
            except Exception as e:
                st.error(f"Error processing item: {str(e)}")
        
        # Convert to DataFrame
        df = pd.DataFrame(processed_data)
        
        # Apply date filter if provided
        if date_filter:
            df = df[df['date'] == date_filter]
            
        # Apply status filter if provided
        if state_filter and state_filter != "All":
            df = df[df['status'] == state_filter]
            
        return df
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return pd.DataFrame()

def create_sample_data(n=50):
    """Create sample data for testing."""
    np.random.seed(42)  # For reproducibility
    
    # Generate random data
    vehicle_types = ["Car", "Motorcycle", "Bus", "Truck"]
    vehicle_ids = [f"V{i:03d}" for i in range(1, n+1)]
    
    today = datetime.now().date()
    dates = [(today - timedelta(days=np.random.randint(0, 7))).strftime('%Y-%m-%d') for _ in range(n)]
    
    # Generate random entry times
    entry_hours = np.random.randint(6, 22, n)  # Between 6 AM and 10 PM
    entry_minutes = np.random.randint(0, 60, n)
    entry_seconds = np.random.randint(0, 60, n)
    entry_times = [f"{h:02d}:{m:02d}:{s:02d}" for h, m, s in zip(entry_hours, entry_minutes, entry_seconds)]
    
    # Generate durations and exit times
    durations = np.random.randint(30, 600, n)  # 30 seconds to 10 minutes
    exit_times = []
    for i in range(n):
        try:
            entry_dt = datetime.strptime(entry_times[i], "%H:%M:%S")
            exit_dt = entry_dt + timedelta(seconds=durations[i])
            exit_times.append(exit_dt.strftime("%H:%M:%S"))
        except:
            exit_times.append("")
    
    # Randomly make some vehicles still active
    in_roi = np.random.choice([True, False], n, p=[0.2, 0.8])
    status = ["Active" if roi else "Completed" for roi in in_roi]
    
    # For active vehicles, clear exit time
    for i in range(n):
        if in_roi[i]:
            exit_times[i] = ""
            durations[i] = 0
    
    # Create DataFrame
    df = pd.DataFrame({
        'vehicle_id': vehicle_ids,
        'entry_time': entry_times,
        'exit_time': exit_times,
        'duration': durations,
        'in_roi': in_roi,
        'date': dates,
        'vehicle_type': np.random.choice(vehicle_types, n),
        'status': status
    })
    
    return df

def main():
    # Create session state for persistence
    if 'data' not in st.session_state:
        st.session_state.data = pd.DataFrame()
    
    # Header
    st.markdown('<div class="header-style">Vehicle Analytics Dashboard</div>', unsafe_allow_html=True)
    
    # Create tabs for different analytics views
    tabs = st.tabs(["📊 Dashboard", "📋 Data Table", "📈 Advanced Analytics", "📤 Export Tools"])
    
    with tabs[0]:  # Dashboard Tab
        # Filter section
        st.markdown('<div class="subheader-style">Filters & Controls</div>', unsafe_allow_html=True)
        
        with st.container():
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                petrol_pump_id = st.text_input("🏪 Petrol Pump ID", value="IOCL-1")
            
            with col2:
                date_filter = st.date_input("📅 Date Filter", 
                                          value=datetime.now().date(),
                                          max_value=datetime.now().date())
                date_filter_str = date_filter.strftime('%Y-%m-%d')
            
            with col3:
                state_filter = st.selectbox("🚦 Status Filter", 
                                          options=["All", "Active", "Completed"],
                                          index=0)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            vehicle_id = st.text_input("🚗 Vehicle ID (Optional)", help="Filter by specific vehicle")
        
        with col2:
            use_sample_data = st.checkbox("📊 Use Sample Data", value=False, 
                                        help="Use generated sample data instead of API call")
        
        # Load data button
        if st.button("🔄 Load Data", help="Fetch data from API or generate sample data"):
            with st.spinner("Loading data..."):
                df = load_data(petrol_pump_id, vehicle_id, date_filter_str, use_sample_data, state_filter)
                st.session_state.data = df
                
                if len(df) > 0:
                    st.success(f"Loaded {len(df)} vehicle records!")
                else:
                    st.warning("No data found for the given filters.")
        
        # Display metrics and charts if data is available
        if not st.session_state.data.empty:
            df = st.session_state.data
            
            # Calculate metrics
            metrics = calculate_metrics(df)
            
            # Display key metrics
            st.markdown('<div class="subheader-style">Key Metrics</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metrics['total_vehicles']}</div>
                    <div class="metric-label">Total Vehicles</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_time = format_duration(metrics['average_filling_time'])
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{avg_time}</div>
                    <div class="metric-label">Average Filling Time</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metrics['active_vehicles']}</div>
                    <div class="metric-label">Active Vehicles</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                completed = metrics['total_vehicles'] - metrics['active_vehicles']
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{completed}</div>
                    <div class="metric-label">Completed Vehicles</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Create charts
            st.markdown('<div class="subheader-style">Analytics Charts</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Vehicle type distribution
                vehicle_type_chart = create_vehicle_type_chart(metrics['vehicle_types'])
                if vehicle_type_chart:
                    st.plotly_chart(vehicle_type_chart, use_container_width=True)
                else:
                    st.info("No vehicle type data available.")
            
            with col2:
                # Hourly distribution
                hourly_df = create_hourly_distribution(df)
                hourly_chart = create_hourly_trend_chart(hourly_df)
                st.plotly_chart(hourly_chart, use_container_width=True)
            
            # Duration histogram
            duration_hist = create_duration_histogram(df)
            if duration_hist:
                st.plotly_chart(duration_hist, use_container_width=True)
            else:
                st.info("No duration data available to create histogram.")
            
    
    with tabs[1]:  # Data Table Tab
        st.markdown('<div class="subheader-style">Vehicle Data Table</div>', unsafe_allow_html=True)
        
        # Search and filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_term = st.text_input("🔍 Search", help="Search in any column")
        
        with col2:
            sort_by = st.selectbox("🔄 Sort By", 
                                 options=["entry_time", "exit_time", "duration", "vehicle_type"],
                                 index=0)
        
        with col3:
            sort_order = st.radio("Order", ["Ascending", "Descending"], horizontal=True)
        
        # Apply filters to data
        if not st.session_state.data.empty:
            df = st.session_state.data.copy()
            
            # Apply search
            if search_term:
                mask = np.column_stack([df[col].astype(str).str.contains(search_term, case=False, na=False) 
                                     for col in df.columns])
                df = df[mask.any(axis=1)]
            
            # Apply sorting
            if sort_by in df.columns:
                df = df.sort_values(by=sort_by, ascending=(sort_order == "Ascending"))
            
            # Format duration for display
            if 'duration' in df.columns:
                df['formatted_duration'] = df['duration'].apply(format_duration)
            
            # Show data table
            st.dataframe(
                df,
                column_config={
                    "vehicle_id": "Vehicle ID",
                    "entry_time": "Entry Time",
                    "exit_time": "Exit Time",
                    "formatted_duration": st.column_config.TextColumn("Duration"),
                    "vehicle_type": "Vehicle Type",
                    "status": st.column_config.TextColumn(
                        "Status",
                        help="Current status of the vehicle",
                        width="medium",
                    ),
                    "date": "Date",
                    "in_roi": "In Area"
                },
                use_container_width=True,
                hide_index=True,
                height=400
            )
            
            # Show summary stats
            st.markdown('<div class="subheader-style">Summary Statistics</div>', unsafe_allow_html=True)
            
            if 'duration' in df.columns and len(df) > 0:
                duration_stats = df['duration'].describe()
                
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                
                with stats_col1:
                    st.metric("Count", f"{int(duration_stats['count'])}")
                
                with stats_col2:
                    st.metric("Mean Duration", format_duration(duration_stats['mean']))
                
                with stats_col3:
                    st.metric("Min Duration", format_duration(duration_stats['min']))
                
                with stats_col4:
                    st.metric("Max Duration", format_duration(duration_stats['max']))
                
                # Vehicle type counts
                if 'vehicle_type' in df.columns:
                    st.markdown("#### Vehicle Type Counts")
                    type_counts = df['vehicle_type'].value_counts().reset_index()
                    type_counts.columns = ['Vehicle Type', 'Count']
                    st.dataframe(type_counts, use_container_width=True, hide_index=True)
            else:
                st.info("No numeric data available for statistics.")
        else:
            st.info("No data loaded. Use the Load Data button in the Dashboard tab.")
    
    with tabs[2]:  # Advanced Analytics Tab
        st.markdown('<div class="subheader-style">Advanced Analytics</div>', unsafe_allow_html=True)
        
        if not st.session_state.data.empty:
            df = st.session_state.data
            
            # Time-based analysis
            st.markdown("### Time-Based Analysis")
            
            # Add date column if it doesn't exist
            if 'date' not in df.columns:
                df['date'] = datetime.now().strftime('%Y-%m-%d')
            
            # Group by date if multiple dates
            if len(df['date'].unique()) > 1:
                daily_counts = df.groupby('date').size().reset_index(name='count')
                
                # Plot daily counts
                fig = px.line(daily_counts, x='date', y='count', 
                             title='Daily Vehicle Count',
                             labels={'date': 'Date', 'count': 'Number of Vehicles'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Peak hour analysis
            if 'entry_time' in df.columns:
                st.markdown("### Peak Hour Analysis")
                
                # Create hourly distribution
                hourly_df = create_hourly_distribution(df)
                
                # Find peak hours
                peak_hour = hourly_df.loc[hourly_df['count'].idxmax()]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Peak Hour", f"{int(peak_hour['hour']):02d}:00", f"{int(peak_hour['count'])} vehicles")
                
                with col2:
                    # Calculate morning vs afternoon split
                    morning = hourly_df[hourly_df['hour'] < 12]['count'].sum()
                    afternoon = hourly_df[hourly_df['hour'] >= 12]['count'].sum()
                    st.metric("Morning vs Afternoon", f"{morning} : {afternoon}")
                
                # Create heatmap for weekday/hour if multiple dates
                if len(df['date'].unique()) > 1:
                    try:
                        df['weekday'] = pd.to_datetime(df['date']).dt.day_name()
                        df['hour'] = df['entry_time'].apply(
                            lambda x: parse_time(x).hour if parse_time(x) else 0
                        )
                        
                        # Create pivot table
                        heatmap_data = df.pivot_table(
                            index='weekday', 
                            columns='hour',
                            values='vehicle_id',
                            aggfunc='count',
                            fill_value=0
                        )
                        
                        # Create heatmap
                        fig = px.imshow(heatmap_data, 
                                       labels=dict(x="Hour of Day", y="Day of Week", color="Vehicle Count"),
                                       title="Vehicle Traffic Heatmap",
                                       color_continuous_scale='YlGnBu')
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not create weekday/hour heatmap: {str(e)}")
            
            # Vehicle type analysis
            if 'vehicle_type' in df.columns and 'duration' in df.columns:
                st.markdown("### Vehicle Type Analysis")
                
                # Group by vehicle type
                type_stats = df.groupby('vehicle_type')['duration'].agg(['mean', 'count']).reset_index()
                
                # Create bar chart
                fig = px.bar(type_stats, x='vehicle_type', y='mean', 
                            title='Average Duration by Vehicle Type',
                            labels={'vehicle_type': 'Vehicle Type', 'mean': 'Average Duration (seconds)'},
                            text='count',
                            color='vehicle_type')
                fig.update_traces(texttemplate='%{text} vehicles', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data loaded. Use the Load Data button in the Dashboard tab.")
    
    with tabs[3]:  # Export Tools Tab
        st.markdown('<div class="subheader-style">Export Tools</div>', unsafe_allow_html=True)
        
        if not st.session_state.data.empty:
            df = st.session_state.data
            
            # Export options
            st.markdown("### Export Options")
            
            export_format = st.radio("Select Export Format", 
                                   ["CSV", "Excel", "JSON"], 
                                   horizontal=True)
            
            # Columns to include
            available_columns = df.columns.tolist()
            selected_columns = st.multiselect(
                "Select Columns to Include",
                available_columns,
                default=available_columns
            )
            
            # Filter data
            if not selected_columns:  # If nothing selected, use all columns
                selected_columns = available_columns
            
            export_df = df[selected_columns].copy()
            
            # Show preview
            st.markdown("### Export Preview")
            st.dataframe(export_df.head(5), use_container_width=True)
            
            # Generate export
            filename_base = f"vehicle_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if export_format == "CSV":
                csv = export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"{filename_base}.csv",
                    mime='text/csv'
                )
            
            elif export_format == "Excel":
                buffer = io.BytesIO()
                export_df.to_excel(buffer, index=False)
                buffer.seek(0)
                st.download_button(
                    label="📥 Download Excel",
                    data=buffer,
                    file_name=f"{filename_base}.xlsx",
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
            
            elif export_format == "JSON":
                json_str = export_df.to_json(orient='records', date_format='iso')
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name=f"{filename_base}.json",
                    mime='application/json'
                )
            
            # Report generation
            st.markdown("### Generate Report")
            
            report_type = st.selectbox(
                "Select Report Type",
                ["Basic Summary", "Detailed Analysis", "Daily Performance"]
            )
            
            if st.button("🔍 Generate Report"):
                with st.spinner("Generating report..."):
                    # Calculate metrics
                    metrics = calculate_metrics(df)
                    
                    # Create report DataFrame
                    if report_type == "Basic Summary":
                        report_data = {
                            "Metric": [
                                "Total Vehicles",
                                "Active Vehicles",
                                "Completed Vehicles",
                                "Average Filling Time (seconds)",
                                "Max Filling Time (seconds)",
                                "Min Filling Time (seconds)"
                            ],
                            "Value": [
                                metrics['total_vehicles'],
                                metrics['active_vehicles'],
                                metrics['total_vehicles'] - metrics['active_vehicles'],
                                round(metrics['average_filling_time'], 2),
                                metrics['max_filling_time'],
                                metrics['min_filling_time']
                            ]
                        }
                        
                        # Add vehicle type counts
                        for vtype, count in metrics['vehicle_types'].items():
                            report_data["Metric"].append(f"Vehicle Type: {vtype}")
                            report_data["Value"].append(count)
                        
                        report_df = pd.DataFrame(report_data)
                    
                    elif report_type == "Detailed Analysis":
                        # Hour-by-hour analysis
                        hourly_df = create_hourly_distribution(df)
                        
                        # Type performance
                        if 'vehicle_type' in df.columns and 'duration' in df.columns:
                            type_stats = df.groupby('vehicle_type')['duration'].agg(['mean', 'count']).reset_index()
                            type_stats.columns = ['Vehicle Type', 'Avg Duration', 'Count']
                            
                            # Create the report using both dataframes
                            report_df = pd.concat([
                                pd.DataFrame({
                                    "Report Section": ["Hour Analysis"] * len(hourly_df),
                                    "Hour": hourly_df['hour'],
                                    "Vehicle Count": hourly_df['count']
                                }),
                                pd.DataFrame({
                                    "Report Section": ["Vehicle Type Analysis"] * len(type_stats),
                                    "Vehicle Type": type_stats['Vehicle Type'],
                                    "Avg Duration": type_stats['Avg Duration'],
                                    "Count": type_stats['Count']
                                })
                            ])
                        else:
                            report_df = pd.DataFrame({
                                "Report Section": ["Hour Analysis"] * len(hourly_df),
                                "Hour": hourly_df['hour'],
                                "Vehicle Count": hourly_df['count']
                            })
                    
                    elif report_type == "Daily Performance":
                        if 'date' in df.columns:
                            # Group by date
                            daily_counts = df.groupby('date').size().reset_index(name='count')
                            daily_duration = df.groupby('date')['duration'].mean().reset_index()
                            
                            # Merge the dataframes
                            daily_stats = pd.merge(daily_counts, daily_duration, on='date')
                            daily_stats.columns = ['Date', 'Vehicle Count', 'Avg Duration']
                            
                            # Format for report
                            report_df = daily_stats
                        else:
                            st.warning("No date information available for daily report.")
                            report_df = pd.DataFrame()
                    
                    # Display and provide download for the report
                    if not report_df.empty:
                        st.markdown("### Report Preview")
                        st.dataframe(report_df, use_container_width=True)
                        
                        # Export options
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # CSV download
                            csv = report_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Report as CSV",
                                data=csv,
                                file_name=f"report_{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime='text/csv'
                            )
                        
                        with col2:
                            # Excel download
                            buffer = io.BytesIO()
                            report_df.to_excel(buffer, index=False)
                            buffer.seek(0)
                            st.download_button(
                                label="📥 Download Report as Excel",
                                data=buffer,
                                file_name=f"report_{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                            )
                    else:
                        st.warning("Could not generate report with the available data.")
        else:
            st.info("No data loaded. Use the Load Data button in the Dashboard tab.")

# Run the app
if __name__ == "__main__":
    main()