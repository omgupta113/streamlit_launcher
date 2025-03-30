import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from api_client import get_vehicle_details
import time
import os

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
        font-size: 25px;
        font-weight: bold;
        color: #2E86C1;
        padding: 10px;
        border-bottom: 2px solid #2E86C1;
    }
    .metric-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #F8F9F9;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .dataframe th {
        background-color: #2E86C1 !important;
        color: white !important;
    }
    .stButton button {
        background-color: #2E86C1 !important;
        color: white !important;
        border-radius: 5px;
    }
    .source-selection {
        padding: 15px;
        border: 1px solid #ddd;
        border-radius: 10px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Initialize session state
    if 'tracked_vehicles' not in st.session_state:
        st.session_state.tracked_vehicles = {}
    
    st.markdown('<p class="header-style">Vehicle Analytics Dashboard</p>', unsafe_allow_html=True)
    
    # Add petrol pump ID input
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        petrol_pump_id = st.text_input("🔧 Enter Petrol Pump ID", value="IOCL-1", help="Enter the Petrol Pump ID to fetch details")
    
    with col_filter2:
        vehicle_id = st.text_input("🚗 Enter Vehicle ID (Optional)", help="Filter by specific vehicle if needed")

    # Add refresh button
    refresh_data = st.button("🔄 Refresh Data", help="Fetch latest vehicle details from server")

    # Option to use sample data
    use_sample_data = st.checkbox("Use Sample Data", value=False, help="Use generated sample data instead of API call")

    if refresh_data:
        if use_sample_data:
            # Generate sample data
            with st.spinner("Generating sample data..."):
                sample_data = generate_sample_data()
                st.session_state.tracked_vehicles = sample_data
                st.success("✅ Sample data generated successfully!")
        else:
            try:
                # Show loading indicator
                with st.spinner("Fetching data from API..."):
                    # Fetch data from API
                    api_data = get_vehicle_details(petrol_pump_id, vehicle_id if vehicle_id else None)
                    
                    # Process API response
                    if api_data and isinstance(api_data, list) and len(api_data) > 0:
                        # Convert API data to tracked_vehicles format
                        processed_data = {}
                        
                        for item in api_data:
                            try:
                                # Extract filling time duration safely
                                duration = 0
                                filling_time_str = item.get('FillingTime', '')
                                if filling_time_str and isinstance(filling_time_str, str):
                                    try:
                                        # Extract number from string like "30 seconds"
                                        duration_parts = filling_time_str.split()
                                        if len(duration_parts) > 0:
                                            duration = float(duration_parts[0])
                                    except (ValueError, IndexError) as e:
                                        st.warning(f"Could not parse filling time '{filling_time_str}'")
                                
                                # Get vehicle ID safely
                                vehicle_id = item.get('VehicleID', f"unknown-{len(processed_data)}")
                                
                                # Get entry and exit times
                                entry_time = item.get('EnteringTime', '')
                                exit_time = item.get('ExitTime', '')
                                
                                # Add to processed data
                                processed_data[vehicle_id] = {
                                    'vehicle_id': vehicle_id,
                                    'entry_time': entry_time,
                                    'exit_time': exit_time,
                                    'duration': duration,
                                    'in_roi': False,  # Default to False
                                    'last_seen': exit_time if exit_time else entry_time,
                                    'vehicle_type': item.get('VehicleType', 'Unknown')
                                }
                            except Exception as e:
                                st.error(f"Error processing vehicle: {str(e)}")
                        
                        st.session_state.tracked_vehicles = processed_data
                        st.success("✅ Data updated successfully!")
                    else:
                        st.warning("⚠️ No data found for this Petrol Pump ID")
            
            except Exception as e:
                st.error(f"🔴 Error fetching data: {str(e)}")

    # Display tracked vehicle data
    if st.session_state.tracked_vehicles:
        col1, col2, col3 = st.columns(3)
        
        # Current Vehicles in ROI
        with col1:
            current_vehicles = len([v for v in st.session_state.tracked_vehicles.values() 
                                if v.get('in_roi', False) and not v.get('exit_time')])
            st.markdown(f"""
            <div class="metric-card">
                <h3>Active Vehicles</h3>
                <p style="font-size: 24px; margin: 0;">{current_vehicles}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Total Vehicles
        with col2:
            total_vehicles = len(st.session_state.tracked_vehicles)
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Vehicles</h3>
                <p style="font-size: 24px; margin: 0;">{total_vehicles}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Average Filling Time
        with col3:
            valid_times = [v.get('duration', 0) for v in st.session_state.tracked_vehicles.values() 
                        if v.get('duration', 0) > 0]
            avg_time = np.mean(valid_times) if valid_times else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>Avg. Filling Time</h3>
                <p style="font-size: 24px; margin: 0;">{avg_time:.1f} secs</p>
            </div>
            """, unsafe_allow_html=True)

        # Enhanced Data Table
        st.subheader("Detailed Vehicle Logs")
        try:
            df = pd.DataFrame.from_dict(st.session_state.tracked_vehicles, orient='index')
            
            # Add status indicator column
            df['status'] = np.where(
                df['exit_time'].isnull() | (df['exit_time'] == ''), 
                'Active 🟢', 
                'Completed 🔴'
            )
            
            # Convert duration from seconds to a readable format
            df['duration_str'] = df['duration'].apply(
                lambda x: f"{int(x // 60)}m {int(x % 60)}s" if not pd.isna(x) and x > 0 else ""
            )
            
            # Display enhanced table
            st.dataframe(
                df[['vehicle_id', 'entry_time', 'exit_time', 
                'duration_str', 'status', 'vehicle_type', 'last_seen']],
                column_config={
                    "vehicle_id": "Vehicle ID",
                    "entry_time": "Entry Time",
                    "exit_time": "Exit Time",
                    "duration_str": "Duration",
                    "status": "Status",
                    "vehicle_type": "Vehicle Type",
                    "last_seen": "Last Update"
                },
                use_container_width=True,
                height=500,
                hide_index=True
            )
            
            # Add export button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export to CSV",
                data=csv,
                file_name=f"vehicle_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv'
            )
        except Exception as e:
            st.error(f"Error processing vehicle data: {str(e)}")
    else:
        st.info("ℹ️ No vehicle data available. Use the 'Refresh Data' button to load data.")

def generate_sample_data(num_vehicles=20):
    """Generate sample vehicle data for demonstration purposes."""
    sample_data = {}
    vehicle_types = ["Car", "Motorcycle", "Bus", "Truck"]
    
    current_time = datetime.now()
    
    for i in range(num_vehicles):
        # Generate random entry time (within last 2 hours)
        minutes_ago = np.random.randint(5, 120)
        entry_time = (current_time - pd.Timedelta(minutes=minutes_ago)).strftime("%H:%M:%S")
        
        # For 70% of vehicles, generate exit time
        has_exit = np.random.random() < 0.7
        
        # Generate random duration for completed vehicles
        duration = 0
        exit_time = ""
        if has_exit:
            duration = np.random.randint(30, 300)  # 30 to 300 seconds
            exit_dt = (current_time - pd.Timedelta(minutes=minutes_ago) + 
                      pd.Timedelta(seconds=duration))
            exit_time = exit_dt.strftime("%H:%M:%S")
        
        # Generate random vehicle type
        vehicle_type = np.random.choice(vehicle_types)
        
        # Create vehicle entry
        vehicle_id = f"SAMPLE-{i+1:03d}"
        sample_data[vehicle_id] = {
            'vehicle_id': vehicle_id,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'duration': duration,
            'in_roi': not has_exit,  # In ROI if no exit time
            'last_seen': exit_time if exit_time else entry_time,
            'vehicle_type': vehicle_type
        }
    
    return sample_data

if __name__ == "__main__":
    main()
