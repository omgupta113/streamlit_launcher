import streamlit as st
from multiapp import MultiApp
import polygon_coords
import video_analytics
import importlib

def main():
    # Set page config first (must be the first Streamlit command)
    st.set_page_config(
        page_title="Vehicle Tracking System",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS
    st.markdown("""
    <style>
        .app-header {
            font-size: 28px;
            font-weight: bold;
            color: #2E86C1;
            text-align: center;
            margin-bottom: 20px;
            padding: 20px;
            border-bottom: 2px solid #2E86C1;
        }
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }
        .app-description {
            font-style: italic;
            color: #555;
            margin-bottom: 30px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.markdown('<div class="app-header">Vehicle Tracking System</div>', unsafe_allow_html=True)
    
    # Description
    st.markdown('<div class="app-description">Monitor and analyze vehicle traffic at your petrol station</div>', unsafe_allow_html=True)
    
    # Initialize multi-app
    app = MultiApp()
    
    # Make sure modules are freshly imported for each run
    importlib.reload(polygon_coords)
    importlib.reload(video_analytics)
    
    # Add applications
    app.add_app("Video Monitoring & ROI Selection", 
               polygon_coords.main, 
               "Monitor video feeds and track vehicles in a defined region of interest")
    
    app.add_app("Analytics Dashboard", 
               video_analytics.main, 
               "Visualize and analyze vehicle data with interactive charts and reports")
    
    # Run the selected application
    app.run()

if __name__ == "__main__":
    main()