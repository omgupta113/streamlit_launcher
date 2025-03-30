import streamlit as st
from streamlit_extras.switch_page_button import switch_page

class MultiApp:
    """Framework for combining multiple Streamlit applications."""
    def __init__(self):
        self.apps = []
        
    def add_app(self, title, func, description=""):
        """Add a new application.
        
        Args:
            title: Title of the app
            func: Function to render the app
            description: Description of the app
        """
        self.apps.append({
            "title": title,
            "function": func,
            "description": description
        })

    def run(self):
        """Run the selected application."""
        # Add a sidebar for navigation
        st.sidebar.title('Navigation')
        
        # Add app selection
        app_selection = st.sidebar.radio(
            'Select Application',
            self.apps,
            format_func=lambda app: app['title']
        )
        
        # Display description if available
        if app_selection['description']:
            st.sidebar.info(app_selection['description'])
        
        # Run the selected application
        app_selection['function']()