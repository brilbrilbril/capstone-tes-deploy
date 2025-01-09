import streamlit as st
import streamlit.components.v1 as components

# Fungsi untuk menampilkan dashboard
def dashboard_function():
    # Title of the app
    st.header("Dashboard Shopping Copilot")

    # Looker Studio embed link
    looker_studio_url = "https://lookerstudio.google.com/embed/reporting/131e08c8-31a9-4c60-9624-10b3c466d1c4/page/8ALcE"

    # Embed the dashboard using an iframe
    components.html(
        f"""
        <iframe width="800" height="600" 
                src="{looker_studio_url}" 
                frameborder="0" 
                style="border:0" 
                allowfullscreen 
                sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox">
        </iframe>
        """,
        height=600,  # This sets the height of the iframe in Streamlit
        width=800
    )