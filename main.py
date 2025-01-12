import streamlit as st
from dashboard import dashboard_function  # Pastikan untuk mengimpor fungsi atau komponen dari Dashboard.py
from app import chatbot_function  # Pastikan untuk mengimpor fungsi atau komponen dari app.py yang berhubungan dengan chatbot

def main():
    st.title("Personalized Shopping Copilot")

    # Membuat tombol untuk memilih antara chatbot dan dashboard
    menu = st.sidebar.radio("Select Page", ("Chatbot", "Dashboard"))

    if menu == "Chatbot":
        chatbot_function()  # Menampilkan fungsi chatbot

    elif menu == "Dashboard":
        dashboard_function()  # Menampilkan fungsi dashboard

if __name__ == "__main__":
    main()
