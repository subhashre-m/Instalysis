import streamlit as st
from instaloader_fetcher import fetch_instagram_captions, is_private_account
from model import predict_sentiment
import pandas as pd
import matplotlib.pyplot as plt

# Custom CSS for modern, full redesign
st.markdown("""
    <style>
    /* General Body Styling */
    body {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f4f4f9;
        margin: 0;
        color: #444;
    }
    
    /* Header Section Styling */
    .header {
        background-color: #2a9d8f;
        padding: 40px 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 2.5em;
        font-weight: 600;
        margin-bottom: 30px;
    }
    
    /* Container */
    .main-container {
        max-width: 900px;
        margin: auto;
    }

    /* Section Styling */
    .section {
        background-color: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }

    /* Section Headers */
    .section h3 {
        color: #2a9d8f;
        font-size: 1.5em;
        font-weight: 600;
        margin-bottom: 15px;
    }

    /* Text Input Styling */
    .stTextInput input {
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #ddd;
        font-size: 1em;
        margin-bottom: 10px;
    }

    /* Button Styling */
    .stButton > button {
        background-color: #264653;
        color: #fff;
        font-size: 1em;
        padding: 15px 30px;
        border-radius: 8px;
        border: none;
        font-weight: 500;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #1d3557;
    }

    /* Results Box */
    .results-box {
        text-align: center;
        padding: 25px;
        background-color: #fefae0;
        border: 1px solid #e9c46a;
        border-radius: 10px;
        margin-top: 20px;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 30px;
        font-size: 0.9em;
        color: #aaa;
        padding: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header">
        Instalysis - Sentiment Analysis
    </div>
""", unsafe_allow_html=True)

# Main Container for Centered Layout
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

# Session States for account type and results
if 'account_type' not in st.session_state:
    st.session_state.account_type = None
if 'captions' not in st.session_state:
    st.session_state.captions = None
if 'sentiments' not in st.session_state:
    st.session_state.sentiments = None

# Account Type Section
st.markdown("""
    <div class="section">
        <h3>Analyze Your Instagram Account</h3>
        <p>Select the type of account you'd like to analyze and enter your username.</p>
    </div>
""", unsafe_allow_html=True)

# Account Type Buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Public Account"):
        st.session_state.account_type = "Public"
with col2:
    if st.button("Private Account"):
        st.session_state.account_type = "Private"

# Input Form and Analysis
if st.session_state.account_type == "Public":
    username = st.text_input("Enter Public Username")
    if st.button("Analyze Public Account"):
        try:
            st.session_state.captions = fetch_instagram_captions(username)
            st.session_state.sentiments = predict_sentiment(st.session_state.captions)
        except Exception as e:
            st.error(f"Error: {e}")
elif st.session_state.account_type == "Private":
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Analyze Private Account"):
        try:
            if is_private_account(username, password):
                st.session_state.captions = fetch_instagram_captions(username, password)
                st.session_state.sentiments = predict_sentiment(st.session_state.captions)
            else:
                st.error("Account is not private.")
        except Exception as e:
            st.error(f"Error: {e}")

# Display Results if Available
if st.session_state.sentiments is not None and len(st.session_state.sentiments) > 0:
    sentiments = st.session_state.sentiments
    counts = {
        0: (sentiments == 0).sum(),
        1: (sentiments == 1).sum(),
        2: (sentiments == 2).sum()
    }
    overall_sentiment = (
        "Positive" if counts[1] > counts[0] and counts[1] > counts[2]
        else "Negative" if counts[0] > counts[1] and counts[0] > counts[2]
        else "Neutral"
    )
    
    advice = (
        "Keep up the positivity!" if overall_sentiment == "Positive" else
        "It’s okay to reach out if you're feeling down." if overall_sentiment == "Negative" else
        "Keep reflecting on your feelings!"
    )

    # Results Box
    st.markdown(f"""
    <div class="results-box">
        <h3>Sentiment Analysis Result</h3>
        <h4>Overall Sentiment: {overall_sentiment}</h4>
        <p>{advice}</p>
    </div>
    """, unsafe_allow_html=True)

    # Pie Chart for Sentiment Distribution
    fig, ax = plt.subplots()
    ax.pie(counts.values(), labels=["Negative", "Positive", "Neutral"], autopct='%1.1f%%', colors=['#e76f51', '#2a9d8f', '#f4a261'])
    ax.set_title("Sentiment Distribution")
    st.pyplot(fig)

# Footer
st.markdown("""
    <div class="footer">
        © 2024 Subhu Instalysis | All rights reserved.
    </div>
""", unsafe_allow_html=True)

# Close Main Container
st.markdown("</div>", unsafe_allow_html=True)
