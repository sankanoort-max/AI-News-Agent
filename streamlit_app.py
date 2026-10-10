
import streamlit as st
import requests
import feedparser
import json
import re
import hashlib
import unicodedata
import smtplib
import html

from datetime import datetime, timezone
from dateutil import parser as dateutil_parser
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, urlencode
from dataclasses import dataclass, field, asdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI News Agent",
    page_icon="📰",
    layout="wide"
)

st.title("📰 AI News Agent")
st.caption(
    "Fetch, clean, rank and summarize news using AI."
)


# ==========================================
# CONFIGURATION
# ==========================================

TOP_N_NEWS = 10
GEMINI_MODEL = "gemini-2.5-flash"


# ==========================================
# GEMINI API CONNECTION
# ==========================================

def gemini_generate(prompt: str) -> str:
    """Generate text using the hosted Gemini API."""

    api_key = st.secrets["GEMINI_API_KEY"]

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{GEMINI_MODEL}:generateContent"
    )

    response = requests.post(
        url,
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json"
        },
        json={
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 300
            }
        },
        timeout=60
    )

    response.raise_for_status()
    data = response.json()

    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


# ==========================================
# TEST THE GEMINI CONNECTION
# ==========================================

with st.expander("Test Gemini API connection"):
    st.write(
        "Use this test to verify that your hosted AI API "
        "is configured correctly."
    )

    if st.button("Test Gemini API"):
        try:
            with st.spinner("Connecting to Gemini..."):
                result = gemini_generate(
                    "Reply with exactly: Gemini connection successful."
                )

            st.success(result)

        except Exception:
            st.error(
                "Gemini connection failed. Check your API key, "
                "model availability and API usage limits."
            )
