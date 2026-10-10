
import streamlit as st
import requests
import feedparser
from bs4 import BeautifulSoup


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI News Agent",
    page_icon="📰",
    layout="wide",
)

st.title("📰 AI News Agent")
st.caption(
    "Fetch recent news from multiple sources, "
    "with hosted Gemini AI connectivity."
)


# ==========================================
# CONFIGURATION
# ==========================================

TOP_N_NEWS = 10
GEMINI_MODEL = "gemini-3.8-flash"


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
            "Content-Type": "application/json",
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
                "maxOutputTokens": 300,
            },
        },
        timeout=60,
    )

    response.raise_for_status()
    data = response.json()

    return (
        data["candidates"][0]["content"]["parts"][0]["text"]
        .strip()
    )


# ==========================================
# RSS NEWS SOURCES
# ==========================================

RSS_FEEDS = {
    "BBC News": {
        "url": "https://feeds.bbci.co.uk/news/rss.xml",
        "category": "General",
    },
    "CNN": {
        "url": "http://rss.cnn.com/rss/edition.rss",
        "category": "General",
    },
    "Reuters": {
        "url": "https://feeds.reuters.com/reuters/topNews",
        "category": "General",
    },
    "Al Jazeera": {
        "url": "https://www.aljazeera.com/xml/rss/all.xml",
        "category": "General",
    },
    "The Guardian": {
        "url": "https://www.theguardian.com/world/rss",
        "category": "General",
    },
    "NPR News": {
        "url": "https://feeds.npr.org/1001/rss.xml",
        "category": "General",
    },
    "TechCrunch": {
        "url": "https://techcrunch.com/feed/",
        "category": "Technology",
    },
    "Ars Technica": {
        "url": "https://feeds.arstechnica.com/arstechnica/index",
        "category": "Technology",
    },
    "Wired": {
        "url": "https://www.wired.com/feed/rss",
        "category": "Technology",
    },
    "CNBC": {
        "url": (
            "https://search.cnbc.com/rs/search/combinedcms/"
            "view.xml?partnerId=wrss01&id=100003114"
        ),
        "category": "Business",
    },
    "NASA": {
        "url": "https://www.nasa.gov/rss/dyn/breaking_news.rss",
        "category": "Science",
    },
}


# ==========================================
# FETCH NEWS FROM RSS FEEDS
# ==========================================

def fetch_rss_news():
    """Fetch articles and report the status of each RSS feed."""

    all_articles = []
    feed_status = []

    for source_name, feed_info in RSS_FEEDS.items():
        try:
            response = requests.get(
                feed_info["url"],
                headers={
                    "User-Agent": "AI-News-Agent/1.0"
                },
                timeout=15,
            )
            response.raise_for_status()

            feed = feedparser.parse(response.content)
            entries = feed.entries[:15]
            source_count = 0

            for entry in entries:
                title_html = entry.get("title", "")
                title = BeautifulSoup(
                    title_html,
                    "html.parser",
                ).get_text(" ", strip=True)

                article_url = entry.get("link", "").strip()

                if not title or not article_url:
                    continue

                description_html = (
                    entry.get("summary", "")
                    or entry.get("description", "")
                )

                description = BeautifulSoup(
                    description_html,
                    "html.parser",
                ).get_text(" ", strip=True)

                published = (
                    entry.get("published")
                    or entry.get("updated")
                    or "Date unavailable"
                )

                all_articles.append({
                    "title": title,
                    "url": article_url,
                    "source": source_name,
                    "category": feed_info["category"],
                    "published": published,
                    "summary": description,
                })

                source_count += 1

            feed_status.append({
                "Source": source_name,
                "Articles fetched": source_count,
                "Status": (
                    "Success"
                    if source_count > 0
                    else "No articles returned"
                ),
            })

        except Exception as exc:
            feed_status.append({
                "Source": source_name,
                "Articles fetched": 0,
                "Status": f"Failed: {str(exc)[:100]}",
            })

    # Remove duplicate URLs
    unique_articles = []
    seen_urls = set()

    for article in all_articles:
        normalized_url = article["url"].split("#")[0].rstrip("/")

        if normalized_url not in seen_urls:
            seen_urls.add(normalized_url)
            unique_articles.append(article)

    return unique_articles, feed_status


# ==========================================
# GEMINI CONNECTION TEST
# ==========================================

with st.expander("Test Gemini API connection"):
    st.write(
        "Verify that your hosted Gemini API is configured correctly."
    )

    if st.button("Test Gemini API"):
        try:
            with st.spinner("Connecting to Gemini..."):
                result = gemini_generate(
                    "Reply with exactly: "
                    "Gemini connection successful."
                )

            st.success(result)

        except requests.exceptions.HTTPError as exc:
            status_code = (
                exc.response.status_code
                if exc.response is not None
                else "Unknown"
            )

            st.error(f"Gemini API returned HTTP {status_code}.")

            if exc.response is not None:
                st.code(exc.response.text[:1500])

        except Exception as exc:
            st.error(
                f"{type(exc).__name__}: {exc}"
            )


# ==========================================
# NEWS DASHBOARD
# ==========================================

st.divider()
st.header("🗞️ Latest News")
st.write(
    "Fetch recent articles from your configured RSS sources."
)

if st.button("Fetch Latest News", type="primary"):
    with st.spinner("Fetching news from RSS feeds..."):
        articles, feed_status = fetch_rss_news()

    st.session_state["rss_articles"] = articles
    st.session_state["rss_feed_status"] = feed_status


# ==========================================
# DISPLAY FEED STATUS
# ==========================================

if "rss_feed_status" in st.session_state:
    st.subheader("RSS Feed Status")

    st.dataframe(
        st.session_state["rss_feed_status"],
        use_container_width=True,
        hide_index=True,
    )


# ==========================================
# DISPLAY FETCHED ARTICLES
# ==========================================

if "rss_articles" in st.session_state:
    articles = st.session_state["rss_articles"]

    st.metric(
        "Unique articles fetched",
        len(articles),
    )

    if articles:
        for index, article in enumerate(articles, start=1):
            with st.container(border=True):
                st.subheader(
                    f"{index}. {article['title']}"
                )

                st.caption(
                    f"{article['source']} | "
                    f"{article['category']} | "
                    f"{article['published']}"
                )

                if article["summary"]:
                    st.write(article["summary"][:600])

                st.markdown(
                    f"[Read full article]({article['url']})"
                )

    else:
        st.warning(
            "No articles were fetched. "
            "Check the RSS Feed Status table above."
        )
