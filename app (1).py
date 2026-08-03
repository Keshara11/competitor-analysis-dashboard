"""
app.py - Competitor Analysis Dashboard (Streamlit)
-----------------------------------------------------
Self-contained dashboard covering the SAME 15-category SME industry
taxonomy used by the BizBuddyBot chatbot's guided journey, so a business
idea classified in the chatbot (e.g. "office wear" -> "Retail - Fashion &
Apparel") maps to a REAL, matching category here - instead of the old
5-category dataset that had no equivalent.

Deploy this as its own Streamlit Community Cloud app (one file +
requirements.txt is enough - data is generated and cached on first run,
no external files needed).
"""

import random
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------
# Shared taxonomy - MUST stay in sync with agentic/industry_classifier.py
# and market_research/train_model.py in the chatbot backend.
# ---------------------------------------------------------------------------
INDUSTRY_TAXONOMY = [
    "Retail - Fashion & Apparel",
    "Retail - General & Convenience",
    "Food & Beverage",
    "Technology & Software",
    "Healthcare & Wellness",
    "Education & Training",
    "Beauty & Personal Care",
    "Home, Furniture & Decor",
    "Electronics & Gadgets",
    "Agriculture & Food Production",
    "Professional Services",
    "Hospitality & Tourism",
    "Handicrafts & Artisan Goods",
    "Transportation & Logistics",
    "Finance & Fintech",
]

REGIONS = [
    "Colombo", "Kandy", "Galle", "Jaffna", "Gampaha",
    "Kurunegala", "Negombo", "Batticaloa", "Anuradhapura", "Matara"
]

FEATURE_CATEGORIES = ["Product Quality", "Pricing", "Customer Service", "Delivery/Speed", "Store Experience"]
ASPECTS = ["price", "quality", "service", "speed", "packaging", "variety"]
SOURCES = ["Google Reviews", "Facebook", "Instagram", "Daraz Reviews"]

INDUSTRY_SENTIMENT_BASE = {
    "Retail - Fashion & Apparel":       (0.55, 0.25, 0.20),
    "Retail - General & Convenience":   (0.50, 0.30, 0.20),
    "Food & Beverage":                  (0.60, 0.20, 0.20),
    "Technology & Software":            (0.45, 0.30, 0.25),
    "Healthcare & Wellness":            (0.58, 0.27, 0.15),
    "Education & Training":             (0.52, 0.30, 0.18),
    "Beauty & Personal Care":           (0.57, 0.25, 0.18),
    "Home, Furniture & Decor":          (0.48, 0.32, 0.20),
    "Electronics & Gadgets":            (0.42, 0.28, 0.30),
    "Agriculture & Food Production":    (0.50, 0.35, 0.15),
    "Professional Services":            (0.53, 0.30, 0.17),
    "Hospitality & Tourism":            (0.56, 0.24, 0.20),
    "Handicrafts & Artisan Goods":      (0.62, 0.23, 0.15),
    "Transportation & Logistics":       (0.40, 0.30, 0.30),
    "Finance & Fintech":                (0.47, 0.33, 0.20),
}

BRAND_PREFIXES = ["Prime", "Nova", "Urban", "Ceylon", "Lanka", "Bright", "Metro", "Elite", "Swift", "Aura"]
BRAND_SUFFIXES = {
    "Retail - Fashion & Apparel": ["Fashion", "Wear", "Threads", "Styles"],
    "Retail - General & Convenience": ["Mart", "Store", "Shop"],
    "Food & Beverage": ["Foods", "Kitchen", "Eats", "Cafe"],
    "Technology & Software": ["Tech", "Soft", "Labs", "Systems"],
    "Healthcare & Wellness": ["Health", "Care", "Clinic"],
    "Education & Training": ["Academy", "Institute", "Learning"],
    "Beauty & Personal Care": ["Beauty", "Glow", "Salon"],
    "Home, Furniture & Decor": ["Home", "Furnish", "Decor"],
    "Electronics & Gadgets": ["Electronics", "Gadgets", "Digital"],
    "Agriculture & Food Production": ["Farms", "Agro", "Harvest"],
    "Professional Services": ["Consulting", "Advisory", "Partners"],
    "Hospitality & Tourism": ["Hotels", "Tours", "Stays"],
    "Handicrafts & Artisan Goods": ["Crafts", "Artisan", "Handmade"],
    "Transportation & Logistics": ["Logistics", "Express", "Movers"],
    "Finance & Fintech": ["Finance", "Capital", "Pay"],
}


# ---------------------------------------------------------------------------
# Data generation + model training (cached - runs once per app session)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_dataset(reviews_per_industry: int = 150, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)
    rows = []
    review_id = 1

    for industry in INDUSTRY_TAXONOMY:
        pos_p, neu_p, neg_p = INDUSTRY_SENTIMENT_BASE[industry]
        suffixes = BRAND_SUFFIXES[industry]
        brands = [f"{random.choice(BRAND_PREFIXES)} {random.choice(suffixes)}" for _ in range(6)]

        for _ in range(reviews_per_industry):
            sentiment = np.random.choice(["positive", "neutral", "negative"], p=[pos_p, neu_p, neg_p])
            rating = {"positive": random.choice([4, 5]), "neutral": 3, "negative": random.choice([1, 2])}[sentiment]
            sentiment_score = {
                "positive": round(random.uniform(0.4, 1.0), 2),
                "neutral": round(random.uniform(-0.2, 0.2), 2),
                "negative": round(random.uniform(-1.0, -0.4), 2),
            }[sentiment]

            rows.append({
                "review_id": review_id,
                "brand": random.choice(brands),
                "industry": industry,
                "review_text": f"{sentiment.capitalize()} experience regarding {random.choice(ASPECTS)}.",
                "rating": rating,
                "review_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=random.randint(0, 550)),
                "source": random.choice(SOURCES),
                "price_level": random.choice([1, 2, 3]),
                "feature_category": random.choice(FEATURE_CATEGORIES),
                "sentiment_label": sentiment,
                "sentiment_score": sentiment_score,
                "aspect": random.choice(ASPECTS),
                "region": random.choice(REGIONS),
                "market_share_est": round(random.uniform(2, 35), 2),
                "competitor_type": random.choice(["sme", "enterprise"]),
            })
            review_id += 1

    return pd.DataFrame(rows)


@st.cache_resource(show_spinner=False)
def train_sentiment_model(df: pd.DataFrame):
    X, y = df[["industry", "region"]], df["sentiment_label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["industry", "region"])])
    model = Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000))])
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Competitor Analysis | BizBuddyBot", page_icon="📊", layout="wide")

df = generate_dataset()
model, model_accuracy = train_sentiment_model(df)

st.title("📊 Competitor Analysis Dashboard")
st.caption("Part of the BizBuddyBot SME Toolkit - covers the same 15 industry categories used by the chatbot's guided business journey.")

# ── Sidebar filters ──
st.sidebar.header("Filters")
selected_industry = st.sidebar.selectbox("Industry", INDUSTRY_TAXONOMY, index=0)
selected_region = st.sidebar.selectbox("Region (optional)", ["All Regions"] + REGIONS, index=0)

subset = df[df["industry"] == selected_industry]
if selected_region != "All Regions":
    subset = subset[subset["region"] == selected_region]

if subset.empty:
    st.warning("No data for this filter combination. Try a different region.")
    st.stop()

# ── KPI row ──
col1, col2, col3, col4 = st.columns(4)
col1.metric("Average Rating", f"{subset['rating'].mean():.2f} / 5")
col2.metric("Reviews Analyzed", f"{len(subset):,}")
col3.metric("Avg. Market Share", f"{subset['market_share_est'].mean():.1f}%")
top_sentiment = subset["sentiment_label"].value_counts(normalize=True).idxmax()
col4.metric("Dominant Sentiment", top_sentiment.capitalize())

st.divider()

# ── Charts row 1 ──
c1, c2 = st.columns(2)

with c1:
    st.subheader("Sentiment Breakdown")
    sentiment_counts = subset["sentiment_label"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    fig = px.pie(
        sentiment_counts, names="Sentiment", values="Count", hole=0.45,
        color="Sentiment",
        color_discrete_map={"positive": "#22c55e", "neutral": "#eab308", "negative": "#ef4444"},
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Competitor Type")
    comp_counts = subset["competitor_type"].value_counts().reset_index()
    comp_counts.columns = ["Type", "Count"]
    fig2 = px.bar(comp_counts, x="Type", y="Count", color="Type", text="Count")
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

# ── Charts row 2 ──
c3, c4 = st.columns(2)

with c3:
    st.subheader("Rating Distribution")
    fig3 = px.histogram(subset, x="rating", nbins=5, color="sentiment_label",
                         color_discrete_map={"positive": "#22c55e", "neutral": "#eab308", "negative": "#ef4444"})
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    st.subheader("Most-Mentioned Feature Areas")
    feat_counts = subset["feature_category"].value_counts().reset_index()
    feat_counts.columns = ["Feature Area", "Mentions"]
    fig4 = px.bar(feat_counts, x="Mentions", y="Feature Area", orientation="h")
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ── Sentiment predictor tool ──
st.subheader("🔮 Sentiment Predictor")
st.caption("Predict the likely customer sentiment for any industry + region combination.")
pc1, pc2, pc3 = st.columns([2, 2, 1])
pred_industry = pc1.selectbox("Industry ", INDUSTRY_TAXONOMY, key="pred_industry")
pred_region = pc2.selectbox("Region ", REGIONS, key="pred_region")
if pc3.button("Predict", use_container_width=True):
    sample = pd.DataFrame({"industry": [pred_industry], "region": [pred_region]})
    prediction = model.predict(sample)[0]
    emoji = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}[prediction]
    st.success(f"{emoji} Predicted sentiment: **{prediction.capitalize()}**")

with st.expander("About this dashboard"):
    st.write(
        f"This dashboard uses a synthetically generated dataset covering all {len(INDUSTRY_TAXONOMY)} SME "
        "industry categories, matching the taxonomy used by the BizBuddyBot chatbot's guided business "
        f"journey. Sentiment prediction model accuracy on held-out test data: **{model_accuracy*100:.1f}%** "
        "(industry + region alone are inherently weak predictors of any single review's sentiment - the "
        "aggregate charts above are the primary insight, not this single-review predictor)."
    )
