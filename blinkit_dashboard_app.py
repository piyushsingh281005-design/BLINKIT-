import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
from PIL import Image
import warnings
from itertools import combinations
from collections import Counter
import os

warnings.filterwarnings("ignore")

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="Blinkit Analytics Dashboard",
    page_icon="🛒",
    layout="wide"
)

# -------------------------------------------------------
# MODERN PREMIUM CSS UI
# -------------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.big-title {
    background: linear-gradient(to right, #00C853, #69F0AE);
    padding: 25px; border-radius: 12px;
    text-align: center; font-size: 40px !important;
    color: white !important; font-weight: 700;
    letter-spacing: 1px;
}
.section-header {
    font-size: 28px; font-weight: 700;
    margin-top: 15px; color: #d1d1d1 !important;
}
.card {
    padding: 22px; border-radius: 15px;
    background: #111418; border: 1px solid #343a40;
    color: white !important;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.4);
    text-align: center;
}
.metric-title { font-size: 16px; opacity: 0.8; }
.metric-number { font-size: 34px; font-weight: 800; color: #00E676; }
.about-box {
    background: #1b1f24; padding: 20px;
    border-radius: 12px; border: 1px solid #2f3338;
    color: #e6e6e6; line-height: 1.7;
}
.logo-container img { filter: brightness(3) saturate(2); }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# LOAD NEW MODIFIED CSV (MAIN DATASET)
# -------------------------------------------------------
@st.cache_data
def load_data():
    file = "blinkit_old_dataset_ml_ready.csv"
    if not os.path.exists(file):
        st.error("❌ CSV not found. Please place 'blinkit_old_dataset_ml_ready.csv' in the same folder.")
        st.stop()

    df = pd.read_csv(file)

    # ---- Rename to match your old dashboard code structure ----
    df = df.rename(columns={
        "order_date": "Date",
        "city": "City",
        "category": "Category",
        "product_name": "Product_Name",
        "quantity": "Quantity",
        "final_amount": "Sales",
        "customer_rating": "Rating",
        "sentiment_score": "Sentiment",
        "customer_id": "Customer_ID"
    })

    # ---- Date parsing ----
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.to_period("M")
    df["Day"] = df["Date"].dt.date

    return df

df = load_data()

# -------------------------------------------------------
# HEADER WITH LOGO
# -------------------------------------------------------
logo = Image.open("blinkit_logo.png")

col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.markdown("<div class='logo-container'>", unsafe_allow_html=True)
    st.image(logo, width=120)
    st.markdown("</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("<div class='big-title'>Blinkit Analytics Dashboard</div>", unsafe_allow_html=True)

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
st.sidebar.title("🛒 Blinkit Dashboard")

choice = st.sidebar.radio(
    "Navigate",
    [
        "📊 EDA",
        "📈 Forecasting",
        "😊 Sentiment",
        "👥 Segmentation",
        "🛍 Market Basket",
        "🤖 Delivery Time Prediction",
        "ℹ️ About"
    ]
)

# -------------------------------------------------------
# 📊 EDA
# -------------------------------------------------------
if choice == "📊 EDA":
    st.markdown("<div class='section-header'>Exploratory Data Analysis</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"<div class='card'><div class='metric-title'>Total Orders</div><span class='metric-number'>{len(df):,}</span></div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<div class='card'><div class='metric-title'>Total Revenue</div><span class='metric-number'>₹ {df['Sales'].sum():,}</span></div>", unsafe_allow_html=True)

    with col3:
        st.markdown(f"<div class='card'><div class='metric-title'>Avg Order Value</div><span class='metric-number'>₹ {df['Sales'].mean():.0f}</span></div>", unsafe_allow_html=True)

    st.write("---")

    col1, col2 = st.columns(2)

    # -------- MONTHLY SALES LINE CHART --------
    with col1:
        st.markdown("<div class='section-header'>📅 Monthly Sales Trend</div>", unsafe_allow_html=True)
        monthly = df.groupby("Month")["Sales"].sum()
        fig, ax = plt.subplots()
        monthly.plot(kind="line", marker="o", ax=ax, linewidth=2)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # -------- CATEGORY SALES BAR CHART --------
    with col2:
        st.markdown("<div class='section-header'>📦 Sales by Category</div>", unsafe_allow_html=True)
        category_sales = df.groupby("Category")["Sales"].sum()
        fig, ax = plt.subplots()
        category_sales.plot(kind="barh", ax=ax)
        st.pyplot(fig)

    st.write("---")

    # -------- CITY-WISE SALES (NEW) --------
    st.markdown("<div class='section-header'>🏙️ City-wise Sales</div>", unsafe_allow_html=True)
    city_sales = df.groupby("City")["Sales"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10,4))
    city_sales.plot(kind="bar", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# -------------------------------------------------------
# 📈 FORECASTING
# -------------------------------------------------------
elif choice == "📈 Forecasting":
    st.markdown("<div class='section-header'>Sales Forecasting</div>", unsafe_allow_html=True)

    daily = df.groupby("Day")["Sales"].sum()

    months = st.slider("Forecast Months", 1, 12, 3)
    steps = months * 30

    model = ARIMA(daily, order=(1,1,1))
    fitted = model.fit()

    forecast = fitted.get_forecast(steps=steps).predicted_mean

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(daily.index, daily.values, label="History")
    ax.plot(forecast.index, forecast.values, "--", label="Forecast")
    total_forecast_amount = forecast.sum()

    col1, col2 = st.columns(2)
    col1.metric("Model AIC", f"{fitted.aic:.0f}")
    col2.metric(f"Forecasted Revenue ({months} Months)", f"₹ {total_forecast_amount:,.0f}")

    st.write("---")

    plt.xticks(rotation=45)
    ax.legend()
    st.pyplot(fig)

# -------------------------------------------------------
# 😊 SENTIMENT
# -------------------------------------------------------
elif choice == "😊 Sentiment":
    st.markdown("<div class='section-header'>Customer Sentiment</div>", unsafe_allow_html=True)

    sentiment_counts = df["Sentiment"].value_counts()

    col1, col2, col3 = st.columns(3)
    col1.metric("Positive", sentiment_counts.get(1, 0))
    col2.metric("Neutral", sentiment_counts.get(0, 0))
    col3.metric("Negative", sentiment_counts.get(-1, 0))

    fig, ax = plt.subplots()
    ax.pie(sentiment_counts.values, labels=["Positive","Neutral","Negative"], autopct="%1.1f%%")
    st.pyplot(fig)

# -------------------------------------------------------
# 👥 SEGMENTATION
# -------------------------------------------------------
elif choice == "👥 Segmentation":
    st.markdown("<div class='section-header'>Customer Segmentation</div>", unsafe_allow_html=True)

    rfm = df.groupby("Customer_ID").agg({
        "Date": "max",
        "Sales": "sum",
        "Rating": "mean",
        "Customer_ID": "count"
    })

    rfm.columns = ["LastPurchase", "Monetary", "AvgRating", "Frequency"]

    ref_date = df["Date"].max()
    rfm["Recency"] = (ref_date - rfm["LastPurchase"]).dt.days

    X = rfm[["Recency","Frequency","Monetary"]].values
    X_norm = (X - X.mean()) / X.std()

    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm["Segment"] = kmeans.fit_predict(X_norm)

    st.bar_chart(rfm["Segment"].value_counts())

# -------------------------------------------------------
# 🛍 MARKET BASKET
# -------------------------------------------------------
elif choice == "🛍 Market Basket":
    st.markdown("<div class='section-header'>Market Basket Analysis</div>", unsafe_allow_html=True)

    orders = df.groupby(["Customer_ID","Date"])["Product_Name"].apply(list)

    pair_counts = Counter()
    for items in orders:
        for pair in combinations(set(items), 2):
            pair_counts[tuple(sorted(pair))] += 1

    pairs = pair_counts.most_common(10)

    if pairs:
        df_pairs = pd.DataFrame(pairs, columns=["Pair","Count"])
        df_pairs["Pair"] = df_pairs["Pair"].apply(lambda x: " & ".join(x))
        st.dataframe(df_pairs)
        st.bar_chart(df_pairs.set_index("Pair")["Count"])
    else:
        st.warning("No product pairs found.")

# -------------------------------------------------------
# 🤖 DELIVERY TIME PREDICTION
# -------------------------------------------------------
elif choice == "🤖 Delivery Time Prediction":
    st.markdown("<div class='section-header'>Delivery Time Prediction (ML)</div>", unsafe_allow_html=True)

    df_ml = pd.read_csv("blinkit_old_dataset_ml_ready.csv")

    df_ml = df_ml.dropna()

    X = df_ml[["quantity", "price_per_unit", "order_hour", "customer_age"]]
    y = df_ml["delivery_time_minutes"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    st.markdown(f"<div class='card'><div class='metric-title'>Model R² Score</div><span class='metric-number'>{score:.2f}</span></div>", unsafe_allow_html=True)

    st.write("---")

    st.subheader("Feature Importance")
    imp = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    st.bar_chart(imp.set_index("Feature"))

    st.write("---")
    st.subheader("Predict Delivery Time")

    col1, col2, col3, col4 = st.columns(4)
    with col1: q = st.number_input("Quantity", 1, 20, 2)
    with col2: p = st.number_input("Price per Unit", 5.0, 2000.0, 120.0)
    with col3: h = st.slider("Order Hour", 0, 23, 14)
    with col4: age = st.number_input("Customer Age", 18, 70, 30)

    if st.button("Predict"):
        pred = model.predict([[q,p,h,age]])[0]
        st.success(f"🚚 Estimated Delivery Time: {pred:.1f} minutes")

# -------------------------------------------------------
# ℹ️ ABOUT
# -------------------------------------------------------
elif choice == "ℹ️ About":
    st.markdown("<div class='section-header'>About This Project</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='about-box'>
    <h3>📌 Blinkit Analytics Dashboard</h3>
    <p>This dashboard analyzes sales, sentiment, customer behavior, and delivery performance using a realistic Blinkit dataset.</p>
    <ul>
        <li>EDA</li>
        <li>Forecasting</li>
        <li>Sentiment</li>
        <li>RFM Segmentation</li>
        <li>Market Basket</li>
        <li>ML Delivery Time Prediction</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.write("---")
st.caption("Dashboard powered by Blinkit dataset (modified CSV)")
