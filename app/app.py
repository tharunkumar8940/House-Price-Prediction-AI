from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "house_prices_clean.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "house_price_model.pkl"
)

RESULTS_FILE = (
    BASE_DIR
    / "outputs"
    / "model_comparison.csv"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="House Price Prediction AI",
    page_icon="🏠",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 2rem;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #0f172a,
            #1e3a8a
        );
        color: white;
        margin-bottom: 2rem;
    }

    .hero h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }

    .hero p {
        font-size: 1.05rem;
        color: #dbeafe;
    }

    .metric-card {
        padding: 1.2rem;
        border-radius: 15px;
        background: white;
        border: 1px solid #e5e7eb;
        box-shadow: 0 3px 12px rgba(
            15,
            23,
            42,
            0.06
        );
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        color: #0f172a;
    }

    .prediction-box {
        padding: 1.5rem;
        border-radius: 18px;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        text-align: center;
        margin-top: 1rem;
    }

    .prediction-price {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1d4ed8;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        DATA_FILE
    )


@st.cache_resource
def load_model():

    return joblib.load(
        MODEL_FILE
    )


df = load_data()
model = load_model()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏠 House Price AI")

st.sidebar.markdown(
    "AI-powered property price prediction"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Executive Dashboard",
        "🔮 Price Prediction",
        "📊 Market Analysis",
        "📈 Price Insights",
        "🤖 Model Performance",
        "ℹ️ About",
    ],
)


# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

if page == "🏠 Executive Dashboard":

    st.markdown(
        """
        <div class="hero">

        <h1>
        🏠 House Price Prediction AI
        </h1>

        <p>
        Machine learning powered property price
        prediction and market intelligence system.
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Properties",
            f"{len(df):,}",
        )

    with col2:
        st.metric(
            "Locations",
            df["location"].nunique(),
        )

    with col3:
        st.metric(
            "Average Price",
            f"₹{df['price'].mean()/100000:.1f} L",
        )

    with col4:
        st.metric(
            "Average Area",
            f"{df['area_sqft'].mean():,.0f} sqft",
        )

    st.markdown(
        '<div class="section-title">'
        'Market Overview'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        location_prices = (
            df.groupby("location")["price"]
            .mean()
            .sort_values(
                ascending=False
            )
            .reset_index()
        )

        location_prices["price_lakh"] = (
            location_prices["price"]
            / 100000
        )

        fig = px.bar(
            location_prices,
            x="location",
            y="price_lakh",
            title="Average House Price by Location",
            labels={
                "price_lakh": "Average Price (₹ Lakh)",
                "location": "Location",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with col2:

        fig = px.scatter(
            df,
            x="area_sqft",
            y="price",
            color="location",
            title="Area vs House Price",
            labels={
                "area_sqft": "Area (sqft)",
                "price": "Price (₹)",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# PRICE PREDICTION
# ============================================================

elif page == "🔮 Price Prediction":

    st.title("🔮 AI House Price Prediction")

    st.write(
        "Enter property details to estimate "
        "the expected house price."
    )

    col1, col2 = st.columns(2)

    with col1:

        location = st.selectbox(
            "Location",
            sorted(
                df["location"].unique()
            ),
        )

        area_sqft = st.number_input(
            "Area (sqft)",
            min_value=300,
            max_value=10000,
            value=1500,
            step=50,
        )

        bedrooms = st.number_input(
            "Bedrooms",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
        )

        bathrooms = st.number_input(
            "Bathrooms",
            min_value=1,
            max_value=8,
            value=2,
            step=1,
        )

        age_years = st.number_input(
            "Property Age (years)",
            min_value=0,
            max_value=100,
            value=5,
            step=1,
        )

    with col2:

        parking_spaces = st.number_input(
            "Parking Spaces",
            min_value=0,
            max_value=10,
            value=1,
            step=1,
        )

        floor = st.number_input(
            "Floor",
            min_value=1,
            max_value=50,
            value=5,
            step=1,
        )

        total_floors = st.number_input(
            "Total Floors",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
        )

        distance_to_city_km = st.number_input(
            "Distance to City (km)",
            min_value=0.0,
            max_value=100.0,
            value=8.0,
            step=0.5,
        )

    if floor > total_floors:

        st.error(
            "Floor cannot be greater than "
            "the total number of floors."
        )

    else:

        if st.button(
            "🔮 Predict House Price",
            type="primary",
            use_container_width=True,
        ):

            input_data = pd.DataFrame(
                {
                    "location": [location],
                    "area_sqft": [area_sqft],
                    "bedrooms": [bedrooms],
                    "bathrooms": [bathrooms],
                    "age_years": [age_years],
                    "parking_spaces": [
                        parking_spaces
                    ],
                    "floor": [floor],
                    "total_floors": [
                        total_floors
                    ],
                    "distance_to_city_km": [
                        distance_to_city_km
                    ],
                }
            )

            prediction = model.predict(
                input_data
            )[0]

            st.markdown(
                f"""
                <div class="prediction-box">

                <div>
                Estimated Property Price
                </div>

                <div class="prediction-price">
                ₹{prediction:,.0f}
                </div>

                <div>
                Approx. ₹{prediction/100000:.2f} Lakh
                </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# MARKET ANALYSIS
# ============================================================

elif page == "📊 Market Analysis":

    st.title("📊 Market Analysis")

    col1, col2 = st.columns(2)

    with col1:

        location_summary = (
            df.groupby("location")
            .agg(
                properties=("house_id", "count"),
                average_price=("price", "mean"),
                average_area=("area_sqft", "mean"),
            )
            .reset_index()
        )

        location_summary[
            "average_price_lakh"
        ] = (
            location_summary[
                "average_price"
            ]
            / 100000
        )

        st.dataframe(
            location_summary[
                [
                    "location",
                    "properties",
                    "average_price_lakh",
                    "average_area",
                ]
            ].round(2),
            use_container_width=True,
        )

    with col2:

        selected_location = st.selectbox(
            "Select Location",
            sorted(
                df["location"].unique()
            ),
        )

        location_data = df[
            df["location"]
            == selected_location
        ]

        st.metric(
            "Average Price",
            f"₹{location_data['price'].mean():,.0f}",
        )

        st.metric(
            "Average Area",
            f"{location_data['area_sqft'].mean():,.0f} sqft",
        )

        st.metric(
            "Average Bedrooms",
            f"{location_data['bedrooms'].mean():.1f}",
        )


# ============================================================
# PRICE INSIGHTS
# ============================================================

elif page == "📈 Price Insights":

    st.title("📈 Price Insights")

    fig = px.histogram(
        df,
        x="price",
        nbins=40,
        title="House Price Distribution",
        labels={
            "price": "Price (₹)"
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.scatter(
            df,
            x="area_sqft",
            y="price",
            color="bedrooms",
            title="Area vs Price",
            labels={
                "area_sqft": "Area (sqft)",
                "price": "Price (₹)",
                "bedrooms": "Bedrooms",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with col2:

        bedroom_prices = (
            df.groupby("bedrooms")["price"]
            .mean()
            .reset_index()
        )

        fig = px.line(
            bedroom_prices,
            x="bedrooms",
            y="price",
            markers=True,
            title="Average Price by Bedrooms",
            labels={
                "bedrooms": "Bedrooms",
                "price": "Average Price (₹)",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "🤖 Model Performance":

    st.title("🤖 Model Performance")

    results = pd.read_csv(
        RESULTS_FILE
    )

    display_results = results.copy()

    display_results["MAE"] = (
        display_results["MAE"]
        .round(0)
    )

    display_results["RMSE"] = (
        display_results["RMSE"]
        .round(0)
    )

    display_results["R2"] = (
        display_results["R2"]
        .round(4)
    )

    st.dataframe(
        display_results,
        use_container_width=True,
    )

    best_model = results.iloc[0]

    st.success(
        f"Best Model: "
        f"{best_model['model']} | "
        f"R² Score: "
        f"{best_model['R2']:.4f}"
    )

    fig = px.bar(
        results,
        x="model",
        y="R2",
        title="Model R² Comparison",
        labels={
            "model": "Model",
            "R2": "R² Score",
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About the Project")

    st.markdown(
        """
        ## 🏠 House Price Prediction AI

        This project uses machine learning to estimate
        residential property prices from property
        characteristics.

        ### Machine Learning

        The system compares:

        - Linear Regression
        - Random Forest
        - Gradient Boosting

        The best-performing model is automatically
        selected for prediction.

        ### Features

        - Location
        - Area
        - Bedrooms
        - Bathrooms
        - Property age
        - Parking spaces
        - Floor
        - Total floors
        - Distance to city

        ### Technology

        Python • Pandas • NumPy • Scikit-learn •
        Plotly • Streamlit • Joblib

        ### Important Note

        The current dataset is synthetic development data
        created for demonstrating the machine-learning
        workflow. It should not be treated as real
        property-market pricing.
        """
    )

    st.info(
        "This application provides estimated prices "
        "for educational and demonstration purposes."
    )