import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Student Score Prediction",
    page_icon="🎓",
    layout="wide",
)

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("student_performance.csv")

df = load_data()

X = df[["hours"]]
y = df["final_score"]

# -----------------------------
# Train Model
# -----------------------------
model = LinearRegression()
model.fit(X, y)

prediction = model.predict(X)

mae = mean_absolute_error(y, prediction)
mse = mean_squared_error(y, prediction)
rmse = np.sqrt(mse)
r2 = r2_score(y, prediction)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🎓 Student Score Prediction")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dataset",
        "📈 Visualization",
        "🎯 Prediction",
        "📉 Model Performance",
    ],
)

# -----------------------------
# HOME
# -----------------------------
if page == "🏠 Home":

    st.title("🎓 Student Score Prediction App")

    st.write(
        """
Predict a student's final score based on study hours
using **Linear Regression**.

This project demonstrates a complete Machine Learning
workflow using Python, Scikit-Learn and Streamlit.
"""
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Dataset Size", len(df))
    c2.metric("Features", X.shape[1])
    c3.metric("Model", "Linear Regression")

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(df)

# -----------------------------
# DATASET
# -----------------------------
elif page == "📊 Dataset":

    st.title("📊 Dataset")

    st.dataframe(df, use_container_width=True)

    st.subheader("Dataset Information")

    info = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values,
        }
    )

    st.table(info)

    st.subheader("Statistical Summary")

    st.dataframe(df.describe())

# -----------------------------
# VISUALIZATION
# -----------------------------
elif page == "📈 Visualization":

    st.title("📈 Data Visualization")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.scatter(
        X,
        y,
        color="royalblue",
        label="Actual Data"
    )

    ax.plot(
        X,
        prediction,
        color="red",
        linewidth=3,
        label="Regression Line"
    )

    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Final Score")
    ax.set_title("Study Hours vs Final Score")

    ax.legend()

    st.pyplot(fig)

    st.subheader("Correlation")

    st.write(df.corr(numeric_only=True))

# -----------------------------
# PREDICTION
# -----------------------------
elif page == "🎯 Prediction":

    st.title("🎯 Predict Student Score")

    st.write("Move the slider to select study hours.")

    hours = st.slider(
        "Study Hours",
        min_value=0.0,
        max_value=12.0,
        value=5.0,
        step=0.5,
    )

    if st.button("Predict Score"):

        score = model.predict([[hours]])[0]

        st.success(f"🎉 Predicted Score: {score:.2f}")

        if score >= 90:
            st.balloons()
            st.success("Excellent Performance! 🌟")

        elif score >= 75:
            st.info("Great Job! 👍")

        elif score >= 60:
            st.warning("Good, but there is room for improvement.")

        else:
            st.error("Increase your study hours for better performance.")

# -----------------------------
# MODEL PERFORMANCE
# -----------------------------
elif page == "📉 Model Performance":

    st.title("📉 Model Evaluation")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("MAE", f"{mae:.2f}")
        st.metric("RMSE", f"{rmse:.2f}")

    with col2:
        st.metric("MSE", f"{mse:.2f}")
        st.metric("R² Score", f"{r2:.2f}")

    st.divider()

    st.subheader("Prediction vs Actual")

    result = pd.DataFrame({
        "Study Hours": X["hours"],
        "Actual Score": y,
        "Predicted Score": prediction
    })

    st.dataframe(result, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8,5))

    ax.scatter(
        y,
        prediction,
        color="green"
    )

    ax.plot(
        [y.min(), y.max()],
        [y.min(), y.max()],
        "r--",
        linewidth=2
    )

    ax.set_xlabel("Actual Score")
    ax.set_ylabel("Predicted Score")
    ax.set_title("Actual vs Predicted")

    st.pyplot(fig)

# -----------------------------
# FOOTER
# -----------------------------
st.divider()

st.markdown(
    """
<div style='text-align:center;'>
<h4>🎓 Student Score Prediction App</h4>
<p>Built with ❤️ using Streamlit, Scikit-Learn & Python</p>
</div>
""",
    unsafe_allow_html=True,
)    