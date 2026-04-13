import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import PartialDependenceDisplay

import lime
import lime.lime_tabular

st.title("🌞 Solar Power Prediction App with XAI")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(data.head())

    st.write("Columns:", data.columns)
    st.write("Shape:", data.shape)

    # Data cleaning
    data = data.drop_duplicates()
    data = data.fillna(data.mean(numeric_only=True))

    # Features
    features = ['irradiance','module_temperature','ambient_temperature','humidity']
    target = 'dc_power'

    X = data[features]
    y = data[target]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=features)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Metrics
    st.subheader("📊 Model Performance")
    st.write("MAE:", mean_absolute_error(y_test, y_pred))
    st.write("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    st.write("R2 Score:", r2_score(y_test, y_pred))

    # Graph 1
    st.subheader("Actual vs Predicted")
    fig1, ax1 = plt.subplots()
    ax1.plot(y_test.values[:100])
    ax1.plot(y_pred[:100])
    ax1.legend(["Actual","Predicted"])
    st.pyplot(fig1)

    # Feature importance
    st.subheader("Feature Importance")
    importance = model.feature_importances_
    fig3, ax3 = plt.subplots()
    ax3.bar(features, importance)
    st.pyplot(fig3)

    for f, imp in zip(features, importance):
        st.write(f, "impact:", round(imp,3))

    # Partial Dependence
    st.subheader("Partial Dependence Plot")
    fig4, ax4 = plt.subplots()
    PartialDependenceDisplay.from_estimator(
        model, X_test, features=[0], feature_names=features, ax=ax4
    )
    st.pyplot(fig4)

    # LIME
    st.subheader("LIME Explanation")
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=features,
        mode='regression'
    )

    sample = 5
    exp = explainer.explain_instance(
        X_test.values[sample],
        model.predict,
        num_features=4
    )

    st.write(exp.as_list())

    # Heatmap
    st.subheader("Correlation Heatmap")
    fig5, ax5 = plt.subplots()
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax5)
    st.pyplot(fig5)

    

    