import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Water Potability Prediction",
    page_icon="💧",
    layout="wide"
)

model = joblib.load("models/water_potability_model.pkl")
if "history" not in st.session_state:
    st.session_state.history = []
st.title("💧 Water Potability Prediction System")
st.markdown("### AI-based Water Quality Analysis")
st.write(
    "Enter the water quality parameters below to predict "
    "whether the water is potable or not."
)

st.divider()

st.subheader("🔬 Water Quality Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    ph = st.number_input(
        "pH",
        min_value=0.0,
        max_value=14.0,
        value=7.0
    )

    hardness = st.number_input(
        "Hardness",
        min_value=0.0,
        value=200.0
    )

    solids = st.number_input(
        "Solids",
        min_value=0.0,
        value=20000.0
    )

with col2:
    chloramines = st.number_input(
        "Chloramines",
        min_value=0.0,
        value=7.0
    )

    sulfate = st.number_input(
        "Sulfate",
        min_value=0.0,
        value=350.0
    )

    conductivity = st.number_input(
        "Conductivity",
        min_value=0.0,
        value=400.0
    )

with col3:
    organic_carbon = st.number_input(
        "Organic Carbon",
        min_value=0.0,
        value=15.0
    )

    trihalomethanes = st.number_input(
        "Trihalomethanes",
        min_value=0.0,
        value=60.0
    )

    turbidity = st.number_input(
        "Turbidity",
        min_value=0.0,
        value=4.0
    )

st.divider()

if st.button("🔍 Analyze Water Quality", use_container_width=True):

    input_data = pd.DataFrame({
        "ph": [ph],
        "Hardness": [hardness],
        "Solids": [solids],
        "Chloramines": [chloramines],
        "Sulfate": [sulfate],
        "Conductivity": [conductivity],
        "Organic_carbon": [organic_carbon],
        "Trihalomethanes": [trihalomethanes],
        "Turbidity": [turbidity]
    })

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    potable_probability = probabilities[1] * 100
    not_potable_probability = probabilities[0] * 100

    st.divider()
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.success("💧 WATER IS PREDICTED TO BE POTABLE")
    else:
        st.error("⚠️ WATER IS PREDICTED TO BE NOT POTABLE")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "💧 Potable Probability",
            f"{potable_probability:.2f}%"
        )

    with col2:
        st.metric(
            "⚠️ Not Potable Probability",
            f"{not_potable_probability:.2f}%"
        )

    st.progress(int(potable_probability / 100))

    st.caption(
        "Prediction generated using the trained Random Forest model."
    )
    st.session_state.history.append({
    "pH": ph,
    "Hardness": hardness,
    "Solids": solids,
    "Chloramines": chloramines,
    "Sulfate": sulfate,
    "Conductivity": conductivity,
    "Organic Carbon": organic_carbon,
    "Trihalomethanes": trihalomethanes,
    "Turbidity": turbidity,
    "Prediction": "POTABLE" if prediction == 1 else "NOT POTABLE",
    "Potable Probability": f"{potable_probability:.2f}%"
})
    
    st.divider()
    st.subheader("📈 Feature Importance")

    feature_names = [
        "pH",
        "Hardness",
        "Solids",
        "Chloramines",
        "Sulfate",
        "Conductivity",
        "Organic Carbon",
        "Trihalomethanes",
        "Turbidity"
    ]

    importance = pd.DataFrame({
        "Parameter": feature_names,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    st.bar_chart(
        importance.set_index("Parameter")
    )
   

    if all(column in batch_data.columns for column in required_columns):

        batch_input = batch_data[required_columns].copy()

        for column in required_columns:
            batch_input[column] = batch_input[column].fillna(
                batch_input[column].median()
            )

        predictions = model.predict(batch_input)
        probabilities = model.predict_proba(batch_input)

        batch_data["Prediction"] = [
            "POTABLE" if p == 1 else "NOT POTABLE"
            for p in predictions
        ]

        batch_data["Potable Probability"] = (
            probabilities[:, 1] * 100
        ).round(2)

        st.success(
            f"Successfully predicted {len(batch_data)} water samples."
        )

        st.dataframe(
            batch_data,
            use_container_width=True
        )

    else:
        st.error(
            "Invalid CSV. Please make sure all required water quality columns are present."
        )
    st.divider()
st.subheader("📂 Batch Prediction")

st.write("Upload a CSV file containing water quality parameters.")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    batch_data = pd.read_csv(uploaded_file)

    required_columns = [
        "ph",
        "Hardness",
        "Solids",
        "Chloramines",
        "Sulfate",
        "Conductivity",
        "Organic_carbon",
        "Trihalomethanes",
        "Turbidity"
    ]

    if all(column in batch_data.columns for column in required_columns):

        batch_input = batch_data[required_columns].copy()

        for column in required_columns:
            batch_input[column] = batch_input[column].fillna(
                batch_input[column].median()
            )

        predictions = model.predict(batch_input)
        probabilities = model.predict_proba(batch_input)

        batch_data["Prediction"] = [
            "POTABLE" if p == 1 else "NOT POTABLE"
            for p in predictions
        ]

        batch_data["Potable Probability"] = (
            probabilities[:, 1] * 100
        ).round(2)

        st.success(
            f"Successfully predicted {len(batch_data)} water samples."
        )

        st.dataframe(
            batch_data,
            use_container_width=True
        )

        csv_data = batch_data.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Prediction Results",
            data=csv_data,
            file_name="water_prediction_results.csv",
            mime="text/csv"
        )

    else:
        st.error(
            "Invalid CSV. Please make sure all required water quality columns are present."
        )