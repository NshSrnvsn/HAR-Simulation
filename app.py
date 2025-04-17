import streamlit as st
import pandas as pd
import joblib
import time

# Load the model
model = joblib.load("har_model.pkl")

# Activity label mapping
activity_labels = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}

safe_activities = ["SITTING", "STANDING", "LAYING"]

# Load dataset
dataset_path = "data/UCI HAR Dataset"
features = pd.read_csv(f"{dataset_path}/features.txt", sep=r'\s+', header=None, names=["index", "feature"])
feature_names = features["feature"].tolist()
deduped_features = [f"{name}_{i}" for i, name in enumerate(feature_names)]
X_test = pd.read_csv(f"{dataset_path}/test/X_test.txt", sep=r'\s+', header=None, names=deduped_features)
y_test = pd.read_csv(f"{dataset_path}/test/y_test.txt", header=None, names=["Activity"])

# Create a balanced dataset
sample_df = y_test.copy()
sample_df["idx"] = sample_df.index
balanced_idxs = sample_df.groupby("Activity").sample(n=20, random_state=42)["idx"].tolist()
X_balanced = X_test.loc[balanced_idxs].reset_index(drop=True)
y_balanced = y_test.loc[balanced_idxs].reset_index(drop=True)

# UI
st.title("\U0001F3C3 Real-Time Activity Recognition")
st.markdown("Simulating real-time predictions using the UCI HAR dataset and a Random Forest model.")

# Initialize session state
if 'prediction_log' not in st.session_state:
    st.session_state.prediction_log = []
if 'chart_df' not in st.session_state:
    st.session_state.chart_df = pd.DataFrame(columns=["Activity Code"])

# Start simulation
if st.button("Start Simulation"):
    activity_order = list(activity_labels.values())
    label_map = {label: i for i, label in enumerate(activity_order)}

    chart_placeholder = st.empty()
    log_placeholder = st.empty()

    for i in range(len(X_balanced)):
        sample = X_balanced.iloc[i].values.reshape(1, -1)
        prediction = model.predict(sample)[0]
        actual = y_balanced.iloc[i]['Activity']

        pred_label = activity_labels.get(prediction, "Unknown")
        true_label = activity_labels.get(actual, "Unknown")

        risk_status = "✅ Safe" if pred_label in safe_activities else "⚠️ Potential Risk"

        # Log the prediction
        st.session_state.prediction_log.append({
            "Time Step": i + 1,
            "Predicted": pred_label,
            "Actual": true_label,
            "Risk": risk_status
        })

        # Update line chart
        activity_code = label_map[pred_label]
        st.session_state.chart_df.loc[i] = [activity_code]
        chart_placeholder.line_chart(st.session_state.chart_df)

        # Display recent log entries
        log_lines = [
            f"**Step {row['Time Step']}** | Predicted: `{row['Predicted']}` | Actual: `{row['Actual']}` | Risk: {row['Risk']}"
            for row in st.session_state.prediction_log[-10:]
        ]
        log_placeholder.markdown("\n".join(log_lines), unsafe_allow_html=True)

        time.sleep(0.5)

# Export log
if st.session_state.prediction_log:
    df_log = pd.DataFrame(st.session_state.prediction_log)
    st.download_button(
        label="📁 Download Prediction Log as CSV",
        data=df_log.to_csv(index=False).encode('utf-8'),
        file_name='har_prediction_log.csv',
        mime='text/csv'
    )
