import streamlit as st
import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt

# Load the model
model = joblib.load("har_model.pkl")

activity_labels = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}

# Load test data (same preprocessing as before)
dataset_path = "data/UCI HAR Dataset"
features = pd.read_csv(f"{dataset_path}/features.txt", sep=r'\s+', header=None, names=["index", "feature"])
feature_names = features["feature"].tolist()
deduped_features = [f"{name}_{i}" for i, name in enumerate(feature_names)]
X_test = pd.read_csv(f"{dataset_path}/test/X_test.txt", sep=r'\s+', header=None, names=deduped_features)
y_test = pd.read_csv(f"{dataset_path}/test/y_test.txt", header=None, names=["Activity"])

# Title
st.title("🏃 Real-Time Activity Recognition")
st.markdown("Simulated real-time predictions using UCI HAR dataset and a Random Forest model.")

# Start button
if st.button("Start Simulation"):
    for i in range(len(X_test)):
        sample = X_test.iloc[i].values.reshape(1, -1)
        prediction = model.predict(sample)[0]
        true_label = y_test.iloc[i]['Activity']
        
        pred_label = activity_labels.get(prediction, "Unknown")
        true_label = activity_labels.get(true_label, "Unknown")

        st.write(f"**Time Step {i+1}**")
        st.metric("Predicted Activity", pred_label)
        st.metric("Actual Activity", true_label)

        time.sleep(0.5)
        st.empty()



# Optional: Frequency plot of predictions
st.subheader("📊 Predicted Activity Distribution (so far)")
if 'pred_counts' not in st.session_state:
    st.session_state.pred_counts = {label: 0 for label in activity_labels.values()}

st.session_state.pred_counts[pred_label] += 1
pred_df = pd.DataFrame(list(st.session_state.pred_counts.items()), columns=["Activity", "Count"])

fig, ax = plt.subplots()
pred_df.plot(kind='bar', x='Activity', y='Count', ax=ax, legend=False)
st.pyplot(fig)
