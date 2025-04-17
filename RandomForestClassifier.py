import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# path to the dataset
dataset_path = "data/UCI HAR Dataset"

# Load feature names
features = pd.read_csv(f"{dataset_path}/features.txt", sep=r'\s+', header=None, names=["index", "feature"])
feature_names = features["feature"].tolist()

# Load train/test data
X_train = pd.read_csv(f"{dataset_path}/train/X_train.txt", delim_whitespace=True, header=None)
y_train = pd.read_csv(f"{dataset_path}/train/y_train.txt", header=None, names=["Activity"])
X_test = pd.read_csv(f"{dataset_path}/test/X_test.txt", delim_whitespace=True, header=None)
y_test = pd.read_csv(f"{dataset_path}/test/y_test.txt", header=None, names=["Activity"])

# Train a classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train.values.ravel())

# Predict and evaluate
y_pred = clf.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(clf, "har_model.pkl")
print("📦 Model saved to har_model.pkl")
