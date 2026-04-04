import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

print("Loading dataset...")

df = pd.read_csv("fertilizer_dataset.csv")

print("Dataset Loaded Successfully")
print(df.head())

# -----------------------------
# Encode ALL categorical columns
# -----------------------------

label_encoders = {}

for col in df.columns:

    if df[col].dtype == "object":

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col])

        label_encoders[col] = le

# -----------------------------
# Define target column
# -----------------------------

target_column = "Recommended_Fertilizer"

X = df.drop(target_column, axis=1)
y = df[target_column]

# -----------------------------
# Train/Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train Model
# -----------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("Model Accuracy:", accuracy)

# -----------------------------
# Save files
# -----------------------------

joblib.dump(model, "fertilizer_model.pkl")
joblib.dump(label_encoders, "encoders.pkl")

print("Model and encoders saved successfully!")
