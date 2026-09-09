import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
# LOAD DATASET
df = pd.read_csv("data/customer_churn.csv")

print("=" * 60)
print("CUSTOMER CHURN PREDICTION")
print("=" * 60)
print("\nFirst 5 rows:")
print(df.head())

# BASIC DATASET INFORMATION
print("\nDataset Shape:")
print(df.shape)
print("\nColumn Names:")
print(df.columns)
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# CHURN DISTRIBUTION
print("\nChurn Distribution:")
print(df["Churn"].value_counts())
print("\nChurn Percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)

# SELECT FEATURES AND TARGET
features = [
    "Age",
    "Total_Purchase",
    "Account_Manager",
    "Years",
    "Num_Sites"
]
X = df[features]
y = df["Churn"]
print("\nSelected Features:")
print(features)
print("\nFeature Data:")
print(X.head())
print("\nTarget Data:")
print(y.head())

# TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print("\nTraining Data Shape:")
print(X_train.shape)
print("\nTesting Data Shape:")
print(X_test.shape)

# FEATURE SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MODEL 1 (LOGISTIC REGRESSION)
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_scaled, y_train)
logistic_pred = logistic_model.predict(X_test_scaled)
logistic_prob = logistic_model.predict_proba(X_test_scaled)[:, 1]

# MODEL 2 (RANDOM FOREST)
random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)
random_forest_model.fit(X_train, y_train)
random_forest_pred = random_forest_model.predict(X_test)
random_forest_prob = random_forest_model.predict_proba(X_test)[:, 1]\
    
# EVALUATION FUNCTION
def evaluate_model(model_name, y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }
    
# EVALUATE BOTH MODELS
logistic_results = evaluate_model(
    "Logistic Regression",
    y_test,
    logistic_pred,
    logistic_prob
)
random_forest_results = evaluate_model(
    "Random Forest",
    y_test,
    random_forest_pred,
    random_forest_prob
)

# MODEL COMPARISON
results = pd.DataFrame([
    logistic_results,
    random_forest_results ])
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(results)

#SAVE RESULTS
results.to_csv("model_comparison.csv", index=False)
print("\nModel comparison saved as: model_comparison.csv")

# FEATURE IMPORTANCE (RANDOM FOREST)
feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": random_forest_model.feature_importances_})
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)
print("\n" + "=" * 60)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)
print(feature_importance)

# FEATURE IMPORTANCE GRAPH
plt.figure(figsize=(8, 5))
plt.bar(
    feature_importance["Feature"],
    feature_importance["Importance"])
plt.title("Random Forest Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# FINAL MODEL
best_model = results.loc[
    results["F1 Score"].idxmax(),
    "Model" ]
print("\n" + "=" * 60)
print("FINAL RESULT")
print("=" * 60)
print(f"Best Model based on F1 Score: {best_model}")
print("\nTask 2 Classification Project Completed Successfully!")