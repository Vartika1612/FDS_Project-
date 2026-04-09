import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------------------
# 1. Load dataset
# -------------------------------
df = pd.read_csv("../Data/pca_data.csv")

# OPTIONAL (for speed)
df = df.sample(200000)

print("Dataset shape:", df.shape)

# -------------------------------
# 2. Split features & target
# -------------------------------
target_column = df.columns[-1]

X = df.drop(columns=[target_column])
y = df[target_column]

# -------------------------------
# 3. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 4. Initialize Models
# -------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=10),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# -------------------------------
# 5. Train + Evaluate + Store Results
# -------------------------------
results_list = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"{name} Accuracy: {acc}")

    results_list.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1 Score": f1_score(y_test, y_pred, average='weighted')
    })

# -------------------------------
# 6. Final Comparison
# -------------------------------
print("\n--- Final Comparison ---")
for r in results_list:
    print(f"{r['Model']}: {r['Accuracy']}")

# -------------------------------
# 7. Save results (FIXED PATH)
# -------------------------------
output_path = os.path.join("..", "Results")
os.makedirs(output_path, exist_ok=True)

file_path = os.path.join(output_path, "all_model_results.csv")
results_df = pd.DataFrame(results_list)
results_df.to_csv(file_path, index=False)

print(f"\n✅ Results saved at: {file_path}")