import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
# 1. Load dataset
# -------------------------------
df = pd.read_csv("pca_data.csv")   # or "svd_data.csv"

print("Dataset shape:", df.shape)

# OPTIONAL (VERY IMPORTANT for speed)
df = df.sample(200000)

# -------------------------------
# 2. Separate features & target
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
# 4. Train KNN
# -------------------------------
model = KNeighborsClassifier(
    n_neighbors=5,     # K value
    metric='euclidean'
)

model.fit(X_train, y_train)

# -------------------------------
# 5. Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 6. Evaluation
# -------------------------------
print("\n--- KNN Results ---")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))