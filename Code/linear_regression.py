import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# 1. Load PCA dataset
# -------------------------------
df = pd.read_csv("pca_data.csv")

print("Dataset shape:", df.shape)

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
# 4. Train Linear Regression
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# 5. Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 6. Evaluation
# -------------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Linear Regression Results ---")
print("MSE:", mse)
print("R2 Score:", r2)