import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD

# -------------------------------
# 1. Load cleaned dataset
# -------------------------------
df = pd.read_csv("cleaned_data.csv.gz", compression='gzip')

print("Dataset shape:", df.shape)

# -------------------------------
# 2. Separate features & target
# -------------------------------
target_column = df.columns[-1]

X = df.drop(columns=[target_column])
y = df[target_column]

print("Feature shape:", X.shape)

# -------------------------------
# 3. FINAL CLEANING (VERY IMPORTANT)
# -------------------------------
# Remove any remaining NaN / inf
X = X.replace([np.inf, -np.inf], np.nan)

# Fill remaining NaN with 0
X = X.fillna(0)

# Check
print("NaN count:", X.isna().sum().sum())

# -------------------------------
# 4. Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 5. PCA
# -------------------------------
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

print("\n--- PCA ---")
print("Shape:", X_pca.shape)
print("Total Variance:", sum(pca.explained_variance_ratio_))

# -------------------------------
# 6. SVD
# -------------------------------
svd = TruncatedSVD(n_components=10)
X_svd = svd.fit_transform(X_scaled)

print("\n--- SVD ---")
print("Shape:", X_svd.shape)
print("Total Variance:", sum(svd.explained_variance_ratio_))

# -------------------------------
# 7. Save results
# -------------------------------
pca_df = pd.DataFrame(X_pca)
pca_df[target_column] = y
pca_df.to_csv("pca_data.csv", index=False)

svd_df = pd.DataFrame(X_svd)
svd_df[target_column] = y
svd_df.to_csv("svd_data.csv", index=False)

print("\n✅ PCA & SVD completed and saved successfully!")