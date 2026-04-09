import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load dataset
# -------------------------------
df = pd.read_csv("../cleaned_data.csv.gz", compression='gzip')

# OPTIONAL (faster)
df = df.sample(100000)

print("Dataset shape:", df.shape)

# -------------------------------
# 2. Target Distribution
# -------------------------------
target_column = df.columns[-1]

plt.figure()
df[target_column].value_counts().plot(kind='bar')
plt.title("Target Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.savefig("../Results/target_distribution.png")
plt.show()

# -------------------------------
# 3. Feature Distribution (Histogram)
# -------------------------------
plt.figure()
df.iloc[:, 0:5].hist()
plt.suptitle("Feature Distribution")
plt.savefig("../Results/feature_distribution.png")
plt.show()

# -------------------------------
# 4. Correlation Heatmap (Simple)
# -------------------------------
corr = df.corr()

plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.title("Correlation Heatmap")
plt.savefig("../Results/correlation_heatmap.png")
plt.show()

# -------------------------------
# 5. Boxplot (Outliers)
# -------------------------------
plt.figure()
df.iloc[:, 0:5].plot(kind='box')
plt.title("Outlier Detection (Boxplot)")
plt.savefig("../Results/boxplot.png")
plt.show()

print("✅ EDA graphs saved in Results folder")