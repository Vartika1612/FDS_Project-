import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# 1. Load datasets (FIXED WARNINGS)
# -------------------------------
df1 = pd.read_csv(
    "UNSW_2018_IoT_Botnet_Dataset_19.csv",
    header=None,
    low_memory=False   # fixes dtype warning
)

df2 = pd.read_csv(
    "UNSW_2018_IoT_Botnet_Dataset_73.csv",
    header=None,
    low_memory=False
)

# -------------------------------
# 2. Merge datasets
# -------------------------------
df = pd.concat([df1, df2], ignore_index=True)
print("Shape after merge:", df.shape)

# -------------------------------
# 3. Assign column names
# -------------------------------
df.columns = [f"col_{i}" for i in range(df.shape[1])]

# -------------------------------
# 4. Replace infinite values
# -------------------------------
df = df.replace([np.inf, -np.inf], np.nan)

# -------------------------------
# 5. Handle Missing Values
# -------------------------------
# Numeric columns
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Categorical columns (FIXED WARNING)
cat_cols = df.select_dtypes(include=['object', 'string']).columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# -------------------------------
# 6. Encode Categorical
# -------------------------------
le = LabelEncoder()

for col in cat_cols:
    df[col] = df[col].astype(str)
    df[col] = le.fit_transform(df[col])

# -------------------------------
# 7. Target Column
# -------------------------------
target_column = df.columns[-1]
print("Using target column:", target_column)

# -------------------------------
# 8. Done
# -------------------------------
print("Final dataset shape:", df.shape)
print("Preprocessing completed successfully ✅")

# -------------------------------
# 9. Save cleaned dataset
# -------------------------------
df.to_csv("cleaned_data.csv.gz", index=False, compression='gzip')

print("✅ Cleaned dataset saved as cleaned_UNSW_IoT_Botnet.csv")