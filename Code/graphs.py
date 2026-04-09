import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load results
# -------------------------------
df = pd.read_csv("../Results/all_model_results.csv")

print(df)

# -------------------------------
# 2. Accuracy Graph
# -------------------------------
plt.figure()
plt.bar(df["Model"], df["Accuracy"])
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.savefig("../Results/accuracy_graph.png")
plt.show()

# -------------------------------
# 3. Precision Graph
# -------------------------------
plt.figure()
plt.bar(df["Model"], df["Precision"])
plt.title("Precision Comparison")
plt.xlabel("Models")
plt.ylabel("Precision")
plt.xticks(rotation=30)
plt.savefig("../Results/precision_graph.png")
plt.show()

# -------------------------------
# 4. Recall Graph
# -------------------------------
plt.figure()
plt.bar(df["Model"], df["Recall"])
plt.title("Recall Comparison")
plt.xlabel("Models")
plt.ylabel("Recall")
plt.xticks(rotation=30)
plt.savefig("../Results/recall_graph.png")
plt.show()

# -------------------------------
# 5. F1 Score Graph
# -------------------------------
plt.figure()
plt.bar(df["Model"], df["F1 Score"])
plt.title("F1 Score Comparison")
plt.xlabel("Models")
plt.ylabel("F1 Score")
plt.xticks(rotation=30)
plt.savefig("../Results/f1_graph.png")
plt.show()

print("✅ Graphs saved in Results folder")