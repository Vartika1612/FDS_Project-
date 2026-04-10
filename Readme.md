# Foundations of Data Science Group Project

## Project Title

IoT Botnet Detection using Machine Learning (UNSW Dataset)

## Team

- Group: 9
- Members:

| Reg. No.   | Name                     |
| ---------- | ------------------------ |
| 23BCE11137 | VARTIKA VASHISHTHA       |
| 23BCE11046 | KUSH PATHAK              |
| 23BCE11031 | VAIBHAVI AGARWAL         |
| 23BCE11136 | SUDEEKSHA PACHORI        |
| 23BCE11041 | DHRUV                    |
| 23BCE11111 | RAJAT RANJAN MISRA       |

All Members Contributed

---

## Project Summary

This project focuses on detecting IoT botnet attacks using the UNSW IoT Botnet dataset by building a complete machine learning pipeline.

The workflow includes:

- Data preprocessing and cleaning  
- Exploratory Data Analysis (EDA)  
- Dimensionality reduction using PCA and SVD  
- Training and evaluation of multiple machine learning models  

The objective is to compare model performance and create a reproducible pipeline suitable for academic submission.

---

## Dataset

- Source: UNSW IoT Botnet Dataset  
- Local data path: Data/  

Files used:

- UNSW_2018_IoT_Botnet_Dataset_19.csv  
- UNSW_2018_IoT_Botnet_Dataset_73.csv  

---

## Repository Structure
FDS-PROJECT/
│
├── Code/
│ ├── Preprocessing.py
│ ├── SVD_PCA.py
│ ├── comparison.py
│ ├── eda.py
│ └── graphs.py
│
├── Data/
│ └── (large dataset files - not uploaded to GitHub)
│
├── Results/
│ ├── all_model_results.csv
│ ├── accuracy_graph.png
│ ├── precision_graph.png
│ ├── recall_graph.png
│ ├── f1_graph.png
│ ├── target_distribution.png
│ ├── correlation_heatmap.png
│ └── boxplot.png
│
└── README.md


---

## Main Scripts

- Preprocessing.py  
  - Loads and merges datasets  
  - Handles missing and infinite values  
  - Encodes categorical features  

- SVD_PCA.py  
  - Applies PCA and Truncated SVD  
  - Reduces feature dimensionality  

- comparison.py  
  - Trains multiple models  
  - Compares performance (Accuracy, Precision, Recall, F1 Score)  
  - Saves results  

- eda.py  
  - Performs exploratory data analysis  
  - Generates statistical insights  

- graphs.py  
  - Generates visualization graphs for model comparison and EDA  

---

## Data Processing Workflow

1. Load raw UNSW IoT dataset files  
2. Merge datasets into a unified structure  
3. Handle missing and infinite values  
4. Encode categorical features  
5. Normalize and clean data  
6. Apply dimensionality reduction (PCA, SVD)  
7. Train machine learning models  
8. Evaluate and compare model performance  

---

## Models Used

- Logistic Regression  
- Decision Tree  
- K-Nearest Neighbors (KNN)  

---

## Output Artifacts

- Reduced datasets (PCA & SVD)  
- Model evaluation results (CSV)  
- Performance graphs  
- EDA visualizations  

---

## Results

- Decision Tree achieved highest accuracy  
- KNN provided competitive performance but was slower  
- Logistic Regression was fastest but slightly less accurate  

---

## Environment and Dependencies

Recommended environment:

- Python 3.9+  
- VS Code / Jupyter Notebook  

Required libraries:

- pandas  
- numpy  
- matplotlib  
- scikit-learn  

---

How to Run
Open project folder in VS Code or terminal
Ensure dataset files are present in Data/
Run:
python Code/Preprocessing.py
python Code/SVD_PCA.py
python Code/comparison.py
python Code/graphs.py
View outputs in Results/ folder



