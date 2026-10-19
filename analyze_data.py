import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# 1. Load the Custom CSV Data
try:
    data = pd.read_csv("custom_shopping_data.csv")
    print("Data loaded successfully. First 5 rows:")
    print(data.head(), "\n")
except FileNotFoundError:
    print("Error: 'custom_shopping_data.csv' not found. Please run the generation script first.")
    exit()

# 2. Prepare Data for Clustering
# Using Time on Site and Cart Abandonment Rate for the K-Means algorithm
X = data[['Time_on_Site_Minutes', 'Cart_Abandonment_Rate']].values

# Scale features so minutes and percentages are weighted equally
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. The Elbow Method to verify the optimal number of clusters
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Customer Segmentation', fontsize=16)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# 4. Apply K-Means with 2 clusters
km = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=42)
y_means = km.fit_predict(X_scaled)

# 5. Visualize the Clusters
plt.figure(figsize=(10, 6))

# Plotting the raw (unscaled) data so the axes make sense to a human reader
plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s=50, c='salmon', label='Cluster 0 (High Risk)')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s=50, c='lightgreen', label='Cluster 1 (High Intent)')

plt.title('Customer Segmentation: Time on Site vs. Abandonment', fontsize=16)
plt.xlabel('Time on Site (Minutes)')
plt.ylabel('Cart Abandonment Rate (0 to 1)')
plt.legend()
plt.grid(True)
plt.show()

# 6. Evaluate against actual "Made_Purchase" labels
labels_true = data['Made_Purchase'].values
labels_pred = y_means

# Calculate Adjusted Rand Index
ari_score = metrics.adjusted_rand_score(labels_true, labels_pred)
print(f"Adjusted Rand Index (Similarity to actual purchases): {ari_score:.4f}\n")

# Display Confusion Matrix
cm = metrics.confusion_matrix(labels_true, labels_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Cluster 0', 'Cluster 1'], 
            yticklabels=['No Purchase', 'Made Purchase'])
plt.title('Confusion Matrix: Clusters vs Actual Purchase')
plt.xlabel('Predicted Cluster')
plt.ylabel('Actual Purchase')
plt.show()
