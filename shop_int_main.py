import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# 1. Generate Synthetic Example Data
# Simulating 500 customers with 'Time_on_Site', 'Cart_Abandonment_Rate', and 'Made_Purchase'
np.random.seed(42)

# Group A: High time on site, low abandonment (Likely Purchasers)
time_A = np.random.normal(loc=25, scale=5, size=250)
abandon_A = np.random.normal(loc=0.2, scale=0.1, size=250)
purchase_A = np.random.choice([1, 0], size=250, p=[0.85, 0.15])

# Group B: Low time on site, high abandonment (Window Shoppers)
time_B = np.random.normal(loc=5, scale=3, size=250)
abandon_B = np.random.normal(loc=0.8, scale=0.1, size=250)
purchase_B = np.random.choice([1, 0], size=250, p=[0.10, 0.90])

# Combine into a single DataFrame
time_on_site = np.concatenate([time_A, time_B])
abandonment_rate = np.concatenate([abandon_A, abandon_B])
made_purchase = np.concatenate([purchase_A, purchase_B])

# Ensure abandonment rates stay between 0 and 1, and time is positive
abandonment_rate = np.clip(abandonment_rate, 0, 1)
time_on_site = np.clip(time_on_site, 1, None)

data = pd.DataFrame({
    'Time_on_Site_Minutes': time_on_site,
    'Cart_Abandonment_Rate': abandonment_rate,
    'Made_Purchase': made_purchase
})

print("First 5 rows of our custom dataset:")
print(data.head(), "\n")

# 2. Prepare Data for Clustering
# We will cluster based on Time on Site and Cart Abandonment Rate
X = data[['Time_on_Site_Minutes', 'Cart_Abandonment_Rate']].values

# It's good practice to scale features for K-Means so one variable doesn't dominate
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. The Elbow Method to find optimal clusters
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

# 4. Apply K-Means with 2 clusters (as suggested by our data generation and the elbow)
km = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=42)
y_means = km.fit_predict(X_scaled)

# 5. Visualize the Clusters
plt.figure(figsize=(10, 6))

# Plotting the unscaled data for better readability on axes, using cluster labels for color
plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s=50, c='salmon', label='Cluster 0 (High Risk)')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s=50, c='lightgreen', label='Cluster 1 (High Intent)')

plt.title('Customer Segmentation: Time on Site vs. Abandonment', fontsize=16)
plt.xlabel('Time on Site (Minutes)')
plt.ylabel('Cart Abandonment Rate (0 to 1)')
plt.legend()
plt.grid(True)
plt.show()

# 6. Evaluate against actual "Made_Purchase" labels
# Since K-Means is unsupervised, we check how well our clusters map to actual purchases
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
