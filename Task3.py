import pandas as pd

df = pd.read_csv("/Users/aaditya/Downloads/Mall_Customers.csv")

print(df.info())

print(df.isnull().sum())

print(df.head())

from sklearn.preprocessing import StandardScaler

features = ["Age", "Annual_Income_(k$)", "Spending_Score"]  

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_scaled)
    score = silhouette_score(df_scaled, labels)
    print(f"Silhouette Score for {k} clusters: {score}")

optimal_k = 5

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(df_scaled)

df.to_csv("clustered_customers.csv", index=False)
print(df.head())

import seaborn as sns

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1], hue=df["Cluster"], palette="viridis")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Customer Clusters (PCA Projection)")
plt.show()

sns.pairplot(df, hue="Cluster", vars=features, palette="husl")
plt.show()
