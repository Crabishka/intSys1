import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data_df = pd.read_csv('usa_elections.dat', delimiter=';')
column_means = data_df.iloc[:, 1:].mean()
data_df.iloc[:, 1:] = data_df.iloc[:, 1:].fillna(column_means)

selected_data = data_df.loc[:, 'X1856':]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(selected_data)


state_names = np.array(data_df['state.name'])
linkage_matrix = linkage(scaled_data, method='ward')
plt.figure(figsize=(10, 8))
dendrogram(linkage_matrix, labels=state_names)
plt.xticks(rotation=90)
plt.title('Дендрограмма')
plt.show()

num_clusters = 4
cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

mds = MDS(n_components=2)
mds_data = mds.fit_transform(scaled_data)

cluster_colors = ['r', 'g', 'b', 'y']

plt.figure(figsize=(8, 6))
for cluster_num, color in zip(range(1, num_clusters + 1), cluster_colors):
    cluster_points = mds_data[cluster_labels == cluster_num]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=color, label=f'Cluster {cluster_num}')


plt.title('Кластеризация')
plt.show()

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Число кластеров')
plt.title('График "каменистая осыпь"')
plt.show()
