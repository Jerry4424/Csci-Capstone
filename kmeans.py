import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os

dataset_folder = r"c:\Users\pompk\Desktop\SeniorCaptone\Dataset"
csv_files = [f for f in os.listdir(dataset_folder) if f.endswith('.csv')]

for csv_file in csv_files:
    file_path = os.path.join(dataset_folder, csv_file)
    df = pd.read_csv(file_path)
    
    X = df.iloc[:, :-1].select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inertias = []
    silhouette_scores = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    kneedle = kneelocator(K_range, inertias, curve='convex', direction='decreasing')
    optimal_k = kneedle.elbow if kneedle.elbow else 3
    
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans_final.fit_predict(X_scaled)
    
    df['Cluster'] = clusters
    
    output_file = os.path.join(dataset_folder, f"clustered_{csv_file}")
    df.to_csv(output_file, index=False)
    
    print(f"Processed {csv_file}: Optimal clusters = {optimal_k}, Silhouette Score = {silhouette_scores[optimal_k-2]:.4f}")