import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

path = r'C:\Users\pompk\Desktop\SeniorCaptone\Dataset\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
df = pd.read_csv(path)

# 2. Removes hidden collumn spaces
df.columns = df.columns.str.strip()

# 3. Select features
features = ['Destination Port', 'Total Length of Fwd Packets', 'Flow IAT Mean', 'SYN Flag Count']
X = df[features].copy()

# 4. Handle math errors
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)

# 5. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Train Model
print("Training model... please wait.")
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# 7. Save files
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("'kmeans_model.pkl' and 'scaler.pkl' created.")