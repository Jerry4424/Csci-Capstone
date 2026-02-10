from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import json
import torch
import torch.nn as nn

app = Flask(__name__)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# removed invalid reference to `outputlist` (was causing NameError)

# K-means Model
model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

# Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = Autoencoder(input_dim=4, hidden_dim=5)
autoencoder.load_state_dict(torch.load('autoencoder_model.pt'))
autoencoder.eval()
autoencoder_scaler = joblib.load('autoencoder_scaler.pkl')
threshold_stats = joblib.load('autoencoder_threshold.pkl')
ae_threshold = threshold_stats['threshold']
mse_loss = nn.MSELoss(reduction='none')

path = r'C:\Users\pompk\Desktop\SeniorCaptone\Dataset\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
df = pd.read_csv(path)
df.columns = df.columns.str.strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update_data')
def update_data():
    # Pick 1 random row
    row = df.sample(n=1)
    features = ['Destination Port', 'Total Length of Fwd Packets', 'Flow IAT Mean', 'SYN Flag Count']

    # Extract values and clean
    raw_values = row[features].copy()
    raw_values.replace([np.inf, -np.inf], np.nan, inplace=True)
    raw_values.fillna(0, inplace=True)

    # K-means Prediction
    scaled_point = scaler.transform(raw_values)
    kmeans_prediction = model.predict(scaled_point)[0]
    kmeans_status = "NORMAL" if kmeans_prediction == 0 else "ANOMALY"

    # Autoencoder Prediction (using reconstruction error)
    ae_scaled_point = autoencoder_scaler.transform(raw_values)
    ae_tensor = torch.tensor(ae_scaled_point, dtype=torch.float32)
    with torch.no_grad():
        ae_output = autoencoder(ae_tensor)
        reconstruction_error = torch.mean((ae_output - ae_tensor) ** 2).item()
    
    # Use threshold calculated from training data
    ae_status = "ANOMALY" if reconstruction_error > ae_threshold else "NORMAL"

    # Prepare data
    data = {
        'port': int(row['Destination Port'].values[0]),
        'length': int(row['Total Length of Fwd Packets'].values[0]),
        'iat': float(row['Flow IAT Mean'].values[0]),
        'syn': int(row['SYN Flag Count'].values[0]),
        'kmeans_status': kmeans_status,
        'kmeans_cluster': int(kmeans_prediction),
        'autoencoder_status': ae_status,
        'reconstruction_error': float(reconstruction_error)
    }

    # Save to outputlist.json (single latest sample)
    try:
        outlist_path = r'C:\Users\pompk\Desktop\SeniorCaptone\outputlist.json'
        with open(outlist_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        # If either model detected anomaly, append to output.json
        if data['kmeans_status'] == 'ANOMALY' or data['autoencoder_status'] == 'ANOMALY':
            output_path = r'C:\Users\pompk\Desktop\SeniorCaptone\output.json'
            if os.path.exists(output_path):
                try:
                    with open(output_path, 'r', encoding='utf-8') as f:
                        existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = [existing]
                except (json.JSONDecodeError, OSError):
                    existing = []
            else:
                existing = []

            existing.append(data)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(existing, f, indent=4, ensure_ascii=False)
    except OSError as e:
        return jsonify({"error": f"Failed to write file: {e}"}), 500

    return jsonify({
        'port': int(row['Destination Port'].values[0]),
        'length': int(row['Total Length of Fwd Packets'].values[0]),
        'iat': float(row['Flow IAT Mean'].values[0]),
        'syn': int(row['SYN Flag Count'].values[0]),
        'kmeans_status': kmeans_status,
        'kmeans_cluster': int(kmeans_prediction),
        'autoencoder_status': ae_status,
        'reconstruction_error': float(reconstruction_error)
    })

if __name__ == '__main__':
    app.run(debug=True)