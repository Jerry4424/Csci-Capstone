from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import json

app = Flask(__name__)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# removed invalid reference to `outputlist` (was causing NameError)

model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

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

    # Scale and Predict
    scaled_point = scaler.transform(raw_values)
    prediction = model.predict(scaled_point)[0]

    # Prepare data
    data = {
        'port': int(row['Destination Port'].values[0]),
        'length': int(row['Total Length of Fwd Packets'].values[0]),
        'iat': float(row['Flow IAT Mean'].values[0]),
        'syn': int(row['SYN Flag Count'].values[0]),
        'status': "NORMAL" if prediction == 0 else "ANOMALY"
    }

    # Save to outputlist.json (single latest sample)
    try:
        outlist_path = r'C:\Users\pompk\Desktop\SeniorCaptone\outputlist.json'
        with open(outlist_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        # If anomaly, append to output.json (store list of anomalies)
        if data['status'] == 'ANOMALY':
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
        'status': "NORMAL" if prediction == 0 else "ANOMALY"
    })

if __name__ == '__main__':
    app.run(debug=True)