import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import os
from sklearn.preprocessing import StandardScaler

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load dataset from CSV
path = r'C:\Users\pompk\Desktop\SeniorCaptone\Dataset\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
df = pd.read_csv(path)
df.columns = df.columns.str.strip()

# Select the same features as kmeans model for comparison
features = ['Destination Port', 'Total Length of Fwd Packets', 'Flow IAT Mean', 'SYN Flag Count']
X = df[features].copy()

X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)

# Scaling
scaler = StandardScaler()
data_normalized = scaler.fit_transform(X)

D, K = data_normalized.shape
print(f"Dataset shape: {D} samples, {K} features")

# Convert the numpy array to a PyTorch tensor
data_tensor = torch.tensor(data_normalized, dtype=torch.float32)
print(f"Tensor shape: {data_tensor.shape}")

# Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Hyperparameters
input_dim = K
hidden_dim = 5  # Hidden layer dimension, you need adjusted

# Initialize the model, loss function, and optimizer
model = Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training parameters
num_epochs = 30
batch_size = 32

# Training loop
print("Training model... please wait.")
for epoch in range(num_epochs):
    for i in range(0, D, batch_size):
        batch_data = data_tensor[i:i+batch_size]

        # Forward pass
        outputs = model(batch_data)
        loss = criterion(outputs, batch_data)

        # Backward pass and optimization
        optimizer.zero_grad()  
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Get the encoder output for clustering (latent representation)
with torch.no_grad():
    encoded_output = model.encoder(data_tensor).numpy()

# Save model and scaler
torch.save(model.state_dict(), 'autoencoder_model.pt')
joblib.dump(scaler, 'autoencoder_scaler.pkl')

print("'autoencoder_model.pt' and 'autoencoder_scaler.pkl' created.")
print(f"Latent representation shape: {encoded_output.shape}")