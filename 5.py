import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Define the neural network class
# -------------------------------
class ChebyshevNet(nn.Module):
    def __init__(self):
        super(ChebyshevNet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 200)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

# -------------------------------
# Load trained model and scalers
# -------------------------------
model = ChebyshevNet()
model.load_state_dict(torch.load('chebyshev_model.pth'))
model.eval()

scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# -------------------------------
# Load dataset and preprocess
# -------------------------------
df = pd.read_csv('chebyshev_dataset.csv')

# Split input and output
X = df[['order', 'ripple', 'cutoff']].values
y = df.drop(columns=['order', 'ripple', 'cutoff']).values

# Normalize using loaded scalers
X_scaled = scaler_X.transform(X)
y_scaled = scaler_y.transform(y)

# Split into test data
_, X_test, _, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# -------------------------------
# Predict with model
# -------------------------------
with torch.no_grad():
    y_pred = model(X_test_tensor).numpy()

# Inverse-transform outputs
y_pred_real = scaler_y.inverse_transform(y_pred)
y_test_real = scaler_y.inverse_transform(y_test_tensor.numpy())

# -------------------------------
# Plot comparisons
# -------------------------------
freqs_rad = np.logspace(1, 5, 200)
freqs_kHz = freqs_rad / (2 * np.pi * 1000)  # Convert to kHz

num_examples = 3  # Number of samples to compare
plt.figure(figsize=(12, 4 * num_examples))

for i in range(num_examples):
    plt.subplot(num_examples, 1, i + 1)
    plt.semilogx(freqs_kHz, y_test_real[i], label='Actual', linewidth=2)
    plt.semilogx(freqs_kHz, y_pred_real[i], label='Predicted', linestyle='--')
    plt.xlabel('Frequency [kHz]')
    plt.ylabel('Magnitude [dB]')
    plt.title(f'Sample {i+1} | Model vs Actual')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()
