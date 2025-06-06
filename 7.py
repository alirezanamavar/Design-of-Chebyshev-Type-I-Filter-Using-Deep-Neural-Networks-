import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import joblib

# -------------------------------
# Define the model architecture
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
# Load the trained model and scalers
# -------------------------------
model = ChebyshevNet()
model.load_state_dict(torch.load('chebyshev_model.pth'))
model.eval()

scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# -------------------------------
# User-defined filter parameters
# -------------------------------
order = 8         # Change as needed
ripple = 1.0      # dB
cutoff = 2000     # Hz

# Prepare input
X_input = np.array([[order, ripple, cutoff]])
X_scaled = scaler_X.transform(X_input)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# Predict response
with torch.no_grad():
    y_pred_scaled = model(X_tensor).numpy()
    y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]

# -------------------------------
# Plot the predicted magnitude response
# -------------------------------
freqs_rad = np.logspace(1, 5, 200)
freqs_kHz = freqs_rad / (2 * np.pi * 1000)

plt.figure(figsize=(10, 5))
plt.semilogx(freqs_kHz, y_pred, label='Predicted Magnitude')
plt.axvline(cutoff / 1000, color='red', linestyle='--', label='Cutoff Frequency')
plt.xlabel('Frequency [kHz]')
plt.ylabel('Magnitude [dB]')
plt.title(f'Predicted Chebyshev Type I Filter Response\nOrder={order}, Ripple={ripple} dB, Cutoff={cutoff} Hz')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
