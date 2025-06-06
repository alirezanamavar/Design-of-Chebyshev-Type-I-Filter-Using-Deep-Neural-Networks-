import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.signal import cheby1, freqs
import joblib

# -------------------------------
# Define the neural network model
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
# Frequency vector (same for both methods)
# -------------------------------
freqs_rad = np.logspace(1, 5, 200)
freqs_kHz = freqs_rad / (2 * np.pi * 1000)

# -------------------------------
# Parameters to compare
# -------------------------------
orders = [4, 6, 8, 10]
ripple = 1.0      # dB
cutoff = 2000     # Hz

plt.figure(figsize=(12, 3.5 * len(orders)))

# -------------------------------
# Main loop over different filter orders
# -------------------------------
for idx, order in enumerate(orders):
    # --- Neural Network prediction ---
    X_input = np.array([[order, ripple, cutoff]])
    X_scaled = scaler_X.transform(X_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        y_pred_scaled = model(X_tensor).numpy()
        y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]

    # --- Traditional method ---
    Wn = 2 * np.pi * cutoff
    b, a = cheby1(order, ripple, Wn, btype='low', analog=True)
    _, h = freqs(b, a, worN=freqs_rad)
    magnitude_db = 20 * np.log10(np.abs(h))

    # --- Plotting both curves side-by-side ---
    plt.subplot(len(orders), 1, idx + 1)
    plt.semilogx(freqs_kHz, magnitude_db, label='Traditional', linewidth=2)
    plt.semilogx(freqs_kHz, y_pred, '--', label='Neural Network')
    plt.axvline(cutoff / 1000, color='red', linestyle='--', label='Cutoff')
    plt.title(f'Order = {order} | Ripple = {ripple} dB | Cutoff = {cutoff} Hz')
    plt.xlabel('Frequency [kHz]')
    plt.ylabel('Magnitude [dB]')
    plt.grid(True)
    if idx == 0:
        plt.legend()

plt.tight_layout()
plt.show()
