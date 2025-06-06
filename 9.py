import tkinter as tk
from tkinter import ttk
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import cheby1, freqs
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib

# --- Define the trained model architecture ---
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

# --- Load model and scalers ---
model = ChebyshevNet()
model.load_state_dict(torch.load('chebyshev_model.pth'))
model.eval()

scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# --- GUI setup ---
root = tk.Tk()
root.title("Chebyshev Filter Designer (Neural Network vs Traditional)")

# --- Input fields ---
tk.Label(root, text="Order:").grid(row=0, column=0)
order_entry = ttk.Entry(root)
order_entry.grid(row=0, column=1)

tk.Label(root, text="Ripple (dB):").grid(row=1, column=0)
ripple_entry = ttk.Entry(root)
ripple_entry.grid(row=1, column=1)

tk.Label(root, text="Cutoff (Hz):").grid(row=2, column=0)
cutoff_entry = ttk.Entry(root)
cutoff_entry.grid(row=2, column=1)

# --- Matplotlib figure ---
fig, ax = plt.subplots(figsize=(6, 4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=5, column=0, columnspan=2)

# --- Function to update plot ---
def plot_filter_response():
    order = int(order_entry.get())
    ripple = float(ripple_entry.get())
    cutoff = float(cutoff_entry.get())

    # Frequency axis
    freqs_rad = np.logspace(1, 5, 200)
    freqs_kHz = freqs_rad / (2 * np.pi * 1000)

    # Neural network prediction
    X_input = np.array([[order, ripple, cutoff]])
    X_scaled = scaler_X.transform(X_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).numpy()
        y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]

    # Traditional design
    Wn = 2 * np.pi * cutoff
    b, a = cheby1(order, ripple, Wn, btype='low', analog=True)
    _, h = freqs(b, a, worN=freqs_rad)
    y_true = 20 * np.log10(np.abs(h))

    # Clear and redraw
    ax.clear()
    ax.semilogx(freqs_kHz, y_true, label='Traditional', linewidth=2)
    ax.semilogx(freqs_kHz, y_pred, '--', label='Neural Network')
    ax.axvline(cutoff / 1000, color='red', linestyle='--', label='Cutoff Frequency')
    ax.set_title(f'Order={order}, Ripple={ripple} dB, Cutoff={cutoff} Hz')
    ax.set_xlabel('Frequency [kHz]')
    ax.set_ylabel('Magnitude [dB]')
    ax.grid(True)
    ax.legend()
    canvas.draw()

# --- Button ---
plot_button = ttk.Button(root, text="Plot Filter Response", command=plot_filter_response)
plot_button.grid(row=4, column=0, columnspan=2, pady=10)

# --- Run the GUI ---
root.mainloop()
