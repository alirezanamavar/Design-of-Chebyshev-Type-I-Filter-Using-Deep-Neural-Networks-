import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cheby1, freqs

# Cutoff frequency
cutoff_freq = 1000  # Hz
Wn = 2 * np.pi * cutoff_freq  # rad/s
w = np.logspace(1, 5, 1000)   # Frequency range for plotting

# --------- 1. Compare Different Orders (Ripple fixed) ---------
orders = [2, 6, 12, 20]     # Sharply different filter orders
ripple_fixed = 1            # Ripple in dB

plt.figure(figsize=(10, 6))
for order in orders:
    b, a = cheby1(order, ripple_fixed, Wn, btype='low', analog=True)
    _, h = freqs(b, a, worN=w)
    plt.semilogx(w, 20 * np.log10(np.abs(h)), label=f'Order = {order}')
plt.axvline(Wn, color='red', linestyle='--', label='Cutoff Frequency')
plt.title(f'Chebyshev Type I - Magnitude Response\nFixed Ripple = {ripple_fixed} dB')
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Magnitude [dB]')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --------- 2. Compare Different Ripples (Order fixed) ---------
ripples = [0.05, 0.5, 2, 5]  # Widely spaced ripple values
order_fixed = 8             # Fixed order

plt.figure(figsize=(10, 6))
for ripple in ripples:
    b, a = cheby1(order_fixed, ripple, Wn, btype='low', analog=True)
    _, h = freqs(b, a, worN=w)
    plt.semilogx(w, 20 * np.log10(np.abs(h)), label=f'Ripple = {ripple} dB')
plt.axvline(Wn, color='red', linestyle='--', label='Cutoff Frequency')
plt.title(f'Chebyshev Type I - Magnitude Response\nFixed Order = {order_fixed}')
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Magnitude [dB]')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
