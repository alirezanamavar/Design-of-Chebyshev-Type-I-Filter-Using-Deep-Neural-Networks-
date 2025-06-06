import numpy as np
import pandas as pd
from scipy.signal import cheby1, freqs
from itertools import product
from tqdm import tqdm

# Define parameter ranges
orders = [2, 4, 6, 8, 10, 12]
ripples = [0.1, 0.5, 1.0, 2.0, 3.0]
cutoffs = [500, 1000, 1500, 2000, 3000]  # Hz

# Define frequency points for magnitude response
freqs_rad = np.logspace(1, 5, 200)  # 200 points from 10^1 to 10^5 rad/s

# List to collect data
dataset = []

# Generate all combinations of parameters
for order, ripple, cutoff in tqdm(product(orders, ripples, cutoffs)):
    Wn = 2 * np.pi * cutoff  # Convert cutoff to rad/s
    try:
        b, a = cheby1(order, ripple, Wn, btype='low', analog=True)
        _, h = freqs(b, a, worN=freqs_rad)
        magnitude_db = 20 * np.log10(np.abs(h))  # Gain in dB
        row = [order, ripple, cutoff] + list(magnitude_db)
        dataset.append(row)
    except Exception as e:
        print(f"Skipped: Order={order}, Ripple={ripple}, Cutoff={cutoff} | Error: {e}")

# Column names
input_features = ['order', 'ripple', 'cutoff']
mag_cols = [f'Mag_{i}' for i in range(len(freqs_rad))]
columns = input_features + mag_cols

# Convert to DataFrame and save
df = pd.DataFrame(dataset, columns=columns)
df.to_csv('chebyshev_dataset.csv', index=False)
print("Dataset saved as 'chebyshev_dataset.csv'")
