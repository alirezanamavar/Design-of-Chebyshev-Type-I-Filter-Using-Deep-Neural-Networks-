# Chebyshev Type I Filter Design Using Deep Neural Networks

## Overview

This project presents a comprehensive pipeline for the design and prediction of analog Chebyshev Type I low-pass filters using deep neural networks (DNNs). The project blends classical filter design theory with modern machine learning, especially deep learning using PyTorch, to build a model capable of predicting the frequency response of a Chebyshev filter from its design parameters.

The repository includes Python code, dataset generation, model training, result analysis, GUI implementation, and packaging instructions. This project is ideal for those interested in the intersection of signal processing and AI.

---

## Table of Contents

* [Introduction](#introduction)
* [Theoretical Background](#theoretical-background)
* [Project Structure](#project-structure)
* [Phases Explained](#phases-explained)

  * Phase 1: Classical Filter Design
  * Phase 2: Dataset Generation
  * Phase 3: Dataset Justification
  * Phase 4: Data Preprocessing
  * Phase 5: Neural Network Training
  * Phase 6: Prediction and Visualization
  * Phase 7: GUI Interface (Optional)
  * Phase 8: Model Evaluation
  * Phase 9: Packaging (Optional)
* [Results](#results)
* [Installation and Running](#installation-and-running)
* [Dependencies](#dependencies)
* [Author](#author)

---

## Introduction

Traditional analog filter design methods, such as those for Chebyshev Type I filters, rely heavily on solving equations and analyzing frequency responses. This project proposes a deep learning-based approach to predict the magnitude response of Chebyshev filters, allowing for fast approximation and potentially embedded implementations.

The idea is to treat filter design as a regression problem: Given design parameters (filter order, ripple, and cutoff frequency), predict the filter's frequency response over a defined range.

---

## Theoretical Background

**Chebyshev Type I filters** are analog low-pass filters with an equiripple behavior in the passband and a monotonic stopband. The degree of ripple and the filter order define its steepness and accuracy. The analog transfer function is obtained via standard design techniques.

**Neural Networks**, especially fully connected feedforward networks (MLPs), are capable of approximating nonlinear mappings. Here, a neural network is trained to approximate the mapping:

```
(order, ripple, cutoff) -> frequency response (magnitude in dB)
```

The resulting model can generalize to unseen combinations of filter specifications.

---

## Project Structure

```
├── chebyshev_dataset.csv          # Generated dataset
├── chebyshev_model.pth           # Trained PyTorch model
├── scaler_X.pkl / scaler_y.pkl   # Saved scalers for input/output
├── filter_design.py              # Classical Chebyshev design
├── dataset_generator.py          # Dataset creation script
├── train_model.py                # PyTorch training script
├── predict_response.py           # Use trained model for prediction
├── gui_app.py                    # GUI (optional)
├── README.md                     # This file
```

---

## Phases Explained

### Phase 1: Classical Filter Design

In this phase, we employ `scipy.signal.cheby1` to create analog Chebyshev Type I low-pass filters. These filters are characterized by a user-defined order, ripple in the passband (in dB), and a cutoff frequency. The function returns the filter coefficients which are then used to compute and plot the magnitude and phase response using `scipy.signal.freqs`. This stage ensures the basic understanding and verification of classical filter behavior.

### Phase 2: Dataset Generation

A synthetic dataset is created by sweeping through multiple values of order (e.g., 2 to 10), ripple (e.g., 0.5 to 2 dB), and cutoff frequencies (e.g., 500 to 2000 Hz). For each configuration, the magnitude response over a log-spaced frequency range (10^1 to 10^5 rad/s) is calculated and saved in tabular format. This dataset will serve as the ground truth for training the neural network in subsequent phases.

### Phase 3: Dataset Justification

This phase explains the rationale for the dataset configuration. The frequency response range is chosen to reflect the relevant spectrum for most analog filters. The use of log-space frequencies ensures better resolution at lower frequencies, which are more sensitive to filter design. The magnitude response is sampled at 200 points, balancing computational cost with accuracy. The dataset includes sufficient variation to enable the model to generalize well.

### Phase 4: Data Preprocessing

Before training the model, the dataset is preprocessed:

* The input parameters (order, ripple, cutoff) are standardized using `StandardScaler` to zero-mean and unit variance.
* The outputs (magnitude responses) are similarly normalized.
* The dataset is split into 80% training and 20% testing sets.
* Scalers are saved using `joblib` for use during prediction.
  This standardization ensures better convergence during neural network training and removes scaling bias.

### Phase 5: Neural Network Training

The neural network is implemented in PyTorch as a multilayer perceptron with ReLU activations. The architecture includes four dense layers transforming a 3-element input into a 200-element output vector:

* Input Layer: 3 neurons (order, ripple, cutoff)
* Hidden Layers: 128 → 256 → 128 neurons
* Output Layer: 200 neurons (magnitude response)

The model is trained using MSELoss and Adam optimizer over 200 epochs. Loss is printed every 20 epochs to monitor convergence. The model's weights are saved for later use in prediction.

### Phase 6: Prediction and Visualization

This phase demonstrates the model’s practical usage. A user can input filter parameters (e.g., order = 6, ripple = 1.5 dB, cutoff = 1500 Hz), and the model predicts the corresponding magnitude response. This prediction is inverse-transformed to the original scale and visualized. Optionally, the predicted response is compared with the classical `cheby1` result for qualitative evaluation.

### Phase 7: GUI Interface (Optional)

To make the model accessible without requiring Python knowledge, a simple GUI is developed using `tkinter`. Users can enter filter parameters, press a button, and instantly see the predicted frequency response. This interface demonstrates the feasibility of deploying the model in real-world tools and educational platforms.

### Phase 8: Model Evaluation

To quantitatively assess the model’s performance, we compare predictions with classical designs across several filter orders. Evaluation metrics include:

* **Mean Squared Error (MSE)**: Measures the average squared difference.
* **Mean Absolute Error (MAE in dB)**: Indicates the average deviation in decibels.

Example summary:

```
Order 4 -> MSE: 2.53, MAE: 0.75 dB
Order 6 -> MSE: 4.25, MAE: 0.97 dB
Order 10 -> MSE: 7.32, MAE: 1.23 dB
```

These results show that the neural network maintains reasonable accuracy even at higher orders, which are more complex.

### Phase 9: Packaging

The trained model and GUI are packaged using PyInstaller into a `.exe` file, making the tool runnable on Windows systems without requiring Python installation. This enhances portability and facilitates distribution in non-programming environments.

---

## Results

The trained network achieved excellent accuracy with low MSE and high visual similarity to classical results. In practical use, the network can produce filter responses instantly, offering great advantage in real-time or embedded systems.

Plots show that even for high-order filters with sharp transitions, the neural network accurately reconstructs ripple and steepness behavior.

---

## Installation and Running

### Clone Repository

```bash
git clone https://github.com/your-username/chebyshev-filter-nn.git
cd chebyshev-filter-nn
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Model Training (Optional)

```bash
python train_model.py
```

### Predict Response

```bash
python predict_response.py
```

### Run GUI (Optional)

```bash
python gui_app.py
```

---

## Dependencies

* numpy
* scipy
* matplotlib
* pandas
* scikit-learn
* torch (PyTorch)
* joblib
* tkinter (for GUI)

---

## Author

Developed by \[Your Name] as a final project for "Filter and Circuit Synthesis" course. For questions or contributions, please open an issue or contact via GitHub.

---

## License

This project is open-source under the MIT License.

## Overview

This project presents a comprehensive pipeline for the design and prediction of analog Chebyshev Type I low-pass filters using deep neural networks (DNNs). The project blends classical filter design theory with modern machine learning, especially deep learning using PyTorch, to build a model capable of predicting the frequency response of a Chebyshev filter from its design parameters.

The repository includes Python code, dataset generation, model training, result analysis, GUI implementation, and packaging instructions. This project is ideal for those interested in the intersection of signal processing and AI.

---

## Table of Contents

* [Introduction](#introduction)
* [Theoretical Background](#theoretical-background)
* [Project Structure](#project-structure)
* [Phases Explained](#phases-explained)

  * Phase 1: Classical Filter Design
  * Phase 2: Dataset Generation
  * Phase 3: Dataset Justification
  * Phase 4: Data Preprocessing
  * Phase 5: Neural Network Training
  * Phase 6: Prediction and Visualization
  * Phase 7: GUI Interface (Optional)
  * Phase 8: Model Evaluation
  * Phase 9: Packaging (Optional)
* [Results](#results)
* [Installation and Running](#installation-and-running)
* [Dependencies](#dependencies)
* [Author](#author)

---

## Introduction

Traditional analog filter design methods, such as those for Chebyshev Type I filters, rely heavily on solving equations and analyzing frequency responses. This project proposes a deep learning-based approach to predict the magnitude response of Chebyshev filters, allowing for fast approximation and potentially embedded implementations.

The idea is to treat filter design as a regression problem: Given design parameters (filter order, ripple, and cutoff frequency), predict the filter's frequency response over a defined range.

---

## Theoretical Background

**Chebyshev Type I filters** are analog low-pass filters with an equiripple behavior in the passband and a monotonic stopband. The degree of ripple and the filter order define its steepness and accuracy. The analog transfer function is obtained via standard design techniques.

**Neural Networks**, especially fully connected feedforward networks (MLPs), are capable of approximating nonlinear mappings. Here, a neural network is trained to approximate the mapping:

```
(order, ripple, cutoff) -> frequency response (magnitude in dB)
```

The resulting model can generalize to unseen combinations of filter specifications.

---

## Project Structure

```
├── chebyshev_dataset.csv          # Generated dataset
├── chebyshev_model.pth           # Trained PyTorch model
├── scaler_X.pkl / scaler_y.pkl   # Saved scalers for input/output
├── filter_design.py              # Classical Chebyshev design
├── dataset_generator.py          # Dataset creation script
├── train_model.py                # PyTorch training script
├── predict_response.py           # Use trained model for prediction
├── gui_app.py                    # GUI (optional)
├── README.md                     # This file
```

---

## Phases Explained

### Phase 1: Classical Filter Design

In this phase, we employ `scipy.signal.cheby1` to create analog Chebyshev Type I low-pass filters. These filters are characterized by a user-defined order, ripple in the passband (in dB), and a cutoff frequency. The function returns the filter coefficients which are then used to compute and plot the magnitude and phase response using `scipy.signal.freqs`. This stage ensures the basic understanding and verification of classical filter behavior.

### Phase 2: Dataset Generation

A synthetic dataset is created by sweeping through multiple values of order (e.g., 2 to 10), ripple (e.g., 0.5 to 2 dB), and cutoff frequencies (e.g., 500 to 2000 Hz). For each configuration, the magnitude response over a log-spaced frequency range (10^1 to 10^5 rad/s) is calculated and saved in tabular format. This dataset will serve as the ground truth for training the neural network in subsequent phases.

### Phase 3: Dataset Justification

This phase explains the rationale for the dataset configuration. The frequency response range is chosen to reflect the relevant spectrum for most analog filters. The use of log-space frequencies ensures better resolution at lower frequencies, which are more sensitive to filter design. The magnitude response is sampled at 200 points, balancing computational cost with accuracy. The dataset includes sufficient variation to enable the model to generalize well.

### Phase 4: Data Preprocessing

Before training the model, the dataset is preprocessed:

* The input parameters (order, ripple, cutoff) are standardized using `StandardScaler` to zero-mean and unit variance.
* The outputs (magnitude responses) are similarly normalized.
* The dataset is split into 80% training and 20% testing sets.
* Scalers are saved using `joblib` for use during prediction.
  This standardization ensures better convergence during neural network training and removes scaling bias.

### Phase 5: Neural Network Training

The neural network is implemented in PyTorch as a multilayer perceptron with ReLU activations. The architecture includes four dense layers transforming a 3-element input into a 200-element output vector:

* Input Layer: 3 neurons (order, ripple, cutoff)
* Hidden Layers: 128 → 256 → 128 neurons
* Output Layer: 200 neurons (magnitude response)

The model is trained using MSELoss and Adam optimizer over 200 epochs. Loss is printed every 20 epochs to monitor convergence. The model's weights are saved for later use in prediction.

### Phase 6: Prediction and Visualization

This phase demonstrates the model’s practical usage. A user can input filter parameters (e.g., order = 6, ripple = 1.5 dB, cutoff = 1500 Hz), and the model predicts the corresponding magnitude response. This prediction is inverse-transformed to the original scale and visualized. Optionally, the predicted response is compared with the classical `cheby1` result for qualitative evaluation.

### Phase 7: GUI Interface (Optional)

To make the model accessible without requiring Python knowledge, a simple GUI is developed using `tkinter`. Users can enter filter parameters, press a button, and instantly see the predicted frequency response. This interface demonstrates the feasibility of deploying the model in real-world tools and educational platforms.

### Phase 8: Model Evaluation

To quantitatively assess the model’s performance, we compare predictions with classical designs across several filter orders. Evaluation metrics include:

* **Mean Squared Error (MSE)**: Measures the average squared difference.
* **Mean Absolute Error (MAE in dB)**: Indicates the average deviation in decibels.

Example summary:

```
Order 4 -> MSE: 2.53, MAE: 0.75 dB
Order 6 -> MSE: 4.25, MAE: 0.97 dB
Order 10 -> MSE: 7.32, MAE: 1.23 dB
```

These results show that the neural network maintains reasonable accuracy even at higher orders, which are more complex.

### Phase 9: Packaging

The trained model and GUI are packaged using PyInstaller into a `.exe` file, making the tool runnable on Windows systems without requiring Python installation. This enhances portability and facilitates distribution in non-programming environments.

---

## Results

The trained network achieved excellent accuracy with low MSE and high visual similarity to classical results. In practical use, the network can produce filter responses instantly, offering great advantage in real-time or embedded systems.

Plots show that even for high-order filters with sharp transitions, the neural network accurately reconstructs ripple and steepness behavior.

---

## Installation and Running

### Clone Repository

```bash
git clone https://github.com/your-username/chebyshev-filter-nn.git
cd chebyshev-filter-nn
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Model Training (Optional)

```bash
python train_model.py
```

### Predict Response

```bash
python predict_response.py
```

### Run GUI (Optional)

```bash
python gui_app.py
```

---

## Dependencies

* numpy
* scipy
* matplotlib
* pandas
* scikit-learn
* torch (PyTorch)
* joblib
* tkinter (for GUI)

---

## Author

Developed by \Alireza Namavar] as a final project for "Filter and Circuit Synthesis" course.

---

## License

This project is open-source under the MIT License.
