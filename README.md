# 🧠 Digit Recognition with PyTorch

A fully explained, end-to-end deep learning project that trains a **Multi-Layer Perceptron (MLP)** neural network to recognize digits (0–9) using the MNIST dataset and **PyTorch** — no TensorFlow involved.

---

## 📸 Project Preview

```
Input Image (28×28)
        ↓
  Flatten → 784
        ↓
  FC Layer (512) + ReLU + Dropout
        ↓
  FC Layer (256) + ReLU + Dropout
        ↓
  FC Layer (128) + ReLU + Dropout
        ↓
  Output (10 classes: 0–9)
```

**Achieved Accuracy: ~98–99% on the MNIST test set**

---

## 📁 Project Structure

```
digit-recognition_NN
│
├── digit_recognition_pytorch.ipynb   ← Main Jupyter Notebook (all code + explanations)
├── README.md                         ← You are here
│
├── data                             ← MNIST dataset (auto-downloaded on first run)
│   └── MNIST
│
├── best_digit_model.pth              ← Best model weights (saved during training)
├── digit_recognizer_final.pth        ← Final model weights
│
└── outputs                          ← Saved plots (generated on run)
    ├── sample_images.png
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── predictions.png
    └── misclassified.png
```
Note: The data folder is not included in this repository because dataset files are large.
The MNIST dataset is automatically downloaded when you run the notebook.
---

---
## 🗺️ Notebook Walkthrough

The notebook is organized into **13 clearly documented steps**:

| Step | Title | What You'll Learn |
|------|-------|-------------------|
| 1 | **Import Libraries** | PyTorch modules, torchvision, matplotlib |
| 2 | **Load MNIST Dataset** | DataLoader, transforms, normalization |
| 3 | **Visualize Samples** | How the raw data looks |
| 4 | **Build Neural Network** | `nn.Module`, Linear layers, ReLU, Dropout |
| 5 | **Loss & Optimizer** | CrossEntropyLoss, Adam, StepLR scheduler |
| 6 | **Train/Eval Functions** | Forward pass, backprop, gradient clipping |
| 7 | **Train the Model** | Full training loop with metrics per epoch |
| 8 | **Training Curves** | Loss and accuracy plots |
| 9 | **Confusion Matrix** | Per-class precision, recall, F1 score |
| 10 | **Visualize Predictions** | ✓ Correct vs ✗ Wrong with confidence % |
| 11 | **Error Analysis** | Browse misclassified examples |
| 12 | **Save & Load Model** | `state_dict`, loading weights |
| 13 | **Final Summary** | Results table + what to try next |

---

## 🏗️ Model Architecture

```
DigitRecognizer(
  (fc1):     Linear(784 → 512)
  (fc2):     Linear(512 → 256)
  (fc3):     Linear(256 → 128)
  (fc4):     Linear(128 → 10)
  (dropout): Dropout(p=0.2)
)
Total trainable parameters: ~567,050
```

Each hidden layer applies: **Linear → ReLU → Dropout**  
The output layer produces raw **logits** (no softmax) — compatible with `CrossEntropyLoss`.

---

## ⚙️ Hyperparameters

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| Batch Size | `64` | Mini-batches for gradient estimation |
| Learning Rate | `0.001` | Initial Adam learning rate |
| Epochs | `10` | Full passes over training data |
| Dropout Rate | `0.2` | 20% of neurons dropped per forward pass |
| Weight Decay | `1e-4` | L2 regularization on all parameters |
| LR Scheduler | `StepLR(step=3, gamma=0.5)` | Halves LR every 3 epochs |
| Weight Init | `Kaiming (He) Normal` | Optimized for ReLU activations |

---

## 📊 Dataset: MNIST

| Property | Value |
|----------|-------|
| Training samples | 60,000 |
| Test samples | 10,000 |
| Image size | 28 × 28 pixels |
| Channels | 1 (grayscale) |
| Classes | 10 (digits 0–9) |
| Mean (for normalization) | 0.1307 |
| Std (for normalization) | 0.3081 |

MNIST is automatically downloaded to `./data/` on first run via `torchvision.datasets.MNIST`.

---

## 🔁 Training Loop — How It Works

Each mini-batch goes through these 7 steps:

```python
optimizer.zero_grad()      # 1. Clear old gradients
outputs = model(images)    # 2. Forward pass → get logits
loss = criterion(outputs, labels)  # 3. Compute cross-entropy loss
loss.backward()            # 4. Backprop → compute gradients
clip_grad_norm_(...)       # 5. Clip gradients (prevent explosion)
optimizer.step()           # 6. Update weights
scheduler.step()           # 7. Adjust learning rate (once per epoch)
```

---

## 📈 Key Concepts Explained

### ReLU Activation
```
f(x) = max(0, x)
```
Introduces non-linearity so the network can learn complex patterns.  
Avoids the vanishing gradient problem compared to sigmoid/tanh.

### Dropout
Randomly sets a fraction of neuron outputs to `0` during training.  
Forces the network to not rely on any single neuron → reduces overfitting.  
Disabled automatically during `model.eval()`.

### CrossEntropyLoss
Internally applies **Softmax + Negative Log Likelihood Loss**:
```
Loss = -log(probability of correct class)
```
Do **not** add softmax to the model's final layer — PyTorch handles it inside the loss.

### Adam Optimizer
Combines momentum and adaptive learning rates per parameter.  
Usually the best default choice for neural networks.

### Confusion Matrix
A 10×10 grid comparing true vs predicted labels.  
The diagonal = correct predictions. Off-diagonal = errors.  
Reveals which digits the model confuses (e.g., 4↔9, 3↔8).

---

## 🚀 Getting Started

### 1. Clone / Download
```bash
git clone https://github.com/your-username/digit-recognition-pytorch.git
cd digit-recognition-pytorch
```

### 2. Install Dependencies
```bash
pip install torch torchvision matplotlib numpy scikit-learn jupyter
```

### 3. Launch Jupyter
```bash
jupyter notebook digit_recognition_pytorch.ipynb
```

### 4. Run All Cells
In Jupyter: **Kernel → Restart & Run All**  
MNIST will auto-download (~11 MB) on first run.

---

## 📦 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
jupyter>=1.0.0
```

Install all at once:
```bash
pip install torch torchvision matplotlib numpy scikit-learn jupyter
```

> **GPU Support:** If you have an NVIDIA GPU with CUDA, PyTorch will automatically use it. Training will be significantly faster.

---

## 🏆 Results

| Metric | Value |
|--------|-------|
| Final Train Accuracy | ~99.2% |
| Final Test Accuracy | ~98.5% |
| Best Test Accuracy | ~98.8% |
| Total Epochs | 10 |
| Training Time (CPU) | ~5–8 min |
| Training Time (GPU) | ~1–2 min |

---

## 💾 Model Saving & Loading

```python
# Save model weights
torch.save(model.state_dict(), 'best_digit_model.pth')

# Load model weights
model = DigitRecognizer(dropout_rate=0.2)
model.load_state_dict(torch.load('best_digit_model.pth'))
model.eval()
```

---

## 🔬 What to Try Next

| Idea | Expected Gain |
|------|--------------|
| **CNN (ConvNet)** instead of MLP | 99.5%+ accuracy |
| **Batch Normalization** layers | Faster convergence |
| **Data Augmentation** (random rotations, shifts) | Better generalization |
| **Learning Rate Finder** | Optimal LR selection |
| **Visualize weight filters** | Understand what neurons detect |
| **Deploy as a web app** | Draw a digit → model predicts live |
| **Try on Fashion-MNIST** | Harder dataset, same architecture |

---

## 🧩 Core PyTorch Concepts Used

| Concept | API Used |
|---------|----------|
| Tensor operations | `torch.Tensor`, `torch.max` |
| Autograd (auto-differentiation) | `loss.backward()` |
| Neural network layers | `nn.Linear`, `nn.Dropout` |
| Activation functions | `F.relu`, `F.softmax` |
| Loss functions | `nn.CrossEntropyLoss` |
| Optimizers | `optim.Adam` |
| LR scheduling | `optim.lr_scheduler.StepLR` |
| Data loading | `DataLoader`, `datasets.MNIST` |
| Data transforms | `transforms.Compose`, `ToTensor`, `Normalize` |
| Device management | `torch.device`, `.to(device)` |
| Model persistence | `torch.save`, `torch.load` |
| Gradient management | `optimizer.zero_grad()`, `clip_grad_norm_` |
| Inference mode | `torch.no_grad()`, `model.eval()` |

---

## 🙏 Acknowledgements

- **MNIST Dataset** — Yann LeCun, Corinna Cortes, Christopher Burges
- **PyTorch** — Meta AI Research
- **torchvision** — PyTorch Vision team
