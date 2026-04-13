
# Dataset Augmentation using GANs

## 📌 Project Overview

This project demonstrates how **Generative Adversarial Networks (GANs)** can be leveraged for dataset augmentation, particularly to handle class imbalance problems in classification tasks. Traditional techniques like oversampling and undersampling either duplicate existing samples or discard valuable data. In contrast, GANs generate **realistic synthetic samples** that help balance class distributions **without redundancy or data loss**.

## 🧠 Concept

**GANs** consist of two neural networks:
- **Generator:** Produces synthetic data samples.
- **Discriminator:** Tries to distinguish between real and fake samples.

These networks are trained simultaneously in a minimax game:
- The generator improves at fooling the discriminator.
- The discriminator improves at spotting fake data.

## 🛠 Technologies Used

- **Python**
- **PyTorch** for deep learning
- **Matplotlib** for data visualization
- **Jupyter Notebook** for development and visualization

## 📂 Project Structure

### 1. **Data Generation**

Synthetic 2D data is generated using a multivariate normal distribution, then transformed using matrix multiplication and bias addition:
```python
X = torch.normal(0.0, 1, (1000, 2))
A = torch.tensor([[1, 2], [-0.1, 0.5]])
b = torch.tensor([1, 2])
data = torch.matmul(X, A) + b
```
This creates a well-defined distribution for GAN training.

### 2. **GAN Implementation**

#### a. **Generator Network**
- Accepts a 2D noise vector
- Outputs synthetic 2D samples matching the original distribution

#### b. **Discriminator Network**
- Takes 2D input (real or generated)
- Outputs a probability (real vs. fake)

Both networks are built using `nn.Sequential` with fully connected layers and activation functions (ReLU, Sigmoid, etc.).

### 3. **Training Procedure**

- Loss Function: Binary Cross-Entropy Loss
- Optimizers: Adam (for both networks)
- Training involves alternating updates to:
  - **Discriminator:** Learns to classify real vs fake
  - **Generator:** Learns to fool the discriminator
- Training proceeds over multiple epochs with loss tracking.

### 4. **Visualization**

- Real vs. generated samples are plotted throughout training.
- Generator’s progress is visualized using:
```python
plt.scatter(...)
plt.title("Generated Samples")
```

## 📈 Results

By the end of training:
- The **generator produces realistic synthetic data** samples that visually resemble the original distribution.
- The discriminator cannot reliably distinguish between real and synthetic samples.

## 🎯 Applications

This GAN-based augmentation pipeline can be used in:
- **Imbalanced classification problems** (e.g., fraud detection, medical diagnoses)
- **Data scarcity situations** where collecting new data is expensive or difficult

## ▶️ How to Run

1. Clone the repository.
2. Open the `GAN_pytorch.ipynb` notebook.
3. Run all cells sequentially in Jupyter Notebook.
4. Observe training metrics and generated sample visualizations.

## 🧪 Future Enhancements

- Extend to image data (e.g., MNIST, CIFAR)
- Add Conditional GANs (cGANs) for label-aware generation
- Apply to real-world class-imbalanced datasets

## 🤝 Contributions

Feel free to fork, improve, and raise PRs. Suggestions and improvements are welcome!
