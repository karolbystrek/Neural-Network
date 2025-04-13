# Neural Network Implementation in Java

## Description

This project provides a basic implementation of a feedforward neural network from scratch using Java. It is designed to understand the fundamental concepts of neural networks, including forward propagation, backpropagation, activation functions (ReLU and Softmax), and optimization using Stochastic Gradient Descent (SGD). The network is demonstrated using the classic MNIST dataset for handwritten digit recognition.

## Features

*   **Feedforward Architecture:** Simple multi-layer perceptron structure.
*   **Configurable Layers:** Easily define the number of layers and neurons per layer.
*   **Activation Functions:**
    *   ReLU (Rectified Linear Unit) for hidden layers.
    *   Softmax for the output layer (suitable for multi-class classification).
*   **Loss Function:** Cross-Entropy loss for evaluating performance.
*   **Optimizer:** Stochastic Gradient Descent (SGD) with mini-batch support.
*   **Weight Initialization:** He initialization for weights.
*   **MNIST Data Loader:** Includes a utility to read the MNIST dataset in its standard IDX format.

## Getting Started

### Prerequisites

*   Java Development Kit (JDK) 21 or later.
*   Apache Maven (for building and managing dependencies).
*   MNIST dataset files (download separately).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd NeuralNetwork
    ```
2.  **Download MNIST Data:**
    *   Download the following files from the official MNIST website (or other sources):
        *   `train-images-idx3-ubyte.gz`
        *   `train-labels-idx1-ubyte.gz`
        *   `t10k-images-idx3-ubyte.gz` (Optional, for testing)
        *   `t10k-labels-idx1-ubyte.gz` (Optional, for testing)
    *   Unzip the files.
    *   Place the unzipped `.idx*-ubyte` files into the `data/MNIST/` directory within the project structure. Create the directory if it doesn't exist.
    ```
    NeuralNetwork/
    ├── data/
    │   └── MNIST/
    │       ├── train-images.idx3-ubyte
    │       ├── train-labels.idx1-ubyte
    │       ├── t10k-images.idx3-ubyte
    │       └── t10k-labels.idx1-ubyte
    ├── src/
    │   └── ...
    └── pom.xml
    ```

### Building

Use Maven to compile the project:

```bash
mvn clean package
