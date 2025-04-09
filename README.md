# Handwritten Digit Recognition Using Machine Learning

This repository contains a comprehensive project that implements and compares multiple machine learning models to perform handwritten digit recognition on the MNIST dataset. The project includes implementations using traditional machine learning techniques (Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors) as well as a simple Neural Network using TensorFlow/Keras.

## Overview

Handwritten digit recognition is a classic problem in the field of machine learning and computer vision. The MNIST dataset, which consists of 60,000 training images and 10,000 testing images of 28×28 pixel grayscale digits, serves as an ideal benchmark for comparing various machine learning algorithms. This project achieves the following:

- **Data Preprocessing:** Loading the MNIST dataset, normalizing pixel values, and flattening images for traditional algorithms.
- **Model Implementation:** Training multiple models including:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - A simple Neural Network
- **Evaluation:** Comparing model performance using metrics such as accuracy, precision, recall, F1-score, and confusion matrices.
- **Documentation:** A detailed LaTeX report is included that outlines the methodology, results, discussion, and conclusions.

## Repository Contents

- **Python Scripts:**
  - `load_mnist.py`: Loads and preprocesses the MNIST dataset.
  - `build_models.py`: Trains and evaluates the models; generates evaluation visuals (accuracy bar chart and confusion matrices) and exports the neural network architecture diagram.
- **Images:**
  - Sample images, model confusion matrices, and the neural network architecture diagram (e.g., `mnist_sample.png`, `nn_architecture.png`, `logistic_regression.png`, etc.).
- **Documentation:**
  - LaTeX report (`report.tex`) that describes the project in detail.
- **.gitignore:**
  - Ensures that large or environment-specific files (like the virtual environment) are excluded.

## Installation and Setup

### Prerequisites
- Python 3.7 or later
- Git

### Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/krishna31102004/mnist-handwritten-digit-recognition-clean.git
   cd mnist-handwritten-digit-recognition-clean
   ```

2. **Create a Virtual Environment:**
   ```bash
   python3 -m venv venv
   ```
   
3. **Activate the Virtual Environment:**
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

4. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Project
- **Data Loading and Preprocessing:**
  ```bash
  python3 load_mnist.py
  ```
  This script loads the MNIST dataset, performs normalization and flattening, and displays a few sample images.

- **Training and Evaluating Models:**
  ```bash
  python3 build_models.py
  ```
  This script trains multiple models, evaluates them (printing accuracy, classification reports, and confusion matrices), and generates a neural network architecture diagram.

## Results

The following summarizes the performance (accuracy) of the models on the MNIST test set:

| Model                  | Accuracy  |
| ---------------------- | --------- |
| Logistic Regression    | 92.6%     |
| Decision Tree          | 87.6%     |
| Random Forest          | 96.9%     |
| K-Nearest Neighbors    | 97.1%     |
| Neural Network         | 97.8%     |

Additional evaluation results are available in the form of confusion matrices and model accuracy bar charts, which are automatically generated when you run the evaluation script.

## Deployment

*(Deployment details will be added later if you choose to create an interactive demo or deploy a web application.)*

## Contributing

Contributions are welcome! If you have suggestions or improvements, please fork the repository and submit a pull request.

## Contact

For any questions or further discussion, please contact Krishna Balaji via [E-Mail](krishna311004@gmail.com) or visit my [GitHub profile](https://github.com/krishna31102004).

```
