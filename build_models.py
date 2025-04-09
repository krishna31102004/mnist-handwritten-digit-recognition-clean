# build_models.py
import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

import matplotlib.pyplot as plt  # For plotting and later use in confusion matrices

# Import the plot_model utility for generating architecture diagrams
from tensorflow.keras.utils import plot_model

# Import the load_and_preprocess function from load_mnist.py
from load_mnist import load_and_preprocess

# Load the data
x_train, y_train, x_test, y_test, x_train_flat, x_test_flat = load_and_preprocess()
print("Data loaded and preprocessed successfully.")

# ----------------------------
# Build and Train Traditional ML Models (scikit-learn)
# ----------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(x_train_flat, y_train)
y_pred_log = log_reg.predict(x_test_flat)
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

# Decision Tree
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(x_train_flat, y_train)
y_pred_dt = dtree.predict(x_test_flat)
print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# Random Forest
rforest = RandomForestClassifier(n_estimators=100, random_state=42)
rforest.fit(x_train_flat, y_train)
y_pred_rf = rforest.predict(x_test_flat)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train_flat, y_train)
y_pred_knn = knn.predict(x_test_flat)
print("\nKNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# ----------------------------
# Build and Train a Simple Neural Network (TensorFlow/Keras)
# ----------------------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

nn_model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
nn_model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
history = nn_model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
nn_loss, nn_accuracy = nn_model.evaluate(x_test, y_test)
print("\nNeural Network Test Accuracy:", nn_accuracy)

# Generate Neural Network Architecture Diagram and save to file
plot_model(nn_model, to_file='nn_architecture.png', show_shapes=True, show_layer_names=True)
print("Neural network architecture diagram saved to 'nn_architecture.png'.")

# ----------------------------
# Compare and Visualize Model Accuracies
# ----------------------------
import pandas as pd

results = {
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'KNN', 'Neural Network'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_log),
        accuracy_score(y_test, y_pred_dt),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_knn),
        nn_accuracy
    ]
}
df_results = pd.DataFrame(results)
print("\nComparison of Models:")
print(df_results)

plt.figure(figsize=(10, 6))
plt.bar(df_results['Model'], df_results['Accuracy'], color='skyblue')
plt.ylim([0, 1])
plt.ylabel("Accuracy")
plt.title("Model Accuracies on MNIST")
plt.xticks(rotation=45)
plt.show()

# ----------------------------
# Evaluate with Confusion Matrices
# ----------------------------
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

plot_confusion_matrix(y_test, y_pred_log, "Confusion Matrix: Logistic Regression")
plot_confusion_matrix(y_test, y_pred_dt, "Confusion Matrix: Decision Tree")
plot_confusion_matrix(y_test, y_pred_rf, "Confusion Matrix: Random Forest")
plot_confusion_matrix(y_test, y_pred_knn, "Confusion Matrix: KNN")

# For the neural network, compute the predicted class labels
import numpy as np
nn_probs = nn_model.predict(x_test)
y_pred_nn = np.argmax(nn_probs, axis=1)
plot_confusion_matrix(y_test, y_pred_nn, "Confusion Matrix: Neural Network")
