import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("/Users/unmesh/PycharmProjects/BreastCancerANN/breast-cancer.csv")

# Extract features (X) and labels (y)
X = data.iloc[:, 2:]  # Features from the 3rd column onwards
y = data['diagnosis']  # Target variable

# Convert diagnosis labels to binary (0 for benign, 1 for malignant)
y = (y == 'M').astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Use the best parameters found from hyperparameter tuning
best_params = {'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (256, 128, 64, 32, 16, 8, 4), 'max_iter': 4000}

# Build the model with the best parameters
best_model = MLPClassifier(**best_params)

# Train the model
best_model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = best_model.predict(X_test)

# Print confusion matrix and classification report
print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", metrics.classification_report(y_test, y_pred))

# Print accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion Matrix Visualization
cm = metrics.confusion_matrix(y_test, y_pred)
class_labels = ["benign", "malignant"]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Extract misclassified examples (optional)
misclassified_indices = y_test != y_pred
misclassified_X = X_test[misclassified_indices]
misclassified_y_true = y_test[misclassified_indices]
misclassified_y_pred = y_pred[misclassified_indices]

# Display details of misclassified examples
for index, (true_label, predicted_label) in enumerate(zip(misclassified_y_true, misclassified_y_pred)):
    print(f"Instance {index + 1}:")
    print(f"True Label: {true_label}, Predicted Label: {predicted_label}")

    # Print features of the misclassified instance
    misclassified_instance = misclassified_X[index, :]
    print("Features:")
    print(misclassified_instance)

    # Visualize the features or any other relevant information
    # Example: You can create a bar plot for features
    plt.bar(range(len(misclassified_instance)), misclassified_instance)
    plt.title(f'Misclassified Instance {index + 1} Features')
    plt.show()

    print("\n" + "-" * 40 + "\n")
