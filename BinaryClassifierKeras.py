# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import learning_curve

# Load the dataset
data = pd.read_csv("/Users/unmesh/PycharmProjects/BreastCancerANN/breast-cancer.csv")


# Check for duplicate values
duplicate_rows = data.duplicated()
if any(duplicate_rows):
    print("Duplicate values found. Removing duplicates.")
    data = data[~duplicate_rows]

# Check for null values
null_values = data.isnull().sum()
if any(null_values):
    print("Null values found. Removing rows with null values.")
    data = data.dropna()

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


# Build a Keras deep learning model
def create_model():
    model = Sequential()
    model.add(Dense(256, activation="relu", input_dim=X_train.shape[1]))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# Wrap the Keras model in a scikit-learn classifier
keras_classifier = KerasClassifier(model=create_model, epochs=100, batch_size=100, verbose=0)

# Use cross_val_score for cross-validation
cv_scores = cross_val_score(keras_classifier, X_train, y_train, cv=10)
print("Cross-Validation Scores:")
for i, score in enumerate(cv_scores, 1):
    print(f"Fold {i}: {score:.4f}")
print("\nMean Cross-Validation Score:\n", cv_scores.mean())

# Fit the model on the full training set
keras_classifier.fit(X_train, y_train)

# Get decision scores on the test set
y_pred_proba = keras_classifier.predict(X_test)  # Use predict instead of predict_proba
threshold = 0.5  # Experiment with different thresholds
y_pred = (y_pred_proba > threshold).astype(int)

# Print confusion matrix and classification report
print("\nConfusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", metrics.classification_report(y_test, y_pred))

# Print accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
# Print accuracy as a percentage
accuracy_percentage = accuracy * 100
print("Accuracy: {:.2f}%\n".format(accuracy_percentage))

# Confusion Matrix Visualization
cm = metrics.confusion_matrix(y_test, y_pred)
class_labels = ["benign", "malignant"]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# Save the confusion matrix plot to a file (e.g., PNG or PDF)
plt.savefig('confusion_matrix.png')  # Change the path as needed

# Display the plot
plt.show()

# Calculate classification error rate
error_rate = 1 - accuracy
print("Classification Error Rate: {:.2%}\n".format(error_rate))

# Extract values from confusion matrix
tn, fp, fn, tp = cm.ravel()

# Calculate False Positive Rate (FPR) and True Positive Rate (TPR)
fpr = fp / (fp + tn)
tpr = tp / (tp + fn)

print("False Positive Rate (FPR): {:.4f}".format(fpr))
print("True Positive Rate (TPR): {:.4f}".format(tpr))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')

# Save ROC curve plot
plt.savefig('roc_curve.png')  # Change the path as needed

# Display the plot
plt.show()

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    keras_classifier, X_train, y_train, cv=10, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)

# Calculate mean and standard deviation for training and test sets
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plot learning curve
plt.figure(figsize=(8, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.title("Learning Curve")
plt.legend(loc="best")

# Save Learning Curve plot
plt.savefig('learning_curve.png')  # Change the path as needed

# Display the plot
plt.show()
