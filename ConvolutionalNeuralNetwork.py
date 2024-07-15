import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import metrics
from scikeras.wrappers import KerasClassifier

# Load the dataset
data = pd.read_csv("breast-cancer.csv")

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

# Reshape the data for CNN
X_train_reshaped = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1], 1)
X_test_reshaped = X_test.values.reshape(X_test.shape[0], 1, X_test.shape[1], 1)

# Convert labels to one-hot encoding
y_train_one_hot = np.eye(2)[y_train]
y_test_one_hot = np.eye(2)[y_test]


# Define the function to create the CNN model
def create_cnn_model(hidden_layers=1, epochs=10, batch_size=64):
    input_shape = (1, X_train.shape[1], 1)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(1, 3), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Flatten())

    # Add hidden layers based on the specified parameter
    for _ in range(hidden_layers):
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Wrap the Keras model in a scikit-learn-compatible class
class KerasClassifierWrapper(KerasClassifier):
    def __init__(self, build_fn, **kwargs):
        super().__init__(build_fn=build_fn, **kwargs)

    def score(self, X, y, **kwargs):
        y_one_hot = np.eye(2)[y]
        return super().score(X, y_one_hot, **kwargs)


# Instantiate the wrapped Keras model
cnn_model_tune = KerasClassifierWrapper(build_fn=create_cnn_model)

# Define the hyperparameters to tune
param_grid = {
    'hidden_layers': [1, 2, 3],
    'epochs': [10, 20, 30],
    'batch_size': [32, 64, 128]
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=cnn_model_tune, param_grid=param_grid, cv=3)
grid_result = grid_search.fit(X_train_reshaped, y_train)

# Print the best hyperparameters
print("Best Hyperparameters: ", grid_result.best_params_)

# Get the best model
best_cnn_model = grid_result.best_estimator_

# Evaluate the best model on the test set
y_pred_proba_best = best_cnn_model.predict(X_test_reshaped)
y_pred_best = np.argmax(y_pred_proba_best, axis=1)

# Print confusion matrix and classification report for the best model
print("\nConfusion Matrix (Best Model):\n", metrics.confusion_matrix(y_test, y_pred_best))
print("\nClassification Report (Best Model):\n",
      metrics.classification_report(y_test, y_pred_best))
