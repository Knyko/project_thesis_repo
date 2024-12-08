import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

class neuralNetwork:
    def __init__(self, input_dim, learning_rate=0.01):
        """
        Initializes the neural network model.

        Parameters:
        - input_dim: int, the number of input features
        - learning_rate: float, the learning rate for the optimizer
        """
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.scaler = MinMaxScaler()
        self.model = self._build_model()

    def _build_model(self):
        """
        Builds and compiles the neural network model.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.input_dim,)),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(1)  # Output layer for regression
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse', metrics=['mae'])
        return model

    def preprocess_data(self, X_train, X_test):
        """
        Normalizes training and test data using StandardScaler.

        Parameters:
        - X_train: np.array, training feature set
        - X_test: np.array, testing feature set

        Returns:
        - Scaled X_train and X_test
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def train(self, X_train, y_train, validation_split=0.2, epochs=50, batch_size=64):
        """
        Trains the neural network model.

        Parameters:
        - X_train: np.array, training feature set
        - y_train: np.array, training target set
        - validation_split: float, proportion of training data used for validation
        - epochs: int, number of training epochs
        - batch_size: int, size of training batches

        Returns:
        - history: tf.keras.callbacks.History, training history object
        """
        history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        # Visualization of training history
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        return history

    def evaluate(self, X_test, y_test):
        """
        Evaluates the neural network model on the test set.

        Parameters:
        - X_test: np.array, testing feature set
        - y_test: np.array, testing target set

        Returns:
        - test_loss: float, loss on the test set
        - test_mae: float, mean absolute error on the test set
        """
        test_loss, test_mae = self.model.evaluate(X_test, y_test, verbose=1)
        return test_loss, test_mae

    def predict(self, X):
        """
        Generates predictions using the neural network model.

        Parameters:
        - X: np.array, feature set to predict on

        Returns:
        - predictions: np.array, predicted values
        """
        predictions = self.model.predict(X)
        return predictions
