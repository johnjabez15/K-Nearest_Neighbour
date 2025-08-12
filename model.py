import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

def train_knn_model():
    """
    This function loads the dataset, trains a K-Nearest Neighbors (KNN)
    classifier, evaluates its performance, and saves the trained model
    and the data scaler to a file.
    """
    try:
        # Load the dataset from the specified path.
        print("Loading dataset...")
        df = pd.read_csv('dataset/flower_species_dataset.csv')

        # Separate features (X) and target variable (y).
        # We will use sepal_length, sepal_width, petal_length, and petal_width as features.
        X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        y = df['species']

        # Split the data into training and testing sets.
        # This allocates 80% of the data for training and 20% for testing.
        print("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # It's crucial to scale the data for KNN because it relies on distances.
        # We'll initialize a scaler and fit it on the training data.
        print("Scaling the features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Initialize the KNN classifier.
        # We'll start with n_neighbors=5, a common default value.
        print("Initializing and training the KNN model...")
        knn = KNeighborsClassifier(n_neighbors=5)

        # Train the model using the scaled training data.
        knn.fit(X_train_scaled, y_train)

        # Make predictions on the scaled test data.
        y_pred = knn.predict(X_test_scaled)

        # Evaluate the model's accuracy.
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy on the test set: {accuracy * 100:.2f}%")

        # Save the trained model and the scaler to disk for later use.
        # We will save them as a single dictionary for convenience.
        model_and_scaler = {
            'model': knn,
            'scaler': scaler
        }
        joblib.dump(model_and_scaler, 'model/knn_model_and_scaler.pkl')
        print("Model and scaler saved successfully to 'model/knn_model_and_scaler.pkl'")

    except FileNotFoundError:
        print("Error: The file 'data/flower_species_dataset.csv' was not found.")
        print("Please ensure the dataset file is in the correct location.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# This ensures the function is called when the script is run directly.
if __name__ == "__main__":
    train_knn_model()
