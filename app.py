from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize the Flask application.
app = Flask(__name__)

# Load the trained KNN model and the data scaler from the saved file.
# The model and scaler were saved together in knn_model_and_scaler.pkl.
try:
    model_and_scaler = joblib.load('model/knn_model_and_scaler.pkl')
    knn_model = model_and_scaler['model']
    scaler = model_and_scaler['scaler']
except FileNotFoundError:
    print("Error: Model file 'model/knn_model_and_scaler.pkl' not found.")
    knn_model = None
    scaler = None
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    knn_model = None
    scaler = None

# Define the home page route.
@app.route('/')
def home():
    """Renders the main form for user input."""
    return render_template('index.html')

# Define the prediction route. This will handle the form submission.
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction logic. It retrieves the form data,
    processes it, and uses the KNN model to predict the species.
    """
    if knn_model is None or scaler is None:
        return "Model not loaded. Please check the server logs.", 500

    # Get data from the form.
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Create a NumPy array with the input features.
    # The reshape(-1, 1) is used to ensure the array has the correct shape for the scaler.
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Scale the input features using the same scaler used during training.
    scaled_features = scaler.transform(features)

    # Use the trained model to make a prediction.
    prediction = knn_model.predict(scaled_features)

    # Render the result template, passing the prediction to it.
    return render_template('result.html', prediction=prediction[0])

# Run the Flask application if the script is executed directly.
if __name__ == '__main__':
    # Use debug=True for development, which enables the reloader.
    app.run(debug=True)
