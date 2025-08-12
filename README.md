# K-Nearest Neighbors (KNN) Project – Flower Species Prediction

## Overview

This project implements a **K-Nearest Neighbors (KNN) Classifier** to predict the species of a flower based on its sepal and petal measurements.

The model is trained using a custom dataset and deployed through a **Flask** web application, allowing users to input measurements and get instant predictions.

## Project Structure

```
DataScience/
│
├── KNN/
│   ├── data/
│   │   └── flower_species_dataset.csv
│   ├── model/
│   │   └── knn_model_and_scaler.pkl
│   ├── static/
│   │   └── style.css
│   ├── templates/
│   │   ├── index.html
│   │   └── result.html
│   ├── knn_model.py
│   ├── app.py
│   └── requirements.txt
```

## Installation & Setup

1.  **Clone the repository**

    ```
    git clone <your-repo-url>
    cd "DataScience/KNN"
    ```

2.  **Create a virtual environment (recommended)**

    ```
    python -m venv venv
    source venv/bin/activate   # For Linux/Mac
    venv\Scripts\activate      # For Windows
    ```

3.  **Install dependencies**

    ```
    pip install -r requirements.txt
    ```

## Dataset

The dataset contains details of flower species with the following features:

* **sepal_length** (numeric)
* **sepal_width** (numeric)
* **petal_length** (numeric)
* **petal_width** (numeric)
* **species** (Target: The species of the flower)

## Problem Statement

Accurately identifying the species of a flower is a common task in botany. This project aims to use a machine learning model to automate this process, providing a quick and reliable way to classify flowers based on their physical attributes.

## Why K-Nearest Neighbors?

* **Simplicity:** It is one of the simplest machine learning algorithms to understand and implement.
* **No Training Phase:** The algorithm learns by simply storing the dataset, making it ideal for smaller datasets.
* **Effective:** It can be very effective for classification tasks where the data has a clear structure.
* **Flexibility:** It can be used for both classification and regression.

## How to Run

1.  **Train the Model**

    ```
    python knn_model.py
    ```

    This will create:

    * `knn_model_and_scaler.pkl` (trained model and data scaler)

2.  **Run the Flask App**

    ```
    python app.py
    ```

    Visit `http://127.0.0.1:5000/` in your browser.

## Frontend Input Example

Example flower measurement input:

```
Sepal Length: 5.1
Sepal Width: 3.5
Petal Length: 1.4
Petal Width: 0.2
```

## Prediction Goal

The application predicts the species of the flower, for example: `Iris-setosa`.

## Tech Stack

* **Python** – Core programming language
* **Pandas & NumPy** – Data manipulation
* **Scikit-learn** – Machine learning model training
* **Flask** – Web framework for deployment
* **HTML/CSS** – Frontend UI design

## Future Scope

* Deploy the model on a cloud platform like Heroku or Render for public access.
* Add a visual representation of the prediction, such as a scatter plot showing the input point relative to the training data.
* Improve the model by testing different values of `k` (the number of neighbors).

## Screen Shots

**Home Page:**
<img width="1920" height="1080" alt="Screenshot (25)" src="https://github.com/user-attachments/assets/c5bf7a1b-49b0-436c-8a7a-7a8d85fde9df" />


**Result Page:**
<img width="1920" height="1080" alt="Screenshot (26)" src="https://github.com/user-attachments/assets/7eaf15ad-836b-49a9-a34a-b2f22177f3ed" />

