import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

# Define species mapping
species_mapping = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    # Retrieve measurements from the form
    Sepal_Length = float(request.form['Sepal_Length'])
    Sepal_Width = float(request.form['Sepal_Width'])
    Petal_Length = float(request.form['Petal_Length'])
    Petal_Width = float(request.form['Petal_Width'])

    # Make prediction using the model
    features = np.array([[Sepal_Length, Sepal_Width, Petal_Length, Petal_Width]])
    prediction_index = model.predict(features)[0]  # Get the predicted index
    predicted_species = species_mapping.get(prediction_index, "Unknown")

    # Prepare prediction text to display in the HTML
    prediction_text = f"The predicted species is {predicted_species}."

    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    flask_app.run(debug=True)
