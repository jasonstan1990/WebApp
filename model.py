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
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction_index = model.predict(features)[0]  # Get the predicted index
    predicted_species = species_mapping.get(prediction_index, "Unknown")
    return render_template("index.html", prediction_text=f"The flower species is {predicted_species}")

if __name__ == "__main__":
    flask_app.run(debug=True)
