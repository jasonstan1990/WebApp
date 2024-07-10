from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# Define a route to render the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    # Make prediction using the loaded model
    prediction = model.predict(final_features)

    # Map numerical prediction to class label
    species = {
        0: 'Setosa',
        1: 'Versicolour',
        2: 'Virginica'
    }
    
    # Get the predicted species name
    predicted_species = species[prediction[0]]

    # Return the result to the HTML page
    return render_template('index.html', prediction_text='Predicted class: {}'.format(predicted_species))

if __name__ == '__main__':
    app.run(debug=True)
