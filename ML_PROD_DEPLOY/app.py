import pickle

import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
with open("models/random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract features from the form
        features = [float(x) for x in request.form.values()]

        # Define feature names matching the training data
        feature_names = [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ]

        # Create DataFrame with feature names
        final_features = pd.DataFrame([features], columns=feature_names)

        # Make prediction
        prediction = model.predict(final_features)[0]

        # Render the result
        return render_template(
            "index.html", prediction_text=f"Predicted Class: {prediction}"
        )
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    print(" App is running on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
