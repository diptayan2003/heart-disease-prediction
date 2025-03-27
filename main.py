import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

print("Loading model. Can take a bit...")
model = joblib.load("model.pkl")
print("model loaded and ready\n")


# Mapping function to convert user-friendly input to numerical values
def map_inputs(data):
    mappings = {
        "sex": {"Male": 1, "Female": 0},
        "fbs": {"≤120 mg/dL": 0, ">120 mg/dL": 1},
        "exang": {"No": 0, "Yes": 1},
        "slope": {"Upsloping": 0, "Flat": 1, "Downsloping": 2},
        "thal": {"Normal": 0, "Fixed Defect": 1, "Reversible Defect": 2},
    }

    return [
        float(data["age"]),
        mappings["sex"][data["sex"]],
        int(data["cp"]),
        float(data["trestbps"]),
        float(data["chol"]),
        mappings["fbs"][data["fbs"]],
        int(data["restecg"]),
        float(data["thalach"]),
        mappings["exang"][data["exang"]],
        float(data["oldpeak"]),
        mappings["slope"][data["slope"]],
        float(data["ca"]),
        mappings["thal"][data["thal"]],
    ]


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", data=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = map_inputs(request.form)
        print("input", input_data)

        # Convert input to numpy array and reshape
        input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

        # Prediction
        prediction = model.predict(input_data_as_numpy_array)

        # Guard clause, but probably wont get triggered.
        if len(prediction) == 0:
            print("prediction failed")
            return jsonify({"error": "failed to predict"}, 500)

        if prediction[0] == 0:
            return f'<p id="result" class="text-center text-green-600 text-lg font-semibold text-gray-700 mt-4">The Person does not have a Heart Disease</p>'
        else:
            return f'<p id="result" class="text-center text-red-600 text-lg font-semibold text-gray-700 mt-4">The Person has a Heart Disease</p>'

    except Exception as e:
        return jsonify({"error": str(e)}, 500)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)
