# from flask import Flask, request, jsonify
# import joblib
# import numpy as np

# # Load the saved model and scaler
# model = joblib.load('model.pkl')
# scaler = joblib.load('scaler.pkl')

# # Initialize Flask app
# app = Flask(__name__)

# # Home route
# @app.route('/')
# def index():
#     return "✅ Heart Disease Prediction API is Running (ML Model)."

# # Prediction route
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # JSON input
#         data = request.get_json(force=True)

#         # Extract features in the correct order
#         input_features = np.array([[
#             data['age'],
#             data['anaemia'],
#             data['creatinine_phosphokinase'],
#             data['diabetes'],
#             data['ejection_fraction'],
#             data['high_blood_pressure'],
#             data['platelets'],
#             data['serum_creatinine'],
#             data['serum_sodium'],
#             data['sex'],
#             data['smoking'],
#             data['time']
#         ]])

#         # Scale the input
#         input_scaled = scaler.transform(input_features)

#         # Predict using the model
#         prediction = model.predict(input_scaled)

#         # Convert output to readable form
#         result = "Death Expected 😟" if prediction[0] == 1 else "No Death Expected 🙂"

#         # Send response
#         return jsonify({
#             'prediction': int(prediction[0]),
#             'result': result
#         })

#     except Exception as e:
#         return jsonify({'error': str(e)})

# # Run the app
# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load ML model & scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        try:
            # Get form inputs in correct order
            data = [
                float(request.form['age']),
                int(request.form['anaemia']),
                float(request.form['cpk']),
                int(request.form['diabetes']),
                float(request.form['ef']),
                int(request.form['hbp']),
                float(request.form['platelets']),
                float(request.form['sc']),
                float(request.form['ss']),
                int(request.form['sex']),
                int(request.form['smoking']),
                float(request.form['time']),
            ]

            # Scale & predict
            input_scaled = scaler.transform([data])
            prediction = model.predict(input_scaled)[0]
            result = "Positive" if prediction == 1 else "Negative"


        except Exception as e:
            result = f"❌ Error: {str(e)}"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
