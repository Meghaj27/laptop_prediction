from flask import Flask, request, render_template_string
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the SVC model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVC Model Prediction</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        body {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #f8fafc;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 16px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
            width: 100%;
            max-width: 520px;
            padding: 32px;
        }
        h2 {
            font-size: 1.75rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 8px;
            color: #38bdf8;
        }
        p.subtitle {
            text-align: center;
            color: #94a3b8;
            font-size: 0.95rem;
            margin-bottom: 28px;
        }
        .form-group {
            margin-bottom: 18px;
        }
        label {
            display: block;
            font-size: 0.875rem;
            font-weight: 600;
            color: #cbd5e1;
            margin-bottom: 6px;
        }
        input, select {
            width: 100%;
            padding: 12px 14px;
            background-color: #0f172a;
            border: 1px solid #475569;
            border-radius: 8px;
            color: #ffffff;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        input:focus, select:focus {
            border-color: #38bdf8;
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.2);
        }
        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%);
            border: none;
            border-radius: 8px;
            color: #ffffff;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            margin-top: 10px;
            transition: opacity 0.2s, transform 0.1s;
        }
        button:hover {
            opacity: 0.92;
        }
        button:active {
            transform: scale(0.98);
        }
        .result-box {
            margin-top: 24px;
            padding: 16px;
            border-radius: 8px;
            text-align: center;
            font-size: 1.1rem;
            font-weight: 600;
            background-color: #0f172a;
            border: 1px solid #38bdf8;
            color: #38bdf8;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Classification Predictor</h2>
        <p class="subtitle">Enter customer demographics to generate a prediction</p>
        
        <form action="/predict" method="POST">
            <div class="form-group">
                <label for="age">Age</label>
                <input type="number" id="age" name="age" step="any" placeholder="e.g. 35" required>
            </div>
            
            <div class="form-group">
                <label for="gender">Gender</label>
                <input type="number" id="gender" name="gender" step="any" placeholder="e.g. 0 (Female) or 1 (Male)" required>
            </div>
            
            <div class="form-group">
                <label for="region">Region</label>
                <input type="number" id="region" name="region" step="any" placeholder="e.g. 1, 2, 3" required>
            </div>
            
            <div class="form-group">
                <label for="occupation">Occupation</label>
                <input type="number" id="occupation" name="occupation" step="any" placeholder="e.g. 0, 1, 2" required>
            </div>
            
            <div class="form-group">
                <label for="income">Income</label>
                <input type="number" id="income" name="income" step="any" placeholder="e.g. 50000" required>
            </div>
            
            <button type="submit">Predict Outcome</button>
        </form>

        {% if prediction_text %}
            <div class="result-box">
                {{ prediction_text }}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features matching: ['Age', 'Gender', 'Region', 'Occupation', 'Income']
        features = [
            float(request.form['age']),
            float(request.form['gender']),
            float(request.form['region']),
            float(request.form['occupation']),
            float(request.form['income'])
        ]
        
        input_data = np.array([features])
        prediction = model.predict(input_data)[0]
        
        result_text = f"Prediction Result: {prediction.upper()}"
        return render_template_string(HTML_TEMPLATE, prediction_text=result_text)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
