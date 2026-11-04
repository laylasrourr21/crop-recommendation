from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template('home.html')
    
@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = np.array([int_features])
    prediction = model.predict(final_features)
    output = prediction[0]
    return render_template('home.html', prediction_text=f"Recommended crop is: {output}")

if __name__ == '__main__':
    app.run(debug=True)
