from flask import Flask, render_template, request
import joblib
import numpy as np



app = Flask(__name__)

# Load the trained model
model = joblib.load('C:\\Users\\simpl\\Downloads\\salary_prediction_model.joblib')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        experience = float(request.form['experience'])
        previous_ctc = float(request.form['previous_ctc'])
        grad_marks = float(request.form['grad_marks'])

        # Make a prediction using the loaded model
        prediction = model.predict([[experience, previous_ctc, grad_marks]])

        return render_template('result.html', prediction=f"${prediction[0]:,.2f}")

if __name__ == '__main__':
    app.run(debug=True)
