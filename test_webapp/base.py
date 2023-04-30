from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # get the input data from the user
    x = float(request.form['x'])

    # load the model
    model = LinearRegression()
    model.load('model.pkl')

    # make a prediction using the model
    y_pred = model.predict([[x]])

    # format the output
    output = f"The predicted value of y for x = {x} is {y_pred[0]:.2f}."

    return render_template('result.html', output=output)

if __name__ == '__main__':
    app.run(debug=True)
