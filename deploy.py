from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open('saved_model.sav', 'rb'))

@app.route('/')
def home():
    result = ''
    return render_template('index.html', result=result)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)