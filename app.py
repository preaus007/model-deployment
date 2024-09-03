# Import the Flask class from the flask module
from flask import Flask, render_template, request
import pickle

# Create an instance of the Flask class
app = Flask(__name__)

# loading models
model = pickle.load(open("models/clf.pkl", "rb"))
tokenizer= pickle.load(open("models/cv.pkl", "rb"))

# Register a route
@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get('content-text')
    tokenized_text = tokenizer.transform([text])
    prediction = model.predict(tokenized_text)
    prediction = 1 if prediction == 1 else -1
    return render_template("index.html", prediction=prediction, text=text)


# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)