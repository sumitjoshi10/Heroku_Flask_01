import pickle
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

with open("model.pkl","rb") as f:
    model = pickle.load(f)
    
@app.route("/")
def home():
    # return "hello"
    return render_template("index.html")

@app.route("/predict",methods = ["POST"])
def predict():
    values = [int(i) for i in request.form.values()]
    X = [np.array(values)]
    predited_salary = model.predict(X)
    
    output = f"Epoyee Salary should be around ${round(predited_salary[0],2)}"
   
    return render_template("index.html",prediction_text = output)

if __name__ == "__main__":
    app.run(debug=True)
