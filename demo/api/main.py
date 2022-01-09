from flask import Flask
from flask import render_template

app = Flask(__name__,
            static_url_path='',
            static_folder = "./public",
            template_folder = "./public")

@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route("/predict")
def predict():
    return render_template("index.html")

@app.route("/model_analysis")
def model_alanysis():
    return render_template("index.html")