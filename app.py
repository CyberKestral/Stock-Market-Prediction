from flask import Flask, render_template, request
from model import train_model
from predictor import predict_price

app = Flask(__name__)
model, scaler = train_model()

@app.route("/", methods=["GET", "POST"])
def home():
    price = None
    if request.method == "POST":
        stock = request.form["stock"]
        price = predict_price(model, scaler, stock)
    return render_template("index.html", price=price)

if __name__ == "__main__":
    app.run(debug=True)
