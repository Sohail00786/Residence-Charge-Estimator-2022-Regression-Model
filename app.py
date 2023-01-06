import pandas as pd
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)
data = pd.read_csv("clean_data.csv")
pipe = pickle.load(open("RandomForestModel.pkl","rb"))

@app.route('/')
def index():
    city = sorted(data['City'].unique())
    furnish = sorted(data['Furnishing Status'].unique())
    at = sorted(data['Area Type'].unique())
    return render_template('index.html', city=city, furnish=furnish, at=at)


@app.route('/predict', methods=["POST"])
def predict():
    a = request.form.get("bhk")
    b = request.form.get("size")
    c = request.form.get("at")
    d = request.form.get("city")
    e = request.form.get("furnish")
    f = request.form.get("bath")
    g = request.form.get("area")
    
    
    print(a,b,c,d,e,f,g)
    input = pd.DataFrame([[a,b,c,d,e,f,g]], columns = ["BHK", "Size", "Area Type","City","Furnishing Status","Bathroom","area"])
    prediction = pipe.predict(input)[0]
    return str(prediction)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
