from flask import Flask, jsonify, render_template, request
import joblib
import os
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def result():
    item_identifier = float(request.form['item_identifier'])
    item_weight = float(request.form['item_weight'])
    item_fat_content = float(request.form['item_fat_content'])
    item_visibility = float(request.form['item_visibility'])
    item_type = float(request.form['item_type'])
    item_mrp = float(request.form['item_mrp'])
    outlet_size = float(request.form['outlet_size'])
    outlet_location_type = float(request.form['outlet_location_type'])
    outlet_type = float(request.form['outlet_type'])
    working_year = float(request.form['working_year'])

    X = np.array([[item_identifier, item_weight, item_fat_content, item_visibility, item_type, item_mrp,
                   outlet_size, outlet_location_type, outlet_type, working_year]])

    scaler_path = r'Notebook/sc1.sav'
    sc = joblib.load(scaler_path)

    X_std = sc.transform(X)

    model_path = r'Notebook/regressor.sav'
    model = joblib.load(model_path)

    Y_pred = model.predict(X_std)
    print(Y_pred)

    return render_template("prediction.html", prediction=Y_pred)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
