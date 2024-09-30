from flask import Flask, request, jsonify, render_template
import joblib 
import numpy as np 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd

#initilaize the flask app
app = Flask(__name__)
 
FEATURE_NAMES =  ['Customer ID','Product Category','Purchase Amount','Customer Age','Customer Gender',
'Store Location','Month','Day']

#load model

lr_model = joblib.load("models/linear_regression_model.pkl")
# rf_model = joblib.load("models/random_forest_regression_model.pkl")
# knn_model = joblib.load("models/knn_regression_model.pkl")

#Defiene for API
@app.route("/")
def home():
    # return render_template("index.html")
    return render_template("index.html")

# ROUTE FOR MAKING PREDICTION 
@app.route('/predict', methods = ['POST'])
def predict():
    try:
        #Get input data fro the request
        data = request.get_json()
        if 'features' not in data:
            raise ValueError("No features provided in the request.")
        features = np.array(data['features']).reshape(1,-1) #reshape to become a list for the mode to use

        #predictions from ecah model 
        #  Predictions from each model 
        lr_prediction = lr_model.predict(features)[0]
        # rf_prediction = rf_model.predict(features)[0]
        # knn_prediction = knn_model.predict(features)[0]

        # return the prediction as JSON
        return jsonify({
            'Linear regression': lr_prediction,
            # 'Random forest': rf_prediction,
            # 'KNN ': knn_prediction
       })
    except Exception as  e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
    # app.run(debug=True)

    
