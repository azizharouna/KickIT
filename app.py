from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('randomforestclassifier_model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    
    if request.method == 'POST': 
        Auction_encoded = (request.form['Auction_encoded'])
        Make_encoded = (request.form['Make_encoded'])
        Model_encoded = (request.form['Model_encoded'])
        Trim_encoded = (request.form['Trim_encoded'])
        SubModel_encoded = (request.form['SubModel_encoded'])
        Color_encoded = (request.form['Color_encoded'])
        Transmission_encoded = (request.form['Transmission_encoded'])
        Nationality_encoded = (request.form['Nationality_encoded'])
        Size_encoded = (request.form['Size_encoded'])
        TopThreeAmericanName_encoded = (request.form['TopThreeAmericanName_encoded'])
        VNST_encoded = (request.form['VNST_encoded'])
        MMRAcquisitionAuctionAveragePrice = (request.form['MMRAcquisitionAuctionAveragePrice'])
        MMRAcquisitionAuctionCleanPrice = (request.form['MMRAcquisitionAuctionCleanPrice'])
        MMRAcquisitionRetailAveragePrice = (request.form['MMRAcquisitionRetailAveragePrice'])
        MMRAcquisitonRetailCleanPrice = (request.form['MMRAcquisitonRetailCleanPrice'])
        MMRCurrentAuctionAveragePrice = (request.form['MMRCurrentAuctionAveragePrice'])
        MMRCurrentAuctionCleanPrice = (request.form['MMRCurrentAuctionCleanPrice'])
        MMRCurrentRetailAveragePrice = (request.form['MMRCurrentRetailAveragePrice'])
        MMRCurrentRetailCleanPrice = (request.form['MMRCurrentRetailCleanPrice'])
        WarrantyCost = (request.form['WarrantyCost'])
        
        
        
prediction = model.predict([['Auction_encoded', 'Make_encoded', 'Model_encoded', 'Trim_encoded',
       'SubModel_encoded', 'Color_encoded', 'Transmission_encoded',
       'Nationality_encoded', 'Size_encoded', 'TopThreeAmericanName_encoded',
       'VNST_encoded', 'MMRAcquisitionAuctionAveragePrice',
       'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice',
       'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice',
       'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice',
       'MMRCurrentRetailCleanPrice', 'WarrantyCost']])
        if output<0:
            return render_template('index.html',prediction_texts="you can buy this car")
        else:
            return render_template('index.html',prediction_text="Sorry , It is a kick (bad Buy)  {}".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

