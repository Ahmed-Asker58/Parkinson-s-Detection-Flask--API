from flask import Flask, request, render_template, url_for, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage import feature
from imutils import paths
import numpy as np
import cv2
import os
from PIL import Image
import pickle
from TrainWave import  ExtractFeatures 


app = Flask(__name__)



file = open('object.pkl', 'rb')
le = pickle.load(file)
file.close()

Wave_Model_Loaded = pickle.load(open('finalized_model.sav', 'rb'))

def Predict(image):
    results = []
	# load the testing image, clone it, and resize it
	# pre-process the image in the same manner we did earlier
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))
    image = cv2.threshold(image, 0, 255,
		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	
    features = ExtractFeatures(image)
    preds = Wave_Model_Loaded.predict([features])
    label = le.inverse_transform(preds)[0]
    results.append(label)
    if("parkinson" in results):
     return "parkinson"
    else:
     return "healthy"
    

#preprossing the image sent from the client-side
def preprossing(image):
    image=Image.open(image)
    image_arr = np.array(image.convert('RGB'))
    return image_arr


@app.route('/')
def index():
    return render_template('index.html', appName="Parkinson's Detection")



@app.route('/predictApi', methods=["POST"])
def api():
    # Get the image from post request
    try:
        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('fileup')
        image = preprossing(image)
        print("Model predicting ...")
        result = Predict(image)
        print("Model predicted")
        print(result)
        return jsonify({'prediction': result})
    except:
        return jsonify({'Error': 'Error occur'})
    



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("run code")
    if request.method == 'POST':
        # Get the image from post request
        print("image loading....")
        image = request.files['fileup']
        print("image loaded....")
        image = preprossing(image)
        result = Predict(image)
        print("predicted ...")
        print(result)

        return render_template('index.html', prediction=result, appName="Parkinson's Detection")
    else:
        return render_template('index.html',appName="Parkinson's Detection")









if __name__ == '__main__':
    app.run(debug=True)


