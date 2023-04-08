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

app = Flask(__name__)

#compute the histogram of every image to extract featues
def ExtractFeatures(image):
	# compute the histogram of oriented gradients feature vector for
	# the input image
	features = feature.hog(image, orientations=9,
		pixels_per_cell=(10, 10), cells_per_block=(2, 2),
		transform_sqrt=True, block_norm="L1")
	# return the feature vector
	return features




#collect features along with its data
def InitializeDataAndLabels(path):
	
	# grab the list of images in the input directory, then initialize
	# the list of data (i.e., images) and class labels
	imagePaths = list(paths.list_images(path))
	data = []
	labels = []
	# loop over the image paths [healthy,parkinson]
	for imagePath in imagePaths:
		# extract the class label from the filename
		label = imagePath.split(os.path.sep)[-2]
		# load the input image, convert it to grayscale, and resize
		# it to 200x200 pixels, ignoring aspect ratio
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.resize(image, (200, 200))
		# threshold the image such that the drawing appears as white
		# on a black background
		image = cv2.threshold(image, 0, 255,
			cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		# quantify the image
		features = ExtractFeatures(image)
		# update the data and labels lists, respectively
		data.append(features)
		labels.append(label)
	# return the data and labels
	return (np.array(data), np.array(labels))




# define the path to the training and testing directories
trainingPath = r"D:\dataset\wave\training"


#training the data
(trainX, trainY) = InitializeDataAndLabels(trainingPath)

# encode the labels as integers
le = LabelEncoder()
trainY = le.fit_transform(trainY)



def Train():
    for i in range(0,5):
        model = RandomForestClassifier(n_estimators=100)
        model.fit(trainX,trainY)
        return model
        




Model = Train()
def Predict(image):
    results = []
	# load the testing image, clone it, and resize it
	# pre-process the image in the same manner we did earlier
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))
    image = cv2.threshold(image, 0, 255,
		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	
    features = ExtractFeatures(image)
    preds = Model.predict([features])
    label = le.inverse_transform(preds)[0]
    results.append(label)
    if("parkinson" in results):
     return "parkinson"
    else:
     return "healthy"
    

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


