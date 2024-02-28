import base64
import numpy as np
import cv2
import os
from flask import Flask, render_template, request, jsonify
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math

detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

app = Flask(__name__)

@app.route('/home.html')
def home():
    return render_template('home.html')

@app.route('/asl.html')
def asl():
    return render_template('asl.html')

@app.route('/video.html')
def video():
    return render_template('video.html')

@app.route('/image.html')
def image():
    return render_template('image.html')

@app.route('/sign', methods=['POST'])
def sign():
    if request.method == 'POST':
        print("YESSSSSS")

        # Parse the JSON data from the request body
        data = request.get_json()

        # Access the base64-encoded image data
        img_data = data.get('imgData')

        img_data = img_data.split(',')[1]
        # Decode the base64 image data
        img_bytes = base64.b64decode(img_data)

        # Convert the bytes to a NumPy array
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)

        # Decode the NumPy array as an image
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            result = {'message': labels[index]}

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            result = {'message': labels[index]}

        # Now you can perform recognition on the "received_image.jpg" file
        # For example, you can use OpenCV or other libraries for recognition tasks

        print("Image Data saved as received_image.jpg")


        # Return a response, if needed
        return jsonify(result), 200

if __name__ == '__main__':
    app.run(debug=True)