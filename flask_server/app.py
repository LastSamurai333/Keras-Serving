import base64
import json
from io import BytesIO

import numpy as np
import requests
from flask import Flask, request, jsonify
from keras.preprocessing import image

# from flask_cors import CORS

app = Flask(__name__)


# Uncomment this line if you are making a Cross domain request
# CORS(app)

# Testing URL
@app.route('/hello/', methods=['GET', 'POST'])
def hello_world():
    return 'Hello, World!'


@app.route('/imageclassifier/predict/', methods=['POST'])
def image_classifier():
    # Decoding and pre-processing base64 image
    #img = image.img_to_array(image.load_img(BytesIO(base64.b64decode(request.form['b64'])), target_size=(224, 224))) / 255.
    image_path = BytesIO(base64.b64decode(request.form['b64']))
    img=Image.open(image_path)
    myimg=img.resize((224,224), resample=Image.BILINEAR )
    img = np.asarray(myimg)

    # this line is added because of a bug in tf_serving < 1.11
    img = img.astype('float16')

    # Creating payload for TensorFlow serving request
    payload = {
        "instances": [{'input_image': img.tolist()}]
    }

    # Making POST request
    r = requests.post('http://localhost:1234/v1/models/ImageClassifier:predict', json=payload)

    # Decoding results from TensorFlow Serving server
    pred = json.loads(r.content.decode('utf-8'))
    
    dict = {}
    dict['Sleeveless'] = pred['predictions'][0][0]
    dict['FullSleeve'] = pred['predictions'][0][1]
    dict['HalfSleeve'] = pred['predictions'][0][2]
    dict['3/4 Sleeve'] = pred['predictions'][0][3]
    y = json.dumps(dict)

    # Returning JSON response to the frontend
    #return jsonify(inception_v3.decode_predictions(np.array(pred['predictions']))[0])
    #return y
    return jsonify(dict)
