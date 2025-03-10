from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2
from flask_cors import CORS, cross_origin

app = Flask(__name__)

CORS(app)

model_path = './model/model.keras'
model = tf.keras.models.load_model(model_path)

def categorizar(url):
    respuesta = requests.get(url)
    img = Image.open(BytesIO(respuesta.content))
    
    img = np.array(img).astype(float)/255
    
    img = cv2.resize(img, (224, 224))
    
    img = img.reshape(1, 224, 224, 3)
    
    prediccion = model.predict(img)
    
    return np.argmax(prediccion[0], axis=-1)

@cross_origin
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    url = data['url']
    
    prediction = categorizar(url)
    
    # return jsonify({'prediction': int(prediction)})
    if int(prediction) == 0:
        return jsonify({'prediction': 'headset'})
    elif int(prediction) == 1:
        return jsonify({'prediction': 'keyboard'})
    elif int(prediction) == 2:
        return jsonify({'prediction': 'mouse'})

# @app.route('/', methods=['GET'])
# def hello_world():
#     return render_template('index.html')

if __name__ == '__main__':
    # app.run()
    # run_with_ngrok(app)
    app.run(port=4040, debug=True)
    # app.run(debug=True)
