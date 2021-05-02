from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np
import keras_preprocessing
from keras_preprocessing import image
from PIL import Image
import datetime


# initialize our Flask application
app= Flask(__name__)

# server
@app.route("/", methods=["POST"])
def predict():
    model = tf.keras.models.load_model('./model.h5')
    # fileName = Image.open(request.files['file'])
    filePath = request.files['file']
    fileName = Image.open(filePath)
    theTime = datetime.datetime.now()
    realtime = theTime.strftime("%b %d, %Y \n%X")

    img = image.load_img(filePath, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)

    paper = [1,0,0]
    rock = [0,1,0]
    scissors = [0,0,1]

    if (classes[0] == paper).all():
        prediction = "paper"
    elif (classes[0] == rock).all():
        prediction = "rock"
    else :
        prediction = "scissor"

    statement = f"\nRock, Paper and Scissors image classification server.\nLeouel Guanzon\n{realtime}\n\nThe image you’ve submitted is classified as a: {prediction}.\n"

    return statement


@app.route("/", methods=["GET"])
def hello():
    return "<HTTML> <BODY> <STRONG> Hello </STRONG> World! </BODY> </HTML>"


if __name__=='__main__':
    app.run(debug=True)
