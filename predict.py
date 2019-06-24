from __future__ import print_function
from flask import Flask, request
import flask
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.datasets import mnist
from keras import layers
import numpy as np
from keras.models import load_model
from six.moves import range
from PIL import Image

app = Flask(__name__)
cloud_dict = {'Ac': 0, 'As': 1, 'Cb': 2, 'Cc': 3, 'Ci': 4, 'Cs': 5, 'Ct': 6, 'Cu': 7, 'Ns': 8, 'Sc': 9, 'St': 10}
labels = dict((v, k) for k, v in cloud_dict.items())
descriptipn = {'Ci': 'Cirrus Fibrous, white feathery clouds of ice crystals.',
               'Cs': 'Cirrostratus Milky, translucent cloud veil of ice crystals.',
               'Cc': '268 Cirrocumulus Fleecy cloud, cloud banks of small, white flakes.',
               'Ac': 'Altocumulus Grey cloud bundles, compound like rough fleecy cloud.',
               'As': 'Altostratus Dense, gray layer cloud, often even and opaque.',
               'Cu': 'Cumulus Heap clouds with flat bases in the middle or lower level.',
               'Cb': 'Cumulonimbus Middle or lower cloud level thundercloud.',
               'Ns': 'Nimbostratus Rain cloud. Grey, dark layer cloud, indistinct outlines.',
               'Sc': 'Stratocumulus Rollers or banks of compound dark gray layer cloud.',
               'St': 'Stratus Low layer cloud, causes fog or fine precipitation.',
               'Ct': 'Contrails Line-shaped clouds produced by aircraft engine.' }


def loadmodel():
    print('start load model')
    global model
    model_dir = 'cloudnet.h5'
    if os.path.exists(model_dir):
        print('yes it is')
        model = load_model(model_dir)
    else:
        print('it doesnt')
    global graph
    graph = tf.get_default_graph()

def cloud_test(datapath, model):
    img_width = 227
    img_height = 227
    datadir = datapath
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = test_datagen.flow_from_directory(
        datadir,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode='categorical')
    with graph.as_default():
        preds = model.predict_generator(validation_generator, steps=1)
        predicted_class_indices = np.argmax(preds, axis=1)
        prediction = labels[predicted_class_indices[0]]
    return prediction

@app.route('/', methods=['GET', 'POST'])
@app.route('/index')
def index():
    return flask.render_template("render.html")

@app.route('/photo', methods=['GET', 'POST'])
def photo():
    data = dict()
    img = request.files.get('file')
    path = "static/photo/"
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
    file_path = os.path.join(path + img.filename)
    datapath = "static"
    img.save(file_path)
    # data["prediction"] = cloud_test(datapath, model)
    # data['descrition'] = descriptipn[data['prediction']]
    data["prediction"] = 'as'
    file_url = "/static/photo/" + img.filename
    data["url"] = file_url
    print(file_path)
    print(data)
    return flask.jsonify(data)

print("start server")
#loadmodel()
if __name__ == "__main__":
    app.run()