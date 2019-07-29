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

os.environ['CUDA_VISIBLE_DEVICES'] = " " # 强制使用CPU
app = Flask(__name__)
cloud_dict = {'Ac': 0, 'As': 1, 'Cb': 2, 'Cc': 3, 'Ci': 4, 'Cs': 5, 'Ct': 6, 'Cu': 7, 'Ns': 8, 'Sc': 9, 'St': 10}
labels = dict((v, k) for k, v in cloud_dict.items())
description = {'Ci': 'Fibrous, white feathery clouds of ice crystals.',
               'Cs': 'Milky, translucent cloud veil of ice crystals.',
               'Cc': 'Fleecy cloud, cloud banks of small, white flakes.',
               'Ac': 'Grey cloud bundles, compound like rough fleecy cloud.',
               'As': 'Dense, gray layer cloud, often even and opaque.',
               'Cu': 'Heap clouds with flat bases in the middle or lower level.',
               'Cb': 'Middle or lower cloud level thundercloud.',
               'Ns': 'Rain cloud. Grey, dark layer cloud, indistinct outlines.',
               'Sc': 'Rollers or banks of compound dark gray layer cloud.',
               'St': 'Low layer cloud, causes fog or fine precipitation.',
               'Ct': 'Line-shaped clouds produced by aircraft engine.' }
name = {'Ci':'Cirrus',
        'Cs':'Cirrostratus',
        'Cc':'Cirrocumulus',
        'Ac':'Altocumulus',
        'As':'Altostratus',
        'Cu':'Cumulus',
        'Cb':'Cumulonimbus',
        'Ns':'Nimbostratus',
        'Sc':'Stratocumulus',
        'St' : 'Stratus',
        'Ct' : 'Contrails'}
weather = {
    'Cumulus' : 'All is Fair',
    'Cirrus' : 'All is Fair(For Now)',
    'Altocumulus': 'Warm With a Risk of Storms',
    'Cirrostratus' : 'Moisture Moving In',
    'Altostratus' : 'Expect Light Rain',
    'Stratus': 'Fog',
    'Cumulonimbus':'Severe Storms',
    'Nimbostratus':'Rain, Rain Go Away!',
    'Stratocumulus': 'weak convection in the atmosphere.',
    'Cirrocumulus':'it\'s cold but fair',
    'Contrails' : 'Caused by air traffic.'
}
def loadmodel():
    print('start load model')
    global model
    model_dir = 'CNN_SENet_50.h5'
    if os.path.exists(model_dir):
        print('yes it is')
        model = load_model(model_dir)
    else:
        print('it doesnt')
    global graph
    graph = tf.get_default_graph()

def cloud_test(datapath, model):
    img_width = 200
    img_height = 200
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
    cloud_abb = cloud_test(datapath, model)
    #cloud_abb = 'As'
    cloud_name = name[cloud_abb]
    cloud_des = description[cloud_abb]
    upcoming_weather = weather[cloud_name]
    data["cloud_name"] = cloud_name
    data['description'] = cloud_des
    data["weather"] = upcoming_weather
    file_url = "/static/photo/" + img.filename
    data["url"] = file_url
    print(file_path)
    print(data)
    return flask.jsonify(data)

print("start server")
loadmodel()
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=1234)