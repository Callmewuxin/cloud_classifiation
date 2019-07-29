import keras
from keras.models import Model
from keras.applications import resnet50
from keras.layers import Dropout, Conv2D, Dense, MaxPooling2D, Flatten, Input
from keras.optimizers import RMSprop, SGD
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import os

train_data_dir = 'CCSN_dataset/train'
test_data_dir = 'CCSN_dataset/test'
batch_size = 8
nb_epoches = 20000
size = 227
input_data = Input(shape=[size, size, 3])
conv1 = Conv2D(filters=96, kernel_size=[11, 11], strides=[4, 4], activation='relu')(input_data)
maxpool1 = MaxPooling2D(pool_size=[3, 3], strides=[2, 2])(conv1)
conv2 = Conv2D(filters=256, kernel_size=[5, 5], padding='same', activation='relu')(maxpool1)
maxpool2 = MaxPooling2D(pool_size=[3, 3], strides=[2, 2])(conv2)
conv3 = Conv2D(filters=384, kernel_size=[3, 3], strides=[1, 1], padding='same', activation='relu')(maxpool2)
conv4 = Conv2D(filters=384, kernel_size=[3, 3], strides=[1, 1], padding='same', activation='relu')(conv3)
maxpool3 = MaxPooling2D(pool_size=[3, 3], strides=[2, 2])(conv4)
conv5 = Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], padding='same', activation='relu')(maxpool3)
flatten = Flatten()(conv5)
dropout1 = Dropout(0.5)(flatten)
fc1 = Dense(units=4096, activation='relu')(dropout1)
drouput = Dropout(0.5)(fc1)
fc2 = Dense(units=11, activation='softmax')(drouput)


model = Model(inputs=input_data, outputs=fc2)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(size, size),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(size, size),
    batch_size=batch_size,
    class_mode='categorical')
model.summary()
filepath="cloud_net.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# In[10]:
STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size
if os.path.exists(filepath):
    model.load_weights(filepath)
    # 若成功加载前面保存的参数，输出下列信息
    print("checkpoint_loaded")
# 模式一训练
history_tl = model.fit_generator(
train_generator,
nb_epoch=nb_epoches,
validation_data=validation_generator,
steps_per_epoch= STEP_SIZE_TRAIN,
validation_steps = STEP_SIZE_VALID,
class_weight='auto',
callbacks=callbacks_list)
fig,ax = plt.subplots(2,1,figsize=(10,10))
ax[0].plot(history_tl.history['loss'], color='r', label='Training Loss')
ax[0].plot(history_tl.history['val_loss'], color='g', label='Validation Loss')
ax[0].legend(loc='best',shadow=True)
ax[0].grid(True)

ax[1].plot(history_tl.history['acc'], color='r', label='Training Accuracy')
ax[1].plot(history_tl.history['val_acc'], color='g', label='Validation Accuracy')
ax[1].legend(loc='best',shadow=True)
ax[1].grid(True)
plt.savefig("cloud_net.png")
plt.show()
