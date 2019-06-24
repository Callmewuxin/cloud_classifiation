import keras
from keras.models import Model
from keras.applications import resnet50
from keras.layers import Dropout, Conv2D, Dense, MaxPooling2D, Flatten, Input
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

train_data_dir = 'CCSN_dataset/train'
test_data_dir = 'CCSN_dataset/test'
batch_size = 32
epochs = 50
size = 227
input_data = Input(shape=[size, size, 3])
conv1 = Conv2D(filters=96, kernel_size=[11, 11], strides=[4, 4], activation='relu')(input_data)
maxpool1 = MaxPooling2D(pool_size=[3, 3], strides=[2, 2])(conv1)
conv2 = Conv2D(filters=256, kernel_size=[5, 5], padding='same', activation='relu')(maxpool1)
maxpool2 = MaxPooling2D(pool_size=[3, 3], strides=[2, 2])(conv2)
conv3 = Conv2D(filters=384, kernel_size=[3, 3], padding='same', activation='relu')(maxpool2)
conv4 = Conv2D(filters=256, kernel_size=[3, 3], padding='same', activation='relu')(conv3)
maxpool3 = MaxPooling2D(pool_size=[3, 3], strides=[2, 2])(conv4)
flatten = Flatten()(maxpool3)
drouput = Dropout(0.5)(flatten)
fc1 = Dense(units=4096, activation='relu')(drouput)
fc2 = Dense(units=11, activation='softmax')(fc1)


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

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size
#print(train_generator.n, validation_generator.n)
print(train_generator.class_indices)
print(validation_generator.class_indices)
'''
history = model.fit_generator(
    train_generator,
    epochs=epochs,
    steps_per_epoch= STEP_SIZE_TRAIN,
    validation_data= validation_generator,
    validation_steps= STEP_SIZE_VALID)

model.save('cloundNet.h5')
history_dict=history.history
loss=history_dict['loss']
val=history_dict['val_loss']
epochs=range(1,len(loss)+1)
plt.plot(epochs,loss,'bo',label='training loss')
plt.plot(epochs,val,'b',label='val loss')
plt.legend()
plt.show()
'''
