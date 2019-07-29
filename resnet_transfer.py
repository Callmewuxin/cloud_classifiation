import keras
from keras.applications import InceptionV3,Xception,MobileNet,VGG16, ResNet50
from keras.layers import Activation,Add,AveragePooling1D,Dense,Dropout,GlobalAveragePooling2D,Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.models import Model, Sequential
from keras.optimizers import Adam,SGD,Adadelta,RMSprop, Adagrad
from keras.utils import np_utils
import os
from keras.callbacks import ModelCheckpoint
import glob
import numpy as np
import matplotlib.pyplot as plt

seed = 7
np.random.seed(seed)
IM_WIDTH, IM_HEIGHT =197,197 #InceptionV3指定的图片尺寸
train_dir = 'train_test/train'  # 训练集数据
val_dir = 'train_test/test' # 验证集数据


nb_classes = 11
nb_epoches = int(5000)                # epoch数量
batch_size = int(8)


def ResNet50_model(lr=0.001, decay=1e-6, momentum=0.9, nb_classes=11, img_rows=197, img_cols=197, RGB=True,
                  ):
    color = 3 if RGB else 1
    base_model = ResNet50(weights='imagenet', include_top=False, pooling=None, input_shape=(img_rows, img_cols, color),
                          classes=nb_classes)

    # 冻结base_model所有层，这样就可以正确获得bottleneck特征
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    # 添加自己的全链接分类层
    x = Flatten()(x)
    # x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)

    # 训练模型
    model = Model(inputs=base_model.input, outputs=predictions)
    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])



    return model




#　图片生成器
datagen =  ImageDataGenerator(height_shift_range=0.2,
                                    width_shift_range=0.2,
                                    rescale=1./255,
                                    vertical_flip=True,
                                    horizontal_flip=True,
                                    validation_split=0.3,
                                    )


# In[7]:


# 训练数据与测试数据
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')#subset='training'，假如设置了（1），可以通过设置subset来分配训练集和验证集

valid_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')#subset='validation'
model = ResNet50_model()
filepath="cloud_resnet.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# In[10]:
STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size
if os.path.exists(filepath):
    model.load_weights(filepath)
    # 若成功加载前面保存的参数，输出下列信息
    print("checkpoint_loaded")
# 模式一训练
history_tl = model.fit_generator(
train_generator,
nb_epoch=nb_epoches,
validation_data=valid_generator,
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
plt.savefig("result+"+str(nb_epoches)+".png")
plt.show()
