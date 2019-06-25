import keras
from keras.applications import InceptionV3,Xception,MobileNet,VGG16
from keras.layers import Activation,Add,AveragePooling1D,Dense,Dropout,GlobalAveragePooling2D,Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.models import Model
from keras.optimizers import Adam,SGD,Adadelta,RMSprop, Adagrad
from keras.utils import np_utils
import os
from keras.callbacks import ModelCheckpoint
import glob
import matplotlib.pyplot as plt




IM_WIDTH, IM_HEIGHT =229,229 #InceptionV3指定的图片尺寸
train_dir = 'train_test/train'  # 训练集数据
val_dir = 'train_test/test' # 验证集数据


nb_classes = 11
nb_epoches = int(5000)                # epoch数量
batch_size = int(8)





#　图片生成器
train_datagen =  ImageDataGenerator(height_shift_range=0.2,
                                    width_shift_range=0.2,
                                    rescale=1./255,
                                    vertical_flip=True,
                                    horizontal_flip=True)
#validation_split=0.3 可以通过设置validation_split只从train文件中提取训练集和验证集（1）
valid_datagen =  ImageDataGenerator(height_shift_range=0.2,
                                    width_shift_range=0.2,
                                    rescale=1./255,
                                    vertical_flip=True,
                                    horizontal_flip=True)


# In[7]:


# 训练数据与测试数据
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
    class_mode='categorical')#subset='training'，假如设置了（1），可以通过设置subset来分配训练集和验证集

valid_generator = valid_datagen.flow_from_directory(
    val_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
    class_mode='categorical')#subset='validation'

base_model = InceptionV3(weights='imagenet', include_top=False,input_shape=(IM_WIDTH,IM_HEIGHT,3)) # 预先要下载no_top模型

x = base_model.output
x = GlobalAveragePooling2D()(x) # GlobalAveragePooling2D 将 MxNxC 的张量转换成 1xC 张量，C是通道数
x = Dense(1024, activation='relu')(x)
predictions = Dense(nb_classes, activation='softmax')(x)
model = Model(inputs=base_model.input,outputs=predictions)

def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable =False#这里可以设置我们导入base_model哪些层可以训练，哪些层参数是固定的
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 定义网络框架
def setup_to_fine_tune(model,base_model):
    GAP_LAYER = 17 # max_pooling_2d_2
    for layer in base_model.layers[:GAP_LAYER+1]:
        layer.trainable = False
    for layer in base_model.layers[GAP_LAYER+1:]:
        layer.trainable = True
    model.compile(optimizer=Adagrad(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

setup_to_transfer_learn(model, base_model)
model.summary()

filepath="cloud_inp.h5"
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

setup_to_fine_tune(model,base_model)
history_ft = model.fit_generator(train_generator,
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



#https://blog.csdn.net/weixin_41972134/article/details/81985944 这个网址说明了loss变化情况来分析训练模型可能遇到的所有情况
#自己实验了InceptionV3,Xception,MobileNet,VGG16,Deasenet121等网络，发现InceptionV3，Xception,Deasenet121都会有过拟合现象，主要原因还是数据少
#轻量级网络MobileNet以及VGG16表现都不错

#我带学生做到的东西：
#（1）迁移学习
#（2）数据增广
#（3）从头搭建了自己的cnn
#（4）学生懂得在别人模型基础上加全连接层，增加模型表达能力，会Dropout等操作。

#存在的不足：
#开始的时候出现了严重的过拟合，现在通过使用参数量少的网络，效果有很好的提升。但也出现了验证集正确率高于训练集，我认为这种情况也可以理解

#可以改进的地方：
#（1）数据整理，因为每一类杂草的图片包含它生长的不同时期图片，这对网络来说是难点，整理图片后网络效果应该能表现的很好
#（2）可以试验的方案有很多，但目前因为时间原因，试验的网络比较少

