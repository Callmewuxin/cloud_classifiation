import test
from test.models import Model
from test.layers import Dropout, Conv2D, Dense, MaxPooling2D, Flatten, Input
from test.optimizers import RMSprop
import matplotlib.pyplot as plt
from test.callbacks import ModelCheckpoint
import input_data



mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
input_data = Input(784)
fc1 = Dense(10, activation='softmax')(input_data)
model = Model(inputs=input_data, outputs=fc1)
model.compile(optimizer=RMSprop(lr=0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history=model.fit(mnist.train.images, mnist.train.labels,validation_data=(mnist.test.images, mnist.test.labels),batch_size=100,epochs=1000, verbose=1)
history_dict=history.history
loss=history_dict['loss']
val=history_dict['val_loss']
epochs=range(1,len(loss)+1)
plt.plot(epochs,loss,'bo',label='training loss')
plt.plot(epochs,val,'b',label='val loss')
plt.legend()
plt.show()