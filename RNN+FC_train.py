import pickle
import json
import numpy as np
from keras.layers import Input, LSTM, Dense
from keras import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt

config = json.load(open('config.json','r'))
SEQ_LEN = config['SEQ_LEN']
input_dim = config['input_dim']
train_units = config['train_units']
BATCH_SIZE = config['BATCH_SIZE']
EPOCHS =config['EPOCHS']

x_train = []
y_train = []

for i in range(5):
    fout = open('../dataset/train/x_train_'+str(i)+'.npy','rb')
    x_train = x_train + pickle.load(fout)
    fout.close()
    fout = open('../dataset/train/y_train_'+str(i)+'.npy','rb')
    y_train = y_train + pickle.load(fout)
    fout.close()

x_train = np.array(x_train)
y_train = np.array(y_train)

inputs = Input(shape=(SEQ_LEN,input_dim),dtype='float32')
lstm = LSTM(300)(inputs)
output = Dense(units=1,activation='softmax')(lstm)
model = Model(inputs,output)
model.compile(optimizer=Adam(beta_1=0.9,beta_2=0.999),loss='binary_crossentropy',metrics=['acc'])
model.summary()

result = model.fit(x_train,y_train,batch_size=BATCH_SIZE,epochs=EPOCHS)

plt.plot(range(1,EPOCHS+1), result.history['acc'], label='acc')
plt.plot(range(1,EPOCHS+1), result.history['loss'], label='loss')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend()
plt.savefig('plt.jpg')

model.save('RNN+FC.h5')