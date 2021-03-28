import json
import pickle
import numpy as np
from keras.models import load_model
from sklearn import metrics

# パラメータ読み込み
config = json.load(open('config.json'))
BATCH_SIZE = config['BATCH_SIZE']
EPOCHS = config['EPOCHS']
SEQ_LEN = config['SEQ_LEN']
test_units = config['test_units']

# テストデータの読み込み
x_test = []
y_test = []
for i in range(5):
    fout = open('../dataset/test/x_test_'+str(i)+'.npy','rb')
    x_test = x_test + pickle.load(fout)
    fout.close()
    fout = open('../dataset/test/y_test_'+str(i)+'.npy','rb')
    y_test = y_test + pickle.load(fout)
    fout.close()
x_test = np.array(x_test)
y_test = np.array(y_test)

model = load_model('RNN+FC.h5')

y_pred = model.predict(x_test)
y_pred = np.split(y_pred,len(y_test))

y_pred_list = []
for y_p in y_pred:
    y_pred_list.append(np.argmax(y_p))

rep = metrics.classification_report(y_test,y_pred_list,digits=3)
print(rep)
with open('result.txt','w') as f:
    f.write(rep)