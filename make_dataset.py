import gensim
import json
import numpy as np
from tqdm import tqdm
import pickle

np.random.seed(0)
config = json.load(open('config.json','r'))
SEQ_LEN = config['SEQ_LEN']
train_units = config['train_units']
test_units = config['test_units']

# 文章のベクトル化
word2vec = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)

def class_vector(sentence):
    vectors_sum = np.zeros(300, dtype=np.float32)
    count = 0
    words = sentence.split(' ')
    for word in words:
        try:
            vectors_sum = vectors_sum + word2vec[word]
            count += 1
        except KeyError:
            pass
    vector = vectors_sum/count
    return vector

def vector(sentence,class_vector,SEQ_LEN):
    zeros = np.zeros(600, dtype=np.float32)
    words = sentence.split(' ')
    vectors = []
    for word in words:
        if len(vectors) < SEQ_LEN:
            try:
                vectors.append(np.append(word2vec[word],class_vector))
            except KeyError:
                pass
    if len(vectors) < SEQ_LEN:
        for _ in range(SEQ_LEN - len(vectors)):
                vectors.append(zeros)
    return vectors

# yahoo topic class
with open('../data/yahootopic/classes.txt','r',encoding='utf-8',errors='ignore')as f:
    yahoo_class = f.read().splitlines()
    yahoo_class_vector = [class_vector(text) for text in yahoo_class]

# traindata
# yahoo topic train_v0
with open('../data/yahootopic/train_pu_half_v0.txt','r',encoding='utf-8') as f:
    v0 = f.read().splitlines()
    v0 = v0[:65000] #消す
ite = 0
ite_range = int(len(v0)/5)
for i in range(5):
    x_train = []
    y_train = []
    for text in tqdm(v0[ite:ite+ite_range],total=len(range(ite_range))):
        text = text.split('\t')
        x_train.append(vector(text[1],yahoo_class_vector[int(text[0])],SEQ_LEN))
        y_train.append(1)
        rand_base = list(range(train_units))
        rand_base.remove(int(int(text[0])/2))
        rand = np.random.choice(rand_base)
        x_train.append(vector(text[1],yahoo_class_vector[rand*2],SEQ_LEN)) # 要確認
        y_train.append(0)

    fout = open('../dataset/train/x_train_'+str(i)+'.npy','wb')
    pickle.dump(x_train, fout, protocol=4)
    fout.close()
    fout = open('../dataset/train/y_train_'+str(i)+'.npy','wb')
    pickle.dump(y_train, fout, protocol=4)
    fout.close()
    ite += ite_range

# # yahoo topic test_v1
with open('../data/yahootopic/test.txt','r',encoding='utf-8') as f:
    yahoo_test = f.read().splitlines()
    yahoo_test_v1 = [y for y in yahoo_test if int(y[0])%2==1]
    print(np.array(yahoo_test_v1).shape)

yahoo_test_v1 = yahoo_test_v1[:5000]
ite = 0
ite_range = int(len(yahoo_test_v1)/5)
for j in range(5):
    x_test = []
    y_test = []
    for texts in tqdm(yahoo_test_v1[ite:ite+ite_range],total=len(range(ite_range))):
        text = texts.split('\t')
        for i in range(test_units):
            x_test.append(vector(text[1],yahoo_class_vector[i*2+1],SEQ_LEN))
            if i == int(int(text[0])/2):
                y_test.append(i)

    fout = open('../dataset/test/x_test_'+str(j)+'.npy','wb')
    pickle.dump(x_test, fout, protocol=4)
    fout.close()
    fout = open('../dataset/test/y_test_'+str(j)+'.npy','wb')
    pickle.dump(y_test, fout, protocol=4)
    fout.close()
    ite += ite_range
        
    
        








