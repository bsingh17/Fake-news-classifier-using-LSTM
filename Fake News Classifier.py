import pandas as pd

dataset=pd.read_csv('train.csv')

dataset=dataset.dropna()

x=dataset.drop(['label'],axis='columns')
y=dataset['label']

print(x.shape)

print(y.shape)

import tensorflow as tf

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import  Dense

voc_size=5000

messages=x.copy()
messages.reset_index(inplace=True)

import nltk
import re
import nltk.corpus
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stemmer=PorterStemmer()
corpus=[]

for i in range(0,len(messages)):
    review=re.sub('[^a-zA-Z]',' ',messages['title'][i])
    review=review.lower()
    review=review.split()
    review=[stemmer.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    
one_hot_representation=[one_hot(words,voc_size) for words in corpus]

sent_length=20
embedded_docs=pad_sequences(one_hot_representation,padding='pre',maxlen=sent_length)


embedded_feature_vectors=40
model=Sequential()
model.add(Embedding(voc_size,embedded_feature_vectors,input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

import numpy as np
x_final=np.array(embedded_docs)
y_final=np.array(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_final,y_final,test_size=0.33,random_state=42)

model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,batch_size=64)

y_predict=model.predict_classes(x_test)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_predict)
print(confusion)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_predict))

    
