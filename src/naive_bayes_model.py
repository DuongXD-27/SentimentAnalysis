import numpy as np
import pandas as pd
import math as m
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pickle

df = pd.read_csv('data/processed/train.csv')
df = df.dropna(subset=["text"])  #Drop NaN rows: vì khi làm khảo sát thì có thể lấy các câu mà người dùng không điền gì
X=df["text"]
Y_train=df["sentiment"]

matranhoa= CountVectorizer()
X_train=matranhoa.fit_transform(X) # lấy tất cả các từ khác nhau từ X train để tạo nên 1 ma trận tần suất

model=MultinomialNB()
model.fit(X_train,Y_train)

with open('model/nb_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(matranhoa, f)
    
def process(text):
    text = text.lower()
    words = word_tokenize(text)
    words=[word for word in words if word not in stopwords.words('english') and word.isalnum()]  # word.isalnum(): nếu là số hoặc là kí tự alphabet thì trả về true nếu là dấu hoặc kí tự đặc biệt thì false
    return " ".join(words)

def polarity(text):
    text=process(text)
    text_new=matranhoa.transform([text])    
    return model.predict(text_new)[0]

def predict_naive_bayes(texts):
    texts_processed = [process(text) for text in texts]
    X_vec = matranhoa.transform(texts_processed)
    return model.predict(X_vec)


