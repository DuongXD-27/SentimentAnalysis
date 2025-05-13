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

df = pd.read_csv('data/processed/train.csv')
df = df.dropna(subset=["text"])  #Drop NaN rows: vì khi làm khảo sát thì có thể lấy các câu mà người dùng không điền gì
X=df["text"]
Y_train=df["sentiment"]

matranhoa= CountVectorizer()
X_train=matranhoa.fit_transform(X) # lấy tất cả các từ khác nhau từ X train để tạo nên 1 ma trận tần suất

model=MultinomialNB()
model.fit(X_train,Y_train)

def process(text):
    text = text.lower()
    words = word_tokenize(text)
    words=[word for word in words if word not in stopwords.words('english') and word.isalnum()]  # word.isalnum(): nếu là số hoặc là kí tự alphabet thì trả về true nếu là dấu hoặc kí tự đặc biệt thì false
    return " ".join(words)

def polarity(text):
    text=process(text)
    text_new=matranhoa.transform([text])    
    return model.predict(text_new)[0] 

text_input=input()
print(polarity(text_input))

#  Đánh giá Accuracy
df1 = pd.read_csv('data/processed/test.csv')
df1 = df1.dropna(subset=["text"])  
X_test = df1["text"]
Y_test = df1["sentiment"]

X_test_vec = matranhoa.transform(X_test)  
Y_pred = model.predict(X_test_vec)  
correct_test = sum(Y_pred == Y_test)  
accuracy = correct_test / len(X_test)  

print("the accuracy of the model is:", accuracy)
