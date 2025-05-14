import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import pickle
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec

# Load dữ liệu test
df_test = pd.read_csv('data/processed/test.csv')
df_test = df_test.dropna(subset=["text"])
X_test = df_test["text"].values
Y_test = df_test["sentiment"].values

# --- Naive Bayes ---
# Tải mô hình và vectorizer
with open('model/nb_model.pkl', 'rb') as f:
    nb_model = pickle.load(f)
with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Dự đoán
from naive_bayes_model import predict_naive_bayes
Y_pred_nb = predict_naive_bayes(X_test)
accuracy_nb = accuracy_score(Y_test, Y_pred_nb)
print(f"Độ chính xác của Naive Bayes trên tập test: {accuracy_nb:.4f}")

# --- RNN ---
# Tải mô hình và Word2Vec
rnn_model = load_model('model/rnn_model.h5')
word2vec_model = Word2Vec.load('model/word2vec_model.model')

# Dự đoán (chuyển nhãn test: -1 → 0, 0 → 1, 1 → 2)
from rnn_model import predict_rnn
Y_test_rnn = Y_test + 1
Y_pred_rnn = predict_rnn(X_test)
accuracy_rnn = accuracy_score(Y_test_rnn, Y_pred_rnn)
print(f"Độ chính xác của RNN trên tập test: {accuracy_rnn:.4f}")