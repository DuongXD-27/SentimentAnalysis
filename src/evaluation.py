import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import pickle
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec
from naive_bayes_model import predict_naive_bayes

# Hàm chuyển văn bản thành vector cho RNN
def texts_to_vectors(texts, model, max_len):
    vector_sequences = []
    for text in texts:
        words = text.split()
        vectors = [model.wv[word] for word in words if word in model.wv]
        if len(vectors) < max_len:
            vectors += [np.zeros(model.vector_size)] * (max_len - len(vectors))
        else:
            vectors = vectors[:max_len]
        vector_sequences.append(np.array(vectors))
    return np.array(vector_sequences)

# Hàm tính độ chính xác cho cả hai mô hình
def compute_accuracies(X_test, Y_test, nb_model, vectorizer, rnn_model, word2vec_model, max_length=20):
    # --- Naive Bayes ---
    Y_pred_nb = predict_naive_bayes(X_test)
    accuracy_nb = accuracy_score(Y_test, Y_pred_nb)
    
    # --- RNN ---
    Y_test_rnn = Y_test + 1  # Chuyển nhãn: -1 → 0, 0 → 1, 1 → 2
    X_vectors = texts_to_vectors(X_test, word2vec_model, max_length)
    Y_pred_onehot = rnn_model.predict(X_vectors)
    Y_pred_rnn = np.argmax(Y_pred_onehot, axis=1)
    accuracy_rnn = accuracy_score(Y_test_rnn, Y_pred_rnn)
    
    return accuracy_nb, accuracy_rnn

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

# --- RNN ---
# Tải mô hình và Word2Vec
rnn_model = load_model('model/rnn_model.h5')
rnn_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
word2vec_model = Word2Vec.load('model/word2vec_model.model')

# Tính độ chính xác
accuracy_nb, accuracy_rnn = compute_accuracies(X_test, Y_test, nb_model, vectorizer, rnn_model, word2vec_model)
print(f"Độ chính xác của Naive Bayes trên tập test: {accuracy_nb:.4f}")
print(f"Độ chính xác của RNN trên tập test: {accuracy_rnn:.4f}")