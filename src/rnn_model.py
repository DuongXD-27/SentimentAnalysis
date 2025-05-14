import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Masking
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import pickle

# 1. Load và xử lý dữ liệu
df = pd.read_csv('data/processed/train.csv')
df = df.dropna(subset=["text"])

X = df["text"].values
Y = df["sentiment"].values

# Chuyển nhãn: -1 → 0, 0 → 1, 1 → 2
Y_shifted = Y + 1
Y_onehot = to_categorical(Y_shifted)

# Chia tập train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_shifted, test_size=0.1, random_state=42)

# Chuẩn bị mô hình Word2Vec
tokenized_sentences = [text.split() for text in X_train]
word2vec_model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1,
    epochs=10
)

# Hàm chuyển văn bản thành vector
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

# Padding theo độ dài câu dài nhất trong tập train
max_length = 20
X_train_vectors = texts_to_vectors(X_train, word2vec_model, max_length)
X_test_vectors = texts_to_vectors(X_test, word2vec_model, max_length)

# One-hot cho Y_train
Y_train_onehot = to_categorical(Y_train)

# Xây dựng mô hình
def build_rnn_model(input_shape, num_classes):
    model = Sequential([
        Masking(mask_value=0.0, input_shape=input_shape),
        SimpleRNN(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def predict_rnn(texts, max_length=20):
    X_vectors = texts_to_vectors(texts, word2vec_model, max_length)
    Y_pred_onehot = model.predict(X_vectors)
    return np.argmax(Y_pred_onehot, axis=1)
model = build_rnn_model((max_length, 100), 3)

model.fit(X_train_vectors, Y_train_onehot, epochs=20, batch_size=32, verbose=1)

model.save('model/rnn_model.h5')
word2vec_model.save('model/word2vec_model.model')

