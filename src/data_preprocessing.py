import json
import os
import pandas as pd
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Tải danh sách stop words và công cụ tokenize
nltk.download('stopwords')
nltk.download('punkt')

def load_dataset(*src_filenames, labels=None):
    data = []
    for filename in src_filenames:
        with open(filename) as f:
            for line in f:
                d = json.loads(line)
                if labels is None or d['gold_label'] in labels:
                    data.append(d)
    return data

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    # Tokenize văn bản và chuyển thành chữ thường
    tokens = word_tokenize(text.lower())
    # Chỉ giữ các token là chữ hoặc số (loại bỏ dấu câu) và không phải stop words
    filtered_tokens = [word for word in tokens if word not in stop_words and word.isalnum()]
    return ' '.join(filtered_tokens)

def split_data(data, train_size, test_size, labels):
    # Convert to DataFrame
    df = pd.DataFrame([(d['sentence'], d['gold_label']) for d in data], columns=['text', 'sentiment'])
    
    # Filter only specified labels
    df = df[df['sentiment'].isin(labels)]

    # Áp dụng tiền xử lý: loại bỏ stop words, dấu câu, và chuyển thành chữ thường
    df['text'] = df['text'].apply(preprocess_text)

    # Shuffle data
    df = shuffle(df, random_state=3160)

    # Initialize lists to store balanced data
    train_data = []
    test_data = []

    # Calculate samples per label
    samples_per_label_train = train_size // len(labels)
    samples_per_label_test = test_size // len(labels)

    # Split data for each label
    for label in labels:
        label_data = df[df['sentiment'] == label]
        train_data.append(label_data[:samples_per_label_train])
        test_data.append(label_data[samples_per_label_train:samples_per_label_train + samples_per_label_test])

    # Concatenate and shuffle
    train_df = shuffle(pd.concat(train_data, ignore_index=True), random_state=3160)
    test_df = shuffle(pd.concat(test_data, ignore_index=True), random_state=3160)
    
    # Map sentiment labels to numerical values
    sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
    train_df['sentiment'] = train_df['sentiment'].map(sentiment_mapping)
    test_df['sentiment'] = test_df['sentiment'].map(sentiment_mapping)
    
    return train_df, test_df

def main():
    # Define paths
    raw_data_dir = os.path.join('data', 'raw')
    processed_data_dir = os.path.join('data', 'processed')

    # Create processed data directory if it doesn't exist
    os.makedirs(processed_data_dir, exist_ok=True)

    # File paths
    train_file = os.path.join(raw_data_dir, 'dynasent_dataset.jsonl')

    # Labels
    ternary_labels = ('positive', 'negative', 'neutral')

    # Load all data
    data = load_dataset(train_file, labels=ternary_labels)

    # Balance and split data
    train_size = 90000  # 3000 samples per label
    test_size = 3000   # 333-334 samples per label
    train_df, test_df = split_data(data, train_size, test_size, ternary_labels)

    # Save to CSV
    train_df.to_csv(os.path.join(processed_data_dir, 'train.csv'), index=False, encoding='utf-8')
    test_df.to_csv(os.path.join(processed_data_dir, 'test.csv'), index=False, encoding='utf-8')

if __name__ == "__main__":
    main()