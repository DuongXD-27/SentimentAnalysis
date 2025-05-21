# Phân tích tình cảm (SentimentAnalysis)

Dự án này thực hiện phân tích tình cảm trên dữ liệu văn bản sử dụng các mô hình học máy, bao gồm Naive Bayes và Mạng nơ-ron hồi quy (RNN).

## Cấu trúc thư mục

```
SentimentAnalysis/
├── src/
│   ├── data_preprocessing.py   # Script tiền xử lý dữ liệu
│   ├── naive_bayes_model.py    # Script huấn luyện và lưu mô hình Naive Bayes
│   ├── rnn_model.py            # Script huấn luyện và lưu mô hình RNN
│   └── evaluation.py           # Script đánh giá các mô hình đã huấn luyện
├── requirements.txt            # Danh sách các thư viện Python cần thiết
└── README.md                   # Tệp này
```

## Yêu cầu

*   Python 3.x
*   Các thư viện được liệt kê trong `requirements.txt`

## Cài đặt

1.  Clone repository này (nếu bạn chưa làm).
2.  Cài đặt các thư viện cần thiết:
    ```bash
    pip install -r requirements.txt
    ```
3.  Tải xuống tài nguyên `punkt` của NLTK (nếu chưa có). Mở một trình thông dịch Python và chạy:
    ```python
    import nltk
    nltk.download('punkt')
    # nltk.download('punkt_tab') # Dường như bạn đã tải punkt_tab, nhưng 'punkt' phổ biến hơn.
    ```

## Cách sử dụng

Các script chính nằm trong thư mục `src/`.

1.  **Tiền xử lý dữ liệu:**
    Chạy script này để chuẩn bị dữ liệu cho việc huấn luyện mô hình.
    ```bash
    python src/data_preprocessing.py
    ```

2.  **Huấn luyện mô hình Naive Bayes:**
    Chạy script này để huấn luyện mô hình Naive Bayes và lưu lại.
    ```bash
    python src/naive_bayes_model.py
    ```

3.  **Huấn luyện mô hình RNN:**
    Chạy script này để huấn luyện mô hình RNN và lưu lại.
    ```bash
    python src/rnn_model.py
    ```

4.  **Đánh giá mô hình:**
    Sau khi đã huấn luyện các mô hình, chạy script này để đánh giá hiệu suất của chúng trên tập dữ liệu kiểm tra.
    ```bash
    python src/evaluation.py
    ```
