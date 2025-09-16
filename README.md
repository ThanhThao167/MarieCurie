
# 🤖 Chatbot Tư Vấn Tuyển Sinh Lớp 10 - THPT Marie Curie

Hệ thống chatbot giúp học sinh và phụ huynh tra cứu thông tin tuyển sinh lớp 10 một cách nhanh chóng và tiện lợi. Chatbot sử dụng kết hợp **truy xuất thông tin ngữ nghĩa (FAISS + SentenceTransformer)** và **mô hình ngôn ngữ GPT (RAG)** để trả lời thông minh.

---

## 📁 Cấu trúc dự án

```plaintext
├── RAG_chatbot.py            # API backend FastAPI xử lý truy vấn và gọi GPT
├── streamlit_chat.py         # Giao diện chatbot sử dụng Streamlit
├── MC_chatbot.csv            # Cơ sở tri thức: Câu hỏi và câu trả lời tuyển sinh
├── chat_history.csv          # Lưu lịch sử hội thoại người dùng
├── feedback.csv              # Phản hồi người dùng về độ hài lòng câu trả lời
├── requirements.txt          # Danh sách thư viện cần thiết
├── .env                      # Biến môi trường (OPENAI_API_KEY, ADMIN_PASSWORD)
├── README.md                 # Tài liệu hướng dẫn sử dụng
````

---

## ⚙️ Cài đặt & chạy local (gồm cả backend và frontend)

### 1️⃣ Cài đặt thư viện

```bash
# Tạo và kích hoạt môi trường ảo (tuỳ chọn)
python -m venv venv
# Windows: venv\Scripts\activate

# Cài đặt các thư viện
pip install -r requirements.txt
```

### 2️⃣ Thiết lập biến môi trường `.env`

Tạo file `.env` và thêm:

```env
OPENAI_API_KEY=your_openai_api_key
ADMIN_PASSWORD=admin123
```

> 🔑 Lấy API key tại: [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)

---

### 3️⃣ Chạy server FastAPI (API backend)

```bash
uvicorn RAG_chatbot:app --reload
```

API sẽ chạy tại: `http://localhost:8000/chat`

---

### 4️⃣ Chạy giao diện Streamlit (frontend)

```bash
streamlit run streamlit_chat.py
```

Giao diện chatbot sẽ xuất hiện tại `http://localhost:8501`

---

## 🧪 Tính năng chính

* 🤖 Chatbot trả lời bằng **tiếng Việt**, thân thiện, dễ hiểu.
* 🔎 Truy xuất thông tin bằng **FAISS + SentenceTransformer**, chính xác, dựa trên ngữ nghĩa.
* 💬 Sử dụng GPT (gpt-4o-mini) để sinh câu trả lời nếu không đủ tương đồng với cơ sở dữ liệu.
* 👍 Cho phép người dùng đánh giá hài lòng / không hài lòng với mỗi câu trả lời.
* 📊 Giao diện quản trị viên với thống kê câu hỏi, số lượt truy cập, câu hỏi ngoài dữ liệu.
* 📁 Lưu toàn bộ lịch sử hội thoại vào `chat_history.csv` để phân tích sau.
* 📝 Quản lý dữ liệu kiến thức `MC_chatbot.csv` và phản hồi `feedback.csv`.

---

## 🚀 Triển khai trên Streamlit Cloud (Chỉ frontend)

> 📝 Yêu cầu: FastAPI backend cần deploy riêng (Heroku, Railway...)

1. Push code lên GitHub
2. Truy cập [https://streamlit.io/cloud](https://streamlit.io/cloud) → Connect GitHub repo
3. Chọn file chính: `streamlit_chat.py`
4. Thiết lập biến môi trường:

   * `OPENAI_API_KEY`
   * `ADMIN_PASSWORD` (tùy chọn)


## 🆕 Phiên bản nâng cấp (2025)

* ✅ Chuyển từ TF-IDF sang FAISS + SentenceTransformer.
* ✅ Bổ sung tính năng feedback (like/unlike).
* ✅ Giới hạn GPT chỉ được dùng nếu không tìm thấy câu trả lời phù hợp.
* ✅ Giao diện Streamlit quản trị: hiển thị dữ liệu, thống kê, phản hồi.
* ✅ Giữ nguyên khả năng dịch từ tiếng Anh sang tiếng Việt và ngược lại.

---

## 👨‍💻 Tác giả & Đóng góp

Dự án được phát triển để hỗ trợ học sinh trong kỳ tuyển sinh lớp 10 tại **THPT Marie Curie**.

Mọi đóng góp, ý tưởng mở rộng hoặc phản hồi xin vui lòng gửi tại GitHub hoặc liên hệ qua email.

