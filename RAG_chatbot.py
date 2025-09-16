import os
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from deep_translator import GoogleTranslator

# ==== 1. Môi trường & mô hình ====
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
logger = logging.getLogger("RAG_Chatbot")
logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

model = SentenceTransformer("all-MiniLM-L6-v2")

class ChatPayload(BaseModel):
    session_id: str
    messages: List[dict]

# ==== 2. Load dữ liệu ====
qa_df = pd.read_csv("MC_chatbot.csv", encoding="utf-8")
questions = qa_df["cauhoi"].tolist()
answers = qa_df["cautraloi"].tolist()

embeddings = model.encode(questions, convert_to_numpy=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# ==== 3. Tiện ích ====
def save_chat(session_id: str, role: str, content: str):
    df = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "role": role,
        "content": content
    }])
    df.to_csv("chat_history.csv", mode="a", index=False, header=not os.path.exists("chat_history.csv"), encoding="utf-8")

def save_feedback(session_id: str, question: str, answer: str, rating: str):
    df = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "rating": rating
    }])
    df.to_csv("feedback.csv", mode="a", index=False, header=not os.path.exists("feedback.csv"), encoding="utf-8")

def detect_language(text: str):
    # Cải thiện: Kiểm tra thêm từ khóa để chính xác hơn
    en_keywords = ["what", "how", "where", "when", "school", "admission", "who", "principal", "gold", "weather", "tomorrow"]
    return "en" if any(w in text.lower() for w in en_keywords) else "vi"

# Dict để dịch ngày sang tiếng Việt
DAY_MAP = {
    'Monday': 'Hai',
    'Tuesday': 'Ba',
    'Wednesday': 'Tư',
    'Thursday': 'Năm',
    'Friday': 'Sáu',
    'Saturday': 'Bảy',
    'Sunday': 'Chủ Nhật'
}

def get_date_info(days_offset=0):
    # Tính ngày dựa trên offset (0: hôm nay, 1: ngày mai, -1: hôm qua)
    now = datetime.now() + timedelta(days=days_offset)
    day_en = now.strftime("%A")
    day_vi = DAY_MAP.get(day_en, day_en)
    prefix = "Hôm nay" if days_offset == 0 else "Ngày mai" if days_offset == 1 else "Hôm qua" if days_offset == -1 else f"Ngày {now.day} tháng {now.month} năm {now.year}"
    return f"{prefix} là thứ {day_vi}, ngày {now.day} tháng {now.month} năm {now.year}."

# ==== 4. Truy xuất context ====
def retrieve_context(query: str, top_k=3):
    q_embed = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_embed, top_k)
    contexts = [
        f"Q: {questions[i]}\nA: {answers[i]}\n"
        for idx, i in enumerate(I[0])
        if D[0][idx] < 1.0
    ]
    best_score = 1.0 - D[0][0] if len(D[0]) > 0 else 0.0
    return contexts, best_score

# ==== 5. API ====
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat_handler(payload: ChatPayload, request: Request):
    user_input = payload.messages[-1]["content"]
    session_id = payload.session_id
    user_lang = detect_language(user_input)

    try:
        translated_input = GoogleTranslator(source="auto", target="vi").translate(user_input) if user_lang == "en" else user_input
    except:
        translated_input = user_input

    context_chunks, similarity = retrieve_context(translated_input)
    logger.info(f"🔍 FAISS similarity = {similarity:.2f}")
    save_chat(session_id, "user", user_input)

    # ==== Xử lý đặc biệt cho câu hỏi về thời gian: Làm chính xác hơn ====
    lower_input = translated_input.lower()
    if "hôm nay" in lower_input or "hiện tại" in lower_input or "thứ mấy hôm nay" in lower_input:
        reply = get_date_info(0)  # Hôm nay
        if user_lang == "en":
            reply = GoogleTranslator(source="vi", target="en").translate(reply)
        save_chat(session_id, "assistant", reply)
        return {
            "response": reply,
            "source": "real_time",
            "similarity": float(round(similarity, 2))
        }
    elif "ngày mai" in lower_input or "mai là thứ mấy" in lower_input:
        reply = get_date_info(1)  # Ngày mai
        if user_lang == "en":
            reply = GoogleTranslator(source="vi", target="en").translate(reply)
        save_chat(session_id, "assistant", reply)
        return {
            "response": reply,
            "source": "real_time",
            "similarity": float(round(similarity, 2))
        }
    elif "hôm qua" in lower_input or "qua là thứ mấy" in lower_input:
        reply = get_date_info(-1)  # Hôm qua
        if user_lang == "en":
            reply = GoogleTranslator(source="vi", target="en").translate(reply)
        save_chat(session_id, "assistant", reply)
        return {
            "response": reply,
            "source": "real_time",
            "similarity": float(round(similarity, 2))
        }

    # ==== 6. Trả lời từ kiến thức nội bộ nếu đủ tương đồng ====
    if similarity >= 0.85 and context_chunks:  # Hạ threshold xuống 0.85 để linh hoạt hơn
        top_answer = context_chunks[0].split("A:", 1)[-1].strip()
        reply = GoogleTranslator(source="vi", target="en").translate(top_answer) if user_lang == "en" else top_answer
        save_chat(session_id, "assistant", reply)
        return {
            "response": reply,
            "source": "knowledge_base",
            "similarity": float(round(similarity, 2))
        }

    # ==== 7. Fallback GPT: Xử lý mọi câu hỏi còn lại, prompt linh hoạt hơn như AI tool thông thường ====
    current_date_info = get_date_info(0)  # Luôn cung cấp ngày hôm nay cho GPT
    prompt = f"{current_date_info}\n\nBạn là một trợ lý AI thông minh, thân thiện, giống như ChatGPT hoặc Grok. Luôn sẵn sàng trả lời bất kỳ câu hỏi nào từ người dùng, dù là về giáo dục, thời gian, giá vàng, thời tiết, kiến thức phổ thông, toán học, lịch sử, hoặc bất kỳ chủ đề gì. \nTrả lời một cách chính xác, đầy đủ, hữu ích và không từ chối yêu cầu nếu có thể. Sử dụng thông tin ngày hiện tại để tính toán nếu cần (ví dụ: nếu hỏi ngày mai, tính từ hôm nay).\nNếu câu hỏi bằng tiếng Anh, trả lời bằng tiếng Anh; nếu bằng tiếng Việt, trả lời bằng tiếng Việt.\n\n" + ("\n\n".join(context_chunks) if context_chunks else "")

    messages = [{"role": "system", "content": prompt}] + payload.messages[-3:]

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        reply = completion.choices[0].message.content
        # Không cần dịch nữa vì prompt yêu cầu GPT xử lý ngôn ngữ trực tiếp
    except Exception as e:
        logger.error(f"❌ GPT error: {e}")
        reply = "Xin lỗi, hệ thống đang gặp sự cố khi tạo câu trả lời." if user_lang == "vi" else "Sorry, the system is experiencing an issue generating the response."

    save_chat(session_id, "assistant", reply)
    return {
        "response": reply,
        "source": "rag_gpt",
        "similarity": float(round(similarity, 2))
    }

# ==== 8. Feedback ====
@app.post("/feedback")
async def feedback(
    session_id: str = Form(...),
    question: str = Form(...),
    answer: str = Form(...),
    rating: str = Form(...)
):
    save_feedback(session_id, question, answer, rating)
    return {"status": "ok", "message": "Đã ghi nhận phản hồi."}