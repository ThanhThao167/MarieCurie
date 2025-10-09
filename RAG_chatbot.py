# RAG_chatbot.py - OPTIMIZED VERSION
# FastAPI backend cho Chatbot Tuyển sinh 10 – THPT Marie Curie (Cloud-ready)
# Tối ưu: Giảm top_k, ngưỡng similarity, context window GPT

import os
import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from openai import OpenAI

# =========================
# 0) ENV & Logging
# =========================
load_dotenv()
logger = logging.getLogger("RAG_Chatbot")
logging.basicConfig(level=logging.INFO)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("⚠️ OPENAI_API_KEY chưa được đặt – fallback GPT sẽ lỗi.")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Thư mục lưu dữ liệu CSV (có thể đổi qua ENV: DATA_DIR), mặc định là root
DATA_DIR = Path(os.getenv("DATA_DIR", "."))
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHAT_CSV = DATA_DIR / "chat_history.csv"
FEED_CSV = DATA_DIR / "feedback.csv"
KB_CSV = DATA_DIR / "MC_chatbot.csv"   # nguồn tri thức

# =========================
# 1) App & CORS
# =========================
app = FastAPI(title="Marie Curie RAG Chatbot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# =========================
# 2) Pydantic payload
# =========================
class ChatPayload(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]  # [{"role":"user/assistant","content":"..."}]

# =========================
# 3) Load Knowledge Base + FAISS
# =========================
def _load_kb(kb_path: Path) -> tuple[list[str], list[str]]:
    if not kb_path.exists():
        logger.warning(f"⚠️ Không tìm thấy {kb_path}. RAG sẽ hoạt động hạn chế.")
        return [], []
    df = pd.read_csv(kb_path, encoding="utf-8")
    # bắt buộc có 2 cột 'cauhoi' và 'cautraloi'
    if not {"cauhoi", "cautraloi"}.issubset(df.columns):
        raise RuntimeError("File MC_chatbot.csv cần có cột 'cauhoi' và 'cautraloi'.")
    return df["cauhoi"].tolist(), df["cautraloi"].tolist()

questions, answers = _load_kb(KB_CSV)

# Model & FAISS index
logger.info("📦 Đang load SentenceTransformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
if questions:
    logger.info(f"🔨 Đang tạo FAISS index cho {len(questions)} câu hỏi...")
    embeddings = model.encode(questions, convert_to_numpy=True, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    logger.info("✅ FAISS index đã sẵn sàng!")
else:
    embeddings = None
    index = None
    logger.warning("⚠️ Không có câu hỏi nào trong KB!")

# =========================
# 4) CSV helpers
# =========================
def _append_csv(path: Path, fieldnames: list[str], row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})

def _read_csv_dicts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def save_chat(session_id: str, role: str, content: str):
    _append_csv(
        CHAT_CSV,
        ["timestamp", "session_id", "role", "content"],
        {"timestamp": datetime.utcnow().isoformat(), "session_id": session_id, "role": role, "content": content},
    )

def save_feedback(session_id: str, question: str, answer: str, rating: str):
    _append_csv(
        FEED_CSV,
        ["timestamp", "session_id", "question", "answer", "rating"],
        {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "rating": rating,
        },
    )

# =========================
# 5) Tiện ích ngôn ngữ & ngày tháng
# =========================
def detect_language(text: str) -> str:
    en_keywords = ["what", "how", "where", "when", "school", "admission", "who", "principal", "gold", "weather", "tomorrow"]
    return "en" if any(w in text.lower() for w in en_keywords) else "vi"

DAY_MAP = {
    "Monday": "Hai", "Tuesday": "Ba", "Wednesday": "Tư",
    "Thursday": "Năm", "Friday": "Sáu", "Saturday": "Bảy", "Sunday": "Chủ Nhật"
}

def get_date_info(days_offset=0) -> str:
    now = datetime.now() + timedelta(days=days_offset)
    day_vi = DAY_MAP.get(now.strftime("%A"), now.strftime("%A"))
    prefix = "Hôm nay" if days_offset == 0 else "Ngày mai" if days_offset == 1 else "Hôm qua" if days_offset == -1 else f"Ngày {now.day} tháng {now.month} năm {now.year}"
    return f"{prefix} là thứ {day_vi}, ngày {now.day} tháng {now.month} năm {now.year}."

# =========================
# 6) RAG – Retrieve (OPTIMIZED)
# =========================
def retrieve_context(query: str, top_k: int = 2) -> tuple[list[str], float]:
    """
    OPTIMIZED: Giảm top_k từ 3 → 2 để tăng tốc
    """
    if index is None or embeddings is None or not questions:
        return [], 0.0
    
    q_embed = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    D, I = index.search(q_embed, top_k)
    
    # Lọc các kết quả quá xa (L2 nhỏ hơn ngưỡng thì gần)
    # Tăng ngưỡng lên 1.2 để chấp nhận kết quả rộng hơn
    contexts = [
        f"Q: {questions[i]}\nA: {answers[i]}\n"
        for idx, i in enumerate(I[0])
        if i >= 0 and D[0][idx] < 1.2
    ]
    
    # Similarity "giả lập" từ L2 distance
    best_sim = float(max(0.0, 1.0 - float(D[0][0]))) if len(D[0]) > 0 else 0.0
    
    return contexts, best_sim

# =========================
# 7) API
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok", 
        "kb_loaded": len(questions) if questions else 0,
        "faiss_ready": index is not None
    }

@app.post("/chat")
async def chat_handler(payload: ChatPayload, request: Request):
    """
    OPTIMIZED: 
    - Giảm ngưỡng similarity từ 0.85 → 0.75
    - Giảm context window GPT từ -3 → -2
    """
    user_input = payload.messages[-1]["content"]
    session_id = payload.session_id
    user_lang = detect_language(user_input)

    # Dịch sang tiếng Việt nếu người dùng nói tiếng Anh (để retrieve tốt hơn)
    try:
        translated_input = GoogleTranslator(source="auto", target="vi").translate(user_input) if user_lang == "en" else user_input
    except Exception as e:
        logger.warning(f"⚠️ Translation failed: {e}")
        translated_input = user_input

    # Retrieve tri thức
    context_chunks, similarity = retrieve_context(translated_input)
    logger.info(f"🔍 FAISS similarity = {similarity:.2f} | Found {len(context_chunks)} contexts")

    # Lưu câu của user
    save_chat(session_id, "user", user_input)

    lower_input = translated_input.lower()
    
    # Xử lý đặc biệt về ngày tháng (real-time info)
    if "hôm nay" in lower_input or "hiện tại" in lower_input or "thứ mấy hôm nay" in lower_input:
        reply = get_date_info(0)
    elif "ngày mai" in lower_input or "mai là thứ mấy" in lower_input:
        reply = get_date_info(1)
    elif "hôm qua" in lower_input or "qua là thứ mấy" in lower_input:
        reply = get_date_info(-1)
    else:
        reply = None

    if reply is not None:
        if user_lang == "en":
            try:
                reply = GoogleTranslator(source="vi", target="en").translate(reply)
            except Exception:
                pass
        save_chat(session_id, "assistant", reply)
        return {"response": reply, "source": "real_time", "similarity": round(float(similarity), 2)}

    # OPTIMIZED: Giảm ngưỡng similarity từ 0.85 → 0.75
    if similarity >= 0.75 and context_chunks:
        logger.info("✅ Sử dụng câu trả lời từ Knowledge Base")
        top_answer = context_chunks[0].split("A:", 1)[-1].strip()
        if user_lang == "en":
            try:
                top_answer = GoogleTranslator(source="vi", target="en").translate(top_answer)
            except Exception:
                pass
        save_chat(session_id, "assistant", top_answer)
        return {"response": top_answer, "source": "knowledge_base", "similarity": round(float(similarity), 2)}

    # Fallback GPT
    logger.info("🤖 Calling GPT-4o-mini as fallback...")
    current_date_info = get_date_info(0)
    
    prompt = (
        f"{current_date_info}\n\n"
        "Bạn là trợ lý AI thân thiện, chính xác của THPT Marie Curie. "
        "Trả lời bằng cùng ngôn ngữ với người dùng (Anh/Việt). "
        "Nếu có ngữ cảnh nội bộ, ưu tiên dùng.\n\n"
        + ("\n".join(context_chunks) if context_chunks else "")
    )
    
    # OPTIMIZED: Giảm context window từ -3 → -2 messages
    messages = [{"role": "system", "content": prompt}] + payload.messages[-2:]

    try:
        if client is None:
            raise RuntimeError("OPENAI_API_KEY not set")
        
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=300,  # Giới hạn độ dài response
            temperature=0.7
        )
        gpt_reply = completion.choices[0].message.content
        logger.info("✅ GPT response received")
        
    except Exception as e:
        logger.error(f"❌ GPT error: {e}")
        gpt_reply = (
            "Xin lỗi, hệ thống đang gặp sự cố khi tạo câu trả lời." 
            if user_lang == "vi" 
            else "Sorry, the system is experiencing an issue generating the response."
        )

    save_chat(session_id, "assistant", gpt_reply)
    return {"response": gpt_reply, "source": "rag_gpt", "similarity": round(float(similarity), 2)}

# -------------------------
# 8) Feedback
# -------------------------
@app.post("/feedback")
async def feedback(
    session_id: str = Form(...),
    question: str = Form(...),
    answer: str = Form(...),
    rating: str = Form(...)
):
    save_feedback(session_id, question, answer, rating)
    logger.info(f"📝 Feedback received: {rating} | Session: {session_id}")
    return {"status": "ok", "message": "Đã ghi nhận phản hồi."}

# -------------------------
# 9) Endpoints dữ liệu cho trang Quản trị
# -------------------------
@app.get("/history")
def get_history():
    """Trả JSON lịch sử chat."""
    return JSONResponse(_read_csv_dicts(CHAT_CSV))

@app.get("/feedbacks")
def get_feedbacks():
    """Trả JSON danh sách feedback."""
    return JSONResponse(_read_csv_dicts(FEED_CSV))

@app.get("/chat_history.csv")
def download_history_csv():
    if not CHAT_CSV.exists():
        return JSONResponse({"detail": "chat_history.csv not found"}, status_code=404)
    return FileResponse(CHAT_CSV, media_type="text/csv", filename="chat_history.csv")

@app.get("/feedback.csv")
def download_feedback_csv():
    if not FEED_CSV.exists():
        return JSONResponse({"detail": "feedback.csv not found"}, status_code=404)
    return FileResponse(FEED_CSV, media_type="text/csv", filename="feedback.csv")

# -------------------------
# 10) Endpoint nạp lại KB khi thay MC_chatbot.csv
# -------------------------
@app.post("/reload_kb")
def reload_kb():
    """Đọc lại MC_chatbot.csv & tái tạo FAISS mà không cần redeploy."""
    global questions, answers, embeddings, index
    
    logger.info("🔄 Reloading Knowledge Base...")
    q, a = _load_kb(KB_CSV)
    
    if not q:
        return JSONResponse({"detail": "MC_chatbot.csv missing or invalid"}, status_code=400)

    questions, answers = q, a
    embeddings = model.encode(questions, convert_to_numpy=True, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    logger.info(f"✅ KB reloaded successfully: {len(questions)} questions")
    return {"status": "ok", "count": len(questions)}

# -------------------------
# 11) Optional: Upload endpoint (nếu cần upload MC_chatbot.csv qua API)
# -------------------------
from fastapi import UploadFile, File

@app.post("/upload_mc_data")
async def upload_mc_data(file: UploadFile = File(...)):
    """Upload file MC_chatbot.csv mới và reload KB."""
    if not file.filename.endswith('.csv'):
        return JSONResponse({"detail": "Chỉ chấp nhận file .csv"}, status_code=400)
    
    try:
        content = await file.read()
        KB_CSV.write_bytes(content)
        logger.info(f"📤 Uploaded new KB file: {KB_CSV}")
        
        # Auto reload
        result = reload_kb()
        return result
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        return JSONResponse({"detail": str(e)}, status_code=500)

# -------------------------
# Startup event
# -------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("🚀 Marie Curie RAG Chatbot API Starting...")
    logger.info(f"📚 Knowledge Base: {len(questions)} questions loaded")
    logger.info(f"🔧 FAISS Index: {'Ready' if index else 'Not Ready'}")
    logger.info(f"🤖 OpenAI API: {'Configured' if client else 'Missing'}")
    logger.info("=" * 60)
