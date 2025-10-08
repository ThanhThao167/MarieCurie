# streamlit_chat.py
# ==================
# UI Chatbot tuyển sinh lớp 10 – THPT Marie Curie (frontend Streamlit)
# - Two-phase render (hiển thị câu hỏi ngay + “🤔 Đang suy nghĩ…”)
# - Đọc BACKEND_URL, ADMIN_PASSWORD từ Secrets/ENV
# - Tab Quản trị: health, lịch sử, feedbacks, tải CSV
# - Có lời chào ban đầu dưới dạng bong bóng chat

from __future__ import annotations
import os, uuid, time
from datetime import datetime
from typing import Dict, Any, List, Optional

import requests
import pandas as pd
import streamlit as st

# =============== Cấu hình trang ===============
st.set_page_config(
    page_title="Chatbot AI tư vấn tuyển sinh 10 - THPT Marie Curie",
    page_icon="🤖",
    layout="wide",
)

# =============== Hằng số & tiện ích ===============
GREETING = (
    "Chào bạn, tôi là chatbot hỗ trợ tuyển sinh 10, "
    "sẵn sàng trả lời mọi câu hỏi của bạn liên quan đến vấn đề tuyển sinh "
    "tại trường THPT Marie Curie."
)

def _normalize_backend(url: str) -> str:
    if not url:
        return "http://localhost:8000"
    url = url.strip()
    if url.startswith("https:/") and not url.startswith("https://"):
        url = url.replace("https:/", "https://", 1)
    if url.startswith("http:/") and not url.startswith("http://"):
        url = url.replace("http:/", "http://", 1)
    if not (url.startswith("http://") or url.startswith("https://")):
        url = "https://" + url.lstrip("/")
    return url.rstrip("/")

BACKEND_URL = _normalize_backend(
    st.secrets.get("BACKEND_URL", os.getenv("BACKEND_URL", "http://localhost:8000"))
)
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", os.getenv("ADMIN_PASSWORD", ""))

DEFAULT_HEADERS = {"Content-Type": "application/json"}

def post_json(
    path: str,
    payload: Dict[str, Any],
    timeout: int = 60,
    retries: int = 2,
    backoff: float = 0.8,
) -> Optional[Dict[str, Any]]:
    """POST JSON tới backend với retry/backoff."""
    url = f"{BACKEND_URL}{path}"
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, headers=DEFAULT_HEADERS, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
            else:
                st.error("Không thể kết nối tới backend. Kiểm tra BACKEND_URL trong Secrets hoặc thử lại sau.")
                with st.expander("Chi tiết lỗi", expanded=False):
                    st.code(f"{e}", language="bash")
                return None
    return None

def get_json(path: str, timeout: int = 30) -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(f"{BACKEND_URL}{path}", timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.warning(f"Lỗi khi gọi {path}: {e}")
        return None

def do_rerun():
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

# =============== Session state ===============
if "session_id" not in st.session_state:
    st.session_state.session_id = f"web-{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    # mỗi item: {"role": "user"/"assistant", "content": str, "ts": iso}
    st.session_state.messages: List[Dict[str, Any]] = []

if "awaiting_response" not in st.session_state:
    st.session_state.awaiting_response = False

if "last_user_text" not in st.session_state:
    st.session_state.last_user_text: Optional[str] = None

if "greeted" not in st.session_state:
    st.session_state.greeted = False

# Thêm lời chào 1 lần khi mở app
if not st.session_state.messages and not st.session_state.greeted:
    st.session_state.messages.append(
        {"role": "assistant", "content": GREETING, "ts": datetime.utcnow().isoformat()}
    )
    st.session_state.greeted = True

# =============== Header & Sidebar ===============
st.markdown(
    """
    <h1 style="margin-top:0">🤖 Chatbot AI — trợ lí ảo hỗ trợ tư vấn tuyển sinh 10 — THPT Marie Curie</h1>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### ⚙️ Cấu hình")
    st.write(f"**Backend**: `{BACKEND_URL}`")
    if st.button("Ping /health"):
        health = get_json("/health")
        if health:
            st.success(health)
        else:
            st.error({"status": "fail"})

tab_user, tab_admin = st.tabs(["👩‍🎓 Người dùng", "🛠 Quản trị"])

# ==========================
# TAB: NGƯỜI DÙNG (CHAT)
# ==========================
with tab_user:
    # Hiển thị lịch sử
    for msg in st.session_state.messages:
        with st.chat_message(msg.get("role", "assistant")):
            st.markdown(msg.get("content", ""))

    # Ô nhập
    user_text = st.chat_input("Nhập câu hỏi của bạn…")

    # PHA 1: nhận input -> append -> rerun để hiển thị ngay
    if user_text:
        st.session_state.messages.append(
            {"role": "user", "content": user_text, "ts": datetime.utcnow().isoformat()}
        )
        st.session_state.awaiting_response = True
        st.session_state.last_user_text = user_text
        do_rerun()

    # PHA 2: đã vẽ câu hỏi; gọi backend + hiển thị “đang suy nghĩ”
    if st.session_state.awaiting_response and st.session_state.last_user_text:
        with st.chat_message("assistant"):
            # status (Streamlit >=1.26); fallback spinner nếu bản cũ
            use_status = hasattr(st, "status")
            ctx_mgr = st.status("🤔 Đang suy nghĩ…", state="running") if use_status else st.spinner("🤔 Đang suy nghĩ…")
            with ctx_mgr:
                data = post_json(
                    "/chat",
                    {
                        "message": st.session_state.last_user_text,
                        "user_id": st.session_state.session_id,
                    },
                    timeout=60, retries=2, backoff=0.8,
                )
                answer = None
                if isinstance(data, dict):
                    answer = data.get("answer") or data.get("reply") or data.get("response") or data.get("message")
                if not answer:
                    answer = "Xin lỗi, hiện chưa có phản hồi."
            st.markdown(answer)

        # Lưu & reset
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "ts": datetime.utcnow().isoformat()}
        )
        st.session_state.awaiting_response = False
        st.session_state.last_user_text = None
        do_rerun()

    # Phản hồi 👍 👎 cho câu trả lời cuối
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Hài lòng", use_container_width=True):
                # cố gắng lấy câu hỏi trước đó nếu có
                q = ""
                if len(st.session_state.messages) >= 2 and st.session_state.messages[-2]["role"] == "user":
                    q = st.session_state.messages[-2]["content"]
                post_json("/feedback", {"rating": 1, "question": q,
                                        "answer": st.session_state.messages[-1]["content"],
                                        "user_id": st.session_state.session_id}, timeout=20)
                st.toast("Cảm ơn phản hồi của bạn!", icon="✅")
        with col2:
            if st.button("👎 Chưa tốt", use_container_width=True):
                q = ""
                if len(st.session_state.messages) >= 2 and st.session_state.messages[-2]["role"] == "user":
                    q = st.session_state.messages[-2]["content"]
                post_json("/feedback", {"rating": -1, "question": q,
                                        "answer": st.session_state.messages[-1]["content"],
                                        "user_id": st.session_state.session_id}, timeout=20)
                st.toast("Đã ghi nhận góp ý!", icon="ℹ️")

# ==========================
# TAB: QUẢN TRỊ
# ==========================
with tab_admin:
    st.caption("Nhập mật khẩu quản trị để xem dữ liệu hệ thống.")
    pwd = st.text_input("Mật khẩu quản trị", type="password")
    if ADMIN_PASSWORD and pwd != ADMIN_PASSWORD:
        st.warning("Sai mật khẩu.")
        st.stop()
    elif not ADMIN_PASSWORD:
        st.info("Chưa thiết lập ADMIN_PASSWORD trong Secrets/ENV. Tạm cho phép truy cập.")

    st.subheader("1) Tình trạng dịch vụ")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🔄 Ping /health", use_container_width=True):
            st.json(get_json("/health") or {"status": "fail"})
    with c2:
        if st.button("🧹 Xoá lịch sử hội thoại (UI)", use_container_width=True):
            st.session_state.messages = []
            st.success("Đã xoá hội thoại trên UI.")
    with c3:
        st.write(f"Session ID: `{st.session_state.session_id}`")

    st.subheader("2) Lịch sử hội thoại (server)")
    if st.button("📜 Tải `/history`", use_container_width=True):
        data = get_json("/history")
        rows = data.get("history") if isinstance(data, dict) and "history" in data else data
        if isinstance(rows, list) and rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "⬇️ Tải CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="chat_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("Không có dữ liệu lịch sử.")

    st.subheader("3) Feedback người dùng (server)")
    if st.button("🗳️ Tải `/feedbacks`", use_container_width=True):
        data = get_json("/feedbacks")
        rows = data.get("feedbacks") if isinstance(data, dict) and "feedbacks" in data else data
        if isinstance(rows, list) and rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "⬇️ Tải CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="feedbacks.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("Không có feedback.")

# =============== Footer ===============
st.write("")
st.caption(
    "© 2025 — Chatbot tư vấn tuyển sinh lớp 10. "
    "Nếu gặp lỗi, hãy kiểm tra **Secrets → BACKEND_URL**, và đảm bảo backend `/health` trả `ok`."
)
