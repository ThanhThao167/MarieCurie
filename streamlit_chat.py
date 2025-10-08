# streamlit_chat.py
# ==================
# UI Chatbot tuyển sinh (frontend) cho backend FastAPI.
# - Two-phase render: show user msg ngay, rồi mới gọi API + hiện "đang suy nghĩ".
# - Đọc BACKEND_URL & ADMIN_PASSWORD từ Streamlit Secrets hoặc ENV.
# - Có tab Quản trị: ping health, xem lịch sử & feedback, tải CSV.
# - Chống treo: retry có backoff, timeout rõ ràng, thông báo lỗi thân thiện.

from __future__ import annotations
import os
import json
import uuid
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

import requests
import pandas as pd
import streamlit as st

# -----------------------
# Cấu hình & tiện ích
# -----------------------

st.set_page_config(
    page_title="Chatbot AI tư vấn tuyển sinh 10 - THPT Marie Curie",
    page_icon="🤖",
    layout="wide",
)

def _normalize_backend(url: str) -> str:
    """
    Chuẩn hoá BACKEND_URL (thêm http/https nếu thiếu, bỏ dấu / cuối).
    """
    if not url:
        return "http://localhost:8000"
    url = url.strip()
    # sửa nhầm 'https:/' thành 'https://'
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
    """
    Gọi POST JSON tới backend với retry tuyến tính + backoff.
    Trả về dict JSON hoặc None nếu lỗi.
    """
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
                # backoff nhẹ để tránh flood khi backend đang warmup
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

# -----------------------
# Session state
# -----------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = f"web-{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    # mỗi item: {"role": "user"|"assistant", "content": str, "ts": iso}
    st.session_state.messages: List[Dict[str, Any]] = []

if "awaiting_response" not in st.session_state:
    st.session_state.awaiting_response = False

if "last_user_text" not in st.session_state:
    st.session_state.last_user_text: Optional[str] = None

if "last_answer" not in st.session_state:
    st.session_state.last_answer: Optional[str] = None

# -----------------------
# Header
# -----------------------

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
        st.success(health if health else {"status": "fail"})

tab_user, tab_admin = st.tabs(["👩‍🎓 Người dùng", "🛠 Quản trị"])

# =========================================================
# TAB NGƯỜI DÙNG (Chat)
# =========================================================
with tab_user:
    # Hiển thị lịch sử hội thoại theo chuẩn chat
    for msg in st.session_state.messages:
        role = msg.get("role", "assistant")
        with st.chat_message(role):
            st.markdown(msg.get("content", ""))

    # Ô nhập chat
    user_text = st.chat_input("Nhập câu hỏi của bạn…")

    # -------- PHA 1: nhận input -> append -> rerun để vẽ ngay --------
    if user_text:
        st.session_state.messages.append(
            {"role": "user", "content": user_text, "ts": datetime.utcnow().isoformat()}
        )
        st.session_state.awaiting_response = True
        st.session_state.last_user_text = user_text
        do_rerun()

    # -------- PHA 2: đã hiển thị câu hỏi; giờ gọi backend + thinking --------
    if st.session_state.awaiting_response and st.session_state.last_user_text:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            status_ctx = None
            # st.status có từ Streamlit >=1.26; fallback spinner nếu bản cũ
            try:
                status_ctx = placeholder.status("🤔 Đang suy nghĩ…", state="running")
            except Exception:
                status_ctx = st.spinner("🤔 Đang suy nghĩ…")

            with status_ctx:
                data = post_json(
                    "/chat",
                    {
                        "message": st.session_state.last_user_text,
                        "user_id": st.session_state.session_id,
                    },
                    timeout=60,
                    retries=2,
                    backoff=0.8,
                )
                # Chuẩn hoá khoá trả về từ backend
                answer = None
                if isinstance(data, dict):
                    answer = (
                        data.get("answer")
                        or data.get("reply")
                        or data.get("response")
                        or data.get("message")
                    )
                if not answer:
                    answer = "Xin lỗi, hiện chưa có phản hồi."

            # thay ‘đang suy nghĩ’ bằng câu trả lời
            try:
                placeholder.empty()
            except Exception:
                pass
            st.markdown(answer)

        # Lưu & reset cờ
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "ts": datetime.utcnow().isoformat()}
        )
        st.session_state.last_answer = answer
        st.session_state.awaiting_response = False
        st.session_state.last_user_text = None
        do_rerun()

    # ---------------------------------------------
    # Phản hồi 👍 👎 cho câu trả lời cuối cùng
    # ---------------------------------------------
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Hài lòng", use_container_width=True):
                post_json(
                    "/feedback",
                    {
                        "rating": 1,
                        "question": st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 else "",
                        "answer": st.session_state.messages[-1]["content"],
                        "user_id": st.session_state.session_id,
                    },
                    timeout=20,
                )
                st.success("Cảm ơn phản hồi của bạn!")
        with col2:
            if st.button("👎 Chưa tốt", use_container_width=True):
                post_json(
                    "/feedback",
                    {
                        "rating": -1,
                        "question": st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 else "",
                        "answer": st.session_state.messages[-1]["content"],
                        "user_id": st.session_state.session_id,
                    },
                    timeout=20,
                )
                st.info("Đã ghi nhận góp ý!")

# =========================================================
# TAB QUẢN TRỊ
# =========================================================
with tab_admin:
    st.caption("Yêu cầu mật khẩu quản trị để xem dữ liệu hệ thống.")
    pwd = st.text_input("Mật khẩu quản trị", type="password")
    if ADMIN_PASSWORD and pwd != ADMIN_PASSWORD:
        st.warning("Sai mật khẩu.")
        st.stop()
    elif not ADMIN_PASSWORD:
        st.info("Chưa thiết lập ADMIN_PASSWORD trong Secrets/ENV. Vẫn cho phép truy cập tạm thời.")

    st.subheader("1) Tình trạng dịch vụ")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🔄 Ping /health", use_container_width=True):
            st.json(get_json("/health") or {"status": "fail"})
    with c2:
        if st.button("🧹 Xoá lịch sử phiên hiện tại", use_container_width=True):
            st.session_state.messages = []
            st.success("Đã xoá hội thoại trên UI.")
    with c3:
        st.write(f"Session ID: `{st.session_state.session_id}`")

    st.subheader("2) Lịch sử hội thoại (server)")
    if st.button("📜 Tải `/history`", use_container_width=True):
        data = get_json("/history")  # backend nên trả list[dict] hoặc {"history": [...]}
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

# -----------------------
# Footer nhỏ
# -----------------------
st.write("")
st.caption(
    "© 2025 — Chatbot tư vấn tuyển sinh lớp 10. "
    "Nếu gặp lỗi, hãy kiểm tra **Secrets → BACKEND_URL**, và đảm bảo backend `/health` trả `ok`."
)
