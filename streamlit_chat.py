# streamlit_chat.py
# ============================================================================
# Chatbot tư vấn tuyển sinh 10 – Marie Curie
# Frontend: Streamlit. Gọi backend qua HTTP.
# Thiết kế:
#   - Lời chào là một message thật (không render tạm, không cần rerun)
#   - Không dùng st.experimental_rerun để tránh "mất" lượt chat đầu tiên
#   - Phân tách module: cấu hình, gọi API, UI User/Admin, tiện ích
#   - Bắt lỗi mạng/timeout rõ ràng, hiển thị thông báo thân thiện
#   - Bảo mật đơn giản cho tab Quản trị bằng ADMIN_PASSWORD
# ============================================================================

from __future__ import annotations

import os
import io
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd
import streamlit as st

# ============================== CẤU HÌNH UI =================================
APP_TITLE = "Chatbot AI – trợ lí ảo hỗ trợ tư vấn tuyển sinh 10 – THPT Marie Curie"
PAGE_ICON = "🤖"
LAYOUT = "wide"
THEME_HINT = """
<style>
    /* Tăng tính dễ đọc */
    .markdown-text-container { font-size: 1.05rem; }
    .stChatMessage { line-height: 1.5; }
    /* Nút nhỏ gọn */
    .stButton > button { border-radius: 0.6rem; padding: 0.4rem 0.9rem; }
</style>
"""

DEFAULT_TIMEOUT = (10, 60)  # (connect, read) seconds
BACKEND_URL = os.getenv("BACKEND_URL", "").rstrip("/")
ADMIN_PASSWORD_ENV = os.getenv("ADMIN_PASSWORD", "admin123")  # nhớ đổi khi deploy

# ============================== TIỆN ÍCH CHUNG ==============================

def now_iso() -> str:
    return datetime.utcnow().isoformat()

def human_err(exc: Exception) -> str:
    return f"Lỗi kết nối tới backend: {exc.__class__.__name__} – {exc}"

def as_dataframe_safe(data: Any) -> pd.DataFrame:
    try:
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            return pd.DataFrame([data])
    except Exception:
        pass
    return pd.DataFrame()

# ============================== LỚP API CLIENT ==============================

class BackendClient:
    """
    Gói gọn việc gọi backend. Chấp nhận các biến thể field trả lời:
    {answer} hoặc {reply} hoặc {response}
    """

    def __init__(self, base_url: str):
        self.base_url = base_url

    def _url(self, path: str) -> str:
        if not self.base_url:
            raise RuntimeError("BACKEND_URL chưa được cấu hình.")
        return f"{self.base_url}{path}"

    def health(self) -> Tuple[bool, str]:
        try:
            r = requests.get(self._url("/health"), timeout=DEFAULT_TIMEOUT)
            if r.ok:
                return True, f"OK ({r.status_code})"
            return False, f"DOWN ({r.status_code})"
        except Exception as e:
            return False, human_err(e)

    def chat(self, user_text: str, history: List[Dict[str, str]]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Gọi POST /chat. Backend nên hiểu trường:
            - user_input (hoặc question, hoặc prompt)
            - history: danh sách {role, content}
        """
        payload = {
            "user_input": user_text,
            "history": history,
        }
        try:
            r = requests.post(self._url("/chat"), json=payload, timeout=DEFAULT_TIMEOUT)
            if not r.ok:
                return False, f"HTTP {r.status_code}: {r.text[:200]}", {}
            data = r.json()
            text = data.get("answer") or data.get("reply") or data.get("response") or ""
            if not isinstance(text, str) or not text.strip():
                return False, "Backend không trả về nội dung hợp lệ (answer/reply/response).", data
            return True, text, data
        except Exception as e:
            return False, human_err(e), {}

    def send_feedback(self, question: str, answer: str, rating: int, meta: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        payload = {
            "question": question,
            "answer": answer,
            "rating": rating,  # +1 or -1
            "meta": meta or {},
            "ts": now_iso(),
        }
        try:
            r = requests.post(self._url("/feedback"), json=payload, timeout=DEFAULT_TIMEOUT)
            if r.ok:
                return True, "Đã ghi nhận phản hồi."
            return False, f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            return False, human_err(e)

    def fetch_json(self, path: str) -> Tuple[bool, Any, str]:
        try:
            r = requests.get(self._url(path), timeout=DEFAULT_TIMEOUT)
            if not r.ok:
                return False, None, f"HTTP {r.status_code}: {r.text[:200]}"
            return True, r.json(), "OK"
        except Exception as e:
            return False, None, human_err(e)

    def fetch_csv_bytes(self, path: str) -> Tuple[bool, bytes, str]:
        try:
            r = requests.get(self._url(path), timeout=DEFAULT_TIMEOUT)
            if not r.ok:
                return False, b"", f"HTTP {r.status_code}: {r.text[:200]}"
            return True, r.content, "OK"
        except Exception as e:
            return False, b"", human_err(e)

    def upload_mc_csv(self, file_bytes: bytes, filename: str) -> Tuple[bool, str]:
        files = {"file": (filename, file_bytes, "text/csv")}
        try:
            r = requests.post(self._url("/upload_mc_data"), files=files, timeout=DEFAULT_TIMEOUT)
            if r.ok:
                return True, "Upload thành công và backend đã nhận dữ liệu."
            return False, f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            return False, human_err(e)

# ============================== KHỞI TẠO STATE ==============================

def init_state() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)
    st.markdown(THEME_HINT, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, Any]] = []
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = ""
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""
    if "admin_authed" not in st.session_state:
        st.session_state.admin_authed = False

    # Lời chào là một message thật, chỉ tạo một lần khi list rỗng
    if len(st.session_state.messages) == 0:
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "Chào bạn! Mình là trợ lí ảo tuyển sinh 10 – THPT Marie Curie. "
                "Bạn có thể hỏi về quy chế tuyển sinh, hồ sơ, mốc thời gian, điểm chuẩn… "
                "Hãy đặt câu hỏi bên dưới nhé!"
            ),
            "ts": now_iso()
        })

# ============================== UI: USER TAB =================================

def render_user_tab(client: BackendClient) -> None:
    st.header("👤 Người dùng")

    # 1) Hiển thị lịch sử đã có
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 2) Nhận input
    user_text = st.chat_input("Nhập câu hỏi của bạn…")
    if not user_text:
        return

    # 3) Append câu hỏi NGAY LẬP TỨC (không rerun)
    st.session_state.messages.append({
        "role": "user",
        "content": user_text,
        "ts": now_iso()
    })
    with st.chat_message("user"):
        st.markdown(user_text)

    # 4) Gọi backend và render "assistant thinking" trong cùng lượt
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("_Đang soạn câu trả lời…_")

        success, answer_text, raw = client.chat(
            user_text=user_text,
            history=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
                if m["role"] in ("user", "assistant")
            ],
        )

        if success:
            thinking_placeholder.empty()
            st.markdown(answer_text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "ts": now_iso()
            })
            st.session_state.last_question = user_text
            st.session_state.last_answer = answer_text
        else:
            thinking_placeholder.empty()
            st.error(answer_text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Xin lỗi, hiện chưa kết nối được: {answer_text}",
                "ts": now_iso()
            })
            st.session_state.last_question = user_text
            st.session_state.last_answer = ""

    # 5) Feedback cho câu trả lời vừa nhận
    col1, col2, col3 = st.columns([1,1,6])
    with col1:
        if st.button("👍 Hữu ích", use_container_width=True):
            ok, msg = client.send_feedback(
                st.session_state.last_question,
                st.session_state.last_answer,
                rating=+1,
                meta={"source": "streamlit", "version": 1},
            )
            st.toast("Cảm ơn phản hồi!" if ok else f"Không ghi được phản hồi: {msg}")
    with col2:
        if st.button("👎 Chưa tốt", use_container_width=True):
            ok, msg = client.send_feedback(
                st.session_state.last_question,
                st.session_state.last_answer,
                rating=-1,
                meta={"source": "streamlit", "version": 1},
            )
            st.toast("Đã ghi nhận góp ý!" if ok else f"Không ghi được phản hồi: {msg}")

# ============================== UI: ADMIN TAB ================================

def render_admin_tab(client: BackendClient) -> None:
    st.header("🛠️ Quản trị")

    # --- Xác thực đơn giản ---
    if not st.session_state.admin_authed:
        with st.form("admin_login", clear_on_submit=False):
            pwd = st.text_input("Mật khẩu quản trị", type="password")
            ok = st.form_submit_button("Đăng nhập")
        if ok:
            if pwd and pwd == ADMIN_PASSWORD_ENV:
                st.session_state.admin_authed = True
                st.success("Đăng nhập thành công.")
            else:
                st.error("Sai mật khẩu.")
        return

    # --- Thông tin hệ thống ---
    st.subheader("Tình trạng Backend")
    colA, colB, colC = st.columns([1,1,2])
    with colA:
        ping = st.button("Ping /health", use_container_width=True)
    with colB:
        st.write(f"**BACKEND_URL:** `{BACKEND_URL or 'CHƯA CẤU HÌNH'}`")
    if ping:
        ok, msg = client.health()
        (st.success if ok else st.error)(msg)

    st.divider()

    # --- Dữ liệu/CSV quản trị ---
    st.subheader("Lịch sử & Phản hồi")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Tải JSON /history", use_container_width=True):
            ok, data, msg = client.fetch_json("/history")
            if ok:
                df = as_dataframe_safe(data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.error(msg)
    with col2:
        if st.button("Tải JSON /feedbacks", use_container_width=True):
            ok, data, msg = client.fetch_json("/feedbacks")
            if ok:
                df = as_dataframe_safe(data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.error(msg)
    with col3:
        if st.button("Tải CSV /chat_history.csv", use_container_width=True):
            ok, b, msg = client.fetch_csv_bytes("/chat_history.csv")
            if ok:
                st.download_button("⬇️ Lưu chat_history.csv", b, file_name="chat_history.csv", mime="text/csv", use_container_width=True)
            else:
                st.error(msg)
    with col4:
        if st.button("Tải CSV /feedback.csv", use_container_width=True):
            ok, b, msg = client.fetch_csv_bytes("/feedback.csv")
            if ok:
                st.download_button("⬇️ Lưu feedback.csv", b, file_name="feedback.csv", mime="text/csv", use_container_width=True)
            else:
                st.error(msg)

    st.divider()

    # --- Upload dữ liệu câu hỏi/FAQ (MC_chatbot.csv) ---
    st.subheader("Cập nhật dữ liệu tư vấn (CSV)")
    up = st.file_uploader("Chọn file CSV (ví dụ: MC_chatbot.csv)", type=["csv"])
    if up is not None:
        bytes_io = up.read()
        if st.button("📤 Upload lên backend", use_container_width=True):
            ok, msg = client.upload_mc_csv(bytes_io, up.name)
            (st.success if ok else st.error)(msg)

# ============================== MAIN APP =====================================

def main() -> None:
    init_state()

    # Nhắc cấu hình nếu thiếu BACKEND_URL
    if not BACKEND_URL:
        st.warning("⚠️ Chưa cấu hình biến môi trường BACKEND_URL. Hãy đặt BACKEND_URL trỏ tới Railway (ví dụ: https://your-app.up.railway.app).")

    client = BackendClient(BACKEND_URL) if BACKEND_URL else None
    tabs = st.tabs(["Người dùng", "Quản trị"])

    with tabs[0]:
        if client:
            render_user_tab(client)
        else:
            st.info("Chưa có BACKEND_URL nên không thể gửi câu hỏi. Vui lòng cấu hình rồi tải lại trang.")

    with tabs[1]:
        if client:
            render_admin_tab(client)
        else:
            st.info("Chưa có BACKEND_URL – các tính năng quản trị sẽ hoạt động sau khi cấu hình.")

if __name__ == "__main__":
    main()
