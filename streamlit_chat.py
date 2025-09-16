# streamlit_chat.py
# Frontend cho Chatbot tư vấn tuyển sinh, chạy trên Streamlit (Cloud hoặc local)

import os
import uuid
import json
import requests
import pandas as pd
from datetime import datetime
import streamlit as st

# =========================
# 1) Cấu hình & tiện ích
# =========================
st.set_page_config(page_title="Chatbot Tư Vấn Tuyển Sinh", page_icon="🤖")

# Đọc cấu hình từ ENV hoặc st.secrets (ưu tiên ENV)
BACKEND_URL = os.getenv("BACKEND_URL") or st.secrets.get("BACKEND_URL", "http://localhost:8000")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD") or st.secrets.get("ADMIN_PASSWORD", "admin123")

# Timeout mặc định cho mọi request ra backend (giây)
DEFAULT_TIMEOUT = 60

def post_json(path: str, payload: dict):
    """POST JSON lên backend và trả về dict (hoặc raise)."""
    url = f"{BACKEND_URL.rstrip('/')}/{path.lstrip('/')}"
    resp = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()

def post_form(path: str, form: dict):
    """POST form-data lên backend và trả về dict (hoặc raise)."""
    url = f"{BACKEND_URL.rstrip('/')}/{path.lstrip('/')}"
    resp = requests.post(url, data=form, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()

def get_json(path: str):
    """GET JSON từ backend; trả về None nếu 404/không có."""
    url = f"{BACKEND_URL.rstrip('/')}/{path.lstrip('/')}"
    try:
        resp = requests.get(url, timeout=DEFAULT_TIMEOUT)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        # cho phép plain text "ok"
        try:
            return resp.json()
        except Exception:
            return {"text": resp.text}
    except requests.RequestException:
        return None

def get_csv_as_df(path: str):
    """Thử tải một CSV (backend phải phục vụ URL này) → DataFrame; lỗi thì trả None."""
    url = f"{BACKEND_URL.rstrip('/')}/{path.lstrip('/')}"
    try:
        df = pd.read_csv(url)
        return df
    except Exception:
        return None

# =========================
# 2) State khởi tạo
# =========================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    # Định dạng openai style: [{"role":"user"/"assistant","content":"..."}]
    st.session_state.messages = []

if "last_reply" not in st.session_state:
    st.session_state.last_reply = ""

# =========================
# 3) UI: Tabs
# =========================
tab_user, tab_admin = st.tabs(["👨‍🎓 Người dùng", "🛠 Quản trị"])

# =========================
# 4) Tab Người dùng (Chat)
# =========================
with tab_user:
    st.title("🤖 Chatbot Tư vấn Tuyển sinh lớp 10 – THPT Marie Curie")

    # Hiển thị lịch sử hội thoại
    if len(st.session_state.messages) == 0:
        with st.chat_message("assistant"):
            st.markdown(
                "Chào bạn! Mình là **Chatbot tư vấn tuyển sinh lớp 10**. "
                "Bạn có thể hỏi về chỉ tiêu, cách đăng ký, thời khóa biểu, học phí, chính sách, v.v."
            )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Ô nhập chat
    user_input = st.chat_input("Nhập câu hỏi của bạn...")

    if user_input:
        # Append tin nhắn của user vào UI ngay
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Gọi backend /chat
        try:
            payload = {
                "messages": st.session_state.messages,
                "session_id": st.session_state.session_id
            }
            data = post_json("/chat", payload)
            # Backend dự kiến trả về {"reply": "...", "source": "...", "similarity": 0.95, ...}
            reply = data.get("reply") or data.get("response") or "Xin lỗi, hiện chưa có phản hồi."
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.session_state.last_reply = reply

            with st.chat_message("assistant"):
                st.markdown(reply)

                # Nút feedback cho câu trả lời hiện tại
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Hài lòng", key=f"fb_up_{len(st.session_state.messages)}"):
                        try:
                            post_form(
                                "/feedback",
                                {
                                    "session_id": st.session_state.session_id,
                                    "question": user_input,
                                    "answer": reply,
                                    "rating": "up",
                                },
                            )
                            st.success("Đã gửi phản hồi 👍")
                        except requests.RequestException as e:
                            st.warning(f"Gửi feedback thất bại: {e}")

                with col2:
                    if st.button("👎 Chưa tốt", key=f"fb_dn_{len(st.session_state.messages)}"):
                        try:
                            post_form(
                                "/feedback",
                                {
                                    "session_id": st.session_state.session_id,
                                    "question": user_input,
                                    "answer": reply,
                                    "rating": "down",
                                },
                            )
                            st.success("Đã gửi phản hồi 👎")
                        except requests.RequestException as e:
                            st.warning(f"Gửi feedback thất bại: {e}")

        except requests.RequestException as e:
            with st.chat_message("assistant"):
                st.error(
                    "Không thể kết nối tới backend. "
                    "Vui lòng kiểm tra `BACKEND_URL` trên Streamlit Secrets hoặc thử lại sau."
                )
                st.code(str(e))

    # Thanh thông tin nhỏ
    st.caption(
        f"Phiên: `{st.session_state.session_id}` • Backend: `{BACKEND_URL}` • "
        f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

# =========================
# 5) Tab Quản trị
# =========================
with tab_admin:
    st.header("🛠 Khu vực Quản trị")

    pwd = st.text_input("Nhập mật khẩu quản trị", type="password")
    if pwd != ADMIN_PASSWORD:
        st.info(
            "Nhập đúng mật khẩu để xem công cụ quản trị. "
            "Bạn đặt mật khẩu trong **ENV** hoặc **`st.secrets`** với key `ADMIN_PASSWORD`."
        )
        st.stop()

    st.success("Đăng nhập quản trị thành công ✅")

    # 5.1. Kiểm tra tình trạng backend
    st.subheader("✅ Kiểm tra tình trạng Backend")
    colA, colB = st.columns(2)
    with colA:
        if st.button("Ping /health"):
            result = get_json("/health")
            if result:
                st.write(result)
            else:
                st.error("Không gọi được `/health`. Kiểm tra BACKEND_URL hoặc triển khai backend.")
    with colB:
        st.write(f"**BACKEND_URL:** `{BACKEND_URL}`")

    st.divider()

    # 5.2. Xem lịch sử chat (nếu backend có endpoint)
    st.subheader("🗂 Lịch sử hội thoại")
    st.caption("Yêu cầu backend cung cấp endpoint JSON hoặc CSV. Thử các phương án phổ biến bên dưới.")

    tabs_hist = st.tabs(["/history (JSON)", "/chat_history.csv (CSV)"])
    with tabs_hist[0]:
        hist = get_json("/history")
        if hist and isinstance(hist, list):
            df = pd.DataFrame(hist)
            st.dataframe(df, use_container_width=True)
        elif hist:
            st.write(hist)
        else:
            st.info("Không có endpoint `/history` hoặc không thể truy cập.")

    with tabs_hist[1]:
        df = get_csv_as_df("/chat_history.csv")
        if df is not None:
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Không tìm thấy `/chat_history.csv` từ backend.")

    st.divider()

    # 5.3. Xem feedback (nếu backend có endpoint)
    st.subheader("📝 Feedback người dùng")
    tabs_fb = st.tabs(["/feedbacks (JSON)", "/feedback.csv (CSV)"])
    with tabs_fb[0]:
        fjson = get_json("/feedbacks")
        if fjson and isinstance(fjson, list):
            df = pd.DataFrame(fjson)
            st.dataframe(df, use_container_width=True)
        elif fjson:
            st.write(fjson)
        else:
            st.info("Không có endpoint `/feedbacks` hoặc không thể truy cập.")

    with tabs_fb[1]:
        df = get_csv_as_df("/feedback.csv")
        if df is not None:
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Không tìm thấy `/feedback.csv` từ backend.")

    st.divider()

    # 5.4. Hướng dẫn chỉnh sửa dữ liệu MC_chatbot.csv (phụ thuộc backend)
    st.subheader("✏️ Chỉnh sửa dữ liệu MC_chatbot.csv (tuỳ chọn)")
    st.markdown(
        "- Trên môi trường Cloud, frontend **không thể** ghi trực tiếp vào ổ đĩa của backend.\n"
        "- Bạn có thể:\n"
        "  1) Tạo **endpoint upload** ở backend (ví dụ: `POST /upload_mc_data` nhận file CSV), rồi dùng file uploader dưới đây để gửi lên.\n"
        "  2) Hoặc gắn cơ sở dữ liệu (Postgres/Supabase/Google Sheets) để lưu bền vững.\n"
    )

    uploaded = st.file_uploader("Chọn file MC_chatbot.csv để tải lên backend (nếu backend có endpoint).", type=["csv"])
    if uploaded and st.button("➡️ Gửi lên backend (POST /upload_mc_data)"):
        try:
            url = f"{BACKEND_URL.rstrip('/')}/upload_mc_data"
            files = {"file": ("MC_chatbot.csv", uploaded.getvalue(), "text/csv")}
            resp = requests.post(url, files=files, timeout=DEFAULT_TIMEOUT)
            if resp.status_code == 200:
                st.success("Đã gửi file lên backend.")
            else:
                st.warning(f"Backend trả về mã {resp.status_code}: {resp.text}")
        except requests.RequestException as e:
            st.error(f"Lỗi gửi file: {e}")

# ============ Hết ============
