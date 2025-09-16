# streamlit_chat.py
# Frontend Streamlit cho Chatbot tuyển sinh 10 (Cloud-ready)

import os
import uuid
import requests
import pandas as pd
from datetime import datetime
from contextlib import suppress
import streamlit as st

# =========================
# 0) Page config & CSS
# =========================
st.set_page_config(
    page_title="Chatbot AI- trợ lí ảo hỗ trợ tư vấn tuyển sinh 10- THPT Marie Curie",
    page_icon="🤖",
    layout="wide",
)

# Tabs + buttons + input styling
st.markdown(
    """
<style>
/* Tabs: nền cho tab được chọn */
div.stTabs [data-baseweb="tab-list"] { gap: .25rem; }
div.stTabs [data-baseweb="tab-list"] button[role="tab"] {
  background-color: transparent; border: 1px solid transparent; border-bottom: none;
  padding: .5rem 1rem; margin: 0; border-radius: 10px 10px 0 0;
}
div.stTabs [data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {
  background-color: rgba(31,111,235,0.2); border-color: rgba(31,111,235,0.35); color: white;
}
div.stTabs [data-baseweb="tab-list"] button p { font-size: 1rem; font-weight: 600; }

/* Chat input viền nổi bật nhẹ */
.stChatInput textarea {
  border: 2px solid rgba(255,255,255,0.15) !important;
  border-radius: 12px !important;
}

/* Thu nhỏ padding nút feedback */
.small-btn > button { padding: .25rem .5rem; min-width: 0; border-radius: 10px; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# 1) Config from ENV/Secrets
# =========================
BACKEND_URL = os.getenv("BACKEND_URL") or st.secrets.get("BACKEND_URL", "http://localhost:8000")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD") or st.secrets.get("ADMIN_PASSWORD", "admin123")
DEFAULT_TIMEOUT = 60  # seconds

def _join(path: str) -> str:
    return f"{BACKEND_URL.rstrip('/')}/{path.lstrip('/')}"

def post_json(path: str, payload: dict):
    r = requests.post(_join(path), json=payload, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return {"text": r.text}

def post_form(path: str, form: dict):
    r = requests.post(_join(path), data=form, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return {"text": r.text}

def get_json(path: str):
    try:
        r = requests.get(_join(path), timeout=DEFAULT_TIMEOUT)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return {"text": r.text}
    except requests.RequestException:
        return None

def get_csv_as_df(path: str):
    try:
        return pd.read_csv(_join(path))
    except Exception:
        return None

# =========================
# 2) Session state
# =========================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_reply" not in st.session_state:
    st.session_state.last_reply = ""

# =========================
# 3) Tabs
# =========================
tab_user, tab_admin = st.tabs(["👨‍🎓 Người dùng", "🛠 Quản trị"])

# =========================
# 4) Tab Người dùng (Chat)
# =========================
with tab_user:
    st.title("🤖 Chatbot AI- trợ lí ảo hỗ trợ tư vấn tuyển sinh 10- THPT Marie Curie")

    # Lời chào đầu tiên
    if len(st.session_state.messages) == 0:
        with st.chat_message("assistant"):
            st.markdown(
                "CHào bạn! mình là chatbot tuyển sinh 10, sẵn sàng giải đáp mọi thắc mắc của bạn. "
                "Hãy đặt câu hỏi cho mình nhé!"
            )

    # ---- hiển thị toàn bộ lịch sử trước ----
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
        # hiện 2 nút feedback chỉ cho **tin nhắn trợ lí cuối cùng**
        is_last_assistant = (
            msg["role"] == "assistant" and i == len(st.session_state.messages) - 1
        )
        if is_last_assistant:
            # tìm câu hỏi liền trước (nếu có)
            prev_q = ""
            if i >= 1 and st.session_state.messages[i-1]["role"] == "user":
                prev_q = st.session_state.messages[i-1]["content"]

            # đặt hai nút cùng một hàng, sát nhau
            c1, c2, _ = st.columns([0.07, 0.07, 0.86])
            with c1:
                if st.button("👍", key=f"fb_up_{i}", help="Hài lòng", type="secondary", kwargs=None):
                    with suppress(Exception):
                        post_form("/feedback", {
                            "session_id": st.session_state.session_id,
                            "question": prev_q,
                            "answer": msg["content"],
                            "rating": "up",
                        })
                        st.success("Đã gửi phản hồi 👍")
            with c2:
                if st.button("👎", key=f"fb_dn_{i}", help="Chưa tốt", type="secondary"):
                    with suppress(Exception):
                        post_form("/feedback", {
                            "session_id": st.session_state.session_id,
                            "question": prev_q,
                            "answer": msg["content"],
                            "rating": "down",
                        })
                        st.success("Đã gửi phản hồi 👎")

    # ---- ô nhập luôn ở cuối trang ----
    user_input = st.chat_input("Nhập câu hỏi của bạn...")

    if user_input:
        # cập nhật state nhưng KHÔNG render ngay tại đây
        st.session_state.messages.append({"role": "user", "content": user_input})

        try:
            data = post_json(
                "/chat",
                {"messages": st.session_state.messages, "session_id": st.session_state.session_id},
            )
            reply = (data or {}).get("reply") or (data or {}).get("response") or "Xin lỗi, hiện chưa có phản hồi."
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.session_state.last_reply = reply
        except requests.RequestException as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Không thể kết nối tới backend. Kiểm tra `BACKEND_URL` trong Secrets hoặc thử lại sau.\n\n"
                           f"Chi tiết lỗi: `{e}`"
            })

        # rerender để tất cả tin nhắn hiển thị **phía trên**,
        # còn ô nhập vẫn nằm **cuối cùng**
        try:
            st.rerun()  # Streamlit >=1.30
        except Exception:
            st.experimental_rerun()

    # Ẩn debug trừ khi SHOW_DEBUG=1
    if os.getenv("SHOW_DEBUG") == "1":
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
        st.info("Nhập đúng mật khẩu để truy cập công cụ quản trị. Đặt `ADMIN_PASSWORD` trong ENV hoặc st.secrets.")
        st.stop()

    st.success("Đăng nhập quản trị thành công ✅")

    # 5.1 Kiểm tra backend
    st.subheader("✅ Kiểm tra tình trạng Backend")
    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("Ping /health"):
            result = get_json("/health")
            if result:
                st.write(result)
            else:
                st.error("Không gọi được `/health`. Kiểm tra BACKEND_URL/Service.")
    with colB:
        st.write(f"**BACKEND_URL:** `{BACKEND_URL}`")

    st.divider()

    # 5.2 Lịch sử hội thoại
    st.subheader("🗂 Lịch sử hội thoại")
    st.caption("Yêu cầu backend cung cấp endpoint JSON hoặc CSV. Thử các phương án phổ biến bên dưới.")
    tabs_hist = st.tabs(["/history (JSON)", "/chat_history.csv (CSV)"])
    with tabs_hist[0]:
        hist = get_json("/history")
        if hist and isinstance(hist, list):
            df_hist_json = pd.DataFrame(hist)
            st.dataframe(df_hist_json, use_container_width=True)
        elif hist:
            st.write(hist)
        else:
            st.info("Không có endpoint `/history` hoặc không truy cập được.")
    with tabs_hist[1]:
        df_hist_csv = get_csv_as_df("/chat_history.csv")
        if df_hist_csv is not None:
            st.dataframe(df_hist_csv, use_container_width=True)
        else:
            st.info("Không tìm thấy `/chat_history.csv` từ backend.")

    st.divider()

    # 5.3 Feedback
    st.subheader("📝 Feedback")
    tabs_fb = st.tabs(["/feedbacks (JSON)", "/feedback.csv (CSV)"])
    with tabs_fb[0]:
        fjson = get_json("/feedbacks")
        if fjson and isinstance(fjson, list):
            st.dataframe(pd.DataFrame(fjson), use_container_width=True)
        elif fjson:
            st.write(fjson)
        else:
            st.info("Không có endpoint `/feedbacks` hoặc không truy cập được.")
    with tabs_fb[1]:
        df_fb_csv = get_csv_as_df("/feedback.csv")
        if df_fb_csv is not None:
            st.dataframe(df_fb_csv, use_container_width=True)
        else:
            st.info("Không tìm thấy `/feedback.csv` từ backend.")

    st.divider()

    # 5.4 Thống kê: Top 10 câu hỏi được hỏi nhiều nhất
    st.subheader("📈 Top 10 câu hỏi được hỏi nhiều nhất")

    def _load_questions_series() -> pd.Series | None:
        """
        Trả về Series các câu hỏi (string) từ /history JSON hoặc /chat_history.csv.
        Cố gắng dò tên cột phổ biến: 'question', 'user_input', 'content', ...
        Hoặc lọc từ lịch sử theo role='user'.
        """
        # Ưu tiên JSON
        data = get_json("/history")
        df = None
        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)

        if df is None:
            df = get_csv_as_df("/chat_history.csv")

        if df is None or len(df) == 0:
            return None

        # Ưu tiên cột question
        for col in ["question", "user_input", "prompt", "text", "content"]:
            if col in df.columns:
                s = df[col].dropna().astype(str)
                if "role" in df.columns:
                    try:
                        s = df.loc[df["role"].astype(str).str.lower().eq("user"), col].dropna().astype(str)
                    except Exception:
                        pass
                return s if len(s) else None

        if {"role", "content"}.issubset(set(df.columns)):
            s = df.loc[df["role"].astype(str).str.lower().eq("user"), "content"].dropna().astype(str)
            return s if len(s) else None

        return None

    s_questions = _load_questions_series()
    if s_questions is None or len(s_questions) == 0:
        st.info("Chưa có dữ liệu câu hỏi để thống kê (cần `/history` JSON hoặc `/chat_history.csv`).")
    else:
        s_norm = (
            s_questions.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
        )
        top_counts = s_norm.value_counts().head(10)

        rep_map = {}
        for orig in s_questions:
            key = str(orig).strip().lower().replace("\n", " ")
            key = " ".join(key.split())
            if key not in rep_map:
                rep_map[key] = str(orig).strip()

        rows = [{"Câu hỏi": rep_map.get(k, k), "Số lần": int(v)} for k, v in top_counts.items()]
        df_top = pd.DataFrame(rows)

        st.dataframe(df_top, use_container_width=True, hide_index=True)
        st.bar_chart(df_top.set_index("Câu hỏi")["Số lần"])

    st.divider()

    # 5.5 Upload MC_chatbot.csv (nếu backend hỗ trợ)
    st.subheader("⬆️ Cập nhật MC_chatbot.csv (tuỳ chọn)")
    st.caption(
        "Frontend không ghi trực tiếp tệp vào server. Tạo endpoint `POST /upload_mc_data` ở backend "
        "để nhận file CSV nếu muốn cập nhật dữ liệu."
    )
    uploaded = st.file_uploader("Chọn file MC_chatbot.csv để tải lên backend", type=["csv"])
    if uploaded and st.button("Gửi lên backend (/upload_mc_data)"):
        try:
            url = _join("/upload_mc_data")
            files = {"file": ("MC_chatbot.csv", uploaded.getvalue(), "text/csv")}
            r = requests.post(url, files=files, timeout=DEFAULT_TIMEOUT)
            if r.status_code == 200:
                st.success("Đã gửi file lên backend.")
            else:
                st.warning(f"Backend trả {r.status_code}: {r.text}")
        except requests.RequestException as e:
            st.error(f"Lỗi gửi file: {e}")
