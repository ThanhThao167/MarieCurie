# streamlit_chat.py — Giao diện giống ảnh mẫu (không thay đổi phần Quản trị)
# - Tabs: 👨‍🎓 Người dùng / 🛠 Quản trị
# - Tiêu đề lớn, nền tối, bong bóng chat dạng thẻ, icon avatar
# - Nút 👍/👎 dưới câu trả lời mới nhất; bong bóng "Đang suy nghĩ…" ở cuối
# - KHÔNG SỬA phần Quản trị ở cuối file

import os
import uuid
import requests
import pandas as pd
from datetime import datetime
from contextlib import suppress
from typing import Optional
import streamlit as st

# ---------------- Page setup & CSS ----------------
st.set_page_config(
    page_title="Chatbot AI- trợ lí ảo hỗ trợ tư vấn tuyển sinh 10- THPT Marie Curie",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Tabs: bo tròn + nhấn tab đang chọn */
    div.stTabs [data-baseweb="tab-list"]{ gap:.35rem; }
    div.stTabs [data-baseweb="tab-list"] button[role="tab"]{
      background:transparent; border:1px solid transparent; border-bottom:none;
      padding:.5rem 1rem; border-radius:12px 12px 0 0;
    }
    div.stTabs [data-baseweb="tab-list"] button[role="tab"][aria-selected="true"]{
      background:rgba(255,255,255,.08); border-color:rgba(255,255,255,.15); color:#fff;
    }
    div.stTabs [data-baseweb="tab-list"] button p{ font-size:1rem; font-weight:700; }

    /* Tiêu đề lớn giống ảnh */
    .hero-title{ font-size:2.2rem; font-weight:800; margin:.3rem 0 .8rem 0; }

    /* Chat bubble tối, bo lớn */
    .chat-bubble{
      background:#151a22; border:1px solid rgba(255,255,255,.08);
      border-radius:16px; padding:.75rem 1rem; margin:.35rem 0;
    }
    .chat-bubble.user{
      background:#1a1f29; border-color:rgba(244,63,94,.25);  /* đỏ nhạt */
    }
    .chat-bubble.assistant{
      background:#171f17; border-color:rgba(234,179,8,.25);  /* vàng nhạt */
    }

    /* Văn bản TRẮNG trong mọi khung chat */
    .chat-bubble, .chat-bubble *{ color:#fff !important; }

    .soft{ opacity:.9; }

    /* Ô nhập đỏ viền như ảnh */
    .stChatInput textarea{
      border:2px solid rgba(239,68,68,.40) !important; border-radius:12px !important;
    }

    .small-btn > button{ padding:.25rem .5rem; min-width:0; border-radius:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- Utils ----------------
def do_rerun() -> None:
    """Rerun an toàn cho mọi phiên bản Streamlit."""
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

# ---------------- Config ----------------
BACKEND_URL = os.getenv("BACKEND_URL") or st.secrets.get("BACKEND_URL", "http://localhost:8000")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD") or st.secrets.get("ADMIN_PASSWORD", "admin123")
DEFAULT_TIMEOUT = 60

def _join(p:str)->str: return f"{BACKEND_URL.rstrip('/')}/{p.lstrip('/')}"

def post_json(p:str, payload:dict):
    r=requests.post(_join(p), json=payload, timeout=DEFAULT_TIMEOUT); r.raise_for_status()
    try: return r.json()
    except: return {"text": r.text}

def post_form(p:str, form:dict):
    r=requests.post(_join(p), data=form, timeout=DEFAULT_TIMEOUT); r.raise_for_status()
    try: return r.json()
    except: return {"text": r.text}

def get_json(p:str):
    try:
        r=requests.get(_join(p), timeout=DEFAULT_TIMEOUT)
        if r.status_code==404: return None
        r.raise_for_status()
        try: return r.json()
        except: return {"text": r.text}
    except requests.RequestException:
        return None

def get_csv_as_df(p:str):
    try: return pd.read_csv(_join(p))
    except: return None

# ---------------- Session state ----------------
if "session_id" not in st.session_state: st.session_state.session_id=str(uuid.uuid4())
if "messages"   not in st.session_state: st.session_state.messages=[]
if "last_reply" not in st.session_state: st.session_state.last_reply=""
if "awaiting_response" not in st.session_state: st.session_state.awaiting_response=False

# ---------------- Tabs ----------------
tab_user, tab_admin = st.tabs(["👨‍🎓 Người dùng", "🛠 Quản trị"])

# ---------------- User tab ----------------
with tab_user:
    st.markdown('<div class="hero-title">🤖 Chatbot AI- trợ lí ảo hỗ trợ tư vấn tuyển sinh 10- THPT Marie Curie</div>', unsafe_allow_html=True)

    chat_box = st.container()   # toàn bộ đoạn hội thoại
    with chat_box:
        # Lời chào đầu tiên
        if not st.session_state.messages:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown('<div class="chat-bubble assistant soft">Chào bạn! Mình là chatbot tuyển sinh 10. Hãy đặt câu hỏi để mình hỗ trợ nhé!</div>', unsafe_allow_html=True)

        # tìm chỉ số câu trả lời assistant cuối để đặt nút 👍👎
        last_ass_idx = None
        for i, m in enumerate(st.session_state.messages):
            if m.get("role") == "assistant":
                last_ass_idx = i

        # render lịch sử (bong bóng & avatar như ảnh)
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                with st.chat_message("user", avatar="🙂"):
                    st.markdown(f'<div class="chat-bubble user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(f'<div class="chat-bubble assistant">{msg["content"]}</div>', unsafe_allow_html=True)

            # hàng nút phản hồi ngay dưới câu trả lời mới nhất
            if i == last_ass_idx and msg["role"] == "assistant":
                prev_q = ""
                for j in range(i-1, -1, -1):
                    if st.session_state.messages[j]["role"] == "user":
                        prev_q = st.session_state.messages[j]["content"]; break
                c1, c2, _ = st.columns([0.07, 0.07, 0.86])
                with c1:
                    if st.button("👍", key=f"fb_up_{i}", help="Hài lòng"):
                        with suppress(Exception):
                            post_form("/feedback", {
                                "session_id": st.session_state.session_id,
                                "question": prev_q, "answer": msg["content"], "rating": "up"
                            }); st.success("Đã gửi phản hồi 👍")
                with c2:
                    if st.button("👎", key=f"fb_dn_{i}", help="Chưa tốt"):
                        with suppress(Exception):
                            post_form("/feedback", {
                                "session_id": st.session_state.session_id,
                                "question": prev_q, "answer": msg["content"], "rating": "down"
                            }); st.success("Đã gửi phản hồi 👎")

        # nếu đang chờ trả lời: hiện bong bóng thinking ở CUỐI
        if st.session_state.awaiting_response:
            with st.chat_message("assistant", avatar="⏳"):
                st.markdown('<div class="chat-bubble assistant soft">⏳ <em>Đang suy nghĩ…</em></div>', unsafe_allow_html=True)

    # ô nhập luôn ở dưới cùng
    user_input = st.chat_input("Nhập câu hỏi của bạn...")

    # bước 1: thêm câu hỏi & bật chờ -> rerun
    if user_input:
        st.session_state.messages.append({"role":"user","content": user_input})
        st.session_state.awaiting_response = True
        try:
            st.rerun()
        except Exception:
            try: st.experimental_rerun()
            except Exception: pass

# bước 2: nếu đang chờ -> gọi backend, thêm câu trả lời rồi rerun để hiển thị ở cuối
if st.session_state.awaiting_response:
    try:
        data = post_json("/chat", {
            "messages": st.session_state.messages,
            "session_id": st.session_state.session_id
        })
        reply = (data or {}).get("reply") or (data or {}).get("response") or "Xin lỗi, hiện chưa có phản hồi."
    except requests.RequestException as e:
        reply = "Không thể kết nối tới backend. Kiểm tra BACKEND_URL trong Secrets hoặc thử lại sau.\n\n" + f"Chi tiết lỗi: `{e}`"
    st.session_state.messages.append({"role":"assistant","content": reply})
    st.session_state.last_reply = reply
    st.session_state.awaiting_response = False
    try:
        st.rerun()
    except Exception:
        try: st.experimental_rerun()
        except Exception: pass

# ============================================================
#                       PHẦN QUẢN TRỊ (GIỮ NGUYÊN)
# ============================================================
with tab_admin:
    st.header("🛠 Khu vực Quản trị")
    pwd = st.text_input("Nhập mật khẩu quản trị", type="password")
    if pwd != ADMIN_PASSWORD:
        st.info("Nhập đúng mật khẩu để truy cập công cụ quản trị. Đặt `ADMIN_PASSWORD` trong ENV hoặc st.secrets.")
        st.stop()

    st.success("Đăng nhập quản trị thành công ✅")

    st.subheader("✅ Kiểm tra tình trạng Backend")
    colA, colB = st.columns([1,1])
    with colA:
        if st.button("Ping /health"):
            result = get_json("/health")
            st.write(result if result else "Không gọi được `/health`.")
    with colB: st.write(f"**BACKEND_URL:** `{BACKEND_URL}`")

    st.divider()
    st.subheader("🗂 Lịch sử hội thoại")
    st.caption("Đọc từ `/history` (JSON) hoặc `/chat_history.csv` (CSV).")
    tabs_hist = st.tabs(["/history (JSON)", "/chat_history.csv (CSV)"])
    with tabs_hist[0]:
        hist = get_json("/history")
        if isinstance(hist, list) and hist:
            st.dataframe(pd.DataFrame(hist), use_container_width=True)
        else:
            st.info("Không có endpoint `/history` hoặc không truy cập được.")
    with tabs_hist[1]:
        df_hist = get_csv_as_df("/chat_history.csv")
        if df_hist is not None:
            st.dataframe(df_hist, use_container_width=True)
        else:
            st.info("Không tìm thấy `/chat_history.csv`.")

    st.divider()
    st.subheader("📝 Feedback")
    tabs_fb = st.tabs(["/feedbacks (JSON)", "/feedback.csv (CSV)"])
    with tabs_fb[0]:
        fjson = get_json("/feedbacks")
        if isinstance(fjson, list) and fjson:
            st.dataframe(pd.DataFrame(fjson), use_container_width=True)
        else:
            st.info("Không có endpoint `/feedbacks` hoặc không truy cập được.")
    with tabs_fb[1]:
        df_fb = get_csv_as_df("/feedback.csv")
        if df_fb is not None:
            st.dataframe(df_fb, use_container_width=True)
        else:
            st.info("Không tìm thấy `/feedback.csv`.")

    st.divider()
    st.subheader("📈 Top 10 câu hỏi được hỏi nhiều nhất")
    def _load_questions_series() -> Optional[pd.Series]:
        data = get_json("/history")
        df = pd.DataFrame(data) if isinstance(data, list) and data else get_csv_as_df("/chat_history.csv")
        if df is None or df.empty: return None
        for col in ["question","user_input","prompt","text","content"]:
            if col in df.columns:
                s = df[col].dropna().astype(str)
                if "role" in df.columns:
                    try: s = df.loc[df["role"].astype(str).str.lower().eq("user"), col].dropna().astype(str)
                    except: pass
                return s if not s.empty else None
        if {"role","content"}.issubset(df.columns):
            s = df.loc[df["role"].astype(str).str.lower().eq("user"), "content"].dropna().astype(str)
            return s if not s.empty else None
        return None
    s = _load_questions_series()
    if s is None or s.empty:
        st.info("Chưa có dữ liệu câu hỏi (cần `/history` JSON hoặc `/chat_history.csv`).")
    else:
        s_norm = s.astype(str).str.strip().str.lower().str.replace(r"\\s+"," ", regex=True)
        counts = s_norm.value_counts().head(10)
        rep = {}
        for t in s:
            k=" ".join(str(t).strip().lower().split())
            if k not in rep: rep[k]=str(t).strip()
        df_top = pd.DataFrame({"Câu hỏi":[rep.get(k,k) for k in counts.index], "Số lần":counts.values})
        st.dataframe(df_top, use_container_width=True)
        st.bar_chart(df_top.set_index("Câu hỏi")["Số lần"])

    st.divider()
    st.subheader("⬆️ Cập nhật MC_chatbot.csv (tuỳ chọn)")
    st.caption("Frontend không ghi trực tiếp file lên server. Tạo endpoint `POST /upload_mc_data` ở backend nếu cần.")
    up = st.file_uploader("Chọn file MC_chatbot.csv", type=["csv"])
    if up and st.button("Gửi lên backend (/upload_mc_data)"):
        try:
            r = requests.post(_join("/upload_mc_data"),
                              files={"file": ("MC_chatbot.csv", up.getvalue(), "text/csv")},
                              timeout=DEFAULT_TIMEOUT)
            st.success("Đã gửi file lên backend.") if r.status_code==200 else st.warning(f"Backend trả {r.status_code}: {r.text}")
        except requests.RequestException as e:
            st.error(f"Lỗi gửi file: {e}")
