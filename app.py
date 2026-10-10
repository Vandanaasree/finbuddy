"""FinBuddy web app (Streamlit). Same agent as main.py, with a chat page and
Approve / Reject buttons in place of the terminal prompt.

    streamlit run app.py
"""

import os
import sqlite3
import uuid

import streamlit as st

st.set_page_config(page_title="FinBuddy", page_icon="💸")

# Copy hosting secrets (API key, model name, ...) into environment variables BEFORE the
# finbuddy modules are imported, because they read their settings at import time.
try:
    for _key in st.secrets:
        if isinstance(st.secrets[_key], (str, int, float)):
            os.environ.setdefault(_key, str(st.secrets[_key]))
except Exception:  # no secrets file locally; .env or real env vars are used instead
    pass

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from finbuddy.graph import build_graph  # noqa: E402
from finbuddy.llm import get_llm  # noqa: E402
from seed_demo_data import seed  # noqa: E402

MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", "25"))  # per visitor session, protects your API credits
ACCESS_CODE = os.getenv("ACCESS_CODE", "")  # optional: set it to keep the app private


def text_of(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


@st.cache_resource
def get_memory() -> SqliteSaver:
    conn = sqlite3.connect(os.getenv("FINBUDDY_MEMORY_DB", "memory.db"), check_same_thread=False)
    return SqliteSaver(conn)


@st.cache_resource
def get_model():
    return get_llm()


def new_session() -> None:
    """Every visitor gets their own private user id, so data and chat never mix."""
    user = f"web-{uuid.uuid4().hex[:8]}"
    st.session_state.user = user
    st.session_state.agent = build_graph(get_model(), user, checkpointer=get_memory())
    st.session_state.config = {"configurable": {"thread_id": f"user-{user}"}, "recursion_limit": 30}
    st.session_state.chat = []  # (role, text) pairs shown on screen
    st.session_state.pending = None  # approval request waiting for a click
    st.session_state.count = 0


def run(payload) -> None:
    """Send input to the agent. Either store a pending approval or append the reply."""
    try:
        result = st.session_state.agent.invoke(payload, st.session_state.config)
    except Exception as exc:  # keep the page alive on API/network errors
        st.session_state.pending = None
        st.session_state.chat.append(("assistant", f"Sorry, something went wrong: {type(exc).__name__}: {exc}"))
        return
    if result.get("__interrupt__"):
        st.session_state.pending = result["__interrupt__"][0].value
    else:
        st.session_state.pending = None
        st.session_state.chat.append(("assistant", text_of(result["messages"][-1])))


# ---------- access gate (optional) ----------
if ACCESS_CODE and not st.session_state.get("unlocked"):
    st.title("💸 FinBuddy")
    code = st.text_input("Access code", type="password")
    if code:
        if code == ACCESS_CODE:
            st.session_state.unlocked = True
            st.rerun()
        else:
            st.error("Wrong code.")
    st.stop()

if "agent" not in st.session_state:
    try:
        new_session()
    except Exception as exc:
        st.error(f"Could not start the agent: {exc}\n\nCheck that your API key is set in the app's secrets.")
        st.stop()

# ---------- sidebar ----------
with st.sidebar:
    st.header("FinBuddy")
    st.caption("A personal-finance AI agent. It uses tools on a database and asks you to approve risky actions.")
    if st.button("Load demo data", use_container_width=True):
        n = seed(st.session_state.user)
        st.success(f"Added {n} demo transactions and 3 budgets.")
    if st.button("Start over (new user)", use_container_width=True):
        new_session()
        st.rerun()
    st.markdown("**Try:**")
    st.markdown(
        "- I spent 450 on lunch today\n"
        "- how much did I spend on food this month?\n"
        "- which categories am I over budget in?\n"
        "- I bought a laptop for 55000\n"
        "- delete transaction 5\n"
        "- convert 5000 INR to USD"
    )
    st.caption(f"Messages used: {st.session_state.count}/{MAX_MESSAGES}")
    st.caption("Demo app: your data is private to this session and may be reset when the server restarts.")

# ---------- chat ----------
st.title("💸 FinBuddy")
st.caption("Tell me about your spending in plain English.")

for role, text in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(text)

pending = st.session_state.pending
if pending:
    with st.chat_message("assistant"):
        st.warning("**Approval needed** before I do this:")
        for a in pending["actions"]:
            st.markdown(f"- `{a['tool']}({a['args']})`  \n  *{a['reason']}*")
        col1, col2 = st.columns(2)
        approve = col1.button("✅ Approve", use_container_width=True)
        reject = col2.button("❌ Reject", use_container_width=True)
    if approve or reject:
        with st.spinner("Working..."):
            run(Command(resume={"approved": bool(approve)}))
        st.rerun()

prompt = st.chat_input("e.g. I spent 450 on lunch today", disabled=pending is not None)
if prompt:
    if st.session_state.count >= MAX_MESSAGES:
        st.session_state.chat.append(("assistant", "You've reached the message limit for this demo session."))
    else:
        st.session_state.count += 1
        st.session_state.chat.append(("user", prompt))
        with st.spinner("Thinking..."):
            run({"messages": [HumanMessage(prompt)]})
    st.rerun()
