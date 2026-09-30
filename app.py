"""
NexusAgent MCP - Interactive Web Dashboard
==========================================
Author: Anurag Gurjar
GitHub: https://github.com/anuraggurjar01
LinkedIn: https://www.linkedin.com/in/anuraggurjar1/

A production-grade Streamlit web interface for the NexusAgent LangGraph + MCP system.
Features real-time chat, tool execution tracing, Human-in-the-Loop review, and developer profiling.
"""

import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# Load environment configuration
load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI

# ─── Configuration & Paths ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
NOTES_SERVER = str(PROJECT_ROOT / "mcp_servers" / "notes_server.py")
CALC_SERVER  = str(PROJECT_ROOT / "mcp_servers" / "calculator_server.py")

MCP_CONFIG = {
    "notes": {
        "command": sys.executable,
        "args": [NOTES_SERVER],
        "transport": "stdio",
    },
    "calculator": {
        "command": sys.executable,
        "args": [CALC_SERVER],
        "transport": "stdio",
    },
}

SYSTEM_PROMPT = """You are NexusAgent, an autonomous AI assistant engineered by Anurag Gurjar.
You are equipped with access to two dedicated Model Context Protocol (MCP) tool clusters:

1. Notes Tools:
   - list_notes: List all available markdown notes with metadata
   - read_note: Read the full content of a specific note by filename
   - search_notes: Search across all notes for keywords or phrases
   - create_note: Create a brand new markdown note
   - append_to_note: Safely append information or action items to an existing note

2. Calculator Tools:
   - add: Add two numbers
   - multiply: Multiply two numbers
   - divide: Divide two numbers safely (zero-division protected)
   - percentage: Calculate percentage of a value
   - summarize_numbers: Compute count, sum, mean, min, max, and standard deviation

Operational Directives:
- For general reasoning or knowledge questions, respond directly without invoking tools.
- When queries involve notes, documents, meetings, todos, or project backlogs, leverage the Notes tools.
- When queries require mathematical computations, financial ratios, or statistical analysis, invoke the Calculator tools.
- Chain tools seamlessly: read notes, parse values, compute figures, and summarize findings.
- Format all outputs with structured, professional Markdown.
"""

# ─── Page Setup ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="NexusAgent MCP | Anurag Gurjar",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS Styling ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .profile-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
        color: #F8FAFC;
    }
    .profile-name {
        font-size: 1.25rem;
        font-weight: 700;
        color: #38BDF8;
        margin-bottom: 0.2rem;
    }
    .profile-title {
        font-size: 0.88rem;
        color: #94A3B8;
        margin-bottom: 0.8rem;
    }
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 10px;
    }
    .tech-badge {
        background: #0F172A;
        border: 1px solid #38BDF8;
        color: #38BDF8;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 6px;
    }
    .tool-box {
        background: #0F172A;
        border-left: 3px solid #38BDF8;
        padding: 8px 12px;
        border-radius: 4px;
        margin: 6px 0;
        font-family: monospace;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar: Developer Profile & Configuration ───────────────────────
with st.sidebar:
    st.markdown("""
    <div class="profile-card">
        <div class="profile-name">Anurag Gurjar</div>
        <div class="profile-title">AI & Agentic Systems Developer</div>
        <p style="font-size: 0.85rem; color: #CBD5E1; margin-bottom: 12px;">
            Full-stack AI developer specializing in autonomous multi-agent systems, LangGraph state machines, and Model Context Protocol (MCP) integrations.
        </p>
        <div style="display: flex; gap: 10px; margin-bottom: 12px;">
            <a href="https://github.com/anuraggurjar01" target="_blank" style="text-decoration: none; color: #38BDF8; font-weight: 600; font-size: 0.85rem;">
                🐙 GitHub
            </a>
            <span style="color: #64748B;">•</span>
            <a href="https://www.linkedin.com/in/anuraggurjar1/" target="_blank" style="text-decoration: none; color: #38BDF8; font-weight: 600; font-size: 0.85rem;">
                💼 LinkedIn
            </a>
        </div>
        <div class="badge-container">
            <span class="tech-badge">LangGraph</span>
            <span class="tech-badge">Model Context Protocol</span>
            <span class="tech-badge">FastMCP</span>
            <span class="tech-badge">Python 3.11</span>
            <span class="tech-badge">OpenAI</span>
            <span class="tech-badge">ReAct</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("⚙️ Agent Controls")
    
    # API Key Input
    env_api_key = os.getenv("OPENAI_API_KEY", "")
    api_key_input = st.text_input(
        "OpenAI API Key",
        value=env_api_key,
        type="password",
        help="Leave blank to use OPENAI_API_KEY from environment/.env",
    )
    active_api_key = api_key_input.strip() if api_key_input.strip() else env_api_key

    # Model Selection
    selected_model = st.selectbox(
        "LLM Backbone",
        ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "o1-mini"],
        index=0,
    )

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

    st.markdown("---")
    st.subheader("🔌 Connected MCP Servers")
    st.markdown("""
    - 🟢 **Notes Manager** (`stdio`)
      - `list_notes`, `read_note`, `search_notes`, `create_note`, `append_to_note`
    - 🟢 **Calculator Server** (`stdio`)
      - `add`, `multiply`, `divide`, `percentage`, `summarize_numbers`
    """)

    st.markdown("---")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = st.session_state.get("thread_id", 0) + 1
        st.rerun()


# ─── Session State Initialization ─────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = 1


# ─── Main View Header ─────────────────────────────────────────────────
st.markdown('<div class="main-header">⚡ NexusAgent MCP</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production-Grade LangGraph ReAct Agent with Model Context Protocol (MCP) Tool Clusters</div>', unsafe_allow_html=True)

# Tabs
tab_chat, tab_tools, tab_architecture = st.tabs(["💬 Interactive Agent Chat", "🛠️ MCP Tool Registry", "📐 System Architecture"])

with tab_tools:
    st.markdown("### 🔌 Model Context Protocol (MCP) Tool Inventory")
    st.markdown("The agent dynamically connects to two subprocesses exposing typed tools via the official MCP specification:")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("#### 📝 Notes Server (`notes_server.py`)")
        st.markdown("""
        * **`list_notes()`**: Enumerates all `.md` files in the repository knowledge base with size and timestamps.
        * **`read_note(filename)`**: Reads full text content with built-in path-traversal security verification.
        * **`search_notes(query)`**: High-speed, case-insensitive keyword search across all notes with line matching.
        * **`create_note(filename, content)`**: Atomically writes a new markdown document into the knowledge repository.
        * **`append_to_note(filename, text)`**: Appends action items or summaries to an existing document safely.
        """)

    with col_t2:
        st.markdown("#### 🧮 Calculator Server (`calculator_server.py`)")
        st.markdown("""
        * **`add(a, b)`**: Computes the sum of two real numbers.
        * **`multiply(a, b)`**: Multiplies two numerical arguments.
        * **`divide(a, b)`**: Safe floating-point division with zero-division validation.
        * **`percentage(value, percent)`**: Calculates percentage ratio of a given base value.
        * **`summarize_numbers(numbers)`**: Computes count, sum, mean, min, max, and sample standard deviation.
        """)

with tab_architecture:
    st.markdown("### 📐 System Architecture & Execution Flow")
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                     Streamlit Web UI / CLI                  │
    └──────────────────────────────┬──────────────────────────────┘
                                   │ User Query
                                   ▼
    ┌─────────────────────────────────────────────────────────────┐
    │             LangGraph ReAct State Machine Engine            │
    │                                                             │
    │   START ───► call_model (LLM Inference)                     │
    │                  │                                          │
    │             tools_condition                                 │
    │             /            \                                  │
    │       (Tool Call)     (No Tools)                            │
    │            │               │                                │
    │          tools ─────────► END                               │
    │            │                                                │
    │       (Loop back)                                           │
    └────────────┬────────────────────────────────────────────────┘
                 │ MultiServerMCPClient (stdio IPC)
         ┌───────┴──────────────────┐
         ▼                          ▼
    ┌──────────────────────┐   ┌──────────────────────┐
    │  Notes MCP Server    │   │ Calculator Server    │
    │  (FastMCP stdio)     │   │ (FastMCP stdio)      │
    └──────────────────────┘   └──────────────────────┘
    ```
    """)


# ─── Chat Interface ───────────────────────────────────────────────────
with tab_chat:
    # Display prompt suggestions if conversation is empty
    if not st.session_state.messages:
        st.markdown("**Try a sample prompt:**")
        sample_cols = st.columns(4)
        sample_prompts = [
            "What notes do I have available?",
            "Read meeting notes & calculate 15% of $250K budget",
            "Search notes for all action items and todos",
            "Give me statistical summary of: 85, 92, 78, 95, 88",
        ]
        for col, prompt in zip(sample_cols, sample_prompts):
            if col.button(prompt, use_container_width=True):
                st.session_state.pending_prompt = prompt
                st.rerun()

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if "tool_calls" in msg and msg["tool_calls"]:
                with st.expander(f"🛠️ Executed {len(msg['tool_calls'])} MCP Tool Call(s)", expanded=False):
                    for tc in msg["tool_calls"]:
                        st.markdown(f"**Tool:** `{tc.get('name')}`")
                        st.json(tc.get("args", {}))
                        if "output" in tc:
                            st.markdown("**Result:**")
                            st.code(str(tc["output"]), language="text")
            st.markdown(msg["content"])

    # Input handling
    user_prompt = st.chat_input("Ask NexusAgent anything (e.g., analyze notes, perform calculations)...")
    if "pending_prompt" in st.session_state:
        user_prompt = st.session_state.pop("pending_prompt")

    if user_prompt:
        if not active_api_key:
            st.error("⚠️ Please configure your OpenAI API Key in `.env` or in the sidebar to proceed.")
            st.stop()

        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Agent processing
        with st.chat_message("assistant"):
            with st.spinner("NexusAgent is reasoning and orchestrating MCP tools..."):
                async def execute_agent_turn(query: str, history: list):
                    # Set API key in environment for LangChain
                    os.environ["OPENAI_API_KEY"] = active_api_key

                    client = MultiServerMCPClient(MCP_CONFIG)
                    tools = await client.get_tools()

                    model = ChatOpenAI(model=selected_model, temperature=temperature)

                    def call_model(state: MessagesState):
                        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
                        response = model.bind_tools(tools).invoke(messages)
                        return {"messages": [response]}

                    builder = StateGraph(MessagesState)
                    builder.add_node("call_model", call_model)
                    builder.add_node("tools", ToolNode(tools))
                    builder.add_edge(START, "call_model")
                    builder.add_conditional_edges("call_model", tools_condition)
                    builder.add_edge("tools", "call_model")

                    checkpointer = MemorySaver()
                    graph = builder.compile(checkpointer=checkpointer)

                    # Build message input payload
                    input_messages = []
                    for m in history:
                        if m["role"] == "user":
                            input_messages.append(HumanMessage(content=m["content"]))
                        elif m["role"] == "assistant":
                            input_messages.append(AIMessage(content=m["content"]))
                    input_messages.append(HumanMessage(content=query))

                    config = {"configurable": {"thread_id": f"web-session-{st.session_state.thread_id}"}}
                    result = await graph.ainvoke({"messages": input_messages}, config=config)

                    final_messages = result["messages"]
                    assistant_msg = final_messages[-1].content

                    # Extract all tool calls made during execution
                    captured_tool_calls = []
                    for i, m in enumerate(final_messages):
                        if hasattr(m, "tool_calls") and m.tool_calls:
                            for tc in m.tool_calls:
                                call_info = {
                                    "name": tc.get("name"),
                                    "args": tc.get("args", {}),
                                    "id": tc.get("id"),
                                }
                                # Match with subsequent ToolMessage for output
                                for next_m in final_messages[i+1:]:
                                    if isinstance(next_m, ToolMessage) and next_m.tool_call_id == tc.get("id"):
                                        call_info["output"] = next_m.content
                                        break
                                captured_tool_calls.append(call_info)

                    return assistant_msg, captured_tool_calls

                try:
                    assistant_response, tool_calls_made = asyncio.run(
                        execute_agent_turn(user_prompt, st.session_state.messages[:-1])
                    )

                    if tool_calls_made:
                        with st.expander(f"🛠️ Executed {len(tool_calls_made)} MCP Tool Call(s)", expanded=True):
                            for tc in tool_calls_made:
                                st.markdown(f"**Tool:** `{tc.get('name')}`")
                                st.json(tc.get("args", {}))
                                if "output" in tc:
                                    st.markdown("**Output:**")
                                    st.code(str(tc["output"]), language="text")

                    st.markdown(assistant_response)

                    # Save to state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_response,
                        "tool_calls": tool_calls_made,
                    })

                except Exception as e:
                    st.error(f"Error during agent execution: {e}")
