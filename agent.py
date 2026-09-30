# agent.py
"""
NexusAgent MCP - Production-Grade Agentic Orchestration Engine
=============================================================
Author: Anurag Gurjar (https://github.com/anuraggurjar01)
LinkedIn: https://www.linkedin.com/in/anuraggurjar1/

A ReAct agent built with LangGraph that connects to multiple Model Context Protocol
(MCP) servers (Notes + Calculator) via langchain-mcp-adapters with Human-in-the-Loop.

Features:
- Dual MCP server integration (stdio transport)
- Human-in-the-Loop (HITL) approval / denial checkpoints
- Bi-directional knowledge tool integration (read & write notes)
- Multi-turn conversation memory with session checkpointing
"""

import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Auto-load environment variables from .env
load_dotenv()

from langchain_core.messages import AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI

PROJECT_ROOT = Path(__file__).parent.resolve()
NOTES_SERVER = str(PROJECT_ROOT / "mcp_servers" / "notes_server.py")
CALC_SERVER  = str(PROJECT_ROOT / "mcp_servers" / "calculator_server.py")

LLM_MODEL       = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

SYSTEM_PROMPT = """You are NexusAgent, an autonomous AI assistant engineered by Anurag Gurjar.
You are equipped with access to two dedicated Model Context Protocol (MCP) tool clusters:

1. Notes Tools:
   - list_notes: List all available markdown notes with metadata (size, last modified)
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
- If a tool call is denied by the user under Human-in-the-Loop review, gracefully adapt and provide the best response possible.
- Format all outputs with structured, professional Markdown.
"""

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


async def build_agent(client: MultiServerMCPClient, checkpointer=None):
    """
    Builds the LangGraph ReAct graph.

    When a checkpointer is provided, interrupt_before=["tools"] is enabled,
    which pauses the graph before every tool execution for human review.
    """
    tools = await client.get_tools()
    print(f"\nLoaded {len(tools)} tools from MCP servers:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:60]}...")

    model = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

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

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["tools"] if checkpointer else [],
    )


async def handle_hitl_loop(graph, config: dict) -> tuple[list[str], list[str]]:
    """
    After each graph invocation, check whether the graph is paused before
    the tools node. If so, show the pending tool calls to the user and ask
    for approval. Loop until the graph reaches END.

    Returns (approved_tools, denied_tools) for summary display.
    """
    approved = []
    denied   = []

    while True:
        state = graph.get_state(config)

        if not state.next or "tools" not in state.next:
            break

        last_msg      = state.values["messages"][-1]
        pending_calls = last_msg.tool_calls

        print("\n" + "-" * 50)
        print("  [Human Review] Agent wants to call:")
        for tc in pending_calls:
            args_str = (
                ", ".join(f"{k}={repr(v)}" for k, v in tc["args"].items())
                if tc["args"] else "no args"
            )
            print(f"    - {tc['name']}({args_str})")
        print("-" * 50)

        decision = input("  Approve? (y/n): ").strip().lower()

        if decision == "y":
            for tc in pending_calls:
                approved.append(tc["name"])
            await graph.ainvoke(None, config=config)
        else:
            for tc in pending_calls:
                denied.append(tc["name"])

            # Step 1 — satisfy the graph: inject ToolMessages so every tool_call_id
            # has a matching response. as_node="tools" advances the graph past
            # the tools node without actually executing anything.
            await graph.aupdate_state(
                config,
                {
                    "messages": [
                        ToolMessage(
                            content="Denied.",
                            tool_call_id=tc["id"],
                            name=tc["name"],
                        )
                        for tc in pending_calls
                    ]
                },
                as_node="tools",
            )

            # Step 2 — inject the final reply directly as the agent's response,
            # bypassing the LLM entirely so it cannot compute the answer on its own.
            tool_names = ", ".join(f"`{tc['name']}`" for tc in pending_calls)
            await graph.aupdate_state(
                config,
                {
                    "messages": [
                        AIMessage(
                            content=(
                                f"Action blocked. You denied the {tool_names} tool(s), "
                                f"so I cannot complete this request. "
                                f"Try again and approve the tool when prompted if you'd like the result."
                            )
                        )
                    ]
                },
                as_node="call_model",
            )
            break

    return approved, denied


async def run_interactive_chat():
    print("=" * 65)
    print("  NexusAgent MCP | Enterprise Agentic Orchestration")
    print("  Engineered by: Anurag Gurjar (github.com/anuraggurjar01)")
    print("  Human-in-the-Loop (HITL) : Active")
    print("=" * 65)
    print(f"\nNotes Directory : {PROJECT_ROOT / 'notes'}")
    print(f"LLM Model       : {LLM_MODEL}")
    print("\nConnecting to MCP servers (Notes + Calculator)...")

    client       = MultiServerMCPClient(MCP_CONFIG)
    checkpointer = MemorySaver()
    graph        = await build_agent(client, checkpointer)

    print("\n" + "-" * 65)
    print("NexusAgent ready! Type your questions or instructions below.")
    print("  help   - show sample queries & tool operations")
    print("  clear  - reset conversation memory")
    print("  quit   - exit session")
    print("-" * 65)

    # Each session gets a unique thread_id so the checkpointer stores it separately.
    # Incrementing session_id on 'clear' effectively starts a fresh conversation.
    session_id = 0

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break
        if user_input.lower() == "help":
            print_help()
            continue
        if user_input.lower() == "clear":
            session_id += 1
            print("Conversation cleared.")
            continue

        config = {"configurable": {"thread_id": f"session-{session_id}"}}

        try:
            # Pass only the new user message.
            # The checkpointer appends it to the thread's existing message history.
            await graph.ainvoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
            )

            approved, denied = await handle_hitl_loop(graph, config)

            final_msg = graph.get_state(config).values["messages"][-1]
            print(f"\nNexusAgent: {final_msg.content}")

            if approved:
                print(f"\n  Approved Tools : {', '.join(approved)}")
            if denied:
                print(f"  Denied Tools   : {', '.join(denied)}")

        except Exception as e:
            print(f"\nError: {e}")


def print_help():
    print("""
NexusAgent MCP - Sample Operations & Prompts
--------------------------------------------

Notes Operations:
  "What notes do I have available?"
  "Read my meeting notes"
  "Search for anything about budget or roadmap"
  "Create a new note called 'release_plan.md' with v2.0 checklist"
  "Append an action item to todo.md: Finalize security audit"

Mathematical & Statistical Analysis:
  "What is 15% of 250?"
  "Calculate (45 + 32) x 8"
  "Give me summary statistics for: 85, 92, 78, 95, 88"

Chained Multi-Tool Orchestration:
  "Read the meeting notes and calculate 15% of the $250K budget"
  "Search notes for budget items, calculate total spend, and append to todo.md"

Human-in-the-Loop (HITL) Verification:
  "Read my meeting notes"              -> inspect pending tool, enter 'y' or 'n'
  "Calculate 20% of 500"               -> deny 'percentage' tool, see agent adapt
  "Create note 'test.md' and add 5+5"  -> selectively approve or deny steps

Session Commands:
  clear  - Reset active thread memory
  help   - Show this guide
  quit   - Terminate session
""")


async def run_single_query(query: str):
    """Single query mode — no HITL, no checkpointer."""
    print(f"\nQuery: {query}\n")

    client = MultiServerMCPClient(MCP_CONFIG)
    graph  = await build_agent(client)

    result = await graph.ainvoke({"messages": [{"role": "user", "content": query}]})

    print(f"\nResponse:\n{result['messages'][-1].content}")

    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"\nTool called: {tc['name']}({tc['args']})")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] Configuration Error: OPENAI_API_KEY not found.")
        print("  1. Copy .env.example to .env and insert your API key: OPENAI_API_KEY=sk-...")
        print("  2. Or set it in your shell: set OPENAI_API_KEY=your_key (CMD) or $env:OPENAI_API_KEY='your_key' (PowerShell)")
        sys.exit(1)

    if "--query" in sys.argv:
        idx = sys.argv.index("--query")
        if idx + 1 < len(sys.argv):
            asyncio.run(run_single_query(sys.argv[idx + 1]))
        else:
            print('Usage: python agent.py --query "your question here"')
    else:
        asyncio.run(run_interactive_chat())
