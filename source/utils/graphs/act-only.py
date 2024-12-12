import asyncio
from typing import List, Dict, Any, Optional, Union, Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from utils.models import get_text_generator
from langchain.prompts import ChatPromptTemplate, AIMessagePromptTemplate
from langchain_core.messages import ToolMessage, message_chunk_to_message
from langchain_core.callbacks.manager import adispatch_custom_event
import json

class MessagesState(TypedDict):
    messages: Annotated[list, add_messages]


class AgentNode:
    def __init__(self, llm, tools, role="agent"):
        self.tools = {tool.name: tool for tool in tools}
        self.tooled_llm = llm.bind_tools(tools)
        self.role = role  # @TODO change to agent

    async def __call__(self, state: MessagesState, config: RunnableConfig = None):
        messages = state["messages"]
        data = {"input": messages[-1]}

        await adispatch_custom_event(
            f"on_{self.role}_start",  # @TODO maybe should come from answer's langgraph node
            data=data,
            config=config
        )

        stream = self.tooled_llm.astream(messages, config)

        message = None
        async for chunk in stream:
            message = message + chunk if message else chunk
            chunk_data = {"chunk": message}
            await adispatch_custom_event(
                f"on_{self.role}_stream",
                data=chunk_data,
                config=config
            )

        message = message_chunk_to_message(message)
        data |= {"output": message}

        await adispatch_custom_event(
            f"on_{self.role}_end",
            data=data,
            config=config,
        )

        tool_calls = message.additional_kwargs.get("tool_calls", [])
        for tool_call in tool_calls:
            action = {
                "tool": tool_call["function"]["name"],
                "args": tool_call["function"]["arguments"],
                "index": tool_call["index"],
                "call_id": tool_call["id"],
                "run_id": get_run_id_from_message(message),
            }

            action_data = {"action": action}
            await adispatch_custom_event(
                f"on_{self.role}_action",
                data=action_data,
                config=config
            )

        return {"messages": message}


class ActionNode:
    """A node that runs the tools requested in the last Message."""

    def __init__(self, tools: list, role="action") -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.role = "action"

    async def __call__(
        self,
        inputs: dict,
        config: RunnableConfig = None,
    ):
        messages = inputs.get("messages", [])
        if not messages:
            raise ValueError("No message found in input")

        message = messages[-1]
        outputs = []
        for tool_call in message.tool_calls:
            data = {"tool_call": tool_call}
            await adispatch_custom_event(
                f"on_{self.role}_start",
                data=data,
                config=config
            )

            tool_result = await asyncio.to_thread(
                self.tools_by_name[tool_call["name"]].ainvoke,
                tool_call["args"],
            )

            if asyncio.iscoroutine(tool_result):
                resolved_tool_result = await tool_result
            else:
                resolved_tool_result = tool_result

            tool_message = ToolMessage(
                content=json.dumps(resolved_tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )

            data |= {"output": tool_message}

            await adispatch_custom_event(
                f"on_{self.role}_stream",
                data=data,
                config=config
            )

            await adispatch_custom_event(
                f"on_{self.role}_end",
                data=data,
                config=config
            )

            outputs.append(tool_message)
        return {"messages": outputs}


async def get_agent(tools, kind=None):
    text_generator = get_text_generator(kind=kind)

    agent = AgentNode(text_generator, tools, role="agent")
    action = ActionNode(tools)

    def maybe_act(state: MessagesState):
        if isinstance(state, list):
            message = state[-1]
        elif messages := state.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state: {state}")
        if hasattr(message, "tool_calls") and len(message.tool_calls) > 0:
            return "act"
        return "__end__"

    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node("agent", agent)
    graph_builder.add_node("act", action)

    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges("agent", maybe_act)
    graph_builder.add_edge("act", "agent")
    graph_builder.add_edge("agent", END)

    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    
    return graph


    # chat_prompt = ChatPromptTemplate([
        # AIMessagePromptTemplate.from_template(as_template_str("""
            # Answer the following questions as best you can. You have access to the following tools:

            # {tools}

            # Use the following format:

            # Question: the input question you must answer
            # Thought: you should always think about what to do
            # Action: the action to take, should be one of [{tool_names}]
            # Action Input: the input to the action
            # Observation: the result of the action
            # ... (this Thought/Action/Action Input/Observation can repeat N times)
            # Thought: I now know the final answer
            # Final Answer: the final answer to the original input question

            # Begin!

            # Question: {input}
        # """)),
            # #Thought:{agent_scratchpad}
    # ])

    # chat_prompt = chat_prompt.partial(
        # tools=tools_renderer(list(tools)),
        # tool_names=", ".join([t.name for t in tools]),
    # )

    # reasoner = chat_prompt | text_generator.bind_tools(tools)



    # async def reason(state: MessagesState, config: RunnableConfig = None):
        # messages = state["messages"]
        # response = await reasoner.ainvoke(messages, config)
        # return {"messages": response}




def get_run_id_from_message(message=None):
    return message.id[4:] if getattr(message, "id") else None


role_avatars = {
    "user": "🧑",
    "act": ":material/plumbing:",
    "agent": "🤖",
}

# Customer Support Agent 🤖 💬 📞 🤝 💼 📱 ☎️ 👨‍💻 🤝 📞 💻 👂 🛠️
# Educational Tutor 📚 🧑‍🏫 🎓 🖊️ 📘 🎓 ✍️ 🧑‍🎓 📕 🧠 📖 📝 🎓 👩‍🏫 🧑‍🎓
# Technical Assistant 💻 ⚙️ 🔧 🛠️ 🤖 🖥️ 🧑‍💻 🔌 ⚡ 🖥️ 🧑‍🔧 💡 🔧 🖱️ 🖥️
# Creative Writer ✍️ 📜 💡 🖋️ 📖 🎨 🧠 📓 🖌️ 💭 📘 ✏️ 📜 ✍️ 🧑‍🎨
# Finance Advisor 💵 📊 💼 📈 📉 💰 💼 🧮 📊 💳 📊 🧑‍💼 💸 📑 💹
# Medical Assistant 🩺 💊 👩‍⚕️ 🏥 💉 📋 👩‍⚕️ 💊 📄 🧑‍⚕️ 🧬 📝 🏥 👨‍⚕️ 🧫
# Research Assistant 🔬 📝 📊 📚 📊 🧠 🧪 📖 🔍 🧑‍🔬 🧬 📑 🧠 🔍 🔬
# Coding Assistant 👨‍💻 🖥️ 💡 🧑‍💻 💾 👨‍💻 💻 🔑 ⌨️ 🔍 💡 👩‍💻 🧑‍💻 🛠️ 🔍
# Travel Planner ✈️ 🌍 🧳 🗺️ 🏖️ 🧳 🚢 ✈️ 🏝️ 🧳 🚞 🏔️ 🏨 ✈️ 📅
# Legal Advisor ⚖️ 📜 💼 👩‍⚖️ 📑 💼 🧑‍⚖️ 📚 ⚖️ 📜 🧑‍⚖️ 🖋️ 📝 ⚖️ 🧑‍💼
# Intelligence 🧠 💡 🌌 🔄 ⏳ ♾️ 🧠 🧠 📜 ⚛️ 🌌 🌀 🔬 🤖 🧠 🌐
# User: 👤 👥 🧑 👨 👩
