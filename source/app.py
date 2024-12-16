from typing import List, Dict, Any, Optional, Union
import asyncio
from collections import OrderedDict
from functools import lru_cache
import json
import yaml
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate

from utils.prompts import as_template_str
from utils.graphs import get_agent, role_avatars, get_run_id_from_message
from utils.tools import (
    # search_web_sources_tool,
    search_duckduckgo_sources_tool,
    search_serper_sources_tool,
    open_web_page_tool,
    generate_image_tool,
    #search_law_corpus_tool,
)

# adding edition will need below resource:
# https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/edit-graph-state/#interacting-with-the-agent


#@st.cache_resource  # not supported yet
async def get_graph_with_cache():
    tools = [
        # search_duckduckgo_tool,
        search_serper_sources_tool,
        open_web_page_tool,
        generate_image_tool,
        #search_law_corpus_tool,
    ]
    return await get_agent(tools)


async def initialize_session_state():
    state = st.session_state
    if (attr := "history") not in state:
        setattr(state, attr, [])
    if (attr := "messages") not in state:
        setattr(state, attr, [])
    if (attr := "message_parents") not in state:
        setattr(state, attr, OrderedDict())
    if (attr := "actions") not in state:
        setattr(state, attr, OrderedDict())


async def get_user_input():
    chat_input = st.chat_input("Type your message here...", key="chat_input")
    return chat_input


async def append_user_message(prompt):
    message = HumanMessage(prompt, id=str(uuid.uuid4()))
    st.session_state.messages.append(message)
    st.session_state.history.append({
        "role": "user",
        "message": message,
    })


def st_message_box(role):
    return st.chat_message(role, avatar=role_avatars.get(role))

def get_run(message=None, run_id=None):
    if not run_id:
        run_id = get_run_id_from_message(message)
    runs = st.session_state.runs
    run = runs.get(run_id, {}) if run_id else {}
    return run


async def display_history_chunk(chunk):
    role = chunk["role"]
    message = chunk.get("message")
    # action = chunk.get("action")

    if (
        message
        and message.content
        and isinstance(message, Union[HumanMessage, AIMessage])
    ):
        st_message_box(role).markdown(message.content)
    
    if isinstance(message, ToolMessage):
        content = message.content
        call_id = getattr(message, "tool_call_id")
        action = st.session_state.actions.get(call_id)
        tool = action["tool"]
        args = action["args"]
        st_tool_message(role, tool, args, content)

    # if action:
        # tool = action["tool"]
        # args = action["args"]
        # st_tool_call_message(role, tool, args)


async def display_history():
    for chunk in st.session_state.history:
        await display_history_chunk(chunk)


def get_message_box(role: str, context: Dict[str, Any], new=False):
    message_box = context.get("message_box")
    if new or not message_box:
        message_box = st_message_box(role).empty()
        context["message_box"] = message_box
    return message_box


def get_chunk_message(data: Dict[str, Any], context: Dict[str, Any]):
    chunk = data["chunk"]
    message = context.get("message")
    message = message + chunk if message else chunk
    context["message"] = message
    return message


def format_tool_args(tool_args):
    if isinstance(tool_args, str):
        tool_args = json.loads(tool_args)
    return "<br>".join([
        f"`{key}`: {value}"
        for key, value in tool_args.items()
    ])


def st_tool_message(
    role,
    tool,
    arguments,
    output=None,
    prefix=None,
    suffix=None,
):
    prefix = f"{prefix}\n" if prefix else ""
    suffix = f"\n{suffix}" if suffix else ""
    arguments = format_tool_args(arguments)

    message_box = st_message_box(role)
    with message_box:
        with st.container():
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write(f"{prefix}**{tool}**{suffix}")
            with col2:
                st.write(arguments, unsafe_allow_html=True)
            if output:
                st.expander("Result").markdown(output)


def st_tool_call_message(role, tool, arguments):
    return st_tool_message(
        role,
        tool,
        arguments,
        prefix="Calling tool:",
    )


def remove_from_context(context: Dict[str, Any], keys: List[str]):
    for key in keys:
        if key not in context:
            continue
        del context[key]


async def process_event(
    event: str,
    name: str,
    data: Dict[str, Any],
    metadata: Dict[str, Any],
    tags: List[str],
    run_id: str,
    parent_ids: List[str],
    context: Optional[Dict[str, Any]] = None,
):
    local_args = {**locals()}

    context = context if context is not None else {}
    if not event == "on_custom_event":
        return context

    node = metadata.get("langgraph_node")
    step = metadata.get("langgraph_step")
    thread = metadata.get("thread_id")
    ckpt_ns = metadata.get("checkpoint_ns")
    role = node
    # if run_id not in st.session_state.runs:
        # st.session_state.runs[run_id] = {
            # "node": node,
            # "name": name,
            # "step": step,
            # "thread": thread,
            # "tags": tags,
            # "parents": parent_ids,
            # "ckpt_ns": ckpt_ns,
        # }
     
    if name == "on_agent_start":
        return context
    
    if name == "on_agent_stream":
        message = data["chunk"]
        if not message.content:
            return context
        message_box = context.get("message_box")
        if not message_box:
            message_box = st_message_box(role).empty()
            context["message_box"] = message_box
        message_box.markdown(message.content)
        return context

    if name == "on_agent_end":
        for key in ["message_box"]:
            if key in context:
                del context[key]

        message = data["output"]
        st.session_state.history.append({
            "role": role,
            "message": message,
        })
        return context

    if name == "on_agent_action":
        action = data["action"]
        call_id = action["call_id"]
        st.session_state.actions[call_id] = action
        return context

    if name == "on_action_end":
        message = data["output"]
        call_id = data["tool_call"]["id"]
        action = st.session_state.actions.get(call_id)

        tool = action["tool"]
        args = action["args"]
        output = json.loads(message.content)
        st_tool_message(role, tool, args, output)

        st.session_state.history.append({
            "role": role,
            "message": message,
        })
        return context

    return context

async def process_agent_response(agent, config):
    messages = st.session_state.messages
    inputs = {"messages": messages}
    stream = agent.astream_events(inputs, config=config, version="v2")

    context = None
    async for event_info in stream:
        context = await process_event(**event_info, context=context)

    snapshot = agent.get_state(config)
    messages = snapshot.values["messages"]
    st.session_state.messages = messages

async def main():
    config = {"configurable": {"thread_id": "1"}}
    agent = await get_graph_with_cache()

    await initialize_session_state()
    prompt = await get_user_input()

    if prompt:
        await append_user_message(prompt)

    await display_history()

    if prompt:
        await process_agent_response(agent, config)

if __name__ == "__main__":
    asyncio.run(main())
