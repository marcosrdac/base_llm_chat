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
# from streamlit_extras.stylable_container import stylable_container

from utils.graphs import get_agent, persona_avatars, get_run_id_from_message
from utils.tools import (
    # search_web_sources_tool,
    search_duckduckgo_sources_tool,
    search_serper_sources_tool,
    open_web_page_tool,
    generate_image_tool,
    #search_law_corpus_tool,
)

# TODO Save checkpoint thread (X not!) / id (V yes!) at each new message
# Edit the message in of current state only
# then save the new checkpoint ID
# rerun if the user asks so
# delete following cells if the user asks for deletion
# keep a message cache when displaying them, be them joined with thread info or not
# such cache is going to be useful to sinalize the user he can move between multiverses at that messages thingy

# adding edition will need below resource:
# https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/edit-graph-state/#interacting-with-the-agent

# st.markdown("""
    # <style>
    # .stButton>button {
        # background-color: rgba(0, 123, 255, 0.0); /* Blue background with 50% opacity */
        # position: relative; /* Positioning context */
        # padding: 0px 0px;
        # min-width: 33px;
        # min-height: 31px;
        # left: -40px; /* Move the button to the left */
        # opacity: 0; /* Fully invisible */
    # }
    # .stButton>button:hover {
        # opacity: 1; /* Make visible on hover */
    # }
    # </style>
# """, unsafe_allow_html=True)

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet">
    <style>
        .material-symbols-outlined {
            vertical-align: middle;
        }
    </style>
    """,
    unsafe_allow_html=True
)

style = """
<style>
.container {
   background-color: #D1C4E9; 
   padding: 15px; 
   border-radius: 8px;
}
</style>
"""


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
    if (attr := "snapshot") not in state:
        setattr(state, attr, None)
    if (attr := "thread") not in state:
        default_thread = {"configurable": {"thread_id": "1"}}
        setattr(state, attr, default_thread)
    if (attr := "actions") not in state:
        setattr(state, attr, {})
    if (attr := "agent") not in state:
        agent = await get_graph_with_cache()
        setattr(state, attr, agent)


async def get_user_input():
    chat_input = st.chat_input("Type your message here...", key="chat_input")
    return chat_input

def make_symbol(icon_name, cls=None, **kwargs):
    style_props = {
        key.replace("_", "-"): value
        for key, value in kwargs.items()
        if value is not None
    }
    style_string = 'style="' + "; ".join([f"{k}: {v}" for k, v in style_props.items()]) + ';"' if style_props else ""
    class_string = f'class="{cls}"' if cls else ""
    span_config = " ".join(filter(None, [class_string, style_string]))
    icon_html = f"""<span {span_config}>{icon_name}</span>"""
    return icon_html

def st_message_box(persona, button_key=None):
    avatar = persona_avatars[persona]

    with st.empty():
        icon_col, message_col, other_col = st.columns([1, 17, 1])


    with icon_col:
        st.markdown(make_symbol(avatar, size="24px"), unsafe_allow_html=True)

    with other_col:
        if button_key:
            edit_button = st.button(
                label=":material/more_vert:",
                key=button_key,
                on_click=lambda: message_col.warning(button_key),
                disabled=False,
                type="secondary",
            )
    return message_col


def get_run(message=None, run_id=None):
    if not run_id:
        run_id = get_run_id_from_message(message)
    runs = st.session_state.runs
    run = runs.get(run_id, {}) if run_id else {}
    return run


async def display_history_message(message, key=None):
    persona = message.name

    if (
        message
        and message.content
        and isinstance(message, Union[HumanMessage, AIMessage])
    ):
        st_message_box(persona, button_key=key).markdown(message.content)
    
    if isinstance(message, ToolMessage):
        content = message.content
        call_id = getattr(message, "tool_call_id")
        action = st.session_state.actions.get(call_id)
        # st.write(action)
        tool = action["tool"]
        args = action["args"]
        st_tool_message(persona, tool, args, content)

    # if action:
        # tool = action["tool"]
        # args = action["args"]
        # st_tool_call_message(persona, tool, args)
    # st.write(message.name)


async def display_history():
    agent = st.session_state.agent
    thread = st.session_state.thread

    state = agent.get_state(thread)
    messages = state.values.get("messages", [])
    for message in messages:
        await display_history_message(message)

    # states = [*agent.get_state_history(thread)][::-1]
    # for state in states:
        # messages = state.values.get("messages", [])
        # if not messages:
            # continue
        # await display_history_message(messages[-1], key=state.config)


def get_message_box(persona: str, context: Dict[str, Any], new=False):
    message_box = context.get("message_box")
    if new or not message_box:
        message_box = st_message_box(persona).empty()
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
    persona,
    tool,
    arguments,
    output=None,
    prefix=None,
    suffix=None,
):
    prefix = f"{prefix}\n" if prefix else ""
    suffix = f"\n{suffix}" if suffix else ""
    arguments = format_tool_args(arguments)

    message_box = st_message_box(persona)
    with message_box:
        with st.container():
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write(f"{prefix}**{tool}**{suffix}")
            with col2:
                st.write(arguments, unsafe_allow_html=True)
            if output:
                st.expander("Result").markdown(output)


def st_tool_call_message(persona, tool, arguments):
    return st_tool_message(
        persona,
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
     
    if name == "on_agent_start":
        return context
    
    if name == "on_agent_stream":
        message = data["chunk"]
        if not message.content:
            return context
        message_box = context.get("message_box")
        if not message_box:
            message_box = st_message_box(message.name).empty()
            context["message_box"] = message_box
        message_box.markdown(message.content)
        return context

    if name == "on_agent_end":
        for key in ["message_box"]:
            if key in context:
                del context[key]

        message = data["output"]
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
        st_tool_message("tool", tool, args, output)
        # st_tool_message(message.name, tool, args, output)

        return context

    return context


async def create_user_message(text):
    message = HumanMessage(text, name="user", id=str(uuid.uuid4()))
    await display_history_message(message)
    return message


async def process_agent_response(message):
    inputs = {"messages": [message]}

    agent = st.session_state.agent
    thread = st.session_state.thread

    stream = agent.astream_events(inputs, config=thread, version="v2")

    context = None
    async for event_info in stream:
        context = await process_event(**event_info, context=context)

async def main():
    await initialize_session_state()

    await display_history()

    prompt = await get_user_input()
    if prompt:
        message = await create_user_message(prompt)
        await process_agent_response(message)

if __name__ == "__main__":
    asyncio.run(main())
