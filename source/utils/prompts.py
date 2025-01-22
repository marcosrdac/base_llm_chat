import textwrap
from langchain_core.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate,
)

def as_simple_string(s):
    return textwrap.dedent(s).strip("\n")

def as_template(inputs, joiner="\n"):
    if isinstance(inputs, str):
        inputs = [inputs]
    inputs = joiner.join(as_simple_string(i) for i in inputs)
    return inputs


AGENT_EXPERTISE_SETUP_PROMPT = as_template("""
    You are an expert in {expertise}.
""") 

AGENT_ROLE_SETUP_PROMPT = as_template("""
    You must act as a {role}.
""")

AGENT_STRICTNESS_SETUP_PROMPT = as_template("""
    You must strictly adhere to your designated role and expertise, avoiding topics or actions beyond this scope in your responses.
""")

AGENT_ASSISTANT_SETUP_PROMPT = as_template("""
    You should assist the user in solving their problems.
""")

AGENT_CONCISENESS_SETUP_PROMPT = as_template("""
    You should be as economical and concise as possible in your answers, without sacrificing relevant information.
""")

AGENT_LANGUAGE_SETUP_PROMPT = as_template("""
    You should only speak to the user in the {language} language.
""")

AGENT_AVOID_FILLER_SETUP_PROMPT = as_template("""
    You should only provide direct, relevant information without introductory or concluding statements. Respond in a concise, factual manner, and avoid any unnecessary commentary or phrases such as 'if you need more information' or 'let's analyze this'.
""")

AGENT_BULLET_POINT_STYLE_SETUP_PROMPT = as_template("""
    You should respond to the user in a clear, concise manner by using bullet points for each key idea or action, ensure the information is well-structured and easy to read.
""")

AGENT_REFERENCE_SOURCES_SETUP_PROMPT = as_template("""
    Always use tools to provide responses backed by reliable references. Before answering, ask yourself: "Would using tools improve the quality or accuracy of my response?"
    Only provide information obtained through tools, and do not rely on assumptions. Conduct additional tool usage if necessary to ensure a thorough and accurate answer.
    In the final response, summarize the information instead of simply listing the references.
""")



def make_agent_chat_template(role=None, expertise=None, language=None, initial_message=None):
    template = ChatPromptTemplate.from_messages([
        ("system", ". ".join([
            *([AGENT_EXPERTISE_SETUP_PROMPT] if expertise else []),
            *([AGENT_ROLE_SETUP_PROMPT] if role else []),
            *([AGENT_STRICTNESS_SETUP_PROMPT] if role or expertise else []),
            AGENT_ASSISTANT_SETUP_PROMPT,
            AGENT_CONCISENESS_SETUP_PROMPT,
            *([AGENT_LANGUAGE_SETUP_PROMPT] if language else []),
            AGENT_BULLET_POINT_STYLE_SETUP_PROMPT,
            # AGENT_AVOID_FILLER_SETUP_PROMPT,
            AGENT_REFERENCE_SOURCES_SETUP_PROMPT,
        ]) + "."),
        *([("assistant", as_template("""
            {initial_message}
        """))] if initial_message else []),
        ("user", as_template("""
            {input}
        """)),
    ])
    template = template.partial(**{
        k: v
        for k, v in {
            "role": role,
            "expertise": expertise,
            "language": language,
            "initial_message": initial_message,
        }.items()
        if v
    })
    return template