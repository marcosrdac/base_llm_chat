import asyncio
import re
from typing import Optional
from langchain.tools import Tool, tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from utils.decorators import catch_as_str, add_docstring
from utils.models import OpenAIModels
from utils.prompts import as_template_str
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

from utils.models import get_text_embedder, get_text_generator
from utils.vector_stores import get_chroma_vector_store
from utils.paths import (
    DOCUMENTS_DIR,
    VADE_MECUM_PATH,
    MODELS_DIR,
    EMBEDDER_MODELS_DIR,
    GENERATOR_MODELS_DIR,
)

import numpy as np
#from tqdm.notebook import tqdm
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


DEFAULT_TIME_FMT = "%H:%M"
DEFAULT_DATE_FMT = "%Y-%m-%d"
DEFAULT_DATETIME_FMT = "%Y-%m-%dT%H:%M"
LOCAL_TIMEZONE_DELTA = -3
DEFAULT_LANGUAGE = "Brazilian Portuguese"
DEFAULT_DUCKDUCKGO_REGION = "br-pt"


# from semanticscholar import SemanticScholar
#def search_semanticscholar(
#    query: str,
#    max_results: int = 10,
#    fields: Optional[List[str]] = None,
#    year: Optional[str] = None,
#    fields_of_study: Optional[List[str]] = None,
#    kind: str = 'paper',
#) -> str:
#    """
#    Perform a search for papers or authors on Semantic Scholar and return formatted results.
#
#    Parameters:
#    - query (str): The search query.
#    - max_results (int): Number of results to return (default: 10).
#    - fields (List[str]): Specific fields to return in the response.
#    - year (str): Filter by publication year.
#    - fields_of_study (List[str]): Filter results to specific fields of study.
#    - search_type (str): Type of search ('paper' or 'author').
#
#    Returns:
#    - str: Formatted search results in Markdown.
#    """
#    sch = SemanticScholar(timeout=5)
#    if kind == 'paper':
#        results = sch.search_paper(query, limit=max_results, fields=fields, year=year, fields_of_study=fields_of_study)
#    elif kind == 'author':
#        results = sch.search_author(query, limit=max_results, fields=fields)
#
#    output = f'### Search Results for "{query}"\n'
#    for idx, item in enumerate(results):
#        if kind == 'paper':
#            output += f'- **Paper {idx+1}:** {item.title} ({item.year})\n'
#        elif kind == 'author':
#            #output += f'- **Author {idx+1}:** {item.name}, {item.affiliation}\n'
#            output += f'- **Author {idx+1}:** {item.name}\n'
#    return output


def search_duckduckgo(
    query: str,
    max_results: int = 10,
    how_old: Optional[str] = None,
    region: Optional[str] = None,
    news: bool = False,
    moderate:bool = True,
) -> List[Dict[str, Any]]:
    """
    Perform a web search for the given query and return formatted results.

    Parameters:
    - query (str): The search query.
    - max_results (int): Number of results to return (default: 10).
    - how_old (str): Filter by recency. One of "day", "week", "month", "year", or "any" (default: "any").
    - region (str): Results filtered by region in "{REGION}-{LANGUAGE}" format (default: "wt-wt").
    - news (bool): Use news sources (default: False).
    - moderate (bool): Apply content filtering if True (default: True).

    Note:
    - When searching for news with `news=True`, avoid querying with irrelevant terms like "news" or "latest" as these will filter out news without such terms.

    Returns:
    - str: Formatted search results.
    """
    how_old = how_old[0] if how_old is not None else "a"
    kind = "news" if news else "text"
    region = region or "wt-wt"
    moderation = "moderate" if moderate else "off"
    search = DuckDuckGoSearchAPIWrapper(region=region, time=how_old, safesearch=moderation, backend="api", source=kind)
    return search.results(query, max_results)

def escape_markdown(text: str) -> str:
    markdown_special_chars = r'([\\`*_{}\[\]()#+\-.!|$])'
    return re.sub(markdown_special_chars, r'\\\1', text)
        

def result_to_md(result, local_timezone=None):
    title = result.get("title", "")
    snippet = result.get("snippet", "")
    link = result.get("link", "")
    date = result.get("date", "")
    source = result.get("source", "")
    time = ""

    if date:
        try:
            time = read_iso_time(date)
            if local_timezone:
                time = convert_to_timezone(time, utc_delta=local_timezone)
            time = time_to_pretty_str(time)
        except Exception as e:
            time = ""
    
    if not source:
        parsed_url = urlparse(link)
        source = parsed_url.netloc
        # Remove 'www.' prefix if present
        if source.startswith('www.'):
            source = source[4:]

    source = escape_markdown(source)
    title = escape_markdown(title)
    
    source_and_time = f"{source}@{time}" if time else source
    result_md = "**{title}**: {snippet} ([{source_and_time}]({link}))".format(
        title=title,
        snippet=snippet,
        source_and_time=source_and_time,
        link=link
    )
    return result_md


def read_iso_time(time_str):
    return datetime.datetime.fromisoformat(time_str)


def convert_to_timezone(time, utc_delta=None):
    if utc_delta is None:
        return time
    timezone = datetime.timezone(datetime.timedelta(hours=utc_delta))
    return time.astimezone(timezone)


def time_to_str(time):
    return time.strftime(DEFAULT_DATETIME_FMT)


def date_to_str(time):
    return time.strftime(DEFAULT_DATE_FMT)


def time_to_pretty_str(time):
    now = datetime.datetime.now(time.tzinfo)
    if time.date() == now.date():
        return time.strftime(DEFAULT_TIME_FORMAT)
    elif time.date() == (now.date() - datetime.timedelta(days=1)):
        return f"yesterdayT{time.strftime(DEFAULT_TIME_FORMAT)}"
    else:
        return time.strftime(DEFAULT_DATETIME_FMT)


def search_results_to_md(results, local_timezone=None):
    return "\n".join([
        f"- {result_to_md(r)}" for r in results
    ])


def get_page_content(url):
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.text


def get_page_text(url: str) -> str:
    page_content = get_page_content(url)
    soup = BeautifulSoup(page_content, 'html.parser')
    for unwanted in soup(["script", "style", "header", "nav", "footer"]):
        unwanted.decompose()
    text_content = soup.get_text(separator='\n')
    lines = [line.strip() for line in text_content.splitlines() if line.strip()]
    return '\n'.join(lines)


@tool("search_web_sources")
@catch_as_str()
@add_docstring(search_duckduckgo.__doc__)
async def search_web_sources_tool(
    query: str,
    max_results: int = 10,
    how_old: Optional[str] = None,
    region: Optional[str] = None,
    news: bool = False,
    moderate:bool = True,
    return_steps:bool = False,
) -> str:
    results = await asyncio.to_thread(search_duckduckgo, query=query, max_results=max_results, region=region, how_old=how_old, news=news, moderate=moderate)

    return search_results_to_md(results, local_timezone=LOCAL_TIMEZONE_DELTA)


@tool("open_web_page")
@catch_as_str()
async def open_web_page_tool(
    url: str,
    max_chars: Optional[int] = None,
) -> str:
    """
    Fetch and return the main textual content of a webpage.

    Parameters:
    - url (str): The URL of the page to fetch.
    - max_chars (optional, int): Maximum number of characters returned from the page.

    Returns:
    - str: The main text content of the webpage. 
    """
    return get_page_text(url)[:max_chars]


@tool("generate_image")
@catch_as_str()
async def generate_image_tool(
    prompt: str,
    model: Optional[str] = None,
    size: Optional[str] = None,
    quality: str = "standard",
    n: int = 1,
) -> str:
    """
    Generates images based on a text prompt.

    Parameters:
    - prompt (str): Text prompt to generate images.
    - model (str, optional): Image model to use. Options: 'dall-e-2', 'dall-e-3'. Default is None, which uses the dall-e-3 model.
    - size (str, optional): Image size. For 'dall-e-2', it must be '256x256', '512x512', or '1024x1024'. For 'dall-e-3', it must be '1024x1024', '1792x1024', or '1024x1792'. Default is None.
    - quality (str): Image quality ('standard' or 'hd'). Default is 'standard'.
    - n (int): Number of images to generate. Default is 1.

    Returns:
    - str: Generated image data.

    Notes:
    - Use clear, descriptive language in prompts. Aim for at least one paragraph to provide context and detail.
    - Start with the subject, followed by a description, style, and additional details (lighting, mood).
    - Avoid abstract concepts; focus on concrete nouns and actions.
    """
    response = OpenAIModels().generate_image(
        prompt=prompt,
        model_name=model,
        size=size,
        quality=quality,
        n=n
    )
    return [image.url for image in response.data]


#@tool("search_law_corpus")
#def search_law_corpus_tool(query, reranking_context, max_results=20):
#    """
#    Searches the Brazilian Vade Mecum for relevant law documents relevant.
#    When using this tool, prepare at least 5 very different queries to be able to explore the data well in this tool, even if the user does not ask for it. 20 results seems to have enough useful findings. If you cannot find information, vary the query knowing it searches using a RAG based system and try some times before giving up. As a RAG, you need to truly vary the query if you expect to find different information. BE CREATIVE, don't just change word order. Also, try using law vocabulary if possible.
#
#    Parameters:
#    - query (str): User's search query.
#    - reranking_context (str): Context used to rank and filter results of the user query.
#    - max_results (int, optional): Number of documents to retrieve. Default is 20.
#
#    Returns:
#    - str: Answer in Brazilian Portuguese with referenced Vade Mecum pages.
#
#    Notes:
#    - Answers are simplified for laymen and always in Brazilian Portuguese.
#    - All information must be referenced from Vade Mecum documents.
#    """
#    # search vade mecum, to be converted to an agentic RAG
#    vector_store = get_chroma_vector_store(
#        persist_name="2023_vade_mecum_test",
#        embedder_name="all-minilm-l6-v2",
#    )
#    
#    document_retriever = vector_store.as_retriever(search_kwargs={"k": max_results})
#    
#    retrieval_qa_chat_prompt = PromptTemplate(
#        input_variables=["context", "input", "reranking_context"],
#        template=as_template_str("""
#            You are meant to answer the user's question using information given from context documents.
#            Always answer in Brazilian Portuguese.
#            The documents are from a Vade Mecum book of 2023.
#            Assume the user is a layman, he does not know about laws and such kind of vocabulary, so answer in a way anyonce could understand.
#            Each of those documents were ranked using a similarity measure to the user question.
#            You must find which quotes are related to the user's query and tell about them to him.
#            If the answer cannot be answered with the context refferenes, say you cannot answer that.
#            EVERY information you tell the user MUST be based on references (artigos, seção, inciso, tudo o que estiver disponível) and quoted along with Vade Mecum's page.
#            Your answer must be as concise as possible without sacrificing useful information. Avoid using unneeded introductions or conclusions, cut to the chase.
#
#            
#            Context:
#            {context}
#
#            Reranking context:
#            {reranking_context}
#            
#            User search query:
#            {input}
#            
#            Answer: 
#        """),
#    )
#    
#    document_prompt = PromptTemplate(
#        input_variables=["page_content", "page"],
#        template=as_template_str("""
#            2023 Vade Mecum page: {page}
#            snippet: {page_content}
#        """),
#    )
#    
#    #smart_text_generator = get_text_generator(kind="smart")
#    text_generator = get_text_generator(kind="cheap")
#    
#    combine_docs_chain = create_stuff_documents_chain(
#        text_generator, retrieval_qa_chat_prompt, document_prompt=document_prompt,
#    )
#    
#    retrieval_chain = create_retrieval_chain(document_retriever, combine_docs_chain)
#    
#    return retrieval_chain.invoke({"input": query, "reranking_context": reranking_context})["answer"]
#    # return retrieval_chain.invoke({"input": query})
