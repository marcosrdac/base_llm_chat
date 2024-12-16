import re
import datetime
from urllib.parse import urlparse
from typing_extensions import Literal
from typing import Optional, List, Dict, Any
from langchain_community.utilities import GoogleSerperAPIWrapper, DuckDuckGoSearchAPIWrapper
from utils.credentials import get_credentials


DEFAULT_TIME_FORMAT = "%H:%M"
DEFAULT_DATETIME_FMT = "%Y-%m-%dT%H:%M:%SZ"
DEFAULT_DATE_FMT = "%Y-%m-%d"


class ExtendedGoogleSerperAPIWrapper(GoogleSerperAPIWrapper):
    type: Literal["news", "search", "places", "images", "scholar"] = "search"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.result_key_for_type = {
            **self.result_key_for_type,
            "scholar": "scholar",
        }

def search_serper(
    query: str,
    max_results: int = 10,
    kind: Optional[str] = None,
    region: str = None,
    language: Optional[str] = None,
    how_old: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    #- api_key (str): Your Serper API key (required).
    """
    Perform a web search using the Serper API for the given query and return formatted results.
    Read the parameters with attention.

    Parameters:
    - query (str): The search query.
    - max_results (int): Number of results to return (default: 10).
    - kind (str): The type of search ("general", "news", "scientific", "images", or "places", default: "general").
    - region (str): Results searched from country in "country_code" format (default: "us").
    - language (str): Results filtered by language in "language_code" format (default: "" (empty)).
    - how_old (str): Filter by recency. One of "day", "week", "month", "year", or "any" (default: "any").

    Note:
    - Avoid redundancy between query and parameters (i.e.: try not to search for "technology news". Instead, search for technology with kind="news").
    - Region setting is IMPORTANT!
    - For "images" and "places", ensure the query aligns with the expected type for best results.

    Returns:
    - List[Dict[str, Any]]: A list of formatted search results.
    """
    api_key = api_key or get_credentials("serper").get("key")
    if not api_key:
        raise ValueError("An API key must be provided for Serper.")

    kind = kind or "general"
    if kind == "general":
        kind = "search"
    if kind == "scientific":
        kind = "scholar"
    
    how_old_map = {"d": "qdr:d", "w": "qdr:w", "m": "qdr:m", "y": "qdr:y", "a": None}
    tbs = how_old_map.get(how_old[0].lower(), None) if how_old else None
    region = region or "us"
    language = language or ""

    search = ExtendedGoogleSerperAPIWrapper(
        serper_api_key=api_key,
        k=max_results,
        gl=region,
        hl=language,
        type=kind,
        tbs=tbs,
    )
    
    results = search.results(query)
    for key in [
        "organic",  # general
        "news",     # news kind
        "places",   # places kind
        "images",   # images kind
    ]:
        if not key in results:
            continue
        results = results[key]
        break
    return results


def search_duckduckgo(
    query: str,
    max_results: int = 10,
    kind: Optional[str] = None,
    region_and_language: Optional[str] = None,
    how_old: Optional[str] = None,
    moderate: bool = True,
) -> List[Dict[str, Any]]:
    """
    Perform a web search for the given query and return formatted results.

    Parameters:
    - query (str): The search query.
    - max_results (int): Number of results to return (default: 10).
    - kind (str): The type of search ("general" or "news", default: "general").
    - region_and_language (str): Results filtered by region and language in "{REGION}-{LANGUAGE}" format (default: "wt-wt").
    - how_old (str): Filter by recency. One of "day", "week", "month", "year", or "any" (default: "any").
    - moderate (bool): Apply content filtering if True (default: True).

    Note:
    - When `kind="news"`, avoid adding terms like "news" in the query.

    Returns:
    - List[Dict[str, Any]]: A list of formatted search results.
    """
    kind = "news" if kind == "news" else "general"
    how_old = how_old[0] if how_old else "a"
    region_and_language = region_and_language or "wt-wt"
    moderation = "moderate" if moderate else "off"

    search = DuckDuckGoSearchAPIWrapper(
        region=region_and_language,
        time=how_old,
        safesearch=moderation,
        #backend="api",  # has rate limit
        backend="html",
        source="news" if kind == "news" else "text",
    )
    return search.results(query, max_results)

def escape_markdown(text: str) -> str:
    markdown_special_chars = r'([\\*_{}\[\]()#+\-.!|$])'
    return re.sub(markdown_special_chars, r'\\\1', text)

def read_iso_time(time_str):
    try:
        return datetime.datetime.fromisoformat(time_str)
    except:
        return time_str

def convert_to_timezone(time, utc_delta=None):
    if isinstance(time, str):
        return time
    if utc_delta is None:
        return time
    timezone = datetime.timezone(datetime.timedelta(hours=utc_delta))
    return time.astimezone(timezone)

def time_to_pretty_str(time):
    if isinstance(time, str):
        return time
    now = datetime.datetime.now(time.tzinfo)
    if time.date() == now.date():
        return time.strftime(DEFAULT_TIME_FORMAT)
    elif time.date() == (now.date() - datetime.timedelta(days=1)):
        return f"yesterdayT{time.strftime(DEFAULT_TIME_FORMAT)}"
    else:
        return time.strftime(DEFAULT_DATETIME_FMT)

def beautify_url(url):
    """Extracts a beautiful domain from a URL, removing 'http', 'www', and paths."""
    if not url:
        return ""
    parsed = urlparse(url)
    domain = parsed.netloc
    if domain.startswith('www.'):
        domain = domain[4:]
    return domain

def prepare_common_fields(result, local_timezone=None):
    title = escape_markdown(result.get("title", ""))
    snippet = result.get("snippet", "")
    link = result.get("link", "")
    date = result.get("date", "")
    source = result.get("source", "")
    imageUrl = result.get("imageUrl", "")
    sitelinks = result.get("sitelinks", [])
    category = result.get("category", "")
    address = result.get("address", "")
    latitude = result.get("latitude", "")
    longitude = result.get("longitude", "")
    rating = result.get("rating", "")
    ratingCount = result.get("ratingCount", "")
    website = result.get("website", "")
    cid = result.get("cid", "")
    thumbnailUrl = result.get("thumbnailUrl", "")
    domain = result.get("domain", "")
    
    imageSize = "x".join([str(s) for c in ['imageWidth', 'imageHeight'] if (s:=result.get(c))])
    thumbnailSize = "x".join([str(s) for c in ['thumbnailWidth', 'thumbnailHeight'] if (s:=result.get(c))])

    if not domain:
        domain = beautify_url(link or imageUrl)

    time = ""
    if date:
        parsed_time = read_iso_time(date)
        if not isinstance(parsed_time, str):
            if local_timezone:
                parsed_time = convert_to_timezone(parsed_time, utc_delta=local_timezone)
            time = time_to_pretty_str(parsed_time)
        else:
            time = parsed_time

    if not source and link:
        source = beautify_url(link)
    source = escape_markdown(source)

    return {
        "title": title,
        "snippet": snippet,
        "link": link,
        "time": time,
        "source": source,
        "imageUrl": imageUrl,
        "imageSize": imageSize,
        "thumbnailUrl": thumbnailUrl,
        "thumbnailSize": thumbnailSize,
        "sitelinks": sitelinks,
        "category": category,
        "address": address,
        "latitude": latitude,
        "longitude": longitude,
        "rating": rating,
        "ratingCount": ratingCount,
        "website": website,
        "cid": cid,
        "domain": domain,
    }

def format_links(fields):
    if fields["sitelinks"]:
        links_str = ", ".join([f"[{escape_markdown(s['title'])}]({s['link']})" for s in fields["sitelinks"]])
        return f"; Links: {links_str}"
    return ""

def get_thumbnail_display(fields):
    thumb_url = fields.get("thumbnailUrl") or fields.get("imageUrl")
    thumb_size = fields["thumbnailSize"]
    thumb_name = "thumb" + (f"@{thumb_size}" if thumb_size else "")
    if thumb_url:
        return f"![{thumb_name}]({thumb_url})"

def format_source_and_time(fields):
    # Construct source@time string
    parts = []
    if fields["source"]:
        parts.append(fields["source"])
    if fields["time"]:
        parts.append(fields["time"])
    # Join with '@' if both present
    source_time = "@".join(parts)
    return source_time if source_time else None

def format_result(result, template_name="general", add_thumb=True, add_image_info=True, add_sitelinks=True, local_timezone=None):
    fields = prepare_common_fields(result, local_timezone)
    
    entry = {}
    
    if add_thumb:
        thumb = get_thumbnail_display(fields)
        if thumb:
            entry["thumb"] = thumb

    if add_image_info:
        image_size = fields.get("imageSize")
        if image_size:
            image_info = f"[@{image_size}]({fields['imageUrl']})"
            entry["image_info"] = image_info
        
    title_and_snippet = []
    if fields.get("title"):
        title_and_snippet.append(f"**{fields['title']}**")
    if fields.get("snippet"):
        title_and_snippet.append(f"{fields['snippet']}")
    entry["snippet"] = ": ".join(title_and_snippet)
    
    source_and_time = format_source_and_time(fields)
    source_and_time_linked = f"([{source_and_time}]({fields['link']}))"
    entry["source"] = source_and_time_linked
    
    if add_sitelinks and fields.get("sitelinks"):
        links_str = ", ".join([f"[{escape_markdown(s['title'])}]({s['link']})" for s in fields["sitelinks"]])
        links = f"Other links: {links_str}."
        entry["links"] = links
    
    if template_name in ["general", None]:
        template = "thumb image_info snippet source links".split()
        entry = " ".join(["-"] + [f"{{{k}}}" for k in template if k in entry]).format(**entry)
    
    return entry


def search_results_to_md(results, local_timezone=None):
    return "\n".join([
        format_result(r, local_timezone=local_timezone) for r in results
    ])
