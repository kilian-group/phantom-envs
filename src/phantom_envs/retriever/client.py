"""HTTP client for the retrieval server, with a shared connection pool and retries.

Usage::

    from phantom_envs.retriever.client import SearchClient

    client = SearchClient("http://127.0.0.1:8000/retrieve", topk=3)
    results = client.search("who is Eloy Lovett")  # formatted documents, or an error message
"""

import json
import logging
import threading
import time
import uuid
from typing import Any, ClassVar
from urllib.parse import urlparse

import requests

from phantom_envs._types import RetrievalResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 1


def call_search_api(
    retrieval_service_url: str,
    query: str,
    topk: int = 3,
    return_scores: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
    log_requests: bool = True,
    session: requests.Session | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Call the retrieval server with retry logic.

    Args:
        retrieval_service_url: The URL of the search API.
        query: The query to search for.
        topk: The number of results to return.
        return_scores: Whether to return scores for the results.
        timeout: The timeout for the request.
        log_requests: Whether to log requests.
        session: The session to use for the request. If none is provided, a new session will be created.

    Returns:
        Tuple of (response dict or None, error message or None).
    """
    request_id = str(uuid.uuid4())
    log_prefix = f"[Search Request ID: {request_id}] "

    payload = {"query": query, "topk": topk, "return_scores": return_scores}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    if session is None:
        session = requests.Session()
        should_close_session = True
    else:
        should_close_session = False

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            if log_requests:
                logger.info(
                    f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: Calling search API at {retrieval_service_url}"
                )
            response = session.post(
                retrieval_service_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            if response.status_code in [500, 502, 503, 504]:
                last_error = (
                    f"{log_prefix}API Request Error: Server Error ({response.status_code}) "
                    f"on attempt {attempt + 1}/{MAX_RETRIES}"
                )
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                    time.sleep(delay)
                continue

            response.raise_for_status()

            if log_requests:
                logger.info(f"{log_prefix}Search API call successful on attempt {attempt + 1}")

            if should_close_session:
                session.close()

            return response.json(), None

        except requests.exceptions.ConnectionError as e:
            last_error = f"{log_prefix}Connection Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.Timeout as e:
            last_error = f"{log_prefix}Timeout Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"
            break
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = (
                f"{log_prefix}API Response JSON Decode Error: {e}, Response: {raw_response_text[:200]}"
            )
            break
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"
            break

    logger.error(f"{log_prefix}API Request Failed after {MAX_RETRIES} attempts: {last_error}")

    if should_close_session:
        session.close()

    return None, last_error


class SearchClient:
    """Client for the retrieval server's ``/retrieve`` endpoint.

    Clients for the same server share one pooled ``requests.Session``, so many concurrent
    environments reuse connections instead of overflowing the pool.
    """

    session_pool: ClassVar[dict[str, requests.Session]] = {}
    session_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def get_shared_session(cls, base_url: str, pool_size: int) -> requests.Session:
        """Get or create the shared session for ``base_url`` with ``pool_size`` connections."""
        with cls.session_lock:
            if base_url not in cls.session_pool:
                session = requests.Session()
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=pool_size,
                    pool_maxsize=pool_size,
                    max_retries=0,  # We handle retries ourselves
                    pool_block=False,
                )
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                cls.session_pool[base_url] = session
                logger.info(f"Created shared session pool for {base_url} (pool_size={pool_size})")
            return cls.session_pool[base_url]

    def __init__(
        self,
        search_url: str = "http://127.0.0.1:8000/retrieve",
        topk: int = 3,
        timeout: int = DEFAULT_TIMEOUT,
        log_requests: bool = True,
        pool_size: int = 64,
    ):
        self.search_url = search_url
        self.topk = topk
        self.timeout = timeout
        self.log_requests = log_requests

        parsed_url = urlparse(self.search_url)
        self.base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        self.session = self.get_shared_session(self.base_url, pool_size)

    def search(self, query: str) -> str:
        """Return the top-k documents for ``query`` formatted as text, or an error message."""
        query = query.strip()

        try:
            api_response, error_msg = call_search_api(
                retrieval_service_url=self.search_url,
                query=query,
                topk=self.topk,
                timeout=self.timeout,
                log_requests=self.log_requests,
                session=self.session,
            )
        except Exception as e:
            error_msg = f"API Request Exception during batch search: {e}"
            logger.error(f"Batch search: {error_msg}")

        if error_msg:
            logger.error(f"Search: API error: {error_msg}")
            return f"Search error: {error_msg}"

        if not api_response:
            logger.error("Search: Unknown API state (no response and no error).")
            return "Search request failed or timed out after retries."

        logger.debug(f"Search: API response: {api_response}")
        try:
            parsed = RetrievalResponse.model_validate(api_response)
        except Exception as e:
            logger.error(f"Search: Failed to parse API response: {e}")
            return f"Error processing search results: {e}"

        if not parsed.result:
            if self.log_requests:
                logger.info("Search: No results found.")
            return "No search results found."

        if self.log_requests:
            logger.info(f"Search: Got {len(parsed.result)} results.")
        return str(parsed)
