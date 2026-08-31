

# ================= NOTEBOOK CELL 5 =================

import re
import time
import asyncio
import unicodedata
from typing import List, Optional, Dict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from rapidfuzz import fuzz
from pydantic import BaseModel

from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool


# ================= NOTEBOOK CELL 7 =================

def _try_build_llm(name, env_var, build_fn, fallback_build_fn=None):
    if env_var and not os.environ.get(env_var):
        return None
    try:
        return {"name": name, "llm": build_fn()}
    except Exception as e:
        if fallback_build_fn is not None:
            # FIX: used for Groq's JSON-mode preference below - if this
            # crewai/litellm version doesn't accept the extra
            # response_format kwarg, construction itself would raise
            # here. Rather than losing the provider entirely, retry
            # once with the plain (tool-calling) constructor so Groq
            # still ends up in the chain either way.
            try:
                print(f"\u2139 {name}: structured-JSON construction failed ({e}) - retrying in standard mode.")
                return {"name": name, "llm": fallback_build_fn()}
            except Exception as e2:
                print(f"\u26a0 Could not initialize {name}: {e2}")
                return None
        print(f"\u26a0 Could not initialize {name}: {e}")
        return None


class ProviderChain:
    """Sticky failover across providers for ONE workload. Tries the
    CURRENT provider every time; only moves to the next one on failure."""

    def __init__(self, name, specs):
        self.name = name
        self.providers = [p for p in (_try_build_llm(**s) for s in specs) if p]
        if not self.providers:
            raise RuntimeError(
                f"No usable providers configured for '{name}' - "
                f"set at least one API key (Groq/Gemini/etc.)."
            )
        self.index = 0

    def current(self):
        return self.providers[self.index]

    def advance(self):
        if self.index < len(self.providers) - 1:
            self.index += 1
            print(f"\u26a0 [{self.name}] switching to next provider: {self.current()['name']}")
            return True
        return False


# FIX: none of these passed a `timeout` before, so a stalled Groq/Gemini
# request could hang the whole pipeline indefinitely (crewai/litellm
# honor this kwarg per-request). 45s is generous for a single structured
# parse/verify call but still finite - combined with run_with_fallback's
# existing retry/failover, a genuinely stuck provider now surfaces as a
# transient error and gets failed over instead of hanging forever.
_LLM_TIMEOUT_SECONDS = 45

# FIX: Groq specifically - prefer plain JSON-mode output over
# tool-calling, since Groq's structured-output/tool-call path is the
# one that's been failing (see STRUCTURED_OUTPUT_ERROR_KEYWORDS in the
# next cell). response_format={"type": "json_object"} is a standard
# litellm/OpenAI-compatible passthrough kwarg Groq's API supports, and
# asks the model to just emit JSON text directly rather than going
# through function/tool-calling. crewai still parses the result into
# the same output_pydantic model either way - nothing about how the
# result is USED changes, only how Groq is asked to produce it.
# _try_build_llm's fallback_build_fn (above) means that if this kwarg
# somehow isn't accepted by this crewai/litellm version, Groq falls
# back to its normal construction instead of disappearing from the
# chain. Gemini is untouched - it was already working correctly.
_GROQ_JSON_MODE_KWARGS = {"response_format": {"type": "json_object"}}

PARSER_SPECS = [
    {"name": "Groq (llama-3.3-70b)", "env_var": "GROQ_API_KEY",
     "build_fn": lambda: LLM(model='groq/openai/gpt-oss-120b', temperature=0.3, timeout=_LLM_TIMEOUT_SECONDS, **_GROQ_JSON_MODE_KWARGS),
     "fallback_build_fn": lambda: LLM(model='groq/openai/gpt-oss-120b', temperature=0.3, timeout=_LLM_TIMEOUT_SECONDS)},
    #{"name": "Cerebras (llama-3.3-70b)", "env_var": "CEREBRAS_API_KEY",
     #"build_fn": lambda: LLM(model='cerebras/llama-3.3-70b', temperature=0.3, timeout=_LLM_TIMEOUT_SECONDS)},
    {"name": "Gemini Flash", "env_var": "GOOGLE_API_KEY",
     "build_fn": lambda: LLM(model='gemini-3.1-flash-lite', provider='google', temperature=0.3, timeout=_LLM_TIMEOUT_SECONDS)},
    #{"name": "OpenRouter (free Llama)", "env_var": "OPENROUTER_API_KEY",
    # "build_fn": lambda: LLM(model='openrouter/meta-llama/llama-3.3-70b-instruct:free', temperature=0.3, timeout=_LLM_TIMEOUT_SECONDS)},
]

VERIFIER_SPECS = [
    {"name": "Gemini Flash", "env_var": "GOOGLE_API_KEY",
     "build_fn": lambda: LLM(model='gemini-3.1-flash-lite', provider='google', temperature=0.2, timeout=_LLM_TIMEOUT_SECONDS)},
    #{"name": "Cerebras (llama-3.3-70b)", "env_var": "CEREBRAS_API_KEY",
     #"build_fn": lambda: LLM(model='cerebras/llama-3.3-70b', temperature=0.2, timeout=_LLM_TIMEOUT_SECONDS)},
    #{"name": "OpenRouter (free Llama)", "env_var": "OPENROUTER_API_KEY",
     #"build_fn": lambda: LLM(model='openrouter/meta-llama/llama-3.3-70b-instruct:free', temperature=0.2, timeout=_LLM_TIMEOUT_SECONDS)},
    {"name": "Groq (llama-3.3-70b)", "env_var": "GROQ_API_KEY",
     "build_fn": lambda: LLM(model='groq/openai/gpt-oss-120b', temperature=0.2, timeout=_LLM_TIMEOUT_SECONDS, **_GROQ_JSON_MODE_KWARGS),
     "fallback_build_fn": lambda: LLM(model='groq/openai/gpt-oss-120b', temperature=0.2, timeout=_LLM_TIMEOUT_SECONDS)},
]

parser_chain = ProviderChain("PARSER", PARSER_SPECS)
verifier_chain = ProviderChain("VERIFIER", VERIFIER_SPECS)

print("Parser chain:", [p["name"] for p in parser_chain.providers])
print("Verifier chain:", [p["name"] for p in verifier_chain.providers])


# ================= NOTEBOOK CELL 8 =================


# --------------------------------------------------
# AI usage tracking - every successful call is counted here so you can
# read real per-paper cost from print_usage_summary() instead of guessing.
# --------------------------------------------------

USAGE_STATS = {"calls_by_chain": {}, "calls_by_provider": {},
               "total_input_tokens": 0, "total_output_tokens": 0, "total_tokens": 0}

def _record_usage(chain_name, provider_name, crew_result):
    USAGE_STATS["calls_by_chain"][chain_name] = USAGE_STATS["calls_by_chain"].get(chain_name, 0) + 1
    USAGE_STATS["calls_by_provider"][provider_name] = USAGE_STATS["calls_by_provider"].get(provider_name, 0) + 1
    usage = getattr(crew_result, "token_usage", None)
    if usage:
        p = getattr(usage, "prompt_tokens", 0) or 0
        c = getattr(usage, "completion_tokens", 0) or 0
        t = getattr(usage, "total_tokens", 0) or (p + c)
        USAGE_STATS["total_input_tokens"] += p
        USAGE_STATS["total_output_tokens"] += c
        USAGE_STATS["total_tokens"] += t

def print_usage_summary():
    print("\n" + "=" * 60)
    print("AI USAGE SUMMARY")
    print("=" * 60)
    for chain, count in USAGE_STATS["calls_by_chain"].items():
        print(f"  {chain}: {count} call(s)")
    if USAGE_STATS["total_tokens"]:
        print(f"  Total tokens: {USAGE_STATS['total_tokens']:,} "
              f"(in={USAGE_STATS['total_input_tokens']:,}, out={USAGE_STATS['total_output_tokens']:,})")
    print("=" * 60)


QUOTA_ERROR_KEYWORDS = ["rate", "429", "quota", "resource exhausted"]
TRANSIENT_ERROR_KEYWORDS = ["timeout", "overloaded", "unavailable", "502", "503", "504", "connection"]

# FIX: a Groq structured-output/tool-call failure (the model's response
# doesn't parse into the expected schema) previously matched NONE of
# the keywords above, so `if not is_transient: raise` gave up on the
# ENTIRE chain on the very first attempt instead of failing over to the
# next provider (e.g. Gemini). This is almost always a single
# provider/model hiccup, not evidence the reference itself is bad, so
# it's treated as transient too and goes through the exact same
# failover path as any other transient error - no new retry system.
STRUCTURED_OUTPUT_ERROR_KEYWORDS = [
    "tool call", "tool_call", "function call", "function_call",
    "structured output", "invalid json", "json decode", "jsondecodeerror",
    "does not support", "failed to parse", "validation error",
    "response_format", "schema",
]

async def run_with_fallback(chain, build_agent_fn, build_task_fn, inputs, max_attempts_per_provider=2):
    """Builds a fresh agent+task on the chain's CURRENT provider each
    attempt. FIX: a quota/rate-limit error switches providers
    IMMEDIATELY (retrying the same exhausted quota is pointless) - only
    genuinely transient errors (timeout, temporary server error) get a
    same-provider retry before moving on."""
    attempts_left = len(chain.providers) * max_attempts_per_provider
    last_error = None

    while attempts_left > 0:
        provider = chain.current()
        agent = build_agent_fn(provider["llm"])
        task = build_task_fn(agent)
        crew = Crew(agents=[agent], tasks=[task], verbose=False)

        try:
            await asyncio.sleep(0.3)
            crew_result = await crew.kickoff_async(inputs=inputs)
            _record_usage(chain.name, provider["name"], crew_result)
            return crew_result
        except Exception as e:
            last_error = e
            error_text = str(e).lower()
            attempts_left -= 1
            is_quota_error = any(k in error_text for k in QUOTA_ERROR_KEYWORDS)
            is_transient = (
                is_quota_error
                or any(k in error_text for k in TRANSIENT_ERROR_KEYWORDS)
                or any(k in error_text for k in STRUCTURED_OUTPUT_ERROR_KEYWORDS)
            )
            if not is_transient:
                raise
            print(f"\u26a0 [{chain.name}] {provider['name']} failed: {e}")
            if is_quota_error or not chain.providers[chain.index:chain.index+1]:
                chain.advance()
                continue
            if chain.advance():
                continue
            if attempts_left > 0:
                # FIX: Gemini free tier's quota is PER-MINUTE (e.g. 15
                # requests/min) - a 5s wait almost never lets it reset,
                # so with only 2 providers configured (Groq+Gemini) the
                # chain exhausted for real. 20s gives a real chance of
                # the window clearing before we give up entirely.
                print(f"\u23f3 [{chain.name}] all providers exhausted - waiting 20s for quota to reset...")
                await asyncio.sleep(20)

    raise RuntimeError(f"All providers in the {chain.name} chain failed. Last error: {last_error}")


# ================= NOTEBOOK CELL 10 =================

# --------------------------------------------------
# Shared session: retries + backoff + per-host throttle + proper
# identification (Crossref/NCBI explicitly ask for this).
#
# FIX: safe_get() used to silently swallow the status code - a 429
# (rate limited) looked identical to a 404 (doesn't exist) or a
# genuine network drop. Now every failure is labeled with WHY.
# --------------------------------------------------

from urllib.parse import urlparse

_session = requests.Session()
_retry = Retry(
    total=2, backoff_factor=1.0, status_forcelist=[500, 502, 503, 504],
    allowed_methods=["GET"],
    # FIX: urllib3 defaults to honoring a server's Retry-After header,
    # sleeping INSIDE the blocking .get() call for up to 6 hours
    # (retry_after_max default = 21600s) - completely bypassing our own
    # timeout=(5, timeout) parameter below, which only bounds a single
    # attempt's connect/read time, not this internal retry sleep. This
    # is exactly what caused runs to appear permanently stuck: a 5xx
    # response with a large Retry-After header from an overloaded API
    # silently parked the whole session for hours with no log output
    # (the sleep happens before control ever returns to safe_get()).
    # Disabling this makes urllib3 rely purely on our own bounded
    # backoff_factor instead - our own circuit breaker (below) already
    # handles "stop calling a source that's clearly down" correctly.
    respect_retry_after_header=False,
)
_session.mount("https://", HTTPAdapter(max_retries=_retry))
_session.mount("http://", HTTPAdapter(max_retries=_retry))

CONTACT_EMAIL = "your-email@example.com"  # <-- put your real email here (Crossref/NCBI use this to give you a faster "polite pool")
HEADERS = {"User-Agent": f"AcademicReferenceVerifier/1.0 (mailto:{CONTACT_EMAIL})"}

SEARCH_CACHE = {}
MAX_CACHE_SIZE = 1000

# Minimum seconds between requests to each host - keeps us politely
# under each API's published rate limit instead of bursting and
# getting 429'd.
API_THROTTLES = {
    "api.crossref.org": 0.35,
    "api.openalex.org": 0.15,
    "api.semanticscholar.org": 3.0,   # anonymous tier 429s constantly under real load - be very conservative
    "eutils.ncbi.nlm.nih.gov": 0.4,
    "dblp.org": 0.5,
}
_last_request_times = {}

def _throttle(url):
    host = urlparse(url).netloc
    interval = API_THROTTLES.get(host, 0.25)
    now = time.time()
    wait = interval - (now - _last_request_times.get(host, 0))
    if wait > 0:
        time.sleep(wait)
    _last_request_times[host] = time.time()


# --------------------------------------------------
# Circuit breaker: once a host has given a hard-failure
# signal that means "this will keep failing" - a 429
# (rate limit/quota exhausted) or repeated 5xx/timeout/
# connection failures - stop calling it for the rest of
# THIS run instead of continuing to burn throttle-wait +
# timeout time on a source that's already known to be
# down. Per-host, module-level state; resets on the next
# fresh run of the program.
# --------------------------------------------------
CIRCUIT_BREAKER_THRESHOLD = 3   # consecutive 5xx/timeout/connection failures before tripping
_circuit_breaker_state = {}      # host -> {"failures": int, "open": bool}

def _circuit_is_open(host):
    return _circuit_breaker_state.get(host, {}).get("open", False)

def _record_hard_failure(host, immediate=False):
    """immediate=True trips the breaker on this single failure (used for
    429 - a rate-limit/quota response is an unambiguous "stop calling
    me" signal, unlike a possibly-transient timeout or 5xx)."""
    state = _circuit_breaker_state.setdefault(host, {"failures": 0, "open": False})
    if state["open"]:
        return
    state["failures"] += 1
    if immediate or state["failures"] >= CIRCUIT_BREAKER_THRESHOLD:
        state["open"] = True
        reason = "rate limited/quota exhausted" if immediate else f"{state['failures']} consecutive failures"
        print(f"⛔ {host} DISABLED for the rest of this run ({reason}) "
              f"- further calls to this source will be skipped instantly.", flush=True)

def _record_success(host):
    if host in _circuit_breaker_state:
        _circuit_breaker_state[host] = {"failures": 0, "open": False}


def safe_get(url, params=None, timeout=20):
    host = urlparse(url).netloc

    if _circuit_is_open(host):
        print(f"      ⏭ SKIPPING {host} (circuit breaker open - already known unavailable this run)", flush=True)
        return None

    print(f"      → THROTTLE {host}", flush=True)
    _throttle(url)
    print(f"      → THROTTLE DONE {host}", flush=True)

    print(f"      → REQUEST {host}", flush=True)

    try:
        resp = _session.get(
            url,
            params=params,
            headers=HEADERS,
            timeout=(5, timeout)
        )

        print(
            f"      ← RESPONSE {host}: HTTP {resp.status_code}",
            flush=True
        )

        if resp.status_code == 200:
            _record_success(host)
            return resp

        if resp.status_code == 429:
            print(
                f"⚠ Rate limited by {host} "
                f"(Retry-After={resp.headers.get('Retry-After')}) "
                f"- skipping this source for this reference.",
                flush=True
            )
            _record_hard_failure(host, immediate=True)

        elif resp.status_code in (500, 502, 503, 504):
            print(
                f"⚠ {host} temporary server error "
                f"{resp.status_code}",
                flush=True
            )
            _record_hard_failure(host)

        else:
            print(
                f"⚠ {host} returned {resp.status_code}",
                flush=True
            )

        return None

    except requests.exceptions.Timeout:
        print(
            f"⚠ TIMEOUT contacting {host} "
            f"after {timeout}s - skipping.",
            flush=True
        )
        _record_hard_failure(host)
        return None

    except requests.exceptions.RequestException as e:
        print(
            f"⚠ Request to {host} failed: {e}",
            flush=True
        )
        _record_hard_failure(host)
        return None

    except Exception as e:
        # Defensive catch-all: an unexpected, non-requests error here
        # (e.g. a malformed URL, an unusual SSL/library-internal error)
        # must never propagate and stop the current reference - log it
        # and move on exactly like any other source failure.
        print(
            f"⚠ Unexpected error contacting {host}: {e} - skipping.",
            flush=True
        )
        _record_hard_failure(host)
        return None


# FIX: LLM-parsed DOIs (and occasionally API responses) can be the
# literal STRING "null"/"None"/"N/A"/"nan" instead of a real missing
# value - e.g. a JSON tool-call argument serialized as the text "null"
# rather than the JSON literal null. Since this is truthy, `if not doi`
# alone never catches it, and compare_doi() would then treat it as a
# real submitted DOI and report a false MISMATCH against any candidate.
_DOI_PLACEHOLDER_VALUES = {
    "null", "none", "nan", "n/a", "na", "unknown", "undefined", "-", "--", "?"
}

def normalize_doi(doi):
    """
    Canonical DOI normalization.

    Handles:
      10.1234/ABC
      https://doi.org/10.1234/ABC
      http://doi.org/10.1234/ABC
      doi:10.1234/ABC

    Also repairs DOI line breaks introduced by PDF extraction:
      10.1016/j.jbi.
      2024.104746

    Also collapses placeholder "empty" strings ("null", "N/A", ...) to
    "" so every caller (compare_doi, dedup, cache key, ranking) sees
    exactly the same "no DOI" state it would see for a real None.
    """
    if not doi:
        return ""

    doi = unicodedata.normalize("NFKD", str(doi)).strip()

    if doi.lower() in _DOI_PLACEHOLDER_VALUES:
        return ""

    # Remove DOI URL / prefix.
    doi = re.sub(
        r"^\s*(?:https?://)?(?:dx\.)?doi\.org/",
        "",
        doi,
        flags=re.IGNORECASE
    )

    doi = re.sub(
        r"^\s*doi\s*:\s*",
        "",
        doi,
        flags=re.IGNORECASE
    )

    # Remove whitespace/newlines accidentally inserted inside DOI.
    doi = re.sub(r"\s+", "", doi)

    # Remove surrounding punctuation.
    doi = doi.strip(" \t\r\n.,;:()[]{}<>\"'")

    # Defensive re-check: in case stripping ever leaves behind exactly
    # one of the placeholder tokens (unlikely, but cheap to guard).
    if doi.lower() in _DOI_PLACEHOLDER_VALUES:
        return ""

    return doi.lower()


def cache_result(key, value):
    if len(SEARCH_CACHE) >= MAX_CACHE_SIZE:
        SEARCH_CACHE.pop(next(iter(SEARCH_CACHE)))
    SEARCH_CACHE[key] = value


# ================= NOTEBOOK CELL 11 =================

# ---------- Crossref ----------
def search_crossref(query, rows=40):
    resp = safe_get("https://api.crossref.org/works",
                     {"query.bibliographic": query, "rows": rows, "sort": "score",
                      "order": "desc", "mailto": CONTACT_EMAIL})
    if not resp:
        return []
    papers = []
    for item in resp.json().get("message", {}).get("items", []):
        authors = [(a.get("given", "") + " " + a.get("family", "")).strip() for a in item.get("author", [])]
        papers.append({
            "title": (item.get("title") or [""])[0], "authors": authors,
            "doi": item.get("DOI"),
            "year": (item.get("issued", {}).get("date-parts") or [[None]])[0][0],
            "venue": (item.get("container-title") or [None])[0], "source": "Crossref",
        })
    return papers

def search_crossref_by_doi(doi):
    resp = safe_get(f"https://api.crossref.org/works/{normalize_doi(doi)}", {"mailto": CONTACT_EMAIL})
    if not resp:
        return None
    item = resp.json()["message"]
    authors = [(a.get("given", "") + " " + a.get("family", "")).strip() for a in item.get("author", [])]
    return {"title": (item.get("title") or [""])[0], "authors": authors, "doi": item.get("DOI"),
            "year": (item.get("issued", {}).get("date-parts") or [[None]])[0][0],
            "venue": (item.get("container-title") or [None])[0], "source": "Crossref"}

def search_crossref_title(title, rows=30):
    query = f'"{title}"' if len(title.split()) > 2 else title

    # One request only. The previous version made a second identical
    # Crossref request immediately and discarded the first response,
    # unnecessarily doubling search time and API load.
    resp = safe_get(
        "https://api.crossref.org/works",
        {
            "query.title": query,
            "rows": rows,
            "mailto": CONTACT_EMAIL
        }
    )

    if not resp:
        return []

    papers = []

    for item in resp.json().get("message", {}).get("items", []):
        authors = [
            (a.get("given", "") + " " + a.get("family", "")).strip()
            for a in item.get("author", [])
        ]

        papers.append({
            "title": (item.get("title") or [""])[0],
            "authors": authors,
            "doi": item.get("DOI"),
            "year": (item.get("issued", {}).get("date-parts") or [[None]])[0][0],
            "venue": (item.get("container-title") or [None])[0],
            "source": "Crossref"
        })

    return papers


# ---------- OpenAlex ----------
def search_openalex(query, per_page=20):
    resp = safe_get("https://api.openalex.org/works", {"search": query, "per-page": per_page, "mailto": CONTACT_EMAIL})
    if not resp:
        return []
    papers = []
    for item in resp.json().get("results", []):
        authors = [a["author"]["display_name"] for a in item.get("authorships", []) if a.get("author")]
        doi = item.get("doi")
        if doi:
            doi = normalize_doi(doi)
        loc = item.get("primary_location") or {}
        papers.append({"title": item.get("title") or item.get("display_name"), "authors": authors, "doi": doi,
                        "year": item.get("publication_year"),
                        "venue": (loc.get("source") or {}).get("display_name"), "source": "OpenAlex"})
    return papers

def search_openalex_by_doi(doi):
    resp = safe_get(f"https://api.openalex.org/works/https://doi.org/{normalize_doi(doi)}", {"mailto": CONTACT_EMAIL})
    if not resp:
        return None
    item = resp.json()
    authors = [a["author"]["display_name"] for a in item.get("authorships", []) if a.get("author")]
    doi_val = item.get("doi")
    loc = item.get("primary_location") or {}
    return {"title": item.get("title") or item.get("display_name"), "authors": authors,
            "doi": normalize_doi(doi_val) if doi_val else None, "year": item.get("publication_year"),
            "venue": (loc.get("source") or {}).get("display_name"), "source": "OpenAlex"}


# ---------- DBLP (CS-specific) ----------
def search_dblp(query, hits=20):
    resp = safe_get("https://dblp.org/search/publ/api", {"q": query, "h": hits, "format": "json"})
    if not resp:
        return []
    hits_data = resp.json().get("result", {}).get("hits", {}).get("hit", [])
    if isinstance(hits_data, dict):
        hits_data = [hits_data]
    papers = []
    for paper in hits_data:
        info = paper.get("info", {})
        author_data = info.get("authors", {}).get("author", [])
        if isinstance(author_data, dict):
            author_data = [author_data]
        authors = [a.get("text", "") for a in author_data] if author_data else []
        papers.append({"title": info.get("title"), "authors": authors, "doi": info.get("doi"),
                        "year": int(info["year"]) if info.get("year") else None,
                        "venue": info.get("venue"), "source": "DBLP"})
    return papers


# ---------- Semantic Scholar (CS/AI + general scholarly matching) ----------
def search_semantic_scholar(query, limit=20):
    resp = safe_get("https://api.semanticscholar.org/graph/v1/paper/search",
                     {"query": query, "limit": limit, "fields": "title,authors,year,venue,externalIds"})
    if not resp:
        return []
    papers = []
    for item in resp.json().get("data", []):
        papers.append({"title": item.get("title"), "authors": [a.get("name") for a in item.get("authors", [])],
                        "doi": (item.get("externalIds") or {}).get("DOI"), "year": item.get("year"),
                        "venue": item.get("venue"), "source": "Semantic Scholar"})
    return papers

def search_semantic_scholar_by_doi(doi):
    resp = safe_get(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{normalize_doi(doi)}",
                     {"fields": "title,authors,year,venue,externalIds"})
    if not resp:
        return None
    item = resp.json()
    return {"title": item.get("title"), "authors": [a.get("name") for a in item.get("authors", [])],
            "doi": (item.get("externalIds") or {}).get("DOI"), "year": item.get("year"),
            "venue": item.get("venue"), "source": "Semantic Scholar"}


# ---------- PubMed (biomedical only) ----------
def search_pubmed(query, retmax=20):
    search_resp = safe_get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                            {"db": "pubmed", "term": query, "retmax": retmax, "retmode": "json",
                             "tool": "AcademicReferenceVerifier", "email": CONTACT_EMAIL})
    if not search_resp:
        return []
    ids = search_resp.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    summary_resp = safe_get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                             {"db": "pubmed", "id": ",".join(ids), "retmode": "json",
                              "tool": "AcademicReferenceVerifier", "email": CONTACT_EMAIL})
    if not summary_resp:
        return []
    result = summary_resp.json().get("result", {})
    papers = []
    for pmid in ids:
        item = result.get(pmid)
        if not item:
            continue
        authors = [a.get("name", "") for a in item.get("authors", [])]
        doi = next((idobj.get("value") for idobj in item.get("articleids", []) if idobj.get("idtype") == "doi"), None)
        year_match = re.search(r"\d{4}", item.get("pubdate", ""))
        papers.append({"title": item.get("title"), "authors": authors, "doi": doi,
                        "year": int(year_match.group(0)) if year_match else None,
                        "venue": item.get("fulljournalname") or item.get("source"), "source": "PubMed"})
    return papers


# ---------- Domain detection - decides WHICH extra sources are worth calling ----------
_CS_TERMS = ["machine learning", "deep learning", "artificial intelligence", "computer vision",
             "natural language processing", "nlp", "transformer", "neural network", "software",
             "database", "data mining", "information retrieval", "cybersecurity", "computer science",
             "algorithm", "large language model", "llm", "sql", "text-to-sql", "forensic"]
_BIO_TERMS = ["medicine", "medical", "clinical", "health", "healthcare", "cancer", "disease",
              "patient", "biology", "genomics", "genetic", "neuroscience", "pharmaceutical",
              "drug", "pathology", "epidemiology"]

def detect_domain(parsed_reference):
    text = normalize_text(f"{parsed_reference.title} {parsed_reference.venue or ''}")
    if any(t in text for t in _CS_TERMS):
        return "cs"
    if any(t in text for t in _BIO_TERMS):
        return "biomedical"
    return "general"

def get_extra_sources(parsed_reference):
    """Crossref + OpenAlex always run. This decides what ELSE to try
    if those two aren't enough.

    FIX: DBLP now goes FIRST for CS references, not Semantic Scholar.
    DBLP is the authoritative, low-friction index for exactly this kind
    of paper (ACL/EMNLP/AAAI/IEEE venues) and rarely rate-limits.
    Semantic Scholar's anonymous tier gets 429'd constantly under any
    real load, so pushing it second means we usually already have a
    good candidate from DBLP before we even risk hitting that limit."""
    domain = detect_domain(parsed_reference)
    if domain == "cs":
        return [search_dblp, search_semantic_scholar]
    if domain == "biomedical":
        return [search_pubmed, search_semantic_scholar]
    return [search_semantic_scholar]


# ---------- Deduplication - merges the SAME paper found by multiple sources into one candidate ----------
def deduplicate_candidates(candidates):
    groups = {}
    for c in candidates:
        doi = normalize_doi(c.get("doi"))
        key = f"doi:{doi}" if doi else f"title:{normalize_text(c.get('title', ''))}"
        if not key or key in ("doi:", "title:"):
            continue
        if key not in groups:
            groups[key] = {**c, "sources": [c.get("source")]}
        else:
            existing = groups[key]
            src = c.get("source")
            if src and src not in existing["sources"]:
                existing["sources"].append(src)
            for field in ["title", "authors", "year", "venue", "doi"]:
                if not existing.get(field) and c.get(field):
                    existing[field] = c[field]
    return list(groups.values())
def title_identity_score(submitted_title, candidate_title):
    """
    Measures whether the candidate is actually the same paper
    based primarily on TITLE identity.

    A high semantic/topic similarity is NOT enough.
    """

    if not submitted_title or not candidate_title:
        return 0

    a = normalize_text(submitted_title)
    b = normalize_text(candidate_title)

    if not a or not b:
        return 0

    # Exact normalized title
    if a == b:
        return 100

    # Fuzzy full-title similarity
    full_score = fuzz.token_set_ratio(a, b)

    # Stronger measure that penalizes missing title words
    ratio_score = fuzz.ratio(a, b)

    # Token-set similarity is useful for punctuation/order differences
    token_score = fuzz.token_sort_ratio(a, b)

    return max(
        full_score,
        ratio_score,
        token_score
    )
def is_plausible_title_match(submitted_title, candidate_title):
    """
    Decide whether a candidate is actually plausible based on title.
    This prevents unrelated papers from being treated as 'closest matches'.
    """

    score = title_identity_score(submitted_title, candidate_title)

    if score >= 95:
        return True

    # Slightly tolerant threshold for punctuation,
    # hyphenation, OCR and minor title formatting differences.
    if score >= 90:
        return True

    return False


# ================= NOTEBOOK CELL 12 =================

# --------------------------------------------------
# STRONG MATCH
# --------------------------------------------------

def _is_strong_match(parsed_reference, candidates):

    if not candidates:
        return False

    submitted_doi = normalize_doi(parsed_reference.doi)

    # --------------------------------------------------
    # 1. EXACT DOI = strongest possible evidence
    # --------------------------------------------------
    if submitted_doi:

        for c in candidates:

            candidate_doi = normalize_doi(c.get("doi"))

            if candidate_doi and candidate_doi == submitted_doi:
                return True

    # --------------------------------------------------
    # 2. STRONG TITLE + AUTHOR MATCH
    #
    # Check ALL candidates, not only ranked[0].
    # This prevents a bad top-ranked candidate from
    # hiding a genuinely correct candidate.
    # --------------------------------------------------
    for c in candidates:

        title_score = title_identity_score(
            parsed_reference.title or "",
            c.get("title", "") or ""
        )

        author_score = author_similarity(
            parsed_reference.authors,
            c.get("authors", [])
        )

        if title_score >= 95 and author_score >= 90:
            return True

    return False


# --------------------------------------------------
# SEARCH ALL DATABASES
#
# Staged / evidence-aware search:
#
# 1. DOI present:
#       Crossref + OpenAlex + Semantic Scholar DOI lookup
#       -> exact DOI found = STOP
#       -> otherwise continue to text search
#
# 2. Full title:
#       Crossref + OpenAlex
#       -> strong match = STOP
#
# 3. Title without common words
#       -> strong match = STOP
#
# 4. Author + title keywords
#       -> strong match = STOP
#
# 5. Limited distinctive-keyword searches
#       -> strong match = STOP
#
# 6. Title fragments
#
# 7. Domain-specific sources
#
# 8. Return final candidate pool
# --------------------------------------------------

def search_all_databases(parsed_reference):

    # --------------------------------------------------
    # CACHE KEY
    # --------------------------------------------------
    cache_key = (
        normalize_doi(parsed_reference.doi or "")
        + "|"
        + normalize_text(parsed_reference.title or "")
    )

    if cache_key in SEARCH_CACHE:

        print("  ✓ Search cache hit")

        return SEARCH_CACHE[cache_key]

    candidates = []

    # ==================================================
    # STEP 1: DOI-FIRST EXACT LOOKUP
    # ==================================================

    if parsed_reference.doi:

        print("  → DOI-first exact lookup...")

        for direct_lookup in (
            search_crossref_by_doi,
            search_openalex_by_doi,
            search_semantic_scholar_by_doi
        ):

            try:

                result = direct_lookup(
                    parsed_reference.doi
                )

                if result:
                    candidates.append(result)

            except Exception as e:

                print(
                    f"⚠ DOI lookup failed: {e}"
                )

        # --------------------------------------------------
        # Remove duplicate results from DOI providers
        # --------------------------------------------------
        candidates = deduplicate_candidates(candidates)

        submitted_doi = normalize_doi(
            parsed_reference.doi
        )

        # --------------------------------------------------
        # EXACT DOI CONFIRMATION
        # --------------------------------------------------
        if submitted_doi:

            exact_doi_candidates = [
                c
                for c in candidates
                if normalize_doi(c.get("doi")) == submitted_doi
            ]

            if exact_doi_candidates:

                print(
                    f"  ✓ Exact DOI confirmed "
                    f"({len(exact_doi_candidates)} candidate(s))"
                )

                # Rank ONLY the exact-DOI candidates.
                ranked = rank_candidates(
                    parsed_reference,
                    exact_doi_candidates
                )

                if ranked:

                    candidates = [
                        item["candidate"]
                        for item in ranked
                    ]

                else:

                    candidates = exact_doi_candidates

                cache_result(
                    cache_key,
                    candidates
                )

                return candidates

        # --------------------------------------------------
        # IMPORTANT:
        #
        # DOI was supplied but no database confirmed it.
        #
        # DO NOT RETURN HERE.
        #
        # The DOI may be:
        # - malformed
        # - not indexed
        # - temporarily unavailable
        # - extracted incorrectly
        # - absent from a particular database
        #
        # Continue with title/author search.
        # --------------------------------------------------

        print(
            "  → No exact DOI confirmation; "
            "continuing with text search..."
        )

    # ==================================================
    # STEP 2: CLEAN TITLE
    # ==================================================

    query = normalize_text(
        parsed_reference.title or ""
    )

    if not query:

        cache_result(
            cache_key,
            candidates
        )

        return candidates

    print("  → Searching databases...")

    # ==================================================
    # STRATEGY 1: FULL TITLE SEARCH
    # ==================================================

    print("    • Crossref title search...")

    try:

        candidates.extend(
            search_crossref_title(
                query,
                rows=50
            )
        )

    except Exception as e:

        print(
            f"⚠ Crossref title search failed: {e}"
        )

    print("    • Crossref bibliographic search...")

    try:

        candidates.extend(
            search_crossref(
                query,
                rows=50
            )
        )

    except Exception as e:

        print(
            f"⚠ Crossref bibliographic search failed: {e}"
        )

    print("    • OpenAlex search...")

    try:

        candidates.extend(
            search_openalex(
                query,
                per_page=50
            )
        )

    except Exception as e:

        print(
            f"⚠ OpenAlex search failed: {e}"
        )

    candidates = deduplicate_candidates(
        candidates
    )

    # --------------------------------------------------
    # Strong match?
    # --------------------------------------------------

    if _is_strong_match(
        parsed_reference,
        candidates
    ):

        print(
            f"  ✓ Strong match found "
            f"({len(candidates)} candidate(s))"
        )

        cache_result(
            cache_key,
            candidates
        )

        return candidates

    # ==================================================
    # STRATEGY 2: REMOVE COMMON WORDS
    # ==================================================

    print(
        "  → Trying search without common words..."
    )

    stopwords = {
        "a", "an", "the",
        "of", "for", "on", "at",
        "to", "in", "with",
        "without", "and", "or",
        "but", "by", "from",
        "into", "through"
    }

    query_words = query.split()

    clean_query = " ".join(
        w
        for w in query_words
        if w not in stopwords
        and len(w) > 2
    )

    if (
        clean_query
        and clean_query != query
    ):

        try:

            candidates.extend(
                search_crossref(
                    clean_query,
                    rows=50
                )
            )

        except Exception as e:

            print(
                f"⚠ Crossref reduced-title search failed: {e}"
            )

        try:

            candidates.extend(
                search_openalex(
                    clean_query,
                    per_page=50
                )
            )

        except Exception as e:

            print(
                f"⚠ OpenAlex reduced-title search failed: {e}"
            )

        candidates = deduplicate_candidates(
            candidates
        )

        if _is_strong_match(
            parsed_reference,
            candidates
        ):

            print(
                f"  ✓ Strong match found "
                f"({len(candidates)} candidate(s))"
            )

            cache_result(
                cache_key,
                candidates
            )

            return candidates

    # ==================================================
    # STRATEGY 3: AUTHOR + KEYWORDS
    # ==================================================

    if parsed_reference.authors:

        print(
            "  → Trying author + keyword search..."
        )

        first_author = (
            parsed_reference.authors[0]
        )

        title_words = query.split()

        keywords = (
            " ".join(title_words[:3])
            if len(title_words) >= 3
            else query
        )

        author_query = (
            f"{first_author} {keywords}"
        )

        try:

            candidates.extend(
                search_crossref(
                    author_query,
                    rows=50
                )
            )

        except Exception as e:

            print(
                f"⚠ Crossref author search failed: {e}"
            )

        try:

            candidates.extend(
                search_openalex(
                    author_query,
                    per_page=50
                )
            )

        except Exception as e:

            print(
                f"⚠ OpenAlex author search failed: {e}"
            )

        candidates = deduplicate_candidates(
            candidates
        )

        if _is_strong_match(
            parsed_reference,
            candidates
        ):

            print(
                f"  ✓ Strong match found "
                f"({len(candidates)} candidate(s))"
            )

            cache_result(
                cache_key,
                candidates
            )

            return candidates

    # ==================================================
    # STRATEGY 4: DISTINCTIVE KEY TERMS
    #
    # IMPORTANT:
    # The old implementation tried EVERY possible pair.
    #
    # For 15 terms:
    #
    #     15 * 14 / 2 = 105 API calls
    #
    # That can recreate the 429 problem.
    #
    # We therefore limit this to a small number of
    # distinctive combinations.
    # ==================================================

    print(
        "  → Trying distinctive key terms..."
    )

    key_terms = [
        w
        for w in query_words
        if w not in stopwords
        and len(w) > 4
    ]

    # Limit the number of terms considered.
    # Keeping the first several distinctive terms
    # prevents API explosion on long titles.
    key_terms = key_terms[:8]

    if len(key_terms) >= 3:

        # --------------------------------------------------
        # Try adjacent distinctive terms first.
        # Maximum: 7 searches.
        # --------------------------------------------------
        combinations = []

        for i in range(
            min(len(key_terms) - 1, 4)
        ):

            combinations.append(
                f"{key_terms[i]} {key_terms[i + 1]}"
            )

        # --------------------------------------------------
        # Also try first + last distinctive terms.
        # --------------------------------------------------
        if len(key_terms) >= 5:

            combinations.append(
                f"{key_terms[0]} {key_terms[-1]}"
            )

            combinations.append(
                " ".join(key_terms[:3])
            )

            combinations.append(
                " ".join(key_terms[-3:])
            )

        # Remove duplicate queries while preserving order.
        combinations = list(
            dict.fromkeys(combinations)
        )

        # Hard safety limit.
        combinations = combinations[:8]

        for combo in combinations:

            try:

                candidates.extend(
                    search_crossref(
                        combo,
                        rows=30
                    )
                )

            except Exception as e:

                print(
                    f"⚠ Key-term search failed: {e}"
                )

        candidates = deduplicate_candidates(
            candidates
        )

        if _is_strong_match(
            parsed_reference,
            candidates
        ):

            print(
                f"  ✓ Strong match found "
                f"({len(candidates)} candidate(s))"
            )

            cache_result(
                cache_key,
                candidates
            )

            return candidates

    # ==================================================
    # STRATEGY 5: TITLE FRAGMENTS
    # ==================================================

    print(
        "  → Trying title fragment search..."
    )

    word_count = len(query_words)

    if word_count >= 6:

        midpoint = word_count // 2

        first_half = " ".join(
            query_words[:midpoint]
        )

        last_half = " ".join(
            query_words[midpoint:]
        )

        for fragment in (
            first_half,
            last_half
        ):

            try:

                candidates.extend(
                    search_crossref(
                        fragment,
                        rows=30
                    )
                )

            except Exception as e:

                print(
                    f"⚠ Title fragment search failed: {e}"
                )

        candidates = deduplicate_candidates(
            candidates
        )

    # ==================================================
    # STEP 6: DOMAIN-SPECIFIC SOURCES
    # ==================================================

    if not _is_strong_match(
        parsed_reference,
        candidates
    ):

        print(
            "  → Searching additional databases..."
        )

        extra_sources = get_extra_sources(
            parsed_reference
        )

        for search_fn in extra_sources:

            try:

                result = search_fn(
                    query
                )

                if result:

                    candidates.extend(result)

            except Exception as e:

                print(
                    f"⚠ {search_fn.__name__} failed: {e}"
                )

        candidates = deduplicate_candidates(
            candidates
        )

    # ==================================================
    # STEP 7: FINAL RESULT
    # ==================================================

    print(
        f"  → Final: {len(candidates)} candidate(s)"
    )

    cache_result(
        cache_key,
        candidates
    )

    return candidates


# ================= NOTEBOOK CELL 14 =================

import fitz  # PyMuPDF

REFERENCE_HEADINGS = [
    r"references", r"bibliography", r"works cited", r"literature cited",
    r"reference list", r"references cited", r"cited references",
]
STOP_HEADINGS = [
    r"appendix", r"acknowledg(e)?ments", r"author biograph(y|ies)",
    r"about the authors?", r"biograph(y|ies)", r"author information",
    # FIX: additive - a few more end-of-paper boilerplate sections that
    # some journals place AFTER the reference list rather than before
    # it. Each still requires the FULL line to be just this heading
    # (see heading_pattern's ^...$ anchor below), so this cannot match
    # inside an actual reference's text.
    r"supplementary material(s)?", r"conflicts? of interest",
    r"data availability( statement)?", r"funding( statement)?",
    r"author contributions?", r"declaration of competing interest",
    # FIX: added for the final-entry boundary fix below - "Proof",
    # "Proposition", "Theorem" etc. are common section headings that
    # follow a reference list directly into a paper's proof/appendix
    # section, with no other stop-heading in between.
    r"proof", r"proposition", r"theorem", r"lemma", r"corollary",
]
INLINE_STOP_PATTERNS = [
    r"received (the |his |her |a )?.{0,40}degree",
    r"\((Student |Graduate |Senior |)Member,?\s*IEEE\)",
    r"\(Fellow,?\s*IEEE\)",
    r"is currently (a|an|working|pursuing)",
    r"(his|her) research interests include",
]

# --------------------------------------------------
# Final-entry boundary trim.
#
# Every splitting style in _split_references() ends the LAST reference
# wherever the (already-trimmed-by-STOP_HEADINGS) references_text
# happens to end - correct when a clean stop heading was found earlier,
# but if the paper's post-bibliography content has NO standalone stop
# heading to mark it (e.g. a page footer, a running header repeated on
# every page, or a heading glued onto the same PDF-extracted line as
# the last citation with no following numbered reference to bound it),
# that trailing content silently gets attached to the last reference.
#
# This ONLY ever touches the LAST split entry - every earlier reference
# is already correctly bounded by the start of the next numbered entry
# and is completely unaffected by this.
# --------------------------------------------------
_TAIL_BOUNDARY_LINE_PATTERNS = [
    # Proof/Proposition/etc. are often followed by their statement on
    # the SAME line ("Proposition 3.1: For all epsilon...") rather than
    # standing alone, so this only requires the line to START with the
    # heading word (+ optional number/colon), not consist of only that.
    r"^\s*(proof|proposition|theorem|lemma|corollary|remark|definition)\b\s*\.?\s*\d*(\.\d+)?\s*[:.]?\s*(?=\s|$)",
    r"^\s*appendix\b",
    r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$",
    r"^\s*\u00a9\s*\d{4}",   # copyright line, e.g. "© 2024 IEEE"
]
_TAIL_BOUNDARY_PATTERN = re.compile(
    "|".join(_TAIL_BOUNDARY_LINE_PATTERNS), re.IGNORECASE | re.MULTILINE
)

def _trim_final_entry_boundary(entry_text, min_keep=25):
    """Cuts the LAST split reference at the first line matching a known
    non-reference boundary marker, as long as that still leaves at
    least `min_keep` characters of real content before it (so a
    short-but-real reference is never accidentally gutted). Returns the
    text unchanged if no such marker is found."""
    if not entry_text:
        return entry_text
    for m in _TAIL_BOUNDARY_PATTERN.finditer(entry_text):
        if m.start() >= min_keep:
            return entry_text[:m.start()].strip()
    return entry_text


def extract_full_text_column_aware(pdf_path):
    doc = fitz.open(pdf_path)
    pages_text = []
    for page in doc:
        blocks = page.get_text("blocks")
        mid = page.rect.width / 2
        left = sorted([b for b in blocks if b[0] < mid], key=lambda b: b[1])
        right = sorted([b for b in blocks if b[0] >= mid], key=lambda b: b[1])
        pages_text.append("\n".join(b[4] for b in left + right))
    doc.close()
    return "\n".join(pages_text)


def extract_references_from_text(full_text):
    heading_pattern = re.compile(
        r"^\s*(?:[IVXLC\d]+\.?\s*)?(" + "|".join(REFERENCE_HEADINGS) + r")\s*$",
        re.IGNORECASE | re.MULTILINE
    )
    matches = list(heading_pattern.finditer(full_text))

    if not matches:
        # FIX: fallback only - the strict heading match above requires
        # the heading word to be ALONE on its own line. Some PDF
        # extractions merge the heading and the first citation onto one
        # line (e.g. "References[1] J. Smith..."), which the strict
        # pattern never matches, silently falling back to the WHOLE
        # document as "references text". This loose pattern only runs
        # when the strict one found nothing, and still requires the
        # heading word be immediately followed by a plausible citation
        # opener (a bracketed/numbered marker or end of line) - so it
        # cannot match a stray sentence like "References to prior work
        # show..." inside the body text.
        loose_heading_pattern = re.compile(
            r"^\s*(?:[IVXLC\d]+\.?\s*)?(" + "|".join(REFERENCE_HEADINGS) + r")\s*"
            r"(?=\[\d+\]|\(?\d{1,3}[\.\)]\s|[A-Z][A-Za-z\-']+,\s*[A-Z]\.|\Z)",
            re.IGNORECASE | re.MULTILINE
        )
        matches = list(loose_heading_pattern.finditer(full_text))

    references_text = full_text[matches[-1].end():] if matches else full_text

    stop_pattern = re.compile(r"^\s*(" + "|".join(STOP_HEADINGS) + r")\s*$", re.IGNORECASE | re.MULTILINE)
    stop_match = stop_pattern.search(references_text)
    if stop_match:
        references_text = references_text[:stop_match.start()]

    inline_positions = [m.start() for pat in INLINE_STOP_PATTERNS
                         if (m := re.search(pat, references_text, re.IGNORECASE))]
    if inline_positions:
        references_text = references_text[:min(inline_positions)]

    return _split_references(references_text)


def _split_references(text):
    """FIX: now wraps the original splitter (renamed below) and trims
    only the LAST returned entry at a detected non-reference boundary.
    None of the splitting logic itself changed - see
    _split_references_raw() for the untouched original branches."""
    result = _split_references_raw(text)
    if result:
        result[-1] = _trim_final_entry_boundary(result[-1])
        # A trim could theoretically empty out a final entry that was
        # ALL boilerplate with nothing before it (e.g. min_keep guard
        # was still satisfied by junk text) - keep the existing >20-char
        # filter's spirit by dropping it if that happens, rather than
        # emitting an empty/near-empty "reference".
        if len(result[-1]) <= 20:
            result.pop()
    return result


def _split_references_raw(text):
    # [1] [2] [3]
    pattern = re.compile(r"\[\d+\]")
    matches = list(pattern.finditer(text))
    if len(matches) >= 2:
        positions = [m.start() for m in matches] + [len(text)]
        return [c for i in range(len(positions) - 1)
                if len(c := pattern.sub("", text[positions[i]:positions[i+1]], count=1).strip()) > 20]

    # "1." style
    dot_pattern = re.compile(r"(?:^|\n)\s*(\d{1,3})\.\s+")
    dot_matches = list(dot_pattern.finditer(text))
    if len(dot_matches) >= 2:
        positions = [m.start() for m in dot_matches] + [len(text)]
        return [c for i in range(len(positions) - 1)
                if len(c := dot_pattern.sub("", text[positions[i]:positions[i+1]], count=1).strip()) > 20]

    # "1)" style
    paren_pattern = re.compile(r"(?:^|\n)\s*(\d{1,3})\)\s+")
    paren_matches = list(paren_pattern.finditer(text))
    if len(paren_matches) >= 2:
        positions = [m.start() for m in paren_matches] + [len(text)]
        return [c for i in range(len(positions) - 1)
                if len(c := paren_pattern.sub("", text[positions[i]:positions[i+1]], count=1).strip()) > 20]

    # APA-style, no numbering
    apa_pattern = re.compile(r"(?m)^\s*[A-Z][A-Za-z\-']+,\s+[A-Z]\.")
    starts = [m.start() for m in apa_pattern.finditer(text)]
    if len(starts) >= 2:
        starts.append(len(text))
        return [c for i in range(len(starts) - 1)
                if len(c := text[starts[i]:starts[i+1]].strip()) > 20]

    # Last resort - one per non-empty line
    return [l.strip() for l in text.split("\n") if len(l.strip()) > 20]


def _normalize_pdf_reference_text(text):
    """
    Small PDF-only normalization layer.

    PDF extraction can introduce formatting artifacts that do not exist
    when the same reference is pasted as plain text: soft hyphens,
    hyphenated line breaks, non-breaking spaces, and inconsistent Unicode
    punctuation. Normalize those artifacts BEFORE the shared reference
    splitter/parser sees the text.

    IMPORTANT: newlines are preserved because _split_references() uses
    them for some citation styles.
    """
    if not text:
        return ""

    text = text.replace("\u00ad", "")          # soft hyphen
    text = text.replace("\u00a0", " ")         # non-breaking space

    # Join words split by a PDF line break:
    # "knowl-\nedge" -> "knowledge"
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)

    # Normalize line-ending variants without flattening reference boundaries.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse horizontal whitespace only.
    text = re.sub(r"[ \t]+", " ", text)

    return unicodedata.normalize("NFKD", text)


def extract_references_from_pdf(pdf_path):
    # PDF and pasted-list input still converge on the EXACT SAME
    # extract_references_from_text() + verification pipeline.
    pdf_text = extract_full_text_column_aware(pdf_path)
    pdf_text = _normalize_pdf_reference_text(pdf_text)
    return extract_references_from_text(pdf_text)


def single_reference(reference):
    return [reference.strip()]


# ================= NOTEBOOK CELL 16 =================

class ParsedReference(BaseModel):
    title: str
    authors: List[str]
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None

DOI_PATTERN = re.compile(
    r"""
    (?:
        https?://(?:dx\.)?doi\.org/
        |
        doi\s*:\s*
    )?
    (
        10\.\d{4,9}/
        [^\s<>"\]\[{}]+
    )
    """,
    re.IGNORECASE | re.VERBOSE
)


def _extract_doi(text):
    """
    Extract a DOI robustly from normal text and PDF-extracted text.

    Handles:
    - bare DOI: 10.xxxx/xxxxx
    - https://doi.org/10.xxxx/xxxxx
    - http://dx.doi.org/10.xxxx/xxxxx
    - doi: 10.xxxx/xxxxx
    - DOI split across a single PDF line break
    - DOI containing parentheses
    - trailing punctuation
    - accidental trailing page/reference text
    """

    if not text:
        return None

    # Normalize Unicode first.
    cleaned = unicodedata.normalize("NFKD", text)

    # ---------------------------------------------------------
    # 1. Find the initial DOI
    # ---------------------------------------------------------
    match = DOI_PATTERN.search(cleaned)

    if not match:
        return None

    doi = match.group(1)

    # ---------------------------------------------------------
    # 2. Repair a DOI split across a PDF line/column wrap.
    #
    # By this point all whitespace (including the original
    # newlines) has already been collapsed to single spaces by
    # the caller, so a genuine mid-DOI wrap and an unrelated
    # word gap both just look like "a space". The one reliable
    # signal that separates them: a DOI match that stops on a
    # bare trailing "." is almost always cut off mid-token
    # (e.g. "10.1016/j.jbi." + "2013.09.003") - a complete DOI
    # never legitimately ends with a lone period before a space.
    # Bridge across that ONE gap only when this signal holds -
    # never repeatedly - so page numbers, footers, and the next
    # paragraph of running text (which just happen to follow a
    # complete DOI) are never swallowed.
    # ---------------------------------------------------------
    if doi.endswith("."):
        rest = cleaned[match.end():]
        gap = re.match(r"[ \t]+", rest)
        if gap:
            continuation = re.match(r"[a-z0-9][^\s<>\"\]\[{}]*", rest[gap.end():])
            if continuation:
                doi += continuation.group(0)

    # ---------------------------------------------------------
    # 4. Remove punctuation that belongs to surrounding prose
    #
    # Keep legitimate closing parentheses when they are balanced.
    # ---------------------------------------------------------
    while doi:
        last = doi[-1]

        if last in ".,;":
            doi = doi[:-1]
            continue

        if last == ")":
            # Keep ')' if DOI contains a matching '('.
            if doi.count("(") >= doi.count(")"):
                break
            doi = doi[:-1]
            continue

        break

    # ---------------------------------------------------------
    # 5. Final normalization
    # ---------------------------------------------------------
    doi = normalize_doi(doi)

    return doi or None


def _extract_year(text, doi=None):
    """
    Restored under the name actually used at call sites (a previous
    edit renamed this and broke it - see earlier fix notes). Strips
    URLs before scanning (a URL like ".../E17-2052/" was getting its
    "2052" picked as the year), and clamps to a plausible range.
    """
    t = text.replace(doi, "") if doi else text
    t = re.sub(r"https?://\S+", " ", t)
    matches = re.findall(r"\b(?:19|20)\d{2}\b", t)
    valid = [int(y) for y in matches if 1900 <= int(y) <= 2029]
    return valid[-1] if valid else None


_INITIAL_FIRST_NAME = r"(?:[A-Z]\.\s*){1,3}[A-Z][A-Za-z\-']+(?:\s+[A-Z][A-Za-z\-']+)?"
_SURNAME_FIRST_NAME = r"[A-Z][A-Za-z\-']+,\s*(?:[A-Z]\.\s*)+"
# NEW: "Manktelow M" / "O'Kane M" - surname (possibly 2 words) then bare
# initials with NO period and NO comma before them (BMC/Vancouver style)
_VANCOUVER_NAME = re.compile(r"^[A-Z][A-Za-z\-']+(?:\s[A-Z][A-Za-z\-']+)?\s+[A-Z]{1,4}$")
_ET_AL = re.compile(r"^et\s*al\.?$", re.IGNORECASE)


def _extract_authors_initial_first(block):
    # FIX: some journals separate authors with ';' instead of ',' -
    # added as an alternation alongside the existing separators, so
    # this only changes behavior when a semicolon is actually present
    # (comma-separated blocks are completely unaffected).
    parts = re.split(r",\s*(?:and\s+)?|;\s*|\s+and\s+|\s*&\s*", block)
    return [p.strip().rstrip(',') for p in parts if p.strip() and len(p.strip()) > 1]


def _extract_authors_smart(block):
    """
    FIX: added a THIRD author-block style. Previously only handled
    "M. Z. Ali, S. Rauf" (initial-first) and "Pedregosa, F." (comma
    within name). Many journals - especially BMC/Springer medical
    informatics style - use "Manktelow M, Iftikhar A, O'Kane M": comma
    ONLY between authors, no comma or period within a single name. That
    style matched NONE of the old patterns and fell through to a
    last-resort fallback that produced empty authors and a garbled
    title (the actual bug behind most of the last batch's failures).
    """
    s = block.strip()
    if re.match(r"^[A-Z]\.", s):
        return _extract_authors_initial_first(block)

    if re.match(r"^[A-Z][a-zA-Z\-']+,\s*[A-Z]\.", s):
        matches = re.findall(_SURNAME_FIRST_NAME, block)
        if matches:
            return [m.strip().rstrip(',') for m in matches]

    parts = [p.strip() for p in block.split(",") if p.strip()]
    if parts and all(_VANCOUVER_NAME.match(p) or _ET_AL.match(p) for p in parts):
        return [p for p in parts if not _ET_AL.match(p)]

    return _extract_authors_initial_first(block)


def _extract_title_and_authors(text):
    quote_pattern = re.compile(r"[\u201c\u2018\"']{1,2}([^\u201d\u2019\"']{8,300})[\u201d\u2019\"']{1,2}")
    m = quote_pattern.search(text)
    if m:
        title = m.group(1).strip().rstrip(',').strip()
        authors_block = text[:m.start()].strip().rstrip(',').strip()
        return title, _extract_authors_smart(authors_block), "quoted"

    # FIX: the period after the year-parens is now OPTIONAL.
    # "Smith A (2022). Title." (APA) and "Smith A (2022) Title." (BMC/
    # Vancouver) are now both recognized by the same rule instead of
    # only the first.
    apa_pattern = re.compile(r"\(((?:19|20)\d{2})[a-z]?\)\.?\s+")
    m = apa_pattern.search(text)
    if m:
        authors_block = text[:m.start()].strip().rstrip(',').strip()
        rest = text[m.end():].strip()
        stop_pattern = re.compile(r",?\s*(?:pp\.|vol\.|no\.|doi:|https?://)|\.(?=\s+[A-Z]|\s*$)", re.IGNORECASE)
        stop_match = stop_pattern.search(rest)
        title = rest[:stop_match.start()].strip().rstrip(',.') if stop_match else rest.split(",")[0].strip()
        return title, _extract_authors_smart(authors_block), "apa_or_vancouver"

    # FIX: plain Vancouver/NLM style (very common in medical/scientific
    # journals - PubMed's own citation format), which has NO quotes
    # around the title and NO parentheses around the year:
    #   "Smith J, Doe A. Title of the article. J Abbrev. 2020;15(2):100-110."
    # Signal: a period-terminated segment immediately followed by a bare
    # 4-digit year that is itself immediately followed by ';' or ':'
    # (the volume/issue/page separator this style always uses) - narrow
    # enough that it won't misfire on an APA/quoted reference, since
    # those don't have a bare year glued directly to ';' or ':'.
    vancouver_plain_pattern = re.compile(
        r"\.\s+((?:19|20)\d{2})[a-z]?\s*(?=[;:])"
    )
    m = vancouver_plain_pattern.search(text)
    if m:
        head = text[:m.start()].strip()
        # `head` is "Authors. Title[. Journal]" - split on the FIRST
        # ". " to separate the author block from everything else, then
        # take the segment right after it as the title (stopping at the
        # next ". " if a journal name follows, matching how this style
        # is actually punctuated).
        first_period = re.search(r"\.\s+", head)
        if first_period:
            authors_block = head[:first_period.start()].strip()
            remainder = head[first_period.end():].strip()
            next_period = re.search(r"\.\s+", remainder)
            title = remainder[:next_period.start()].strip() if next_period else remainder.strip()
            title = title.rstrip(',.').strip()
            authors = _extract_authors_smart(authors_block)
            if title and authors:
                return title, authors, "vancouver_plain"

    author_match = re.match(
        r"^\s*(" + _INITIAL_FIRST_NAME + r"(?:,\s*(?:and\s+)?" + _INITIAL_FIRST_NAME + r")*)\s*,\s*",
        text
    )
    if author_match:
        authors_block = author_match.group(1)
        rest = text[author_match.end():]
        segments = [s.strip() for s in re.split(r"[.,]", rest) if s.strip()]
        title = segments[0] if segments else rest.strip()
        return title, _extract_authors_initial_first(authors_block), "fallback"

    # FIX: previously this always returned SOME title (however garbled),
    # which meant "no known citation style matched" was never actually
    # detected - parse_reference() saw "parsed is not None" and used
    # the garbage result instead of escalating to the LLM parser. Now
    # returns "raw" only as a labeled signal; parse_reference_deterministic
    # treats it as a non-match (see below) so the LLM fallback fires
    # for genuinely unrecognized formats.
    segments = [s.strip() for s in re.split(r"[.,]", text) if s.strip()]
    title = max(segments, key=len) if segments else text.strip()
    return title, [], "raw"


_PAGE_RANGE_PATTERN = re.compile(r"^[\d\s\u2013\-,]+$")

def _extract_venue(text, title):
    idx = text.find(title)
    if idx == -1:
        return None
    rest = re.sub(r"^[.,\"'\u201d\u2019\s]+", "", text[idx + len(title):])
    # FIX: added 'isbn' as a stop token so a book's venue string doesn't
    # accidentally swallow "ISBN 978-..." into what should just be the
    # publisher/place - purely cosmetic (avoids a spurious venue
    # mismatch later), doesn't change any comparison thresholds.
    stop = re.search(r"(vol\.|no\.|pp\.|doi:|https?://|isbn|\(\d{4})", rest, re.IGNORECASE)
    venue = rest[:stop.start()] if stop else rest.split(",")[0]
    venue = venue.strip(" ,.")
    # NEW: a page range like "1524-1535" is not a venue - if that's all
    # we captured (no real journal/venue name given in the reference),
    # treat it as genuinely missing rather than a false "venue conflict"
    if venue and _PAGE_RANGE_PATTERN.match(venue):
        return None
    return venue or None


def parse_reference_deterministic(raw_reference):
    raw_reference = unicodedata.normalize("NFKD", raw_reference)
    text = re.sub(r"\s+", " ", raw_reference.strip())
    doi = _extract_doi(text)
    year = _extract_year(text, doi)
    title, authors, style = _extract_title_and_authors(text)

    # FIX: "raw" means no known citation-style pattern matched at all -
    # that's the real "unrecognized format" signal, and should escalate
    # to the LLM parser rather than silently accepting a best-guess
    # title (previously this branch never returned None for any
    # non-empty input, so the LLM fallback almost never fired).
    if style == "raw" or not title or len(title) < 6:
        return None

    venue = _extract_venue(text, title)
    return ParsedReference(title=title, authors=authors, year=year, venue=venue, doi=doi)


def _best_effort_parsed_reference(reference):
    """Shared degrade path used whenever the LLM parser is unavailable -
    reused by BOTH the single-reference and the new batch parser
    fallback below, so this behavior is only defined in one place."""
    cleaned = re.sub(r"\s+", " ", unicodedata.normalize("NFKD", reference).strip())
    return ParsedReference(title=cleaned[:300], authors=[], year=_extract_year(cleaned), venue=None, doi=_extract_doi(cleaned))


async def parse_reference(reference):
    """
    Deterministic parser first (free, instant). Falls back to the LLM
    parser only when the title itself couldn't be found. If the LLM
    fallback is also unavailable (all providers exhausted), degrades
    to a best-effort raw title rather than losing the reference.

    NOTE: the main pipeline (run_batch_verification) no longer calls
    this per-reference - it uses the batched parser below instead, so
    that every reference needing an LLM call is grouped into a handful
    of requests instead of one call each. This function is kept as-is
    for single-reference use (e.g. run_verification_system) and for
    backward compatibility.
    """
    parsed = parse_reference_deterministic(reference)
    if parsed is not None:
        return parsed

    print("  \u2139 Deterministic parser could not find a title - falling back to LLM parser...")
    try:
        result = await run_with_fallback(parser_chain, make_parser_agent, make_parse_task,
                                          inputs={"reference": reference})
        return result.pydantic
    except Exception as e:
        print(f"  \u26a0 LLM parser fallback also unavailable ({e}) - using a best-effort raw title instead.")
        return _best_effort_parsed_reference(reference)


def make_parser_agent(llm_obj):
    return Agent(
        role="Reference Parsing Specialist",
        goal="Extract structured bibliographic information from any citation format.",
        backstory="Expert in APA, IEEE, MLA, Chicago, BibTeX citation styles. Never invents missing fields.",
        verbose=False, llm=llm_obj
    )

def make_parse_task(agent):
    return Task(
        description="Parse this academic reference. Extract title, authors, publication year, venue, DOI (if present).\n\nReference:\n{reference}",
        expected_output="Structured reference metadata.",
        output_pydantic=ParsedReference,
        agent=agent
    )


# ============================================================
# BATCHED LLM PARSER FALLBACK
#
# Only references that FAIL parse_reference_deterministic() ever reach
# here - anything parsed deterministically never touches an LLM at all.
# Mirrors ai_verify_batch()'s exact pattern (same batching approach,
# same run_with_fallback infra via parser_chain, same graceful
# per-reference degrade on total failure) so ~5-10 references needing
# an LLM parse become one request instead of one request each.
# ============================================================

class BatchParsedItem(BaseModel):
    reference_id: int
    title: str
    authors: List[str]
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None

class BatchParseResult(BaseModel):
    results: List[BatchParsedItem]


def make_batch_parser_agent(llm_obj):
    return Agent(
        role="Reference Parsing Specialist",
        goal="Extract structured bibliographic information from MULTIPLE academic references at once, one result per reference_id.",
        backstory="Expert in APA, IEEE, MLA, Chicago, Vancouver/NLM, BibTeX citation styles. Never invents missing fields.",
        verbose=False, llm=llm_obj
    )

def make_batch_parse_task(agent):
    return Task(
        description="""
Parse EACH of the following academic references independently. For every
reference_id below, extract: title, authors (as a list), publication
year, venue, and DOI (use null if not present in that reference's text -
never the literal string "null").

References:
{references}

Return exactly one result per reference_id, in the same order supplied.
Never invent a field that isn't actually present in that specific
reference's text.
""",
        expected_output="A list with exactly one parsed result per reference_id: title, authors, year, venue, doi.",
        agent=agent,
        output_pydantic=BatchParseResult,
    )


async def parse_references_batch(pending, batch_size=8):
    """
    `pending` is a list of {"reference_id": int, "reference": str} for
    references that already failed deterministic parsing. Returns
    {reference_id: ParsedReference}. On any failure (a chunk's LLM call
    fails entirely, or the model returns fewer results than requested),
    each affected reference degrades individually via
    _best_effort_parsed_reference() - never lost, never left unparsed.
    """
    all_parsed = {}

    for start in range(0, len(pending), batch_size):
        chunk = pending[start:start + batch_size]
        payload = [{"reference_id": c["reference_id"], "reference": c["reference"]} for c in chunk]

        print(f"  → LLM batch parsing {len(chunk)} reference(s)...", flush=True)

        try:
            result = await run_with_fallback(
                parser_chain,
                make_batch_parser_agent,
                make_batch_parse_task,
                inputs={"references": payload}
            )
            for item in result.pydantic.results:
                all_parsed[item.reference_id] = ParsedReference(
                    title=item.title, authors=item.authors,
                    year=item.year, venue=item.venue, doi=item.doi,
                )
        except Exception as e:
            print(f"  \u26a0 LLM batch parser unavailable for this chunk ({e}) - "
                  f"using a best-effort raw title for each reference instead.", flush=True)
            for c in chunk:
                all_parsed[c["reference_id"]] = _best_effort_parsed_reference(c["reference"])

        # Defensive: if the model returned fewer results than requested,
        # degrade only the missing ones the same way, instead of losing
        # them or leaving them unparsed.
        expected_ids = {c["reference_id"] for c in chunk}
        missing_ids = expected_ids - set(all_parsed.keys())
        for ref_id in missing_ids:
            c = next(x for x in chunk if x["reference_id"] == ref_id)
            all_parsed[ref_id] = _best_effort_parsed_reference(c["reference"])

    return all_parsed



# ================= NOTEBOOK CELL 18 =================

def normalize_text(text):
    if not text:
        return ""
    text = text.lower()
    # Repair PDF line-break hyphenation:
    # "detec- tion" -> "detection"
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
    text = re.sub(r"<[^>]+>", "", text)  # strip stray HTML/XML tags (Crossref sometimes has <scp>...</scp>)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_similarity(a, b):
    """
    FIX: PDF text extraction (especially two-column IEEE-style layouts
    with line-wrap hyphenation) frequently drops a stray space into the
    middle of a word - e.g. "comprehensive" extracted as "compre
    hensive", or "identification" as "identifi cation". Token-based
    fuzzy matching scores these noticeably lower than they should,
    since "compre" and "hensive" become separate tokens instead of one.
    Fix: also compute similarity with ALL whitespace removed from both
    strings and take whichever score is higher - this neutralizes
    stray mid-word spaces without needing a dictionary.
    """
    if not a or not b:
        return 0
    norm_a, norm_b = normalize_text(a), normalize_text(b)
    token_score = fuzz.token_sort_ratio(norm_a, norm_b)
    despaced_score = fuzz.ratio(norm_a.replace(" ", ""), norm_b.replace(" ", ""))
    return max(token_score, despaced_score)


_GENERIC_SHORT_TITLES = {
    "machine learning", "deep learning", "artificial intelligence",
    "data mining", "computer vision", "natural language processing",
}

def is_abbreviated_title_match(a, b):
    """Handles cases like submitted 'RESDSQL' vs database 'RESDSQL:
    Decoupling Schema Linking...' - the shorter title is a clean prefix
    of the longer one. Conservative: requires an 8+ char prefix and
    excludes generic short titles (submitting just "Machine Learning"
    should NOT auto-match "Machine Learning: A Review of..." - that's
    a real title conflict, not an abbreviation)."""
    if not a or not b:
        return False
    s1, s2 = normalize_text(a), normalize_text(b)
    shorter, longer = (s1, s2) if len(s1) <= len(s2) else (s2, s1)
    if len(shorter) < 8 or not longer.startswith(shorter):
        return False
    return shorter not in _GENERIC_SHORT_TITLES


def normalize_author_name(name):
    if not name:
        return ""
    name = str(name).lower().strip()
    name = re.sub(r"\bet\s+al\.?\b", "", name)
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = re.sub(r"[.,;:]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def author_name_similarity(name1, name2):
    name1, name2 = normalize_author_name(name1), normalize_author_name(name2)
    if not name1 or not name2:
        return 0
    if name1 == name2:
        return 100
    token_score = fuzz.token_set_ratio(name1, name2)
    tokens1, tokens2 = name1.split(), name2.split()
    initials1 = "".join(t[0] for t in tokens1 if t)
    initials2 = "".join(t[0] for t in tokens2 if t)
    if initials1 and initials2:
        shorter, longer = sorted([initials1, initials2], key=len)
        initials_score = 90 if longer.startswith(shorter[0]) and shorter in longer else 0
    else:
        initials_score = 0
    last1 = tokens1[-1] if tokens1 else ""
    last2 = tokens2[-1] if tokens2 else ""
    last_name_score = fuzz.ratio(last1, last2)
    if last_name_score >= 90 and token_score >= 60:
        return max(token_score, 90)
    return max(token_score, initials_score, last_name_score * 0.6)


def author_similarity(original_authors, found_authors):
    if not original_authors or not found_authors:
        return 0
    et_al_present = any(re.search(r"\bet\s+al\.?\b", str(a), re.IGNORECASE) for a in original_authors)
    original = [normalize_author_name(a) for a in original_authors if a and normalize_author_name(a)]
    found = [normalize_author_name(a) for a in found_authors if a and normalize_author_name(a)]
    if not original or not found:
        return 0

    remaining, matched_scores = set(range(len(found))), []
    for oa in original:
        best_score, best_i = 0, None
        for i in remaining:
            s = author_name_similarity(oa, found[i])
            if s > best_score:
                best_score, best_i = s, i
        if best_i is not None:
            matched_scores.append(best_score)
            remaining.remove(best_i)
    if not matched_scores:
        return 0

    average = sum(matched_scores) / len(matched_scores)
    oc, fc = len(original), len(found)
    if et_al_present or fc >= oc:
        bonus = 5 if fc == oc else 0
    else:
        bonus = -10 * (oc - fc)
    score = average + bonus
    unmatched = oc - len(matched_scores)
    if unmatched > 0 and not et_al_present:
        score -= unmatched * 15
    return round(max(0, min(100, score)), 2)


# ================= NOTEBOOK CELL 20 =================

def compare_doi(sub_doi, db_doi):
    sub, db = normalize_doi(sub_doi), normalize_doi(db_doi)
    if not sub and not db:
        return "UNKNOWN"
    if not sub:
        return "MISSING_SUBMITTED"
    if not db:
        return "MISSING_DATABASE"
    if sub == db:
        return "MATCH"
    # Handles PDF line-wrap artifacts that drop/shift a hyphen inside
    # the DOI (e.g. "s41746024-01286-3" vs "s41746-024-01286-3") -
    # DOI hyphen positions don't distinguish two different real works,
    # so a hyphen-insensitive match is still a safe, real match.
    if sub.replace("-", "") == db.replace("-", ""):
        return "MATCH"
    return "MISMATCH"

_GENERIC_SHORT_TITLES = {
    "machine learning", "deep learning", "artificial intelligence",
    "data mining", "computer vision", "natural language processing",
}

def is_abbreviated_title_match(a, b):
    """Handles 'RESDSQL' vs 'RESDSQL: Decoupling...' - but made
    conservative: requires an 8+ char prefix AND excludes generic short
    titles (e.g. submitted 'Machine Learning' should NOT auto-match
    'Machine Learning: A Review of...' - that's a real title conflict,
    not an abbreviation)."""
    if not a or not b:
        return False
    s1, s2 = normalize_text(a), normalize_text(b)
    shorter, longer = (s1, s2) if len(s1) <= len(s2) else (s2, s1)
    if len(shorter) < 8 or not longer.startswith(shorter):
        return False
    if shorter in _GENERIC_SHORT_TITLES:
        return False
    return True


def compare_title(sub_title, db_title):
    if not sub_title or not db_title:
        return "UNKNOWN"
    score = title_similarity(sub_title, db_title)
    if score >= 90 or is_abbreviated_title_match(sub_title, db_title):
        return "MATCH"
    if score < 50:
        return "MISMATCH"
    return "PARTIAL"


def compare_authors(sub_authors, db_authors):
    if not sub_authors and not db_authors:
        return "UNKNOWN"
    if not sub_authors:
        return "MISSING_SUBMITTED"
    if not db_authors:
        return "MISSING_DATABASE"
    score = author_similarity(sub_authors, db_authors)
    if score >= 75:
        return "MATCH"
    if score < 40:
        return "MISMATCH"
    return "PARTIAL"


def compare_year(sub_year, db_year):
    if not sub_year and not db_year:
        return "UNKNOWN"
    if not sub_year:
        return "MISSING_SUBMITTED"
    if not db_year:
        return "MISSING_DATABASE"
    diff = abs(sub_year - db_year)
    if diff == 0:
        return "MATCH"
    if diff <= 1:
        return "PARTIAL"
    return "MISMATCH"


def compare_venue(sub_venue, db_venue):
    if not sub_venue and not db_venue:
        return "UNKNOWN"
    if not sub_venue:
        return "MISSING_SUBMITTED"
    if not db_venue:
        return "MISSING_DATABASE"
    score = fuzz.token_set_ratio(normalize_text(sub_venue), normalize_text(db_venue))
    if score >= 70:
        return "MATCH"
    if score < 40:
        return "MISMATCH"
    return "PARTIAL"


def build_field_states(parsed_reference, candidate):
    return {
        "doi": compare_doi(parsed_reference.doi, candidate.get("doi")),
        "title": compare_title(parsed_reference.title, candidate.get("title")),
        "authors": compare_authors(parsed_reference.authors, candidate.get("authors")),
        "year": compare_year(parsed_reference.year, candidate.get("year")),
        "venue": compare_venue(parsed_reference.venue, candidate.get("venue")),
    }


# ================= NOTEBOOK CELL 22 =================

def score_candidate(parsed_reference, candidate):
    title_score = title_similarity(parsed_reference.title, candidate.get("title"))
    author_score = author_similarity(parsed_reference.authors, candidate.get("authors", []))

    year_state = compare_year(parsed_reference.year, candidate.get("year"))
    year_score = {"MATCH": 100, "PARTIAL": 75, "MISMATCH": 0,
                  "MISSING_SUBMITTED": 50, "MISSING_DATABASE": 50, "UNKNOWN": 50}[year_state]

    weighted = 0.50 * title_score + 0.35 * author_score + 0.15 * year_score

    if is_abbreviated_title_match(parsed_reference.title, candidate.get("title")):
        weighted = max(weighted, 90)

    sub_doi, cand_doi = normalize_doi(parsed_reference.doi), normalize_doi(candidate.get("doi"))
    if sub_doi and cand_doi and sub_doi == cand_doi:
        weighted = 100

    return {"candidate": candidate, "title_score": round(title_score, 1),
            "author_score": round(author_score, 1), "year_score": round(year_score, 1),
            "weighted_score": round(weighted, 1)}


def rank_candidates(parsed_reference, candidates):
    if not candidates:
        return []
    scored = [score_candidate(parsed_reference, c) for c in candidates]
    scored.sort(key=lambda r: r["weighted_score"], reverse=True)
    return scored


def sources_agreeing(candidates, best_candidate):
    """After deduplication, each candidate already carries its own
    merged 'sources' list - just read it, no need to re-scan."""
    if not best_candidate:
        return []
    if best_candidate.get("sources"):
        return sorted(set(best_candidate["sources"]))
    source = best_candidate.get("source")
    return [source] if source else []


# ================= NOTEBOOK CELL 24 =================

def classify_reference(states):
    """
    Returns (status, reason, needs_ai_review).

    IMPORTANT: this rule table remains authoritative. Candidate rescue
    may change WHICH candidate is evaluated, but it does not weaken or
    replace these deterministic rules.
    """
    doi, title, authors, year = states["doi"], states["title"], states["authors"], states["year"]

    # ---- DOI present and confirmed in the submitted reference ----
    if doi == "MATCH":
        if title == "MISMATCH":
            return "NOT VERIFIED", "The DOI matches a database record, but the submitted title conflicts with the title associated with that DOI.", False
        if authors == "MISMATCH":
            return "NOT VERIFIED", "The DOI and title match, but the submitted authors conflict strongly with the authors on record for that DOI - the citation may be misattributed.", False
        if year == "MISMATCH" or (states.get("venue") == "MISMATCH"):
            return "UNCERTAIN", "The DOI, title, and authors match, but the publication year or venue conflicts with the database record.", False
        return "VERIFIED", "Exact DOI match with a compatible title and authors.", False

    if doi == "MISMATCH":
        return "NOT VERIFIED", "The submitted DOI does not match the DOI on record for this title.", False

    # ---- Submitted a DOI, but the matched record doesn't provide one to confirm against ----
    if doi == "MISSING_DATABASE":
        if title == "MATCH" and authors == "MATCH":
            return "UNCERTAIN", "Title and authors match, but the submitted DOI could not be confirmed because the matched database record does not provide a DOI.", False
        if title == "MISMATCH" or authors == "MISMATCH":
            return "NOT VERIFIED", "The submitted DOI could not be confirmed, and the title or authors also conflict with the database record.", False
        return "UNCERTAIN", "The submitted DOI could not be confirmed and the remaining metadata is insufficient for a safe automatic decision.", True

    # ---- No usable DOI on either side ----
    if title == "MATCH" and authors == "MATCH":
        if year == "MISMATCH":
            return "UNCERTAIN", "Title and authors strongly match, but the publication year differs - this can happen with preprint vs. journal versions of the same work.", False
        return "VERIFIED", "Title and authors strongly match a database record.", False

    if authors == "MISMATCH":
        return "NOT VERIFIED", "The submitted authors conflict with the authors on record for the closest-matching title.", False

    if title == "MISMATCH":
        return "NOT VERIFIED", "No candidate's title sufficiently matches the submitted reference.", False

    return "UNCERTAIN", "The available evidence is mixed or incomplete for a safe automatic decision.", True


def missing_field_notes(states):
    notes = []
    labels = {"doi": "DOI", "venue": "venue", "year": "year", "authors": "authors"}
    for field, label in labels.items():
        state = states.get(field)
        if state == "MISSING_SUBMITTED":
            notes.append(f"{label} was not provided in the submitted reference")
        elif state == "MISSING_DATABASE":
            notes.append(f"{label} was not provided by the matched database record")
    return notes


def classify_by_ranking(parsed_reference, candidates, max_candidates_checked=None):
    """
    Surgical candidate-rescue change.

    Ranking is still used exactly as before. The existing
    classify_reference() rule table remains authoritative.

    FIX: deterministic evaluation previously stopped after the top 5
    ranked candidates (max_candidates_checked defaulted to 5). With a
    correct candidate potentially buried far down a large deduplicated
    pool (hundreds of results across Crossref/OpenAlex/etc.), that cap
    could hide it entirely before it was ever checked. max_candidates_checked
    now defaults to None, meaning "evaluate every ranked candidate" -
    no candidate is dropped before deterministic verification. Reducing
    the list to a manageable number only happens later, when building
    the payload sent to AI review (ai_verify_batch already does this by
    taking only ranked[:5] there - that slicing is untouched by this
    change).

    Priority:
      1. any candidate that is deterministically VERIFIED
      2. a specific deterministic UNCERTAIN
      3. a candidate requiring AI review
      4. NOT VERIFIED

    This means a bad ranked[0] can no longer hide a correct lower-ranked
    candidate that satisfies the existing verification rules.
    """
    ranked = rank_candidates(parsed_reference, candidates)
    if not ranked:
        # FIX: zero candidates found means no database coverage (not
        # indexed, temporarily unreachable, or genuinely obscure) - that
        # is an ABSENCE of evidence, not CONTRADICTORY evidence. Reusing
        # "NOT VERIFIED" here conflated the two, incorrectly flagging
        # real-but-unindexed references as if they'd failed a check.
        return {"status": "UNCERTAIN", "reason": "No candidate was found in any academic database - this may mean the reference is not indexed, rather than that it is incorrect.",
                "ranked": [], "best_match": None, "states": {}, "notes": [], "needs_ai_review": False}

    # FIX: reject candidates whose TITLE is clearly unrelated before
    # they ever reach evaluation/AI review, so a high author-similarity
    # score can never "rescue" an unrelated paper into the top-N pool.
    # Reuses the title_score already computed by rank_candidates() -
    # no new scoring system. An exact-DOI match is exempt (same paper,
    # different title wording is not a red flag - e.g. subtitle,
    # translated title, or minor journal formatting differences).
    submitted_doi = normalize_doi(parsed_reference.doi)
    plausible_ranked = [
        r for r in ranked
        if r["title_score"] >= 50
        or (submitted_doi and normalize_doi(r["candidate"].get("doi")) == submitted_doi)
    ]
    ranked = plausible_ranked or ranked  # never end up with zero candidates to evaluate

    candidates_to_check = ranked if max_candidates_checked is None else ranked[:max_candidates_checked]

    evaluated = []
    for index, candidate_result in enumerate(candidates_to_check, start=1):
        states = build_field_states(parsed_reference, candidate_result["candidate"])
        status, reason, needs_ai_review = classify_reference(states)
        evaluated.append({
            "candidate_index": index,
            "candidate_result": candidate_result,
            "states": states,
            "status": status,
            "reason": reason,
            "needs_ai_review": needs_ai_review,
        })

        # A clean deterministic VERIFIED candidate is always the safest rescue.
        if status == "VERIFIED":
            break

    verified = next((e for e in evaluated if e["status"] == "VERIFIED"), None)
    if verified:
        chosen = verified
    else:
        defensible_uncertain = next(
            (e for e in evaluated
             if e["status"] == "UNCERTAIN" and not e["needs_ai_review"]),
            None
        )
        mixed = next(
            (e for e in evaluated if e["status"] == "UNCERTAIN" and e["needs_ai_review"]),
            None
        )
        not_verified = next((e for e in evaluated if e["status"] == "NOT VERIFIED"), None)

        # Prefer a plausible lower-ranked candidate over the first bad
        # NOT VERIFIED candidate. Mixed evidence is sent to AI later.
        chosen = defensible_uncertain or mixed or not_verified or evaluated[0]

    best = chosen["candidate_result"]
    status = chosen["status"]
    reason = chosen["reason"]
    needs_ai_review = chosen["needs_ai_review"]
    states = chosen["states"]
    notes = missing_field_notes(states)

    second = ranked[1] if len(ranked) > 1 else None
    margin = ranked[0]["weighted_score"] - (second["weighted_score"] if second else 0)

    if status == "VERIFIED" and second and margin < 8 and best is ranked[0]:
        status, needs_ai_review = "UNCERTAIN", True
        reason = (f"The top match looks strong, but a second candidate is nearly as close a match "
                  f"(margin {margin:.0f} points) - too ambiguous to auto-verify. Original assessment: {reason}")

    return {
        "status": status,
        "reason": reason,
        "ranked": ranked,
        "best_match": best,
        "states": states,
        "notes": notes,
        "margin": round(margin, 1),
        "needs_ai_review": needs_ai_review,
        "evaluated": evaluated,
    }



# ================= NOTEBOOK CELL 26 =================

class BatchVerdictItem(BaseModel):
    reference_id: int
    status: str  # "VERIFIED", "UNCERTAIN", or "NOT VERIFIED" only
    confidence: Optional[float] = None
    metadata_conflicts: Optional[List[str]] = None
    explanation: str
    selected_candidate_index: Optional[int] = None  # 1-based index in supplied top-5 candidates

class BatchVerificationResult(BaseModel):
    results: List[BatchVerdictItem]


def make_batch_verification_agent(llm_obj):
    return Agent(
        role="Academic Reference Verification Specialist",
        goal=("Compare each ambiguous reference against ALL supplied candidates and identify the "
              "same publication when one exists. Never assume candidate #1 is correct. Missing "
              "metadata is never a contradiction, while a DOI identifying another publication "
              "is a strong contradiction."),
        backstory="Expert in bibliographic verification, weighing DOI, title, authors, year, and venue together - never title alone.",
        llm=llm_obj, verbose=False
    )


def make_batch_verification_task(agent):
    return Task(
        description="""
Verify EACH reference independently against EVERY supplied candidate.

A candidate being ranked first does NOT mean it is correct. Compare the
original citation with candidate #1, #2, #3, etc. independently. If a
lower-ranked candidate is a much better bibliographic identity match,
select that candidate.

Cases:
{uncertain_cases}

Use exactly THREE statuses:
- VERIFIED: one supplied candidate clearly represents the same publication and
  there is no unresolved deterministic contradiction.
- NOT VERIFIED: none of the supplied candidates represents the same publication.
- UNCERTAIN: one candidate is the best plausible match, but evidence remains
  genuinely insufficient or conflicting.

For VERIFIED, selected_candidate_index MUST be the 1-based candidate number.
For UNCERTAIN, selected_candidate_index should be the 1-based best plausible
candidate number when one exists; otherwise null.
For NOT VERIFIED, selected_candidate_index MUST be null.

Rules:
1. Missing metadata is NOT a contradiction.
2. Abbreviated author names are not automatically a mismatch.
3. Formatting/capitalization differences are not contradictions.
4. Venue wording differences alone are not contradictions.
5. A preprint/journal year difference may justify UNCERTAIN when title and
   authors strongly match.
6. A completely different title and author set is strong evidence of
   NOT VERIFIED.
7. A DOI identifying another publication is a strong contradiction.
8. Do not verify merely because one word or topic overlaps.
9. Prefer identity evidence: title + authors + year/venue/DOI.
10. Examine EVERY supplied candidate before returning NOT VERIFIED.
11. Never assume candidate #1 is correct.

IMPORTANT SAFETY RULE:
Do not return VERIFIED for a selected candidate if its bibliographic evidence
contains a strong deterministic contradiction such as:
- exact submitted DOI conflicting with candidate DOI
- exact DOI pointing to another publication
- clearly incompatible title with an exact submitted DOI
- clearly incompatible authors with an exact submitted DOI

Return one verdict for every reference_id with:
status, confidence, metadata_conflicts, explanation, selected_candidate_index.
""",
        expected_output="A list with exactly one verdict per reference_id: status, confidence, metadata_conflicts, explanation, selected_candidate_index.",
        agent=agent,
        output_pydantic=BatchVerificationResult,
    )


async def ai_verify_batch(pending_cases, batch_size=6):
    """Returns {reference_id: BatchVerdictItem}.

    Only unresolved cases reach this function. Each case supplies at most
    five candidates, and the AI must identify the candidate it actually
    used. Existing batch/fallback behavior is preserved.
    """
    all_verdicts = {}

    for start in range(0, len(pending_cases), batch_size):
        chunk = pending_cases[start:start + batch_size]
        payload = [{
            "reference_id": case["reference_id"],
            "original_reference": case["original_reference"],
            "parsed_metadata": case["parsed_reference"],
            "candidates": [{
                "candidate_index": idx,
                "title": r["candidate"].get("title"),
                "authors": r["candidate"].get("authors"),
                "year": r["candidate"].get("year"),
                "doi": r["candidate"].get("doi"),
                "venue": r["candidate"].get("venue"),
                "source": r["candidate"].get("source"),
                "title_similarity": r.get("title_score"),
                "author_similarity": r.get("author_score"),
                "year_score": r.get("year_score"),
                "weighted_score": r["weighted_score"],
                "source_agreement": r["candidate"].get("sources", []),
            } for idx, r in enumerate(case["ranked"][:5], start=1)],
        } for case in chunk]

        print(f"  → AI batch verifying {len(chunk)} reference(s)...")

        try:
            result = await run_with_fallback(
                verifier_chain,
                make_batch_verification_agent,
                make_batch_verification_task,
                inputs={"uncertain_cases": payload}
            )
            for item in result.pydantic.results:
                all_verdicts[item.reference_id] = item
        except Exception as e:
            print(f"  ⚠ AI batch verification unavailable for this chunk ({e}) - "
                  f"falling back to the deterministic estimate for each reference.")
            for case in chunk:
                all_verdicts[case["reference_id"]] = BatchVerdictItem(
                    reference_id=case["reference_id"],
                    status=case.get("deterministic_status", "UNCERTAIN"),
                    confidence=None,
                    metadata_conflicts=[],
                    explanation=(f"AI verification was unavailable (all providers failed), so this reflects the "
                                 f"deterministic estimate instead: {case.get('deterministic_reason', 'insufficient evidence for a confident automatic decision.')}"),
                    selected_candidate_index=None,
                )

        expected_ids = {case["reference_id"] for case in chunk}
        missing_ids = expected_ids - set(all_verdicts.keys())
        for ref_id in missing_ids:
            case = next(c for c in chunk if c["reference_id"] == ref_id)
            all_verdicts[ref_id] = BatchVerdictItem(
                reference_id=ref_id,
                status=case.get("deterministic_status", "UNCERTAIN"),
                confidence=None,
                metadata_conflicts=["ai_missing_verdict"],
                explanation="The AI verifier did not return a verdict for this reference; using the deterministic estimate instead.",
                selected_candidate_index=None,
            )

    return all_verdicts



# ================= NOTEBOOK CELL 27 =================

# ============================================================
# MAIN VERIFICATION ORCHESTRATORS
# ============================================================

async def run_batch_verification(references):
    """
    Main pipeline for verifying multiple references.

    Phase 1:
        - Parse each reference
        - Search academic databases
        - Rank candidates
        - Apply deterministic verification rules
        - Rescue lower-ranked candidates before AI escalation

    Phase 2:
        - Collect ONLY unresolved cases
        - Send them to the AI verifier in batches
        - If AI selects a candidate, replace best_match with that candidate
        - Re-run the existing deterministic rules as a safety gate

    This keeps AI calls limited and lets deterministic rules handle
    straightforward cases.
    """

    if not references:
        return []

    # ========================================================
    # PHASE 0 - PARSING (deterministic first, batched LLM fallback)
    #
    # References that parse deterministically NEVER touch an LLM.
    # References that fail deterministic parsing are grouped into
    # ~5-10-per-request batches instead of one LLM call per reference
    # (mirrors ai_verify_batch's existing batching pattern below).
    # ========================================================

    print("\n" + "=" * 70)
    print("PHASE 0 - PARSING")
    print("=" * 70)

    parsed_map = {}
    llm_pending = []

    for ref_id, reference in enumerate(references, 1):
        try:
            parsed = parse_reference_deterministic(reference)
        except Exception:
            parsed = None
        if parsed is not None:
            parsed_map[ref_id] = parsed
        else:
            llm_pending.append({"reference_id": ref_id, "reference": reference})

    print(f"  \u2713 {len(parsed_map)}/{len(references)} parsed deterministically (no LLM call)", flush=True)

    if llm_pending:
        print(f"  \u2192 {len(llm_pending)} reference(s) need LLM parsing - batching...", flush=True)
        llm_parsed = await parse_references_batch(llm_pending, batch_size=8)
        parsed_map.update(llm_parsed)
    else:
        print("  No references require LLM parsing.", flush=True)

    print("\n" + "=" * 70)
    print("PHASE 1 - DATABASE SEARCH + DETERMINISTIC VERIFICATION")
    print("=" * 70)

    phase1_results = []

    # Process sequentially.
    # This is intentional because the database functions use a shared
    # requests session and per-host throttling.
    for ref_id, reference in enumerate(references, 1):
        print(f"\n[{ref_id}/{len(references)}] Processing reference...")

        try:
            parsed_reference = parsed_map.get(ref_id) or _best_effort_parsed_reference(reference)
            result = await process_reference_phase1(reference, ref_id, parsed_reference)
            phase1_results.append(result)

            status = result.get("status", "UNKNOWN")

            if status == "PENDING_AI":
                print("  → Pending AI review")
            else:
                print(f"  → {status}")

        except Exception as e:
            print(f"  ✗ Error processing reference: {e}")

            # A system/network/quota failure means WE DON'T KNOW,
            # not "this reference is fake".
            phase1_results.append({
                "reference_id": ref_id,
                "original_reference": reference,
                "status": "UNCERTAIN",
                "source": "system error - needs manual review",
                "explanation": (
                    f"This reference could not be automatically processed due to a "
                    f"technical error (not a content judgment): {e}"
                ),
                "error": str(e),
                "parsed_reference": {},
                "ranked": [],
                "best_match": None,
                "sources_agreeing": [],
            })

    # ========================================================
    # PHASE 2 - AI REVIEW ONLY FOR UNRESOLVED CASES
    # ========================================================

    pending_cases = [
        r for r in phase1_results
        if r.get("status") == "PENDING_AI"
    ]

    print("\n" + "=" * 70)
    print("PHASE 2 - AI REVIEW")
    print("=" * 70)

    if pending_cases:
        print(f"AI review required for {len(pending_cases)} reference(s).")

        ai_verdicts = await ai_verify_batch(
            pending_cases,
            batch_size=6
        )

        # Merge AI verdicts back into phase-1 results.
        for result in phase1_results:
            ref_id = result.get("reference_id")

            if ref_id not in ai_verdicts:
                continue

            verdict = ai_verdicts[ref_id]
            selected_index = verdict.selected_candidate_index
            selected_ranked = None

            # The AI index is 1-based and refers to the supplied top-5 list.
            if selected_index is not None:
                try:
                    idx = int(selected_index)
                    if 1 <= idx <= min(5, len(result.get("ranked", []))):
                        selected_ranked = result["ranked"][idx - 1]
                except (TypeError, ValueError):
                    selected_ranked = None

            # If AI chose a candidate, make it the actual best_match.
            # Do NOT allow an invalid index to silently point at ranked[0].
            if selected_ranked is not None:
                selected_candidate = selected_ranked["candidate"]
                selected_states = build_field_states(
                    _parsed_reference_from_result(result),
                    selected_candidate
                )
                deterministic_status, deterministic_reason, _ = classify_reference(
                    selected_states
                )

                # Strong deterministic contradictions remain authoritative.
                strong_contradiction = (
                    selected_states.get("doi") == "MISMATCH"
                    or (
                        selected_states.get("doi") == "MATCH"
                        and selected_states.get("title") == "MISMATCH"
                    )
                    or (
                        selected_states.get("doi") == "MATCH"
                        and selected_states.get("authors") == "MISMATCH"
                    )
                )

                if verdict.status == "VERIFIED" and strong_contradiction:
                    result["status"] = "NOT VERIFIED"
                    result["source"] = "deterministic safety gate"
                    result["explanation"] = (
                        "AI selected a candidate, but the existing deterministic "
                        "rules found a strong bibliographic contradiction: "
                        + deterministic_reason
                    )
                    result["metadata_conflicts"] = verdict.metadata_conflicts or []
                    result["ai_selected_candidate_index"] = selected_index
                    continue

                result["best_match"] = selected_ranked
                result["states"] = selected_states
                result["sources_agreeing"] = sources_agreeing(
                    result.get("ranked", []),
                    selected_candidate
                )
                result["selected_candidate_index"] = selected_index

                # If AI says VERIFIED, the selected candidate must at least
                # pass the existing deterministic safety rules. A deterministic
                # NOT VERIFIED is never overwritten.
                if verdict.status == "VERIFIED" and deterministic_status == "NOT VERIFIED":
                    result["status"] = "NOT VERIFIED"
                    result["source"] = "deterministic safety gate"
                    result["explanation"] = (
                        "AI selected candidate #" + str(selected_index) +
                        ", but the existing deterministic rules rejected it: " +
                        deterministic_reason
                    )
                    result["metadata_conflicts"] = verdict.metadata_conflicts or []
                    result["ai_selected_candidate_index"] = selected_index
                    continue

            result["status"] = verdict.status
            result["explanation"] = verdict.explanation
            result["source"] = "AI verification"

            if verdict.confidence is not None:
                result["confidence"] = verdict.confidence

            result["metadata_conflicts"] = verdict.metadata_conflicts or []
            result["ai_selected_candidate_index"] = selected_index

    else:
        print("No references require AI review.")

    # ========================================================
    # FINAL CLEANUP
    # ========================================================

    for result in phase1_results:
        if result.get("status") == "PENDING_AI":
            result["status"] = "UNCERTAIN"
            result["source"] = "deterministic fallback"
            result["explanation"] = (
                "The reference remained ambiguous and could not "
                "receive a final AI verdict."
            )

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)

    return phase1_results


def _parsed_reference_from_result(result):
    """
    Reconstruct the existing parsed-reference model from the stored
    metadata only for the AI candidate safety check. This avoids creating
    a second verification pipeline.
    """
    data = result.get("parsed_reference", {})

    # parse_reference() already returns the project's Pydantic model.
    # Reuse its declared model rather than duplicating field logic.
    return ParsedReference(**data)


async def run_verification_system(reference):
    """
    Single-reference wrapper.

    Reuses the exact same verification pipeline as batch verification
    so the Single Reference tab does not have a separate verification
    logic.
    """

    if not reference or not reference.strip():
        return {
            "status": "NOT VERIFIED",
            "source": "input validation",
            "explanation": "No reference was provided.",
            "original_reference": reference or "",
        }

    results = await run_batch_verification([reference.strip()])

    if not results:
        return {
            "status": "NOT VERIFIED",
            "source": "system",
            "explanation": "No verification result was produced.",
            "original_reference": reference,
        }

    return results[0]



# ================= NOTEBOOK CELL 29 =================

async def process_reference_phase1(reference, ref_id, parsed_reference):
    """
    FIX: no longer calls parse_reference() itself - the caller
    (run_batch_verification) now parses ALL references up front in
    Phase 0 (deterministic first, batched LLM fallback for the rest)
    and passes the already-parsed result in here. Everything below
    this point - search, ranking, classification - is unchanged.
    """
    candidates = search_all_databases(parsed_reference)

    decision = classify_by_ranking(parsed_reference, candidates)

    result = {
        "reference_id": ref_id,
        "original_reference": reference,
        "parsed_reference": parsed_reference.model_dump(),
        "ranked": decision["ranked"],
        "best_match": decision["best_match"],
        "sources_agreeing": sources_agreeing(
            candidates,
            decision["best_match"]["candidate"]
        ) if decision["best_match"] else [],
        "deterministic_status": decision["status"],
        "deterministic_reason": decision["reason"],
        "notes": decision.get("notes", []),
    }

    # Candidate rescue has already checked the top 5. Only unresolved cases
    # now reach AI. Keep strong deterministic contradictions final.
    states = decision.get("states", {})
    strong_contradiction = (
        states.get("doi") == "MISMATCH"
        or (
            states.get("doi") == "MATCH"
            and states.get("title") == "MISMATCH"
        )
        or (
            states.get("doi") == "MATCH"
            and states.get("authors") == "MISMATCH"
        )
    )

    ai_eligible = (
        bool(decision.get("ranked"))
        and not strong_contradiction
        and (
            decision.get("needs_ai_review", False)
            or decision.get("status") == "NOT VERIFIED"
        )
    )

    if ai_eligible:
        result["status"] = "PENDING_AI"
        return result

    result["status"] = decision["status"]
    result["explanation"] = decision["reason"] + (
        (" (" + "; ".join(decision["notes"]) + ".)")
        if decision.get("notes") else ""
    )
    result["source"] = "deterministic rules"
    return result



# ================= NOTEBOOK CELL 31 =================

import pandas as pd

def export_results_to_excel(results, filename="reference_verification_results.xlsx"):
    """
    Saves all results to a single editable spreadsheet - one row per
    reference, with the reference number for easy cross-checking
    against the original paper's numbered citation list.
    """
    rows = []
    for r in results:
        parsed = r.get("parsed_reference", {}) or {}
        best = r.get("best_match") or {}
        candidate = best.get("candidate", {}) if best else {}
        status = r.get("status", "UNKNOWN")

        rows.append({
            "Ref #": r.get("reference_id"),
            "Status": status,
            # FIX: purely additive column - does not change `status`,
            # `Explanation`, or any classification value above/below it.
            # UNCERTAIN/NOT VERIFIED rules and their reasons are exactly
            # what classify_reference() already produced; this column
            # just makes them easy to filter/sort in the spreadsheet.
            "Manual Check Needed": "Yes" if status in ("UNCERTAIN", "NOT VERIFIED") else "No",
            "Explanation": r.get("explanation", ""),
            "Original Reference": r.get("original_reference", ""),
            "Submitted Title": parsed.get("title", ""),
            "Submitted Authors": ", ".join(parsed.get("authors", []) or []),
            "Submitted Year": parsed.get("year", ""),
            "Submitted DOI": parsed.get("doi", ""),
            "Matched Title": candidate.get("title", ""),
            "Matched Authors": ", ".join(candidate.get("authors", []) or []),
            "Matched Year": candidate.get("year", ""),
            "Matched DOI": candidate.get("doi", ""),
            "Sources Agreeing": ", ".join(r.get("sources_agreeing", []) or []),
            "Verification Source": r.get("source", ""),
        })

    df = pd.DataFrame(rows).sort_values("Ref #")
    df.to_excel(filename, index=False, engine="openpyxl")
    print(f"\u2713 Saved {len(rows)} reference(s) to {filename}")
    return filename


# ================= NOTEBOOK CELL 33 =================

def generate_final_report(results):
    if not results:
        return "No verification results available."

    lines = ["=" * 70, "          ACADEMIC REFERENCE VERIFICATION REPORT", "=" * 70]
    total = len(results)
    verified = sum(1 for r in results if r.get("status") == "VERIFIED")
    uncertain = sum(1 for r in results if r.get("status") == "UNCERTAIN")
    not_verified = sum(1 for r in results if r.get("status") == "NOT VERIFIED")
    errors = sum(1 for r in results if "error" in r)

    lines += ["", "SUMMARY", "-" * 70,
              f"Total references:       {total}",
              f"\u2713 Verified:              {verified}",
              f"\u26a0 Uncertain:             {uncertain}",
              f"\u2717 Not verified:          {not_verified}"]
    if errors:
        lines.append(f"\u274c Errors:               {errors}")
    lines += ["", "=" * 70]

    icons = {"VERIFIED": "\u2713", "UNCERTAIN": "\u26a0", "NOT VERIFIED": "\u2717"}

    for i, result in enumerate(results, 1):
        lines += ["", f"REFERENCE {i}/{total}", "-" * 70]
        lines.append(f"Original Reference:\n{result.get('original_reference', 'Unknown')}")

        if "error" in result:
            lines += ["", f"\u274c ERROR: {result['error']}"]
            continue

        status = result.get("status", "UNKNOWN")
        lines += ["", f"STATUS: {icons.get(status, '?')} {status}"]
        # FIX: purely additive - a clear "Manual check needed" call-out
        # right under the status for UNCERTAIN/NOT VERIFIED results.
        # Doesn't touch `status` or `Reason` below it; those remain
        # exactly what classify_reference() produced.
        if status in ("UNCERTAIN", "NOT VERIFIED"):
            lines.append("\u26a0\ufe0f MANUAL CHECK NEEDED - see reason and evidence below")
        lines.append(f"Source: {result.get('source', 'Unknown')}")
        lines += ["", f"Reason: {result.get('explanation', 'N/A')}"]

        parsed = result.get("parsed_reference", {})
        lines += ["", "SUBMITTED METADATA",
                  f"Title:   {parsed.get('title', 'N/A')}",
                  f"Authors: {', '.join(parsed.get('authors', []))}",
                  f"Year:    {parsed.get('year', 'N/A')}",
                  f"DOI:     {parsed.get('doi', 'N/A')}"]

        best = result.get("best_match")
        if best:
            cand = best.get("candidate", {})
            lines += ["", "BEST MATCH",
                      f"Title:   {cand.get('title', 'N/A')}",
                      f"Authors: {', '.join(cand.get('authors', []) or [])}",
                      f"Year:    {cand.get('year', 'N/A')}",
                      f"DOI:     {cand.get('doi', 'N/A')}",
                      f"Source:  {cand.get('source', 'N/A')}"]

        sources = result.get("sources_agreeing", [])
        if sources:
            lines.append(f"\nSources agreeing: {', '.join(sources)} ({len(sources)})")

        lines.append("-" * 70)

    lines += ["", "=" * 70, "                 END OF REPORT", "=" * 70]
    return "\n".join(lines)