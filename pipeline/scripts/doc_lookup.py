"""
Documentation Lookup Utility

Searches DuckDuckGo for library documentation and fetches page content.
Used by expand_prompts.py and generate_data.py to provide teachers with
relevant documentation context when generating training data.

Special handling:
  - Svelte/SvelteKit: reads from local pipeline/docs/svelte_full.txt
  - GitHub READMEs: fetches raw markdown directly
  - Other docs: searches DuckDuckGo, fetches top result, strips HTML

Usage as library:
  from doc_lookup import DocLookup
  docs = DocLookup()
  text = docs.lookup("Giraffe", "JWT authentication middleware")
  text = docs.lookup("FsToolkit.ErrorHandling", "taskResult computation expression")

Usage as CLI:
  python doc_lookup.py "Giraffe" "HttpHandler composition"
  python doc_lookup.py "Svelte 5" "runes $state $derived"
"""

import hashlib
import json
import logging
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

import httpx
from duckduckgo_search import DDGS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
DOCS_DIR = PIPELINE_DIR / "docs"
CACHE_DIR = DOCS_DIR / ".cache"
SVELTE_DOCS = DOCS_DIR / "svelte_full.txt"

# Rate limit: DuckDuckGo allows ~15-20 requests/minute before throttling
SEARCH_DELAY_S = 3.0

# Known GitHub raw URLs for key libraries (skip search for these)
KNOWN_DOCS = {
    "giraffe": "https://raw.githubusercontent.com/giraffe-fsharp/Giraffe/master/DOCUMENTATION.md",
    "giraffe.viewengine": "https://raw.githubusercontent.com/giraffe-fsharp/Giraffe.ViewEngine/master/README.md",
    "giraffe.openapi": "https://raw.githubusercontent.com/giraffe-fsharp/Giraffe.OpenApi/main/README.md",
    "fsharp.systemtextjson": "https://raw.githubusercontent.com/Tarmil/FSharp.SystemTextJson/master/README.md",
    "fsharp.control.asyncseq": "https://raw.githubusercontent.com/fsprojects/FSharp.Control.AsyncSeq/master/README.md",
    "fsharp.control.reactive": "https://raw.githubusercontent.com/fsprojects/FSharp.Control.Reactive/main/README.md",
    "fsharp.data": "https://raw.githubusercontent.com/fsprojects/FSharp.Data/main/README.md",
    "fstoolkit.errorhandling": "https://raw.githubusercontent.com/demystifyfp/FsToolkit.ErrorHandling/master/gitbook/README.md",
    "akka.net": "https://raw.githubusercontent.com/akkadotnet/akka.net/dev/README.md",
    "akka": "https://raw.githubusercontent.com/akkadotnet/akka.net/dev/README.md",
    "thoth.json": "https://raw.githubusercontent.com/thoth-org/Thoth.Json/main/README.md",
    "thoth.json.net": "https://raw.githubusercontent.com/thoth-org/Thoth.Json/main/README.md",
    "linq2db": "https://raw.githubusercontent.com/linq2db/linq2db/master/README.md",
    "fluentmigrator": "https://raw.githubusercontent.com/fluentmigrator/fluentmigrator/main/README.md",
    "npgsql": "https://raw.githubusercontent.com/npgsql/npgsql/main/README.md",
    "confluent.kafka": "https://raw.githubusercontent.com/confluentinc/confluent-kafka-dotnet/master/README.md",
    "minio": "https://raw.githubusercontent.com/minio/minio-dotnet/master/README.md",
    "serilog": "https://raw.githubusercontent.com/serilog/serilog/dev/README.md",
    "stripe.net": "https://raw.githubusercontent.com/stripe/stripe-dotnet/master/README.md",
}

# Svelte-related queries use the local docs file
SVELTE_KEYWORDS = {
    "svelte",
    "sveltekit",
    "svelte 5",
    "svelte5",
    "runes",
    "$state",
    "$derived",
    "$effect",
    "$props",
}


class HTMLTextExtractor(HTMLParser):
    """Simple HTML to text converter -- strips tags, keeps text content."""

    def __init__(self):
        super().__init__()
        self._text = []
        self._skip = False
        self._skip_tags = {
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "noscript",
            "svg",
        }

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._skip = True
        if tag in ("p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"):
            self._text.append("\n")

    def handle_endtag(self, tag):
        if tag in self._skip_tags:
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self._text.append(data)

    def get_text(self) -> str:
        raw = "".join(self._text)
        # Normalize whitespace: collapse multiple blank lines
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def html_to_text(html: str) -> str:
    """Convert HTML to plain text."""
    extractor = HTMLTextExtractor()
    try:
        extractor.feed(html)
        return extractor.get_text()
    except Exception:
        # Fallback: regex strip
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


class DocLookup:
    """Documentation search and fetch utility."""

    def __init__(self, cache_dir: Path = CACHE_DIR, svelte_docs: Path = SVELTE_DOCS):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.svelte_docs = svelte_docs
        self._svelte_text: Optional[str] = None
        self._last_search_time = 0.0
        self._client = httpx.Client(
            timeout=20.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; TrainingDataBot/1.0)"},
        )
        self._ddgs = DDGS(timeout=20)

    def _cache_key(self, query: str) -> str:
        """Generate a filesystem-safe cache key."""
        return hashlib.md5(query.encode()).hexdigest()

    def _read_cache(self, key: str) -> Optional[str]:
        """Read from disk cache."""
        cache_file = self.cache_dir / f"{key}.txt"
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")
        return None

    def _write_cache(self, key: str, content: str):
        """Write to disk cache."""
        cache_file = self.cache_dir / f"{key}.txt"
        cache_file.write_text(content, encoding="utf-8")

    def _rate_limit(self):
        """Enforce rate limiting for DuckDuckGo searches."""
        elapsed = time.monotonic() - self._last_search_time
        if elapsed < SEARCH_DELAY_S:
            time.sleep(SEARCH_DELAY_S - elapsed)
        self._last_search_time = time.monotonic()

    def _load_svelte_docs(self) -> str:
        """Load the local Svelte docs file."""
        if self._svelte_text is None:
            if self.svelte_docs.exists():
                self._svelte_text = self.svelte_docs.read_text(encoding="utf-8")
                log.info(f"Loaded Svelte docs: {len(self._svelte_text):,} chars")
            else:
                log.warning(f"Svelte docs not found at {self.svelte_docs}")
                self._svelte_text = ""
        return self._svelte_text

    def _search_svelte(self, topic: str, max_chars: int = 12000) -> str:
        """Search within the local Svelte docs for a specific topic."""
        full_text = self._load_svelte_docs()
        if not full_text:
            return ""

        # Search for the topic heading in the docs
        topic_lower = topic.lower()
        keywords = [w for w in topic_lower.split() if len(w) > 2]

        # Try to find a section that matches the topic
        # Split by markdown headings
        sections = re.split(r"\n(?=# )", full_text)

        # Score each section by keyword overlap
        scored = []
        for section in sections:
            section_lower = section.lower()
            score = sum(1 for kw in keywords if kw in section_lower)
            if score > 0:
                scored.append((score, section))

        scored.sort(key=lambda x: -x[0])

        if not scored:
            # Fallback: return the first N chars
            return full_text[:max_chars]

        # Return the top matching sections up to max_chars
        result = []
        total = 0
        for _, section in scored[:5]:
            if total + len(section) > max_chars:
                remaining = max_chars - total
                if remaining > 500:
                    result.append(section[:remaining])
                break
            result.append(section)
            total += len(section)

        return "\n\n".join(result)

    def _fetch_url(self, url: str, max_chars: int = 12000) -> str:
        """Fetch a URL and return cleaned text content."""
        cache_key = self._cache_key(url)
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached[:max_chars]

        try:
            resp = self._client.get(url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")

            if "text/html" in content_type:
                text = html_to_text(resp.text)
            else:
                # Markdown, plain text, etc.
                text = resp.text

            self._write_cache(cache_key, text)
            return text[:max_chars]

        except Exception as e:
            log.warning(f"Failed to fetch {url}: {e}")
            return ""

    def _search_ddg(self, query: str, max_results: int = 3) -> list[dict]:
        """Search DuckDuckGo with rate limiting."""
        self._rate_limit()
        try:
            results = self._ddgs.text(query, max_results=max_results)
            return results or []
        except Exception as e:
            log.warning(f"DuckDuckGo search failed for '{query}': {e}")
            return []

    def lookup(
        self,
        library: str,
        topic: str,
        max_chars: int = 12000,
        max_search_results: int = 3,
    ) -> str:
        """Look up documentation for a library/topic combination.

        Args:
            library: Library name (e.g., "Giraffe", "FsToolkit.ErrorHandling")
            topic: Specific topic to search for (e.g., "JWT authentication", "taskResult CE")
            max_chars: Maximum characters to return
            max_search_results: Number of DuckDuckGo results to consider

        Returns:
            Documentation text (may be empty if nothing found)
        """
        query = f"{library} {topic}".strip()
        cache_key = self._cache_key(f"lookup:{query}")
        cached = self._read_cache(cache_key)
        if cached is not None:
            log.info(f"Cache hit: {query}")
            return cached[:max_chars]

        # Check if this is a Svelte query
        query_lower = query.lower()
        if any(kw in query_lower for kw in SVELTE_KEYWORDS):
            log.info(f"Using local Svelte docs for: {topic}")
            result = self._search_svelte(topic, max_chars)
            if result:
                self._write_cache(cache_key, result)
                return result

        # Check known doc URLs first
        lib_key = library.lower().strip()
        if lib_key in KNOWN_DOCS:
            log.info(f"Using known docs URL for {library}")
            text = self._fetch_url(KNOWN_DOCS[lib_key], max_chars)
            if text:
                # Search within the fetched doc for the specific topic
                topic_text = self._extract_relevant_section(text, topic, max_chars)
                self._write_cache(cache_key, topic_text)
                return topic_text

        # Search DuckDuckGo
        # Sanitize query: F# breaks search engines, replace with "FSharp"
        clean_library = library.replace("F#", "FSharp").replace("C#", "CSharp")
        clean_topic = topic.replace("F#", "FSharp").replace("C#", "CSharp")
        search_query = f"{clean_library} {clean_topic} documentation"
        log.info(f"Searching: {search_query}")
        results = self._search_ddg(search_query, max_search_results)

        if not results:
            # Retry with site-scoped search for known domains
            fallback_sites = {
                "akka": "site:getakka.net",
                "linq2db": "site:linq2db.github.io",
                "fluentmigrator": "site:fluentmigrator.github.io",
                "npgsql": "site:npgsql.org",
                "serilog": "site:github.com/serilog",
                "docker": "site:docs.docker.com",
                "kubernetes": "site:kubernetes.io",
                "asp.net": "site:learn.microsoft.com",
                "aspnet": "site:learn.microsoft.com",
            }
            for key, site in fallback_sites.items():
                if key in clean_library.lower() or key in clean_topic.lower():
                    fallback_query = f"{clean_topic} {site}"
                    log.info(f"Retrying with: {fallback_query}")
                    results = self._search_ddg(fallback_query, max_search_results)
                    break

        if not results:
            log.warning(f"No search results for: {search_query}")
            return ""

        # Filter out obviously irrelevant results
        relevant = [
            r
            for r in results
            if any(
                kw in r.get("title", "").lower() or kw in r.get("href", "").lower()
                for kw in clean_library.lower().split()[:2]
            )
        ]
        if not relevant:
            relevant = results  # Fall back to unfiltered if no matches

        best_url = relevant[0]["href"]
        log.info(f"Fetching: {best_url}")
        text = self._fetch_url(best_url, max_chars)

        if text:
            self._write_cache(cache_key, text)

        return text

    def _extract_relevant_section(
        self, text: str, topic: str, max_chars: int = 12000
    ) -> str:
        """Extract the most relevant section from a large document."""
        if len(text) <= max_chars:
            return text

        keywords = [w.lower() for w in topic.split() if len(w) > 2]
        if not keywords:
            return text[:max_chars]

        # Split into paragraphs/sections
        paragraphs = re.split(r"\n\n+", text)

        # Score paragraphs by keyword density
        scored = []
        for i, para in enumerate(paragraphs):
            para_lower = para.lower()
            score = sum(para_lower.count(kw) for kw in keywords)
            scored.append((score, i, para))

        scored.sort(key=lambda x: -x[0])

        # Take top paragraphs, preserving some order context
        top_indices = sorted([s[1] for s in scored[:20]])
        result = []
        total = 0
        for idx in top_indices:
            para = paragraphs[idx]
            if total + len(para) > max_chars:
                break
            result.append(para)
            total += len(para)

        return "\n\n".join(result) if result else text[:max_chars]

    def lookup_batch(
        self,
        queries: list[tuple[str, str]],
        max_chars: int = 12000,
    ) -> dict[str, str]:
        """Look up documentation for multiple library/topic pairs.

        Args:
            queries: List of (library, topic) tuples
            max_chars: Maximum characters per result

        Returns:
            Dict mapping "{library}:{topic}" to documentation text
        """
        results = {}
        for library, topic in queries:
            key = f"{library}:{topic}"
            results[key] = self.lookup(library, topic, max_chars)
        return results

    def close(self):
        """Clean up HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def main():
    """CLI interface for quick lookups."""
    if len(sys.argv) < 3:
        print("Usage: python doc_lookup.py <library> <topic>")
        print('Example: python doc_lookup.py "Giraffe" "JWT authentication"')
        sys.exit(1)

    library = sys.argv[1]
    topic = " ".join(sys.argv[2:])

    # Fix Windows console encoding
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    with DocLookup() as docs:
        result = docs.lookup(library, topic)
        if result:
            print(f"--- {library}: {topic} ({len(result)} chars) ---")
            print(result[:3000])
            if len(result) > 3000:
                print(f"\n... ({len(result) - 3000} more chars)")
        else:
            print("No documentation found.")


if __name__ == "__main__":
    main()
