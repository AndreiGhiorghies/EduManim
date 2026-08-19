import asyncio
from typing import List, Optional
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════
# SCRAPING FUNCTION
# ═══════════════════════════════════════════════════════════════════
@dataclass
class WebResult:
    """Un rezultat individual de la web search."""
    title: str
    url: str
    snippet: str
    full_text: str = ""
    score: float = 0.5  # baseline; poate fi ajustat
    
    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "full_text": self.full_text,
            "score": self.score,
        }

import trafilatura

def _scrape_url_sync(url: str) -> str:
    """Accesează URL-ul și extrage STRICT textul articolului principal."""
    try:
        # Descărcăm HTML-ul brut
        downloaded = trafilatura.fetch_url(url)
        
        if downloaded:
            # Trafilatura curăță meniurile, reclamele și extrage doar conținutul util
            text = trafilatura.extract(downloaded, include_links=False, include_images=False)
            
            if text:
                # Limităm textul extras pentru a nu umple contextul aiurea
                return text[:3000] 
                
    except Exception as e:
        print(f"[Avertisment] Nu am putut extrage {url}: {e}")
        
    return ""

async def enrich_results_with_content(results: List[WebResult]) -> List[WebResult]:
    """Trece prin toate rezultatele asincron și le descarcă conținutul curat."""
    
    async def fetch_and_assign(result: WebResult):
        text = await asyncio.to_thread(_scrape_url_sync, result.url)
        
        # Dacă site-ul blochează scraping-ul sau nu are un articol clar,
        # dăm fallback la snippet-ul de la DuckDuckGo care e deja dens informațional.
        if len(text) < 100:
            result.full_text = f"Summary snippet: {result.snippet}"
        else:
            result.full_text = text
            
    tasks = [fetch_and_assign(r) for r in results]
    await asyncio.gather(*tasks)
    
    return results

# ═══════════════════════════════════════════════════════════════════
# DATA MODEL
# ══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════
# CORE SEARCH FUNCTION
# ═══════════════════════════════════════════════════════════════════

def _search_sync(
    query: str,
    max_results: int = 5,
    region: str = "wt-wt",  # worldwide
    safesearch: str = "moderate",
) -> List[WebResult]:
    """
    Synchronous DuckDuckGo search. Rulează în thread pool când e chemat async.
    """
    try:
        from ddgs import DDGS
        
        ddgs = DDGS()
        raw_results = ddgs.text(
            query,
            region=region,
            safesearch=safesearch,
            max_results=max_results,
        )
        
        results = []
        for i, r in enumerate(raw_results):
            # DuckDuckGo returnează: title, href, body
            result = WebResult(
                title=r.get("title", ""),
                url=r.get("href", ""),
                snippet=r.get("body", ""),
                # Scor ușor descrescător (primele rezultate = mai bune)
                score=max(0.3, 0.7 - i * 0.08),
            )
            results.append(result)
        
        return results
    
    except Exception as e:
        return []


async def web_search(
    query: str,
    max_results: int = 5,
) -> List[WebResult]:
    
    try:
        results = await asyncio.to_thread(
            _search_sync,
            query,
            max_results,
        )
        
        return results
    
    except Exception as e:
        return []


# ═══════════════════════════════════════════════════════════════════
# VARIANT: News search (pentru info recentă)
# ═══════════════════════════════════════════════════════════════════

async def web_search_news(
    query: str,
    max_results: int = 5,
    days_back: int = 30,
) -> List[WebResult]:
    """
    Search pentru știri recente. Util pentru "latest X" queries.
    """
    try:
        from duckduckgo_search import DDGS
        
        ddgs = DDGS()
        raw_results = await asyncio.to_thread(
            lambda: ddgs.news(
                query,
                max_results=max_results,
                timelimit=f"d{days_back}",  # last N days
            )
        )
        
        results = []
        for r in raw_results:
            result = WebResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=f"{r.get('body', '')} (Source: {r.get('source', 'unknown')})",
                score=0.6,  # news are slightly more relevant for current topics
            )
            results.append(result)
        
        return results
    
    except Exception as e:
        return []


# ═══════════════════════════════════════════════════════════════════
# VARIANT: Cu filtrare (doar anumite domenii)
# ═══════════════════════════════════════════════════════════════════

TRUSTED_DOMAINS = {
    # Educational
    "wikipedia.org", "khanacademy.org", "mit.edu", "stanford.edu",
    "arxiv.org", "nature.com", "sciencedirect.com",
    # Tech docs
    "python.org", "pytorch.org", "tensorflow.org",
    "developer.nvidia.com", "developer.amd.com", "rocm.docs.amd.com",
    "huggingface.co", "github.com",
    # Reference
    "stackoverflow.com", "docs.python.org",
}


async def web_search_filtered(
    query: str,
    max_results: int = 5,
    allowed_domains: Optional[List[str]] = None,
) -> List[WebResult]:
    """
    Web search cu filtrare pe domenii de încredere.
    Implicit folosește TRUSTED_DOMAINS (educational + tech).
    """
    allowed = allowed_domains or list(TRUSTED_DOMAINS)
    
    all_results = await web_search(query, max_results=max_results * 2)
    
    # Filtrează
    filtered = []
    for r in all_results:
        from urllib.parse import urlparse
        domain = urlparse(r.url).netloc.lower()
        
        # Verifică dacă e în allowed domains
        if any(allowed_domain in domain for allowed_domain in allowed):
            r.score = min(1.0, r.score + 0.2)  # boost trusted sources
            filtered.append(r)
    
    return filtered[:max_results]

if __name__ == "__main__":
    # Test rapid
    test_query = "Explain how a llm works"
    print(f"Searching for: {test_query}")
    
    results = asyncio.run(web_search(test_query, max_results=2)) # Limităm la 2 pentru test
    
    # 2. Descărcăm conținutul complet al paginilor
    print("Scraping full content from URLs...")
    results = asyncio.run(enrich_results_with_content(results))

    for r in results:
        print(f"\nTitle: {r.title}\nURL: {r.url}\nSnippet: {r.snippet}\nFull Text: {r.full_text}...\nScore: {r.score}\n\n")
