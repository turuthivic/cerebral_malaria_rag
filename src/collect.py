"""
Phase 1: Collect cerebral malaria documents from PubMed, WHO, and CDC.

Usage:
    python -m src.collect

Output:
    data/raw/*.txt — one file per document (abstract or full text)
    data/raw/metadata.json — metadata for all collected documents
"""

import json
import os
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Session with automatic retries
session = requests.Session()
retries = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))
session.mount("http://", HTTPAdapter(max_retries=retries))

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "CerebralMalariaRAG/1.0 (research project; contact: researcher@example.com)"
}

# --- PubMed via E-utilities (no API key needed, just slower) ---

PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

SEARCH_QUERIES = [
    "cerebral malaria pathogenesis",
    "cerebral malaria treatment",
    "cerebral malaria diagnosis",
    "cerebral malaria children mortality",
    "cerebral malaria neurological sequelae",
    "plasmodium falciparum cerebral malaria",
    "severe malaria brain",
    "cerebral malaria retinopathy",
]


def search_pubmed(query: str, max_results: int = 30) -> list[str]:
    """Search PubMed and return PMIDs."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    resp = session.get(PUBMED_SEARCH_URL, params=params, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])


def fetch_pubmed_abstracts(pmids: list[str]) -> list[dict]:
    """Fetch abstracts for a batch of PMIDs."""
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    resp = session.get(PUBMED_FETCH_URL, params=params, headers=HEADERS, timeout=60)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    articles = []

    for article in soup.find_all("pubmedarticle"):
        pmid = article.find("pmid")
        title = article.find("articletitle")
        abstract = article.find("abstracttext")
        year_elem = article.find("pubdate")
        journal = article.find("journal")

        if not abstract:
            continue

        # Extract all abstract sections (some have labeled sections)
        abstract_parts = article.find_all("abstracttext")
        full_abstract = ""
        for part in abstract_parts:
            label = part.get("label", "")
            if label:
                full_abstract += f"\n{label}: "
            full_abstract += part.get_text(strip=True) + " "

        year = ""
        if year_elem:
            y = year_elem.find("year")
            if y:
                year = y.get_text(strip=True)

        journal_name = ""
        if journal:
            j = journal.find("title")
            if j:
                journal_name = j.get_text(strip=True)

        articles.append({
            "pmid": pmid.get_text(strip=True) if pmid else "",
            "title": title.get_text(strip=True) if title else "",
            "abstract": full_abstract.strip(),
            "year": year,
            "journal": journal_name,
            "source": "pubmed",
        })

    return articles


def collect_pubmed() -> list[dict]:
    """Run all PubMed searches and collect unique articles."""
    all_pmids = set()
    for query in SEARCH_QUERIES:
        print(f"  Searching PubMed: '{query}'")
        pmids = search_pubmed(query, max_results=30)
        all_pmids.update(pmids)
        time.sleep(0.5)  # rate limit

    print(f"  Found {len(all_pmids)} unique PMIDs")

    # Fetch in batches of 50
    pmid_list = list(all_pmids)
    all_articles = []
    for i in range(0, len(pmid_list), 50):
        batch = pmid_list[i : i + 50]
        print(f"  Fetching abstracts {i+1}-{i+len(batch)}...")
        articles = fetch_pubmed_abstracts(batch)
        all_articles.extend(articles)
        time.sleep(0.5)

    print(f"  Collected {len(all_articles)} articles with abstracts")
    return all_articles


# --- WHO Guidelines ---

WHO_URLS = [
    {
        "url": "https://www.who.int/docs/default-source/malaria/malaria-guidelines/who-malaria-guidelines-2023.pdf",
        "title": "WHO Guidelines for Malaria 2023",
        "source": "who",
    },
    {
        "url": "https://www.who.int/news-room/fact-sheets/detail/malaria",
        "title": "WHO Malaria Fact Sheet",
        "source": "who",
    },
]


def collect_who() -> list[dict]:
    """Collect WHO malaria content."""
    articles = []
    for item in WHO_URLS:
        print(f"  Fetching WHO: {item['title']}")
        try:
            resp = session.get(item["url"], headers=HEADERS, timeout=60)
            resp.raise_for_status()

            if item["url"].endswith(".pdf"):
                # Save PDF for later processing
                pdf_path = RAW_DIR / f"who_{item['title'].replace(' ', '_')}.pdf"
                pdf_path.write_bytes(resp.content)
                print(f"    Saved PDF: {pdf_path}")
                articles.append({
                    "title": item["title"],
                    "abstract": f"[PDF saved to {pdf_path}]",
                    "source": item["source"],
                    "url": item["url"],
                    "type": "pdf",
                })
            else:
                soup = BeautifulSoup(resp.text, "html.parser")
                # Get main content
                content = soup.find("main") or soup.find("article") or soup.find("body")
                if content:
                    text = content.get_text(separator="\n", strip=True)
                    # Clean up
                    text = re.sub(r"\n{3,}", "\n\n", text)
                    articles.append({
                        "title": item["title"],
                        "abstract": text[:10000],  # cap at 10k chars
                        "source": item["source"],
                        "url": item["url"],
                        "type": "html",
                    })
                    print(f"    Got {len(text)} chars")
        except Exception as e:
            print(f"    Error: {e}")

    return articles


# --- CDC ---

CDC_URLS = [
    {
        "url": "https://www.cdc.gov/malaria/about/disease.html",
        "title": "CDC - About Malaria Disease",
    },
    {
        "url": "https://www.cdc.gov/malaria/about/biology.html",
        "title": "CDC - Biology of Malaria",
    },
    {
        "url": "https://www.cdc.gov/malaria/hcp/clinical-guidance/index.html",
        "title": "CDC - Malaria Clinical Guidance",
    },
    {
        "url": "https://www.cdc.gov/malaria/hcp/diagnosis-testing/index.html",
        "title": "CDC - Malaria Diagnosis and Testing",
    },
]


def collect_cdc() -> list[dict]:
    """Collect CDC malaria content."""
    articles = []
    for item in CDC_URLS:
        print(f"  Fetching CDC: {item['title']}")
        try:
            resp = session.get(item["url"], headers=HEADERS, timeout=60)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            content = soup.find("main") or soup.find("article") or soup.find("body")
            if content:
                text = content.get_text(separator="\n", strip=True)
                text = re.sub(r"\n{3,}", "\n\n", text)
                articles.append({
                    "title": item["title"],
                    "abstract": text[:10000],
                    "source": "cdc",
                    "url": item["url"],
                    "type": "html",
                })
                print(f"    Got {len(text)} chars")
        except Exception as e:
            print(f"    Error: {e}")
        time.sleep(0.5)

    return articles


# --- PMC Full Text (Open Access) ---

PMC_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PMC_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def collect_pmc_fulltext(max_articles: int = 20) -> list[dict]:
    """Collect full-text open-access articles from PMC."""
    print("  Searching PMC for open-access full text...")
    params = {
        "db": "pmc",
        "term": "cerebral malaria AND open access[filter]",
        "retmax": max_articles,
        "retmode": "json",
        "sort": "relevance",
    }
    resp = session.get(PMC_SEARCH_URL, params=params, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    pmc_ids = resp.json().get("esearchresult", {}).get("idlist", [])
    print(f"  Found {len(pmc_ids)} open-access articles")

    articles = []
    for pmc_id in pmc_ids:
        try:
            params = {
                "db": "pmc",
                "id": pmc_id,
                "retmode": "xml",
            }
            resp = session.get(PMC_FETCH_URL, params=params, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            title = soup.find("article-title")
            body = soup.find("body")

            if body:
                # Extract section text
                text = ""
                for sec in body.find_all("sec"):
                    sec_title = sec.find("title")
                    if sec_title:
                        text += f"\n\n## {sec_title.get_text(strip=True)}\n"
                    for p in sec.find_all("p", recursive=False):
                        text += p.get_text(strip=True) + "\n"

                if len(text) > 500:  # skip very short articles
                    articles.append({
                        "pmcid": pmc_id,
                        "title": title.get_text(strip=True) if title else f"PMC{pmc_id}",
                        "abstract": text[:15000],  # cap full text
                        "source": "pmc",
                        "type": "fulltext",
                    })
                    print(f"    PMC{pmc_id}: {len(text)} chars")
            time.sleep(0.5)
        except Exception as e:
            print(f"    Error fetching PMC{pmc_id}: {e}")

    return articles


def save_documents(all_articles: list[dict]):
    """Save each document as a text file and metadata as JSON."""
    metadata = []

    for i, article in enumerate(all_articles):
        # Create filename
        source = article.get("source", "unknown")
        title_slug = re.sub(r"[^a-z0-9]+", "_", article.get("title", "untitled").lower())[:60]
        filename = f"{source}_{i:04d}_{title_slug}.txt"

        # Write text content
        filepath = RAW_DIR / filename
        content = f"Title: {article.get('title', '')}\n"
        content += f"Source: {article.get('source', '')}\n"
        if article.get("year"):
            content += f"Year: {article['year']}\n"
        if article.get("journal"):
            content += f"Journal: {article['journal']}\n"
        if article.get("url"):
            content += f"URL: {article['url']}\n"
        if article.get("pmid"):
            content += f"PMID: {article['pmid']}\n"
        if article.get("pmcid"):
            content += f"PMCID: {article['pmcid']}\n"
        content += f"\n---\n\n{article.get('abstract', '')}"

        filepath.write_text(content, encoding="utf-8")

        # Track metadata
        meta = {k: v for k, v in article.items() if k != "abstract"}
        meta["filename"] = filename
        meta["char_count"] = len(article.get("abstract", ""))
        metadata.append(meta)

    # Save metadata index
    meta_path = RAW_DIR / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved {len(metadata)} documents to {RAW_DIR}/")
    print(f"Metadata saved to {meta_path}")


def main():
    print("=" * 60)
    print("Cerebral Malaria Document Collection")
    print("=" * 60)

    all_articles = []

    collectors = [
        ("1/4", "PubMed abstracts", lambda: collect_pubmed()),
        ("2/4", "PMC full-text articles", lambda: collect_pmc_fulltext(max_articles=20)),
        ("3/4", "WHO guidelines", lambda: collect_who()),
        ("4/4", "CDC content", lambda: collect_cdc()),
    ]

    for step, name, fn in collectors:
        print(f"\n[{step}] Collecting {name}...")
        try:
            results = fn()
            all_articles.extend(results)
        except Exception as e:
            print(f"  WARNING: {name} collection failed: {e}")
            print("  Continuing with remaining sources...")

    print(f"\n{'=' * 60}")
    print(f"Total documents collected: {len(all_articles)}")
    print(f"{'=' * 60}")

    save_documents(all_articles)


if __name__ == "__main__":
    main()
