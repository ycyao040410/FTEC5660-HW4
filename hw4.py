"""FTEC5660 Homework 4: the retrieval half of a RAG assistant over company reports.

Fill in the five functions marked ### YOUR CODE HERE. You may add imports, constants and helper
functions above them, but do not change the provided code below them (the models, the generator,
the loaders, the runner and the scoring).

    python3 hw4.py --data-folder public_test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek

load_dotenv()


# =============================================================================
# Your code: add imports, constants and helper functions here.
# =============================================================================


from collections import defaultdict
from dataclasses import dataclass

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi


# Defaults are used by the grader. The experiment script changes one option at a time.
@dataclass(frozen=True)
class _Options:
    structured_chunks: bool = True
    normalized_tokens: bool = True
    metadata_routing: bool = True
    evidence_ranking: bool = True
    company_balancing: bool = True
    abstention: bool = True


_OPTIONS = _Options()
_COMPANIES: dict[str, str] = {}
_BROKERS: set[str] = set()
_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_WORD = re.compile(r"[a-z]+|\d+", re.I)
_NAME_WORD = r"[A-Z][A-Za-z0-9]*(?:[-&.][A-Za-z0-9]+)*"
_NAME_SPAN = re.compile(rf"\b{_NAME_WORD}(?:\s+(?:{_NAME_WORD}|&|of|the))*(?:\s+Inc\.|\s+Ltd\.)?")
_SPACE = re.compile(r"\s+")
_STOP = set("""
a an and are as at be been by did do does for from had has have how in is it its
many more much of on or s than that the their to was were what which who with
amount annual company companies consolidated each end ending fiscal
fy held hold higher highest less lower million millions usd dollar dollars
percent percentage value year quarter q4 fourth report reported give set
financial statements shares month months respectively
january february march april may june july august september october november december
""".split())
_NAME_NOISE = set("""
a an and as at by december fiscal for from fy how in of on or q4 the to what
which who whose usd
""".split())
_CANONICAL = {
    "activity": "activities", "announced": "announce", "authorization": "announce",
    "authorisation": "announce", "approved": "announce", "borrowings": "borrowing",
    "buyback": "repurchase", "buybacks": "repurchase", "repurchases": "repurchase",
    "inventories": "inventory",
    "equivalents": "equivalent", "estimates": "estimate", "estimated": "estimate",
    "estimation": "estimate", "forecasts": "estimate", "forecasted": "estimate",
    "forecast": "estimate", "outlook": "guidance", "guided": "guidance",
    "guide": "guidance", "expect": "guidance", "expects": "guidance",
    "employees": "employee", "employed": "employee", "headcount": "employee",
    "headquartered": "headquarters", "audited": "audit", "auditor": "audit",
    "auditors": "audit", "receivables": "receivable", "assets": "asset",
    "liabilities": "liability", "operations": "operating", "program": "programme",
    "expenses": "expense", "region": "geography", "regions": "geography",
    "sales": "revenue", "staff": "employee", "workforce": "employee", "employ": "employee",
    "debts": "borrowing", "debt": "borrowing",
}
_QUERY_FILLER = {"city", "firm", "generate", "investment", "review", "results", "company",
                 "call", "conference", "earnings", "announce", "market", "management",
                 "programme", "large", "name", "organisation", "organization", "q", "period",
                 "posting", "post", "record", "recorded", "disclose", "disclosed", "figure"}
_ROLE_PHRASES = {"chief executive officer": "ceo", "chief financial officer": "cfo",
                 "chief operating officer": "coo", "chief technology officer": "cto"}


def _clean(value: str) -> str:
    return _SPACE.sub(" ", value).strip()


def _text(node: Any) -> str:
    return _clean(node.get_text(" ", strip=True))


def _span_count(tag: Any, attribute: str) -> int:
    raw = str(tag.get(attribute, "1"))
    return min(int(raw), 100) if raw.isdigit() and int(raw) > 0 else 1


def _expand_rows(rows: list) -> list[list[str]]:
    """Expand merged cells so a data cell stays under the appropriate header."""
    grid: dict[tuple[int, int], str] = {}
    result = []
    for y, row in enumerate(rows):
        x = 0
        for cell in row.find_all(["th", "td"], recursive=False):
            while (y, x) in grid:
                x += 1
            width = _span_count(cell, "colspan")
            for dy in range(_span_count(cell, "rowspan")):
                for dx in range(width):
                    grid[y + dy, x + dx] = _text(cell)
            x += width
        positions = [col for row_index, col in grid if row_index == y]
        result.append([grid.get((y, col), "") for col in range(max(positions, default=-1) + 1)])
    return result


def _document_metadata(soup: BeautifulSoup, file_name: str) -> tuple[dict, str]:
    heading = soup.find("h1")
    company = re.sub(r"\s*\([^()]+\)\s*$", "", _text(heading)) if heading else ""
    candidates = []
    if heading:
        for tag in heading.find_all_next(["p", "h2", "h3"], limit=6):
            candidates.append(_text(tag))
    if soup.title:
        candidates.append(_text(soup.title))
    kind = "annual_report"
    year = None
    for line in candidates:
        date = _YEAR.search(line)
        if date and re.search(r"annual report|results review|earnings (?:conference )?call", line, re.I):
            kind = ("research_report" if re.search(r"results review", line, re.I) else
                    "earnings_call" if re.search(r"earnings (?:conference )?call", line, re.I)
                    else "annual_report")
            year = int(date[1])
            break
    if year is None:
        for line in candidates:
            date = _YEAR.search(line)
            if date:
                year = int(date[1])
                break
    broker_tag = soup.select_one(".broker")
    broker = re.split(r"\s+(?:[|—–-])\s+", _text(broker_tag), maxsplit=1)[0] if broker_tag else ""
    if not broker and kind == "research_report" and heading:
        for tag in heading.find_all_previous(["div", "p"], limit=6):
            lead = _text(tag)
            broker_match = re.match(r"(.+?)\s+(?:[|—–-])\s+Equity Research\b", lead, re.I)
            if broker_match:
                broker = broker_match[1]
                break
    return {"source": file_name, "company": company, "fiscal_year": year, "doc_type": kind}, broker


def _documents(text: str, metadata: dict) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_CHUNK_CHARS)
    return [Document(page_content=piece, metadata=metadata.copy())
            for piece in splitter.split_text(text)]


def _table_chunks(table: Any, metadata: dict, section: str) -> list[Document]:
    lines = table_to_text(str(table)).splitlines()
    if not lines:
        return []
    grouped = dict(metadata, section=section, table=True)
    caption_tag = table.find("caption", recursive=False)
    caption = _text(caption_tag) if caption_tag else ""
    grouped["caption"] = caption
    title = "\n".join(dict.fromkeys(item for item in (section, caption) if item))
    lines = lines[1:] if caption else lines
    if len(title) > MAX_CHUNK_CHARS - MAX_LINE_CHARS - 2:
        return _documents("\n".join((title, *lines)), grouped)
    chunks = []
    batch = title
    for line in lines:
        if batch and len(batch) + len(line) + 1 > MAX_CHUNK_CHARS:
            chunks.append(Document(page_content=batch, metadata=grouped.copy()))
            batch = title
        batch = "\n".join(item for item in (batch, line) if item)
    if batch:
        chunks.append(Document(page_content=batch, metadata=grouped.copy()))
    return chunks


def _question_kind(question: str) -> str:
    if re.search(r"\b(?:earnings(?: conference)? (?:call|webcast)|quarter(?:ly)? (?:conference )?call)\b",
                 question, re.I):
        return "earnings_call"
    if re.search(r"\b(?:target[- ]price|price target|(?:investment |broker )?rating|results review|"
                 r"research (?:note|report)|forecast\w*|broker(?:age)?(?:'s)? (?:estimate|recommendation)|"
                 r"\b(?:estimate|project)\w*\s+(?:for|at)\b)", question, re.I):
        return "research_report"
    return "annual_report"


def _name_candidates(question: str) -> list[str]:
    """Identify subject spans even if a company does not occur in the corpus."""
    candidates = []
    for match in _NAME_SPAN.finditer(question):
        name = match[0].strip()
        parts = name.split()
        while parts and (parts[0].casefold() in _NAME_NOISE or
                         re.fullmatch(r"(?:FY|Q)\d+", parts[0])):
            parts.pop(0)
        while parts and (parts[-1].casefold() in _NAME_NOISE or
                         re.fullmatch(r"(?:FY|Q)\d+", parts[-1])):
            parts.pop()
        name = " ".join(parts)
        if len(parts) >= 2:
            candidates.append(name)
    return candidates


def _replace_name(text: str, name: str) -> str:
    words = _WORD.findall(name)
    if not words:
        return text
    pattern = r"(?<!\w)" + r"[\W_]+".join(map(re.escape, words)) + r"(?!\w)"
    return re.sub(pattern, " ", text, flags=re.I)


def _tokenize(text: str, normalized: bool) -> list[str]:
    if not normalized:
        return text.lower().split()
    text = text.casefold()
    for phrase, acronym in _ROLE_PHRASES.items():
        text = text.replace(phrase, acronym)
    return [_CANONICAL.get(word, word) for word in _WORD.findall(text) if word not in _STOP]


def _query_terms(question: str, parsed: dict) -> list[str]:
    for company in parsed["companies"]:
        question = _replace_name(question, company)
    for broker in _BROKERS:
        question = _replace_name(question, broker)
    if re.search(r"\b(?:city|where|location)\b", question, re.I):
        question = re.sub(r"\b(?:based|located)\b", "headquarters", question, flags=re.I)
        question = re.sub(r"\b(?:where|location)\b", " ", question, flags=re.I)
    return list(dict.fromkeys(word for word in _tokenize(question, True)
                              if word.isalpha() and word not in _QUERY_FILLER))


@dataclass
class _Passage:
    chunk: Document
    company: str
    kind: str
    year: int | None
    explicit_years: set[int]
    lines: list[tuple[set[str], set[str], set[int], bool]]
    tokens: set[str]
    header: set[str]


@dataclass
class _Index:
    passages: list[_Passage]
    bm25: Any
    options: _Options
    companies: dict[str, str]
    available_years: dict[tuple[str, str], set[int]]


def _passage(chunk: Document) -> _Passage:
    meta = chunk.metadata
    text = chunk.page_content
    heading = " ".join(meta.get(key, "") for key in ("section", "caption"))
    explicit_years = set()
    lines = []
    for line in text.splitlines():
        label, sep, cells = line.partition(" | ")
        table_row = bool(meta.get("table") and sep)
        # Parse years from table column labels, and never from the cell values.
        dated_text = (" ".join(cell.split(": ", 1)[0] for cell in cells.split(" | "))
                      if table_row else line)
        years = {int(y) for y in _YEAR.findall(dated_text)}
        if table_row or meta["doc_type"] != "annual_report":
            explicit_years.update(years)
        lines.append((set(_tokenize(label, True)), set(_tokenize(line, True)),
                      years, table_row))
    return _Passage(chunk, company_key(meta["company"]), meta["doc_type"], meta["fiscal_year"],
                    explicit_years, lines, set(_tokenize(text, True)), set(_tokenize(heading, True)))


def _context_year(passage: _Passage, year: int | None) -> bool:
    return year is None or year == passage.year or year in passage.explicit_years


def _evidence(passage: _Passage, terms: set[str], year: int | None) -> tuple[float, float]:
    """Return lexical evidence strength and a ranking adjustment."""
    if not terms:
        return 0.0, 0.0
    best_coverage = best_adjustment = 0.0
    for label, full, years, table_row in passage.lines:
        matches = terms & full
        if not matches:
            continue
        supported = matches | (terms & passage.header) if table_row else matches
        coverage = len(supported) / len(terms)
        adjustment = 13.0 * coverage
        if table_row:
            adjustment += 8.0 * len(terms & label) / len(terms)
            if terms <= label:
                adjustment += 4.0 * len(terms) / max(len(label), 1)
            if passage.kind == "annual_report":
                adjustment += 3.0
            if year in years:
                adjustment += 4.0
            elif year is not None and years:
                adjustment -= 5.0
        elif year in years:
            adjustment += 2.0
        best_coverage = max(best_coverage, coverage)
        best_adjustment = max(best_adjustment, adjustment)
    return best_coverage, best_adjustment


def _rank(index: _Index, candidates: list[int], lexical: Any, terms: set[str],
          year: int | None) -> list[tuple[int, float, float]]:
    ranked = []
    scale = max((max(float(lexical[i]), 0.0) for i in candidates), default=0.0) or 1.0
    for i in candidates:
        passage = index.passages[i]
        strength = len(terms & passage.tokens) / len(terms) if terms else 0.0
        if not index.options.evidence_ranking:
            ranked.append((i, float(lexical[i]), strength))
            continue
        strength, adjustment = _evidence(passage, terms, year)
        score = 4.0 * float(lexical[i]) / scale + adjustment
        if year is not None:
            if year == passage.year:
                score += 4.0
            elif not _context_year(passage, year):
                score -= 6.0
        if terms & {"ceo", "cfo", "coo", "cto"} and "executive" in passage.header:
            score += 10.0
        if "audit" in terms and "audit" in passage.header:
            score += 8.0
        if {"cash", "equivalent"} <= terms and {"balance", "sheet"} <= passage.header:
            score += 4.0
        if passage.kind == "research_report" and "estimate" in terms:
            if "estimate" in passage.header and "change" in passage.header:
                score += 10.0
            if passage.chunk.metadata.get("table") and "summary" in passage.header:
                score += 10.0
        ranked.append((i, score, strength))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))


def _balanced_order(ranked: list[tuple[int, float, float]], company_keys: list[str],
                    passages: list[_Passage]) -> list[tuple[int, float, float]]:
    heads = []
    for key in company_keys:
        first = next((item for item in ranked if passages[item[0]].company == key), None)
        if first is not None:
            heads.append(first)
    already = {item[0] for item in heads}
    return sorted(heads, key=lambda item: (-item[1], item[0])) + [
        item for item in ranked if item[0] not in already]


def table_to_text(table_html: str) -> str:
    """Keep the caption and place each cell next to its original row and column labels."""
    table = BeautifulSoup(table_html, "html.parser").find("table")
    if table is None:
        return ""
    rows = [row for row in table.find_all("tr") if row.find_parent("table") is table]
    grid = _expand_rows(rows)
    header_count = 0
    for row in rows:
        cells = row.find_all(["th", "td"], recursive=False)
        if row.find_parent("thead") or (len(cells) > 1 and all(c.name == "th" for c in cells)
                                        and not any(c.get("scope") == "row" for c in cells)):
            header_count += 1
        else:
            break
    width = max((len(row) for row in grid), default=0)
    headers = [" / ".join(dict.fromkeys(
        row[i] for row in grid[:header_count] if i < len(row) and row[i]))
        for i in range(width)]
    caption = table.find("caption", recursive=False)
    lines = [_text(caption)] if caption is not None else []
    for row in grid[header_count:]:
        if not any(row):
            continue
        label, *values = row
        if len(set(row)) == 1:
            lines.append(label)
            continue
        batch = label
        for i, value in enumerate(values, 1):
            if not value:
                continue
            cell = f"{headers[i]}: {value}" if headers[i] else value
            if batch != label and len(batch) + len(cell) + 3 > MAX_LINE_CHARS:
                lines.append(batch)
                batch = label
            batch += (" | " if batch else "") + cell
        if batch:
            lines.append(batch)
    return "\n".join(lines)


def chunk_document(file_name: str, html: str) -> list[Document]:
    """Parse provenance from the HTML and preserve whole table rows whenever possible."""
    soup = BeautifulSoup(html, "html.parser")
    metadata, broker = _document_metadata(soup, file_name)
    key = company_key(metadata["company"])
    if key:
        _COMPANIES[key] = metadata["company"]
    if broker:
        _BROKERS.add(company_key(broker))
        metadata["broker"] = broker
    if not _OPTIONS.structured_chunks:
        # Keep the tutorial's separator and overlap defaults within the homework limit.
        splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_CHUNK_CHARS)
        return [Document(page_content=text, metadata=metadata.copy())
                for text in splitter.split_text(soup.get_text(" ", strip=True))]
    body = soup.body or soup
    for tag in body.find_all(["script", "style", "nav"]):
        tag.decompose()
    chunks = []
    lines = []
    section = ""
    tags = ["h1", "h2", "h3", "h4", "p", "li", "table", "div"]

    def flush() -> None:
        if lines:
            chunks.extend(_documents("\n".join(lines), dict(metadata, section=section)))
            lines.clear()

    for tag in body.find_all(tags):
        if tag.find_parent("table") or (tag.name != "table" and tag.find_parent(["p", "li"])):
            continue
        if tag.name == "div" and tag.find(tags):
            continue
        if tag.name == "table":
            flush()
            chunks.extend(_table_chunks(tag, metadata, section))
        elif tag.name in {"h2", "h3", "h4"}:
            flush()
            section = _text(tag)
            lines.append(section)
        else:
            payload = _text(tag)
            if payload:
                if not lines and section:
                    lines.append(section)
                lines.append(payload)
    flush()
    return chunks


def parse_query(question: str) -> dict:
    """Extract a subject list, requested year and intent without a model call."""
    kind = _question_kind(question)
    text = unicodedata.normalize("NFKC", question).replace("’", "'")
    match = re.search(r"\bFY\s*[- ]?((?:19|20)\d{2})\b", text, re.I) or _YEAR.search(text)
    year = int(match[1]) if match else None

    # Geographic and role phrases are capitalized in some questions. An issuer
    # in the index is trusted only when it matches a full mention.
    candidates = _name_candidates(text)
    names = {}
    unknown = {}
    for candidate in candidates:
        key = company_key(candidate)
        if not key or key in _BROKERS:
            continue
        if key in _COMPANIES:
            names[key] = candidate
        elif not (set(key.split()) & {"quarter", "december", "january", "chief", "equity",
                                      "securities", "auditor", "research", "africa"}):
            unknown[key] = candidate

    normalized = normalize(text)
    for key, company in _COMPANIES.items():
        if f" {key} " in normalized and not any(
                key != candidate and f" {key} " in f" {candidate} " for candidate in unknown):
            names.setdefault(key, company)
    possessive = re.search(r"\b([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*)*)'s?\s", text)
    if possessive:
        candidate = possessive[1]
        key = company_key(candidate)
        if key and key not in _BROKERS:
            names.setdefault(key, candidate)
    if not names and not unknown:
        issuer = re.search(r"\b(?:did|does|is|for|at)\s+([A-Z][A-Za-z0-9.-]*)\b", text)
        if issuer and issuer[1].casefold() not in _STOP:
            names[company_key(issuer[1])] = issuer[1]
    comparison = bool(re.search(
        r"\bwhich\s+(?:company|firm|issuer|of)\b|\b(?:versus|vs\.?|compared (?:with|to))\b",
        text, re.I))
    if not names or comparison:
        for key, candidate in unknown.items():
            names.setdefault(key, candidate)
    return {"companies": list(names.values()), "fiscal_year": year, "doc_type": kind}


def build_index(chunks: list[Document]) -> Any:
    """Index only the current run's source chunks and their document metadata."""
    options = _OPTIONS
    company_names = {company_key(c.metadata["company"]): c.metadata["company"]
                     for c in chunks if c.metadata.get("company")}
    _COMPANIES.clear()
    _COMPANIES.update(company_names)
    _BROKERS.clear()
    _BROKERS.update(company_key(c.metadata["broker"]) for c in chunks if c.metadata.get("broker"))
    passages = [_passage(chunk) for chunk in chunks]
    available_years = defaultdict(set)
    for passage in passages:
        years = available_years[passage.company, passage.kind]
        if passage.year is not None:
            years.add(passage.year)
        years.update(passage.explicit_years)
    tokens = [_tokenize(p.chunk.page_content, options.normalized_tokens) or ["__empty__"]
              for p in passages]
    bm25 = BM25Okapi(tokens) if tokens else None
    return _Index(passages, bm25, options, company_names, dict(available_years))


def retrieve(index: Any, question: str) -> list[Document]:
    """Retrieve at most five unchanged source chunks with optional abstention."""
    if index.bm25 is None:
        return []
    options = index.options
    if not options.metadata_routing:
        scores = index.bm25.get_scores(_tokenize(question, options.normalized_tokens))
        best = sorted(range(len(index.passages)), key=lambda i: (-float(scores[i]), i))
        return [index.passages[i].chunk for i in best[:MAX_RETRIEVED]]

    parsed = parse_query(question)
    keys = list(dict.fromkeys(company_key(name) for name in parsed["companies"]))
    known_keys = [key for key in keys if key in index.companies]
    year, kind = parsed["fiscal_year"], parsed["doc_type"]
    if options.abstention and (len(known_keys) != len(keys) or (
            year is not None and known_keys and any(
                year not in index.available_years.get((key, kind), set()) for key in known_keys))):
        return []

    candidates = [i for i, p in enumerate(index.passages)
                  if p.kind == kind and (not known_keys or p.company in known_keys)]
    if year is not None:
        eligible = [i for i in candidates if _context_year(index.passages[i], year)]
        if eligible:
            candidates = eligible
    if not candidates and not options.abstention:
        candidates = list(range(len(index.passages)))
    if not candidates:
        return []

    question_terms = _query_terms(question, parsed)
    terms = set(question_terms)
    scores = index.bm25.get_scores(_tokenize(question, options.normalized_tokens))
    ranked = _rank(index, candidates, scores, terms, year)
    if options.abstention and terms:
        # A general, positive evidence test: shared words must support the
        # requested relation in one line (with its table caption where relevant).
        min_coverage = 1.0 if len(terms) <= 2 else 0.74
        supported = {index.passages[i].company for i, _, coverage in ranked
                     if coverage >= min_coverage}
        if not supported or any(key not in supported for key in known_keys):
            return []
        ranked = [item for item in ranked if item[2] >= min_coverage]
    if options.company_balancing and len(known_keys) > 1:
        ranked = _balanced_order(ranked, known_keys, index.passages)
    return [index.passages[i].chunk for i, _, _ in ranked[:MAX_RETRIEVED]]


# =============================================================================
# Provided code below. Do not change it.
# =============================================================================

LLM_MODEL = "deepseek-v4-flash"
DOC_TYPES = ("annual_report", "earnings_call", "research_report")

MAX_CHUNK_CHARS = 1500  # characters per chunk
MAX_RETRIEVED = 5       # chunks per question
TIME_LIMIT_SECONDS = 30 * 60

# USD per 1M tokens, only for the cost estimate that the runner prints.
PRICE_LLM_INPUT, PRICE_LLM_OUTPUT = 0.15, 0.60

# Points (Section 2 of the handout).
POINTS = {"1.1": 8, "1.2": 7, "2.1": 5, "2.2": 10, "2.3": 15, "3": 10, "4": 20}
RECALL_ZERO, RECALL_FULL = 0.50, 0.90  # Task 2.3: 0 points at or below RECALL_ZERO, full points from RECALL_FULL
MAX_LINE_CHARS = 400                   # Task 1.1
MIN_PROBES_FOUND = 0.9                 # Task 1.2


class UsageTracker(BaseCallbackHandler):
    """Counts LLM tokens, for the cost estimate. There is no budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = self.llm_input_tokens = self.llm_output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_in = tokens_out = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
        with self._lock:
            self.llm_calls += 1
            self.llm_input_tokens += tokens_in
            self.llm_output_tokens += tokens_out

    def cost(self) -> float:
        return (self.llm_input_tokens * PRICE_LLM_INPUT + self.llm_output_tokens * PRICE_LLM_OUTPUT) / 1e6

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens, "cost_usd": round(self.cost(), 4)}


USAGE = UsageTracker()            # your calls
GENERATOR_USAGE = UsageTracker()  # the generator's calls


def _chat(usage: UsageTracker, temperature: float, thinking: bool, max_tokens: int | None) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=LLM_MODEL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
        # Thinking mode ignores `temperature` and its tokens count as output.
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        callbacks=[usage],
    )


def make_llm(temperature: float = 0.0, thinking: bool = False, max_tokens: int | None = None) -> ChatDeepSeek:
    """deepseek-v4-flash. Use it for every LLM call you make (only in parse_query, build_index and retrieve)."""
    return _chat(USAGE, temperature, thinking, max_tokens)


# ----------------------------------------------------------------------------- the generator

GENERATOR_PROMPT = """You answer questions about company reports, using only the context below.

Context:
{context}

Question: {question}

Rules:
- Use only the context. The header of each passage says which document it comes from.
- Check that the passage you use is about the company and the fiscal year that the question asks about.
- If the context does not contain the answer, reply exactly NOT_FOUND.
- For a number, reply with that one number only, in the unit the question asks for, without units, words or other numbers (for example: 1913.4).
- For a name (a person, a company, a city, an audit firm or a rating), reply with the name only.

Answer:"""

_generator: ChatDeepSeek | None = None


def format_context(chunks: list[Document]) -> str:
    """The retrieved chunks as the generator sees them: a header from the metadata, then the text."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata or {}
        header = " | ".join(f"{key}: {str(meta[key])[:100]}"
                            for key in ("source", "company", "fiscal_year", "doc_type") if meta.get(key) not in (None, ""))
        parts.append(f"[{i}] {header}\n{chunk.page_content}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[Document]) -> str:
    """The fixed generator: deepseek-v4-flash answers from the retrieved chunks only."""
    global _generator
    if not chunks:
        return "NOT_FOUND"
    if _generator is None:
        _generator = _chat(GENERATOR_USAGE, 0.0, False, 64)
    prompt = GENERATOR_PROMPT.format(context=format_context(chunks), question=question)
    return str(_generator.invoke(prompt).content).strip()


# ----------------------------------------------------------------------------- loaders

def load_corpus(folder: str | Path) -> dict[str, str]:
    """{file name: raw HTML} for every .html file in the folder."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(folder).glob("*.html"))}


def load_json(path: str | Path) -> Any:
    """The parsed file, or None if it does not exist (the private set has no answers)."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- scoring

# One number: optional minus sign, optional thousands separators, optional decimals.
NUMBER_RE = re.compile(r"(?<![\d.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
TITLES = {"mr", "mrs", "ms", "dr", "prof", "professor", "sir"}
LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}


def parse_number(text: str) -> float | None:
    """The single number in the text, or None if there is not exactly one."""
    found = NUMBER_RE.findall(text.replace("\u2212", "-"))
    if len(found) != 1:
        return None
    return float(found[0].replace(",", ""))


def normalize(text: str) -> str:
    """Lower case, no punctuation, no titles, padded with spaces for whole-word matching."""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("_", " ")
    words = [w for w in re.sub(r"[^a-z0-9]+", " ", text).split() if w not in TITLES]
    return " " + " ".join(words) + " "


def company_key(name: Any) -> str:
    """A company name without case, punctuation or legal suffix: 'Northgate Robotics Inc.' -> 'northgate robotics'."""
    return " ".join(w for w in normalize(name).split() if w not in LEGAL_SUFFIXES)


def squash(text: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def as_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def answer_correct(answer: str, gold: dict) -> bool:
    answer = str(answer or "")
    said_not_found = " not found " in normalize(answer)
    if gold["type"] == "not_found":
        return said_not_found
    if said_not_found:
        return False
    if gold["type"] == "number":
        value = parse_number(answer)
        if value is None:
            return False
        return abs(value - gold["answer"]) <= max(0.01 * abs(gold["answer"]), 0.05)
    names = [gold["answer"], *gold.get("aliases", [])]
    if not any(normalize(name) in normalize(answer) for name in names):
        return False
    return not any(normalize(other) in normalize(answer) for other in gold.get("reject", []))


def table_score(output: Any, table: dict) -> float:
    """Task 1.1, one table: the share of cells found on a line with their row and column labels and
    without the label of another row, halved if the caption is missing."""
    if not isinstance(output, str) or not table["cells"]:
        return 0.0
    lines = [squash(line) for line in output.splitlines()]
    lines = [line for line in lines if line and len(line) <= MAX_LINE_CHARS]
    labels = {squash(row) for row, _, _ in table["cells"]}

    def kept(row: str, col: str, value: str) -> bool:
        row, col, value = squash(row), squash(col), squash(value)
        others = [o for o in labels if o not in row]  # e.g. "total liabilities" may be in "total liabilities and equity"
        return any(row in line and col in line and value in line and not any(o in line for o in others)
                   for line in lines)

    share = sum(kept(*cell) for cell in table["cells"]) / len(table["cells"])
    return share if squash(table["caption"]) in squash(output) else share / 2


def document_problems(chunks: Any, error: str | None, meta: dict, file_name: str) -> list[str]:
    """Task 1.2, one document: why its chunks fail the checks (an empty list means they pass)."""
    if error:
        return [f"error: {error}"]
    if not isinstance(chunks, list) or not chunks:
        return ["no chunks"]
    if not all(isinstance(c, Document) for c in chunks):
        return ["not a list of Document"]
    problems = []
    longest = max(len(c.page_content) for c in chunks)
    if longest > MAX_CHUNK_CHARS:
        problems.append(f"a chunk has {longest} characters")
    for key, ok in (("source", lambda v: v == file_name),
                    ("company", lambda v: company_key(v) == company_key(meta["company"])),
                    ("fiscal_year", lambda v: as_year(v) == meta["fiscal_year"]),
                    ("doc_type", lambda v: v == meta["doc_type"])):
        wrong = [c.metadata.get(key) for c in chunks if not ok(c.metadata.get(key))]
        if wrong:
            problems.append(f"{key} {wrong[0]!r} (expected {meta.get(key, file_name)!r}) on {len(wrong)} chunk(s)")
    text = squash("\n".join(c.page_content for c in chunks))
    missing = [p for p in meta["probes"] if squash(p) not in text]
    if len(missing) > (1 - MIN_PROBES_FOUND) * len(meta["probes"]):
        problems.append(f"text missing from the chunks: {missing[:3]}")
    return problems


def parse_score(parsed: Any, gold: dict) -> float:
    """Task 2.2, one question: the share of the three fields that are right."""
    if not isinstance(parsed, dict):
        return 0.0
    expected = gold["query"]
    companies = parsed.get("companies")
    if isinstance(companies, str):
        companies = [companies]
    right = 0
    if isinstance(companies, (list, tuple)) and {company_key(c) for c in companies} == {company_key(c) for c in expected["companies"]}:
        right += 1
    if "fiscal_year" in parsed and as_year(parsed["fiscal_year"]) == expected["fiscal_year"] and \
            (parsed["fiscal_year"] is None) == (expected["fiscal_year"] is None):
        right += 1
    if parsed.get("doc_type") == expected["doc_type"]:
        right += 1
    return right / 3


def contains(chunk_text: str, printed: str) -> bool:
    """The fact as printed in the document is in the chunk (a number only as a whole number)."""
    text, printed = squash(chunk_text), squash(printed)
    if re.fullmatch(r"[\d,.%-]+", printed):
        return re.search(rf"(?<![\d.,]){re.escape(printed)}(?![\d]|[.,]\d)", text) is not None
    return printed in text


def retrieval_hit(chunks: list[Document], gold: dict) -> bool:
    """For every fact the question needs, a retrieved chunk of a document that states it contains it."""
    return bool(gold["evidence"]) and all(
        any(c.metadata.get("source") == file and contains(c.page_content, printed)
            for c in chunks for file, printed in group)
        for group in gold["evidence"])


def clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------------- runner

def main() -> None:
    parser = argparse.ArgumentParser(description="FTEC5660 HW4: retrieval for RAG over company reports")
    parser.add_argument("--data-folder", default="public_test",
                        help="folder with corpus/, questions.json and (optionally) the answer files")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = load_corpus(folder / "corpus")
    questions = load_json(folder / "questions.json")
    ground_truth = load_json(folder / "ground_truth.json")
    docs_meta = load_json(folder / "docs.json")
    tables = load_json(folder / "tables.json")
    print(f"{len(corpus)} documents, {len(questions)} questions")
    start = time.perf_counter()
    points: dict[str, float] = {}

    # Task 1.1: tables (no model calls allowed)
    if tables:
        before = USAGE.llm_calls
        scores = []
        for table in tables:
            try:
                scores.append(table_score(table_to_text(table["html"]), table))
            except Exception as e:
                print(f"table_to_text failed on a table of {table['source']}: {e!r}")
                scores.append(0.0)
        if USAGE.llm_calls != before:
            print("WARNING: table_to_text called the LLM, which is not allowed (Task 1.1 = 0).")
            scores = [0.0]
        points["1.1"] = POINTS["1.1"] * sum(scores) / len(scores)
        print(f"\nTask 1.1 table_to_text: {sum(scores) / len(scores):.1%} of the table cells kept with their labels")

    # Task 1.2: chunks (no model calls allowed)
    before = USAGE.llm_calls
    chunks_by_file: dict[str, list] = {}
    errors: dict[str, str] = {}
    for file_name, html in corpus.items():
        try:
            chunks_by_file[file_name] = list(chunk_document(file_name, html) or [])
        except Exception as e:
            traceback.print_exc()
            chunks_by_file[file_name], errors[file_name] = [], repr(e)
    called_models = USAGE.llm_calls != before
    if called_models:
        print("WARNING: chunk_document called the LLM, which is not allowed (Task 1.2 = 0).")
    if docs_meta:
        problems = {f: document_problems(chunks_by_file.get(f), errors.get(f), m, f) for f, m in docs_meta.items()}
        passed = sum(not p for p in problems.values()) / len(problems)
        points["1.2"] = 0.0 if called_models else POINTS["1.2"] * passed
        print(f"Task 1.2 chunk_document: {passed:.1%} of the documents pass")
        for f, p in [(f, p) for f, p in problems.items() if p][:5]:
            print(f"  {f}: {'; '.join(p)}")

    # the chunks that retrieve() may return
    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, Document) and len(c.page_content) <= MAX_CHUNK_CHARS]
    by_text: dict[str, list[Document]] = {}
    for c in all_chunks:
        by_text.setdefault(c.page_content, []).append(c)
    print(f"{len(all_chunks)} chunks to index")

    # Task 2.2: query parsing
    parsed: dict[str, Any] = {}
    for qid, question in questions.items():
        try:
            parsed[qid] = parse_query(question)
        except Exception as e:
            parsed[qid] = f"error: {e!r}"

    # Tasks 2.1, 2.3 and 3: retrieval
    index = build_index(all_chunks)
    index_seconds = time.perf_counter() - start
    print(f"Index built in {index_seconds:.1f}s | {USAGE.summary()}")
    retrieved: dict[str, list[Document]] = {}
    valid: dict[str, bool] = {}
    for qid, question in questions.items():
        try:
            result = retrieve(index, question)
            ok = isinstance(result, (list, tuple)) and len(result) <= MAX_RETRIEVED
            kept = []
            for chunk in list(result or []) if isinstance(result, (list, tuple)) else []:
                matches = by_text.get(getattr(chunk, "page_content", None), [])
                if not matches:
                    ok = False
                    continue
                source = (getattr(chunk, "metadata", None) or {}).get("source")
                kept.append(next((m for m in matches if m.metadata.get("source") == source), matches[0]))
            retrieved[qid], valid[qid] = kept[:MAX_RETRIEVED], ok
        except Exception:
            traceback.print_exc()
            retrieved[qid], valid[qid] = [], False
    retrieval_seconds = time.perf_counter() - start - index_seconds

    # Task 4: the fixed generator
    def answer(qid: str) -> str:
        try:
            return generate(questions[qid], retrieved[qid])
        except Exception as e:
            print(f"Generator failed on {qid}: {e!r}")
            return ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(answer, questions)))
    total_seconds = time.perf_counter() - start

    rows = []
    for qid, question in questions.items():
        sources = sorted({c.metadata.get("source", "") for c in retrieved[qid]})
        row = {"id": qid, "category": "", "question": question, "parsed": json.dumps(parsed[qid], default=str),
               "parse_score": "", "retrieved": ";".join(sources), "valid": valid[qid], "retrieval_hit": "",
               "answer": answers[qid], "correct": "", "expected": ""}
        if ground_truth and qid in ground_truth:
            gold = ground_truth[qid]
            row["category"] = gold["category"]
            row["parse_score"] = round(parse_score(parsed[qid], gold), 3)
            row["retrieval_hit"] = retrieval_hit(retrieved[qid], gold) if gold["evidence"] else ""
            row["correct"] = answer_correct(answers[qid], gold)
            if not row["correct"]:
                row["expected"] = gold["answer"]
        rows.append(row)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")

    if ground_truth:
        graded = [r for r in rows if r["id"] in ground_truth]
        answerable = [r for r in graded if ground_truth[r["id"]]["evidence"]]
        unanswerable = [r for r in graded if not ground_truth[r["id"]]["evidence"]]
        parse_mean = sum(r["parse_score"] for r in graded) / len(graded)
        non_empty = sum(r["valid"] and bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        recall = sum(r["retrieval_hit"] for r in answerable) / max(len(answerable), 1)
        kept = sum(bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        rejected = sum(not r["retrieved"] for r in unanswerable) / max(len(unanswerable), 1)
        accuracy = sum(r["correct"] for r in graded) / len(graded)
        points["2.1"] = POINTS["2.1"] * non_empty
        points["2.2"] = POINTS["2.2"] * parse_mean
        points["2.3"] = POINTS["2.3"] * clip((recall - RECALL_ZERO) / (RECALL_FULL - RECALL_ZERO))
        points["3"] = POINTS["3"] * clip(kept + rejected - 1)
        points["4"] = POINTS["4"] * accuracy
        print(f"Task 2.1 valid, non-empty retrieval on answerable questions: {non_empty:.1%}")
        print(f"Task 2.2 parse_query fields right: {parse_mean:.1%}")
        print(f"Task 2.3 retrieval recall@{MAX_RETRIEVED} on answerable questions: {recall:.1%}")
        print(f"Task 3   answerable questions with chunks: {kept:.1%} | unanswerable questions with []: {rejected:.1%}")
        print(f"Task 4   answer accuracy: {accuracy:.1%}")
        by_category: dict[str, list[tuple]] = {}
        for r in graded:
            by_category.setdefault(r["category"], []).append((r["correct"], r["retrieval_hit"]))
        print("\nBy category       accuracy   recall")
        for category, results in sorted(by_category.items()):
            hits = [h for _, h in results if h != ""]
            recall_text = f"{sum(hits) / len(hits):.1%}" if hits else "-"
            print(f"  {category:<14}{sum(c for c, _ in results) / len(results):>9.1%}{recall_text:>9}")

    if points:
        print("\nEstimated points (the grade uses the private set):")
        for task in POINTS:
            if task in points:
                print(f"  Task {task:<4}{points[task]:6.2f} / {POINTS[task]}")
        print(f"  Total    {sum(points.values()):6.2f} / {sum(POINTS[t] for t in points)}")

    print(f"\nYour usage: {USAGE.summary()}")
    print(f"Generator usage: {GENERATOR_USAGE.summary()}")
    print(f"Time: {total_seconds:.1f}s (index {index_seconds:.1f}s, retrieval {retrieval_seconds:.1f}s)")
    if total_seconds > TIME_LIMIT_SECONDS:
        print(f"WARNING: the run took longer than the {TIME_LIMIT_SECONDS // 60}-minute limit.")


if __name__ == "__main__":
    main()
