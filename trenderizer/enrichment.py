"""
Haiku-powered enrichment: product inference, company inference, topic labels.

All results are committed to the data store (Parquet + vocab JSONs) so that
re-running the pipeline produces zero LLM calls when data is already enriched.

Three prompts (per spec §2.5):
  - Product inference (per event)      → events.parquet.inferred_product
  - Company inference (per domain)     → vocab/company_domains.json
  - Topic summarization (per question) → questions.parquet.topic_label
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

log = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")

PERSONAL_DOMAINS: frozenset[str] = frozenset({
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "live.com",
    "msn.com", "aol.com", "icloud.com", "me.com", "mac.com",
    "protonmail.com", "proton.me", "zoho.com", "ymail.com",
    "yahoo.co.in", "yahoo.co.uk", "yahoo.com.au",
})


def redact_pii(text: str) -> str:
    """Regex-based PII redaction: strips emails and phone numbers."""
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    return text.strip()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _make_client():
    from anthropic import Anthropic
    from .config import ANTHROPIC_API_KEY
    return Anthropic(api_key=ANTHROPIC_API_KEY)


def _call_haiku(system: str, user: str, max_tokens: int = 512) -> str:
    from .config import HAIKU_MODEL
    client = _make_client()
    resp = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text.strip()


def _parse_json(text: str) -> dict | list:
    """Extract JSON from a response that may have surrounding prose or fences."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip markdown fences
    stripped = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    stripped = re.sub(r"\s*```$", "", stripped.strip())
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # Find first {...} or [...]
    for pattern in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    raise ValueError(f"No JSON found in response: {text[:300]}")


# ── Product inference (per event) ─────────────────────────────────────────────

def infer_products(events: list[dict], products_vocab: list[dict]) -> dict[int, str]:
    """
    Infer the canonical Siemens EDA product category for each event.
    Returns {event_id: canonical_label}.
    One Haiku call per event; callers should only pass unenriched events.
    """
    canonical_labels = [p["canonical"] for p in products_vocab]
    products_summary = "\n".join(
        f'- "{p["canonical"]}": {p["category_description"]}. '
        f'Key products: {", ".join((p.get("l1_products") or [])[:3])}. '
        f'Aliases: {", ".join((p.get("aliases") or [])[:5]) or "none"}.'
        for p in products_vocab
    )

    system = (
        "You are a Siemens EDA product expert. Given a webinar title and tag list, "
        "identify which canonical product category it belongs to. "
        "You MUST pick from the provided closed list — never invent a new label. "
        'Respond ONLY with valid JSON: {"product": "<label>", "confidence": "high|medium|low"}'
    )

    result: dict[int, str] = {}
    for event in events:
        event_id = int(event["event_id"])
        title = event.get("title", "Untitled")
        tags_raw = event.get("tags", "[]")
        try:
            tags = json.loads(tags_raw) if isinstance(tags_raw, str) else list(tags_raw or [])
        except (json.JSONDecodeError, TypeError):
            tags = []

        user = (
            f"Webinar title: {title}\n"
            f"Tags: {', '.join(tags) or 'none'}\n\n"
            f"Canonical product categories:\n{products_summary}\n\n"
            f'Pick the single best label, or "Unknown / Cross-cutting" if ambiguous.'
        )

        for attempt in range(2):
            try:
                text = _call_haiku(system, user)
                parsed = _parse_json(text)
                if isinstance(parsed, dict):
                    product = parsed.get("product", "Unknown / Cross-cutting")
                else:
                    product = "Unknown / Cross-cutting"
                if product not in canonical_labels:
                    if attempt == 0:
                        user += f"\n\nYou MUST use one of exactly: {json.dumps(canonical_labels)}"
                        continue
                    product = "Unknown / Cross-cutting"
                result[event_id] = product
                log.info(f"  Product for event {event_id}: {product}")
                break
            except Exception as exc:
                log.warning(f"  Product inference failed for event {event_id}: {exc}")
                result[event_id] = "Unknown / Cross-cutting"
                break

    return result


# ── Company inference (batch, per domain) ─────────────────────────────────────

_COMPANY_BATCH_SIZE = 25


def _infer_companies_batch(domains: list[str]) -> dict[str, dict]:
    """Infer company names for a single batch of ≤25 domains."""
    system = (
        "You are a business intelligence assistant. "
        "Given a list of email domains, return the canonical company name for each one. "
        "For personal email providers (gmail, yahoo, hotmail, outlook, etc.) use 'Personal / Unknown' "
        "and set is_personal to true. "
        "For corporate/academic/government domains, provide the proper organization name "
        "(e.g. intel.com → Intel Corporation, samsung.co.kr → Samsung Electronics). "
        "Respond ONLY with valid JSON: "
        '{"domain1": {"company": "Name", "is_personal": false}, ...}'
    )

    user = (
        "Classify these email domains:\n"
        + "\n".join(f"- {d}" for d in sorted(domains))
    )

    fallback = {
        d: {"company": "Personal / Unknown" if d in PERSONAL_DOMAINS else d, "is_personal": d in PERSONAL_DOMAINS}
        for d in domains
    }

    try:
        text = _call_haiku(system, user, max_tokens=1024)
        parsed = _parse_json(text)
        if not isinstance(parsed, dict):
            log.warning(f"Company inference returned non-dict; using fallback for batch of {len(domains)}")
            return fallback
        result: dict[str, dict] = {}
        for domain in domains:
            info = parsed.get(domain)
            if isinstance(info, dict):
                result[domain] = {
                    "company": str(info.get("company", domain)),
                    "is_personal": bool(info.get("is_personal", domain in PERSONAL_DOMAINS)),
                }
            elif isinstance(info, str):
                result[domain] = {"company": info, "is_personal": domain in PERSONAL_DOMAINS}
            else:
                result[domain] = fallback[domain]
        return result
    except Exception as exc:
        log.warning(f"Batch company inference failed: {exc}")
        return fallback


def infer_companies(domains: list[str]) -> dict[str, dict]:
    """
    Batch-infer company names for a list of email domains.
    Chunks into batches of ≤25 to stay within Haiku's output token budget.
    Returns {domain: {"company": str, "is_personal": bool}}.
    """
    if not domains:
        return {}

    result: dict[str, dict] = {}
    for i in range(0, len(domains), _COMPANY_BATCH_SIZE):
        batch = domains[i: i + _COMPANY_BATCH_SIZE]
        log.info(f"  Company inference batch {i // _COMPANY_BATCH_SIZE + 1}: {len(batch)} domain(s)")
        result.update(_infer_companies_batch(batch))
    return result


# ── Topic assignment (per question) ──────────────────────────────────────────

def assign_topics(questions: list[dict], canonical_topics: list[str]) -> list[dict]:
    """
    Assign a canonical topic label to each question.
    Each question dict must have 'question_id', 'content', and optionally 'product_context'.
    Returns the input list with 'topic_label' and 'topic_label_raw' added.
    """
    if not questions:
        return []

    canonical_json = json.dumps(sorted(canonical_topics)[:60]) if canonical_topics else "[]"

    system = (
        "You are a technical marketing analyst for Siemens EDA. "
        "Given a customer question from a webinar, produce a terse (1–4 word) topic label. "
        "Rules:\n"
        "- Be technical and specific (e.g. 'HB-IJTAG', 'PEX / Parasitic Extraction', "
        "'Power-aware ATPG', 'Formal Verification', 'DFT Coverage', 'Timing Closure').\n"
        "- The same underlying concept must always map to the same label.\n"
        "- Prefer an existing canonical label if one fits; otherwise create a new concise label.\n"
        'Respond ONLY with valid JSON: {"topic": "<label>"}'
    )

    results = []
    for q in questions:
        content = str(q.get("content", "")).strip()
        product = str(q.get("product_context", "Unknown"))

        if not content:
            results.append({**q, "topic_label": "General / Other", "topic_label_raw": "General / Other"})
            continue

        user = (
            f"Product context: {product}\n"
            f"Question: {content}\n\n"
            f"Existing canonical topics (prefer one if it fits):\n{canonical_json}\n\n"
            "Return an existing topic label if the question maps to one, otherwise a new concise label."
        )

        try:
            text = _call_haiku(system, user)
            parsed = _parse_json(text)
            topic = str(parsed.get("topic", "General / Other")) if isinstance(parsed, dict) else "General / Other"
        except Exception as exc:
            log.warning(f"Topic inference failed for question {q.get('question_id')}: {exc}")
            topic = "General / Other"

        results.append({**q, "topic_label": topic, "topic_label_raw": topic})

    return results
