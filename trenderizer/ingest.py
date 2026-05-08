"""
Expert Series Trenderizer — ingest pipeline.

Four-phase pipeline (per spec §3 Phase 1 + Phase 2):
  1. Discover in-scope events via ON24 event listing + tag filter
  2. For each event: fetch registrants (domain map) and Q&A (questions)
  3. Write raw data to partitioned Parquet store
  4. Enrich: product inference (event-level), company inference (domain-level, batched),
     topic assignment (question-level)

Usage:
    python -m trenderizer.ingest           # fetch last 30 days
    python -m trenderizer.ingest --days 90
    python -m trenderizer.ingest --backfill  # fetch last 365 days
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from .config import (
    TARGET_TAGS, BACKFILL_DAYS, DEFAULT_FETCH_DAYS,
    ON24_CLIENT_ID, VOCAB_DIR,
)
from .on24_client import ON24Client, ON24APIError, parse_dt
from .store import (
    load_manifest, save_manifest, log_run,
    read_table, upsert_events, upsert_questions, upsert_registrants,
    load_company_domains, save_company_domains,
    load_topics, save_topics,
)
from .enrichment import (
    infer_products, infer_companies, assign_topics, redact_pii,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_siemens_config() -> tuple[set[str], list[str]]:
    path = VOCAB_DIR / "siemens_domains.json"
    with open(path) as f:
        data = json.load(f)
    return set(data.get("domains", [])), [p.lower() for p in data.get("company_patterns", [])]


def _is_siemens(email: str, company: str, domains: set[str], patterns: list[str]) -> bool:
    if email and "@" in email:
        domain = email.split("@")[-1].strip().lower()
        if domain in domains:
            return True
    if company:
        cl = company.lower()
        if any(p in cl for p in patterns):
            return True
    return False


def _first(record: dict, *keys: str, default=None):
    for k in keys:
        v = record.get(k)
        if v not in (None, ""):
            return v
    return default


def _classify_series(tags: list[str]) -> str:
    tl = [t.lower() for t in tags]
    if any("expert" in t for t in tl):
        return "Expert Series"
    if any("lunch" in t for t in tl):
        return "Lunch and Learn"
    return "Other"


def _stable_qid(event_id: int, content: str) -> int:
    """Generate a stable integer question ID from content hash when ON24 provides none."""
    h = hashlib.sha256(f"{event_id}:{content}".encode()).hexdigest()[:15]
    return int(h, 16)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_ingest(days_back: int = DEFAULT_FETCH_DAYS, trigger: str = "cli") -> dict:
    t_start = datetime.now(timezone.utc)
    log.info(f"[Trenderizer] Ingest started — last {days_back} days (trigger={trigger})")

    siemens_domains, siemens_patterns = _load_siemens_config()
    manifest = load_manifest()
    company_cache = load_company_domains()
    canonical_topics = load_topics()

    with open(VOCAB_DIR / "products.json") as f:
        products_vocab = json.load(f)["products"]

    try:
        client = ON24Client.from_env()
    except ON24APIError as exc:
        log.error(str(exc))
        return {"error": str(exc)}

    now = t_start
    start_date = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")
    end_date = now.strftime("%Y-%m-%d")

    # ── Phase 1: Discover events ───────────────────────────────────────────────
    log.info(f"Fetching events {start_date} → {end_date}")
    try:
        raw_events = client.fetch_events(start_date, end_date)
    except ON24APIError as exc:
        log.error(f"Event fetch failed: {exc}")
        return {"error": str(exc)}

    in_scope: list[dict] = []
    almost_included: list[dict] = []

    for ev in raw_events:
        if ev.get("istestevent"):
            continue
        event_tags = set(ev.get("tags") or [])
        if event_tags & TARGET_TAGS:
            in_scope.append(ev)
        else:
            title_lower = (ev.get("description") or "").lower()
            if any(kw in title_lower for kw in ("expert series", "lunch and learn", "lunch & learn")):
                almost_included.append(ev)

    log.info(f"In-scope: {len(in_scope)} events | Almost-included (tag missing): {len(almost_included)}")
    if almost_included:
        log.info("Events that matched by title but lacked the required tag:")
        for ev in almost_included:
            log.info(f"  [{ev.get('eventid')}] {ev.get('description', '')}")

    # ── Phase 2: Per-event fetch ───────────────────────────────────────────────
    existing_qdf = read_table("questions")
    existing_qids: set[int] = (
        set(existing_qdf["question_id"].tolist())
        if not existing_qdf.empty and "question_id" in existing_qdf.columns
        else set()
    )

    all_event_rows: list[dict] = []
    all_question_rows: list[dict] = []
    all_reg_rows: list[dict] = []
    known_event_ids = {int(k) for k in manifest.get("events", {})}

    new_events = new_questions = 0

    for ev in in_scope:
        event_id = int(_first(ev, "eventid", "eventId", "id") or 0)
        if not event_id:
            continue

        title = str(_first(ev, "description", "title") or "Untitled")
        tags: list[str] = list(ev.get("tags") or [])
        start_ts = parse_dt(_first(ev, "goodafter", "livestart", "liveStart"))

        cached_product = manifest.get("events", {}).get(str(event_id), {}).get("inferred_product")

        all_event_rows.append({
            "event_id": event_id,
            "client_id": int(ON24_CLIENT_ID),
            "title": title,
            "start_ts_pacific": start_ts,
            "tags": json.dumps(tags),
            "inferred_product": cached_product,
            "series": _classify_series(tags),
            "last_fetched_at": now,
        })

        if event_id not in known_event_ids:
            new_events += 1

        # Registrants → domain map
        log.info(f"  [{event_id}] {title[:60]} — fetching registrants")
        regs = client.get_registrants(event_id)
        domain_by_user: dict[int, str] = {}
        company_by_user: dict[int, str] = {}

        for reg in regs:
            uid = int(_first(reg, "eventuserid", "eventUserId") or 0)
            email = str(_first(reg, "email", "emailAddress") or "").strip()
            company = str(_first(reg, "company", "companyName") or "").strip()
            domain = email.split("@")[-1].strip().lower() if "@" in email else ""

            if uid:
                domain_by_user[uid] = domain
                company_by_user[uid] = company

            is_siem = _is_siemens(email, company, siemens_domains, siemens_patterns)
            reg_ts = parse_dt(_first(reg, "createdon", "registrationDate", "created")) or start_ts

            all_reg_rows.append({
                "event_user_id": uid,
                "event_id": event_id,
                "email_domain": domain,
                "inferred_company": None,
                "created_ts_pacific": reg_ts,
            })

        # Questions — extracted from attendee records.
        # The /qanda endpoint is real-time only; historical question text lives
        # inside each attendee record under attendee["questions"][]["content"].
        log.info(f"  [{event_id}] fetching attendees (for Q&A)")
        attendees = client.get_attendees(event_id)

        for att in attendees:
            uid = int(_first(att, "eventuserid", "eventUserId") or 0)
            email = str(_first(att, "email", "emailAddress") or "").strip()
            company = str(_first(att, "company", "companyName") or "").strip()
            domain = domain_by_user.get(uid) or (email.split("@")[-1].strip().lower() if "@" in email else "")
            is_siem = _is_siemens(email, company, siemens_domains, siemens_patterns)
            if not is_siem and domain:
                is_siem = domain in siemens_domains

            # Each attendee may have multiple questions
            raw_questions = _first(att, "questions") or []
            if not isinstance(raw_questions, list):
                continue

            for q in raw_questions:
                if not isinstance(q, dict):
                    continue
                content_raw = str(_first(q, "content", "question", "text") or "").strip()
                if not content_raw:
                    continue

                raw_id = _first(q, "qaid", "qaId", "userquestionid", "questionid", "id")
                qid = int(raw_id) if raw_id else _stable_qid(event_id, f"{uid}:{content_raw}")

                if qid in existing_qids:
                    continue

                content = redact_pii(content_raw)
                created_ts = parse_dt(_first(q, "createdon", "createdOn", "timestamp", "created")) or start_ts

                all_question_rows.append({
                    "question_id": qid,
                    "event_id": event_id,
                    "event_user_id": uid,
                    "created_ts_pacific": created_ts,
                    "content": content,
                    "email_domain": domain,
                    "inferred_company": None,
                    "topic_label": None,
                    "topic_label_raw": None,
                    "is_siemens_employee": is_siem,
                    "enriched_at": None,
                })
                existing_qids.add(qid)
                new_questions += 1

        manifest.setdefault("events", {})[str(event_id)] = {
            "title": title,
            "inferred_product": cached_product,
            "last_fetched_at": now.isoformat(),
        }
        time.sleep(0.2)

    # ── Phase 3: Write raw data ────────────────────────────────────────────────
    log.info(f"Writing raw data: {len(all_event_rows)} events, {len(all_question_rows)} questions")
    if all_event_rows:
        upsert_events(pd.DataFrame(all_event_rows))
    if all_question_rows:
        upsert_questions(pd.DataFrame(all_question_rows))
    if all_reg_rows:
        upsert_registrants(pd.DataFrame(all_reg_rows))

    # ── Phase 4: Enrich ────────────────────────────────────────────────────────

    # 4a. Product inference for unenriched events
    events_df = read_table("events")
    if not events_df.empty and "inferred_product" in events_df.columns:
        unenriched_events = events_df[events_df["inferred_product"].isna()].to_dict("records")
        if unenriched_events:
            log.info(f"Inferring product for {len(unenriched_events)} event(s)")
            products_map = infer_products(unenriched_events, products_vocab)
            for eid, product in products_map.items():
                events_df.loc[events_df["event_id"] == eid, "inferred_product"] = product
                manifest["events"].setdefault(str(eid), {})["inferred_product"] = product
            enriched_mask = events_df["event_id"].isin(products_map.keys())
            upsert_events(events_df[enriched_mask])

    # 4b. Company inference (batch all new domains)
    questions_df = read_table("questions")
    if not questions_df.empty:
        customer_q = questions_df[~questions_df["is_siemens_employee"].fillna(False)]
        new_domains = [
            d for d in customer_q["email_domain"].dropna().unique()
            if d and d not in company_cache
        ]
        if new_domains:
            log.info(f"Inferring company for {len(new_domains)} new domain(s)")
            new_mappings = infer_companies(new_domains)
            company_cache.update(new_mappings)
            save_company_domains(company_cache)

        def _company(domain: str) -> str | None:
            if not domain:
                return None
            info = company_cache.get(domain)
            if isinstance(info, dict):
                return info.get("company")
            return str(info) if info else None

        # Apply company to unenriched question rows
        needs_company = questions_df["inferred_company"].isna()
        if needs_company.any():
            questions_df.loc[needs_company, "inferred_company"] = (
                questions_df.loc[needs_company, "email_domain"].map(_company)
            )
            upsert_questions(questions_df[needs_company])

        # 4c. Topic assignment for unenriched customer questions with content
        needs_topic = (
            questions_df["topic_label"].isna()
            & ~questions_df["is_siemens_employee"].fillna(False)
            & questions_df["content"].notna()
            & (questions_df["content"].str.strip() != "")
        )

        if needs_topic.any():
            to_enrich = questions_df[needs_topic].copy()

            # Join product context from events
            if not events_df.empty and "inferred_product" in events_df.columns:
                product_map = events_df.set_index("event_id")["inferred_product"].to_dict()
                to_enrich = to_enrich.assign(
                    product_context=to_enrich["event_id"].map(product_map).fillna("Unknown")
                )
            else:
                to_enrich = to_enrich.assign(product_context="Unknown")

            records = to_enrich[["question_id", "content", "product_context"]].to_dict("records")
            log.info(f"Assigning topics for {len(records)} question(s)")
            enriched = assign_topics(records, canonical_topics)

            # Update topic vocab with new labels
            new_labels = {r["topic_label"] for r in enriched} - set(canonical_topics)
            if new_labels:
                log.info(f"Adding {len(new_labels)} new canonical topic(s): {sorted(new_labels)}")
                canonical_topics = sorted(set(canonical_topics) | new_labels)
                save_topics(canonical_topics)

            # Apply back to questions_df
            topic_map = {r["question_id"]: r["topic_label"] for r in enriched}
            questions_df.loc[needs_topic, "topic_label"] = questions_df.loc[needs_topic, "question_id"].map(topic_map)
            questions_df.loc[needs_topic, "topic_label_raw"] = questions_df.loc[needs_topic, "question_id"].map(topic_map)
            questions_df.loc[needs_topic, "enriched_at"] = now
            upsert_questions(questions_df[needs_topic])

    # ── Finalize ───────────────────────────────────────────────────────────────
    manifest["last_run_at"] = now.isoformat()
    save_manifest(manifest)

    duration = round((datetime.now(timezone.utc) - t_start).total_seconds(), 1)
    run_record = {
        "timestamp": now.isoformat(),
        "trigger": trigger,
        "days_back": days_back,
        "new_events": new_events,
        "new_questions": new_questions,
        "duration_s": duration,
    }
    log_run(run_record)
    log.info(f"[Trenderizer] Done in {duration}s — {new_events} new events, {new_questions} new questions")
    return run_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Expert Series Trenderizer ingest pipeline")
    parser.add_argument("--days", type=int, default=DEFAULT_FETCH_DAYS,
                        help="Days of events to fetch (default: %(default)s)")
    parser.add_argument("--backfill", action="store_true",
                        help=f"Fetch the last {BACKFILL_DAYS} days (full historical backfill)")
    args = parser.parse_args()
    days = BACKFILL_DAYS if args.backfill else args.days
    result = run_ingest(days_back=days)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
