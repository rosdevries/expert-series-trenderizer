"""Re-fetches email domains from ON24 and restores company + topic data locally."""
import hashlib
import pandas as pd
from trenderizer.store import read_table, upsert_questions, load_company_domains
from trenderizer.on24_client import ON24Client
from trenderizer.ingest import _load_siemens_config, _first, _is_siemens

questions = read_table("questions")
topic_map = dict(zip(questions["question_id"], questions["topic_label"]))
topic_raw_map = dict(zip(questions["question_id"], questions["topic_label_raw"]))
print(f"Saved {len(topic_map)} topic labels")

company_cache = load_company_domains()
print(f"Company cache: {len(company_cache)} domains")

siemens_domains, siemens_patterns = _load_siemens_config()
events = read_table("events")
client = ON24Client.from_env()

domain_map = {}
siem_map = {}

for _, ev_row in events.iterrows():
    event_id = int(ev_row["event_id"])
    attendees = client.get_attendees(event_id)
    for att in attendees:
        uid = int(_first(att, "eventuserid", "eventUserId") or 0)
        email = str(_first(att, "email", "emailAddress") or "").strip()
        company_name = str(_first(att, "company", "companyName") or "").strip()
        domain = email.split("@")[-1].strip().lower() if "@" in email else ""
        is_siem = _is_siemens(email, company_name, siemens_domains, siemens_patterns)
        raw_qs = _first(att, "questions") or []
        if not isinstance(raw_qs, list):
            continue
        for q in raw_qs:
            if not isinstance(q, dict):
                continue
            content_raw = str(_first(q, "content", "question", "text") or "").strip()
            if not content_raw:
                continue
            raw_id = _first(q, "qaid", "qaId", "userquestionid", "questionid", "id")
            if raw_id:
                qid = int(raw_id)
            else:
                h = hashlib.sha256(f"{event_id}:{uid}:{content_raw}".encode()).hexdigest()[:15]
                qid = int(h, 16)
            domain_map[qid] = domain
            siem_map[qid] = is_siem

print(f"Fetched domains for {len(domain_map)} questions")


def _company(domain):
    if not domain:
        return None
    info = company_cache.get(domain)
    if isinstance(info, dict):
        return info.get("company")
    return str(info) if info else None


questions = questions.copy()
questions["email_domain"] = questions["question_id"].map(domain_map)
questions["inferred_company"] = questions["email_domain"].map(_company)
questions["topic_label"] = questions["question_id"].map(topic_map)
questions["topic_label_raw"] = questions["question_id"].map(topic_raw_map)

domains_recovered = int(questions["email_domain"].notna().sum())
companies_recovered = int(questions["inferred_company"].notna().sum())
print(f"Domains recovered: {domains_recovered}")
print(f"Companies recovered: {companies_recovered}")

upsert_questions(questions)
print("Done.")
