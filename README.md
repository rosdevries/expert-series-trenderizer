# Expert Series Trenderizer

A Streamlit dashboard that surfaces topic trends, rising questions, and company engagement across all Siemens EDA Expert Series and Lunch & Learn webinars — powered by ON24 Q&A data enriched with Claude AI.

## What it does

The app pulls every attendee question from ON24, uses Claude Haiku to infer the product covered and the topic of each question, and presents the results across four interactive tabs.

| Tab | What you see |
|---|---|
| **Top Topics** | Bar chart of the most-asked topic categories, filterable by product, series, company, and date range |
| **Rising Topics** | Topics gaining share in the recent window versus the prior period — click a row to see the underlying questions |
| **Companies** | Per-company topic breakdown and trend lines, showing which customers are asking about what |
| **Events Explorer** | All webinars sorted by customer question volume — click an event to see every question asked, with topic and company |

Data is refreshed automatically every Monday via a GitHub Actions workflow and committed back to the repository. The Streamlit Cloud deployment reads directly from these Parquet files, so no database or server is required.

---

## Accessing the app

**URL:** https://expert-series-trenderizer.streamlit.app/

Password is shared internally — ask the team.

---

## Top Topics tab

Answers: *what are attendees asking about, and how has that changed over time?*

- Filter by **Product**, **Company**, **Series** (Expert Series / Lunch & Learn), and **Date range**
- Bar chart ranks topic categories by question count
- Line chart shows how each topic's share of questions has evolved month by month

## Rising Topics tab

Answers: *what's gaining momentum right now?*

- Compares question share in a configurable **recent window** (default: last 90 days) against the prior period of equal length
- Δ (pp) column shows the percentage-point change in share
- Click any row to expand the individual questions for that topic, with company attribution

## Companies tab

Answers: *which customers are most engaged, and what are they asking about?*

> Company names are derived from attendee email domains and are only visible when running locally with ON24 credentials. The public deployment strips this data before committing to the repository.

- Dropdown filters to a single company
- Bar chart shows their top topics
- Line chart shows their topic share trend over time

## Events Explorer tab

Answers: *how did a specific webinar perform, and what did the audience want to know?*

- Table sorted by **Customer Q** (customer questions, descending) by default
- Filterable by Product using the sidebar
- Click any row to expand the full question list for that event — Topic, Question, and Company columns

---

## Things to know

**Company data is local-only.**
Attendee email domains are resolved to company names via Claude Haiku and cached in `vocab/company_domains.json`. This file is committed (domain → company name only, no PII), but the questions Parquet files have `email_domain` and `inferred_company` stripped before being pushed to the public repo. Run `scripts/restore_company_data.py` to re-enrich locally after a pull.

**The Refresh button is local-only.**
The sidebar shows a **Refresh data** button only when ON24 credentials are present in the environment. On Streamlit Cloud, data is updated by the weekly GitHub Actions workflow instead.

**AI enrichment runs at ingest time, not on the fly.**
Topic labels and product inference are generated once by Claude Haiku during ingest and stored in the Parquet files. The dashboard itself makes no AI calls.

**ON24 Q&A comes from the attendee endpoint.**
The `/qanda` endpoint is real-time only and is empty for past events. Questions are extracted from `attendee["questions"]` on the `/attendee` endpoint, which is the correct source for historical data.

---

## Developer setup

### Prerequisites

- Python 3.11+
- ON24 management API credentials (Client ID, Token Key, Token Secret)
- Anthropic API key (for topic labelling and company inference during ingest)

### Local installation

```bash
git clone https://github.com/rosdevries/expert-series-trenderizer
cd expert-series-trenderizer
pip install -r requirements.txt
cp .env.example .env
# Fill in .env with your credentials
streamlit run streamlit_app.py
```

### Environment variables

| Variable | Description |
|---|---|
| `ON24_CLIENT_ID` | ON24 account client ID |
| `ON24_TOKEN_KEY` | ON24 management API token key |
| `ON24_TOKEN_SECRET` | ON24 management API token secret |
| `ANTHROPIC_API_KEY` | Claude API key for topic and company enrichment |

### Running an ingest

```bash
# Fetch the last 30 days (default)
python -m trenderizer.ingest

# Full backfill from a specific date
python -m trenderizer.ingest --backfill --start-date 2024-01-01
```

Ingest fetches all events in the date window, pulls every attendee's questions, enriches them with Claude Haiku (product, topic, company), and writes Parquet files under `data/`.

### Restoring company data after a pull

The public repo ships questions without `email_domain` or `inferred_company`. After pulling, run:

```bash
python scripts/restore_company_data.py
```

This re-fetches attendee emails from ON24, re-applies the company cache, and preserves all existing topic labels without making any new AI calls.

### Streamlit Cloud deployment

Add the following secret in the Streamlit Cloud dashboard under **App settings → Secrets**:

```toml
APP_PASSWORD = "..."
```

The ON24 and Anthropic credentials are **not** needed in Streamlit Cloud — data is pre-built by the GitHub Actions workflow and committed to the repository.

### Automated weekly refresh

`.github/workflows/weekly_refresh.yml` runs every Monday at 06:00 UTC. It installs dependencies, runs `python -m trenderizer.ingest` (last 30 days), and commits any new or updated Parquet files back to `main`. The workflow requires four repository secrets: `ON24_CLIENT_ID`, `ON24_TOKEN_KEY`, `ON24_TOKEN_SECRET`, and `ANTHROPIC_API_KEY`.

### Repository structure

```
streamlit_app.py                # Streamlit dashboard (four tabs)
trenderizer/
  ingest.py                     # Main ingest pipeline (fetch → enrich → store)
  on24_client.py                # ON24 REST API client
  enrichment.py                 # Claude Haiku wrappers (product, topic, company)
  store.py                      # Parquet read/write with year/month partitioning
  config.py                     # Paths, API settings, constants
scripts/
  restore_company_data.py       # Re-enrich company names from ON24 after a pull
  strip_company_data.py         # Strip company/domain columns before a public push
vocab/
  company_domains.json          # Domain → company name cache (180+ entries)
  topics.json                   # Canonical topic taxonomy
  products.json                 # Canonical product list
  siemens_domains.json          # Siemens email domain patterns (employee filter)
data/
  events/                       # Parquet partitioned by year/month
  questions/                    # Parquet partitioned by year/month
  registrants/                  # Parquet partitioned by year/month
.github/workflows/
  weekly_refresh.yml            # Scheduled Monday ingest → commit → push
```
