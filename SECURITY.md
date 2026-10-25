# iClaim - Security and Data Handling

## Current scope

This is a local academic Streamlit prototype. It has no authentication, authorization,
encryption, audit trail, tenant isolation, retention workflow or production deployment
configuration. Do not publish it with real claim, customer, vehicle-identification or insurer
data.

## Implemented controls

- Image uploads are restricted to configured image extensions, a 10 MB byte limit and a
  25-million-pixel decoded-image limit.
- Image validation runs before inference and rejects empty, unreadable, undersized and unsafe
  images.
- Claim IDs are restricted to a safe 1-64 character identifier format, avoiding spreadsheet-formula
  injection in exported CSV files.
- Configuration is parsed with `yaml.safe_load`.
- `.gitignore` excludes virtual environments, Kaggle credentials, environment files, datasets,
  model files, generated figures and prediction logs.
- The interface labels synthetic records and indicative repair costs, and keeps a human-review
  disclaimer visible.

## Before sharing the repository

1. Create a fresh virtual environment with a supported Python version and install
   `requirements.txt`.
2. Run `pytest tests -q` and dependency-advisory tooling available in that environment.
3. Verify that `git status --ignored` does not reveal credentials, customer data, raw images,
   model files or generated claim logs selected for commit.
4. Record the dataset licence, source and consent/privacy basis in `docs/DATASET_GUIDE.md`.
5. Add an explicit software licence before making the repository public.

## Before production use

Add identity and access management, server-side storage controls, encryption in transit and at
rest, audit logging, rate limits, malware scanning, privacy/retention policies, monitoring,
model governance and human approval workflows. Independently validate model performance and
fairness on representative data; never use this prototype as the sole basis for a claim outcome.
