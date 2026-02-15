# HowHungryisAIDashboard

This repository contains an automated pipeline to scrape, clean, and export large language model (LLM) performance and usage metrics from ArtificialAnalysis.ai for a How Hungry is AI. The dashboard estimates the energy, water, and carbon footprint of LLM inference at various scales.

Link: https://app.powerbi.com/view?r=eyJrIjoiZjVmOTI0MmMtY2U2Mi00ZTE2LTk2MGYtY2ZjNDMzODZkMjlmIiwidCI6IjQyNmQyYThkLTljY2QtNDI1NS04OTNkLTA2ODZhMzJjMTY4ZCIsImMiOjF9


## Setup

Install Python (3.10–3.12 recommended) and dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   
   pip install -r requirements.txt

   ```

2. Clean and export:
   ```bash
   python Data_Cleaning_Final.py
   ```




## Automation

A GitHub Actions workflow is included to:
- Run the scraper and cleaner on a schedule
- Commit updated CSVs to `/output`

Enable workflows in the Actions tab and add required secrets under **Settings → Secrets and variables → Actions**.



