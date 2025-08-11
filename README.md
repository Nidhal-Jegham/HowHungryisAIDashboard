# HowHungryisAIDashboard

This repository contains an automated pipeline to scrape, clean, and export large language model (LLM) performance and usage metrics for a Power BI dashboard. The dashboard estimates the energy, water, and carbon footprint of LLM inference at various scales.

```

## Setup

1. Install Python (3.10–3.12 recommended) and dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Install Google Chrome and matching ChromeDriver:
   ```bash
   chromedriver --version
   ```

## Usage

1. Scrape data:
   ```bash
   python ArtificialAnalysisScraping.py
   ```

2. Clean and export:
   ```bash
   python Data_Cleaning_Final.py
   ```

3. In Power BI:
   - Set data source to `/output` CSVs.
   - Refresh to update metrics.

## Model Size Categories

- Nano: <7B parameters (1 GPU)  
- Micro: 7–20B (2 GPUs)  
- Small: 20–40B (4 GPUs)  
- Medium: 40–70B (8 GPUs)  
- Large: >70B (8 GPUs high-memory)  
- Non-disclosed flagship models (e.g., GPT-4o, Claude-3.7 Sonnet) are classified as Large.  
- OpenAI Mini variants (e.g., GPT-4o mini) are classified as Medium.  
- Models labeled "Nano" but with higher performance (e.g., GPT-4.1 nano) are classified as Small.

## Automation

A GitHub Actions workflow is included to:
- Run the scraper and cleaner on a schedule
- Commit updated CSVs to `/output`

Enable workflows in the Actions tab and add required secrets under **Settings → Secrets and variables → Actions**.



