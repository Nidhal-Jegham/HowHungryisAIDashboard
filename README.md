# HowHungryisAIDashboard

This repository contains an automated pipeline to scrape, clean, and export large language model (LLM) performance and usage metrics for a Power BI dashboard. The dashboard estimates the energy, water, and carbon footprint of LLM inference at various scales.


## Setup

1. Install Python (3.10–3.12 recommended) and dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   
   pip install -r requirements.txt
   

2. Install Google Chrome and matching ChromeDriver:
   ```bash
          sudo apt-get update
          sudo DEBIAN_FRONTEND=noninteractive apt-get install -y wget unzip jq
      
          wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
          sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a \
               apt-get install -y ./google-chrome-stable_current_amd64.deb
      
          CFT_JSON="https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json"
          CFT_URL=$(curl -sL "$CFT_JSON" \
            | jq -r '.channels.Stable.downloads.chromedriver[]
                       | select(.platform=="linux64")
                       | .url')
          echo "Downloading ChromeDriver from: $CFT_URL"
      
          wget -qO chromedriver.zip "$CFT_URL"
          unzip -qj chromedriver.zip -d .
          chmod +x chromedriver
          sudo mv chromedriver /usr/local/bin/   ```



## Usage

1. Scrape data:
   ```bash
   python ArtificialAnalysisScraping.py
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



