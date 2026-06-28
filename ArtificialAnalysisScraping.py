from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time
import glob
import os
from datetime import date

TODAY = date.today().strftime("%Y-%m-%d")
SNAPSHOT_DIR = "snapshots"
CUMULATIVE_FILE = "data/artificialanalysis_cumulative.csv"

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

link_list = [
    "https://artificialanalysis.ai/leaderboards/providers/prompt-options/single/100k?deprecation=all",
    "https://artificialanalysis.ai/leaderboards/providers/prompt-options/single/medium?deprecation=all",
    "https://artificialanalysis.ai/leaderboards/providers/prompt-options/single/long?deprecation=all",
]
length_list = [
    "extra_long",
    "medium",
    "long",
]

all_today = []

for i in range(len(link_list)):
    print(f"\n--- Scraping: {length_list[i]} ---")

    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(service=Service(), options=options)

    try:
        driver.get(link_list[i])
        time.sleep(5)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//table"))
        )

        buttons = driver.find_elements(By.TAG_NAME, "button")
        for btn in buttons:
            text = btn.text.strip().replace("\n", " ").lower()
            if "expand columns" in text:
                print("Found 'Expand Columns' button — clicking.")
                try:
                    ActionChains(driver).move_to_element(btn).pause(0.3).click().perform()
                    print("Button clicked.")
                    time.sleep(1)
                except Exception as e:
                    print(f"ActionChains failed: {e}")
                break

        html = driver.page_source
    finally:
        driver.quit()

    soup = BeautifulSoup(html, "html.parser")
    rows = soup.find_all("tr")
    raw_data = []
    for row in rows:
        cells = row.find_all(["td", "th"])
        row_data = [cell.get_text(strip=True) for cell in cells]
        if row_data:
            raw_data.append(row_data)

    if len(raw_data) < 3:
        print(f"Not enough rows found for {length_list[i]}, skipping.")
        continue

    header = raw_data[1]
    cleaned_rows = [r for r in raw_data[2:] if len(r) == len(header)]
    bad_rows = [r for r in raw_data[2:] if len(r) != len(header)]
    if bad_rows:
        print(f"Skipped {len(bad_rows)} rows with mismatched column count.")

    df = pd.DataFrame(cleaned_rows, columns=header)

    df.insert(0, "query_size", length_list[i])
    df.insert(1, "date", TODAY)

    print(df.head())
    all_today.append(df)

if all_today:
    # 1. Save today's snapshot
    snapshot_df = pd.concat(all_today, ignore_index=True)
    snapshot_path = os.path.join(SNAPSHOT_DIR, f"artificialanalysis_{TODAY}.csv")
    snapshot_df.to_csv(snapshot_path, index=False)
    print(f"\nSnapshot saved → {snapshot_path}  ({len(snapshot_df)} rows)")

    # 2. Find all historical daily files in the data directory
    # The pattern "20*.csv" ensures we only grab dated files, ignoring 'clean' or 'cumulative' files
    historical_files = glob.glob("data/artificialanalysis_20*.csv")
    
    dfs_to_combine = []
    for file in historical_files:
        try:
            dfs_to_combine.append(pd.read_csv(file))
        except Exception as e:
            print(f"Skipping {file}: {e}")
            
    # 3. Add today's newly scraped data to the list
    dfs_to_combine.append(snapshot_df)
    
    # 4. Concatenate everything into one master DataFrame
    cumulative_df = pd.concat(dfs_to_combine, ignore_index=True)
    
    # 5. Drop duplicates just in case there are overlapping rows
    cumulative_df = cumulative_df.drop_duplicates()
    
    # 6. Save the newly built cumulative file directly to the data folder
    cumulative_path = "data/artificialanalysis_cumulative.csv"
    cumulative_df.to_csv(cumulative_path, index=False)
    print(f"Cumulative file dynamically rebuilt → {cumulative_path}  ({len(cumulative_df)} rows total)")

else:
    print("No data collected — nothing saved.")
