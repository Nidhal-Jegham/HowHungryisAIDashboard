from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd

link_list= [
    "https://artificialanalysis.ai/leaderboards/providers/prompt-options/single/short?deprecation=all",
    "https://artificialanalysis.ai/leaderboards/providers/prompt-options/single/medium?deprecation=all",
    "https://artificialanalysis.ai/leaderboards/providers/prompt-options/single/long?deprecation=all",]
length_list = [
    "short", 
    "medium",
    "long",]
for i in range(3): 

    
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  
    
    driver = webdriver.Chrome(service=Service(), options=options)

    driver.get(link_list[i])
    time.sleep(5) 

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "//table"))
    )

    buttons = driver.find_elements(By.TAG_NAME, "button")


    for btn in buttons:
        text = btn.text.strip().replace("\n", " ").lower()
        if "expand columns" in text:
            print("Found the'Expand Columns' button.")

            driver.execute_script("""
                arguments[0].style.border = "3px solid red";
                arguments[0].style.background = "#ff0";
            """, btn)

            time.sleep(0.5)

            try:
                from selenium.webdriver.common.action_chains import ActionChains
                actions = ActionChains(driver)
                actions.move_to_element(btn).pause(0.3).click().perform()
                print("Button Clicked")
            except Exception as e:
                print(f" ActionChains failed: {e}")

            break



    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    rows = soup.find_all("tr")
    raw_data = []
    for row in rows:
        cells = row.find_all(["td", "th"])
        row_data = [cell.get_text(strip=True) for cell in cells]
        if row_data:
            raw_data.append(row_data)

    header = raw_data[1]

    cleaned_rows = [r for r in raw_data[2:] if len(r) == len(header)]
    bad_rows = [r for r in raw_data[2:] if len(r) != len(header)]

    if bad_rows:
        print(f"Skipped {len(bad_rows)} rows that didn't match header length of {len(header)}.")

    import pandas as pd
    df = pd.DataFrame(cleaned_rows, columns=header)

    print(df.head())

    df.to_csv(f'artificialanalysis_clean{length_list[i]}.csv', index=False)
    print(f'Data exported to {length_list[i]} artificialanalysis_clean.csv')


