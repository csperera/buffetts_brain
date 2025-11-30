import requests
import os
import time

# --- Configuration ---
DOCS_DIR = 'knowledge_base/docs'
BERKSHIRE_DIR = os.path.join(DOCS_DIR, 'Berkshire_Letters')
MUNGER_DIR = os.path.join(DOCS_DIR, 'Munger_Transcripts')

# Create necessary directories
os.makedirs(BERKSHIRE_DIR, exist_ok=True)
os.makedirs(MUNGER_DIR, exist_ok=True)

print(f"Target directories created or verified: {BERKSHIRE_DIR} and {MUNGER_DIR}")
print("-" * 30)

# --- 1a. Fetch Combined Historical Letters (1977-2002) ---
print("1a. Fetching Combined Historical Letters (1977-2002) from Archive...")
# Using the stable archive link which covers the early years
combined_url = 'https://uploads-ssl.webflow.com/60e3655ca778911eb64b2a00/60f0773bd7a92410fed4ccbb_All-Berkshire-Hathaway-Letters.pdf'
file_path_combined = os.path.join(BERKSHIRE_DIR, '1977-2002_Combined_Archive_Letters.pdf')

if not os.path.exists(file_path_combined):
    try:
        response = requests.get(combined_url, timeout=15)
        if response.status_code == 200:
            with open(file_path_combined, 'wb') as f:
                f.write(response.content)
            print(f'   SUCCESS: Downloaded combined 1977-2002 archive letter.')
        else:
            print(f'   SKIPPED: Combined 1977-2002 letter (Status code {response.status_code})')
    except requests.exceptions.RequestException as e:
        print(f'   ERROR: Could not download combined letters: {e}')
else:
    print(f'   EXISTS: Combined 1977-2002 letter already downloaded.')

print("-" * 30)

# --- 1b. Fetching Individual Annual Letters (1999-2024) ---
print("1b. Fetching Individual Annual Letters (1999-2024)...")
base_url = 'https://www.berkshirehathaway.com/letters/'
years = range(1999, 2025)

for year in years:
    pdf_url = f'{base_url}{year}ltr.pdf'
    file_path = os.path.join(BERKSHIRE_DIR, f'{year}_letter.pdf')

    if os.path.exists(file_path):
        print(f'   EXISTS: {year} letter already downloaded.')
        continue
    
    try:
        response = requests.get(pdf_url, timeout=10)
        
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f'   SUCCESS: Downloaded {year} letter')
        else:
            print(f'   SKIPPED: {year} (Status code {response.status_code})')
            
    except requests.exceptions.RequestException as e:
        print(f'   ERROR: Could not download {year} letter: {e}')
        
    time.sleep(0.5)

print("-" * 30)

# --- 2. Download Poor Charlie's Almanack ---
print("2. Fetching Poor Charlie's Almanack...")
almanack_url = 'https://ia600702.us.archive.org/33/items/poor-charlies-almanack-the-wit-and-wisdom-of-charles-t.-munger-pdfdrive/Poor%20Charlie%E2%80%99s%20Almanack_%20The%20Wit%20and%20Wisdom%20of%20Charles%20T.%20Munger%20%28%20PDFDrive%20%29.pdf'
file_path_almanack = os.path.join(DOCS_DIR, 'Poor_Charlies_Almanack.pdf')

if not os.path.exists(file_path_almanack):
    try:
        response = requests.get(almanack_url, timeout=30)
        with open(file_path_almanack, 'wb') as f:
            f.write(response.content)
        print(f'   SUCCESS: Downloaded: Poor Charlie\'s Almanack to {DOCS_DIR}')
    except requests.exceptions.RequestException as e:
        print(f'   ERROR: Could not download Almanack: {e}')
else:
    print(f'   EXISTS: Poor Charlie\'s Almanack already downloaded.')

print("-" * 30)


# --- 3. Save Munger Transcript Links (as TXT bookmarks) ---
print("3. Saving Munger Transcript URLs (for future web scraping or manual review)...")
dj_transcripts = {
    '2023': 'https://www.kingswell.io/p/charlie-munger-q-and-a-2023-daily',  
    '2022': 'https://latticeworkinvesting.com/2022/06/03/charlie-munger-full-transcript-of-daily-journals-2022-annual-meeting/',
    '2021': 'https://sungcap.com/charlie-munger-daily-journal-2021-transcript/',
    '2019': 'https://latticeworkinvesting.com/2019/03/03/charlie-munger-full-transcript-of-daily-journal-annual-meeting-2019/',
    '2018': 'https://worldlypartners.com/charlie-munger-archive/' 
}
for year, url in dj_transcripts.items():
    file_path = os.path.join(MUNGER_DIR, f'DJ_{year}.txt')
    with open(file_path, 'w') as f:
        f.write(f'Charlie Munger Daily Journal {year} Transcript URL:\n{url}\n\nNote: The full content of this link should be scraped later if you want deep RAG analysis.')
    print(f'   ADDED: DJ {year} transcript link to {MUNGER_DIR}')

print("-" * 30)

# --- 4. Bonus Speeches (TXT transcripts/PDFs) ---
print("4. Fetching Bonus Speeches...")
speeches = {
    # FIX: Using a known stable PDF link for 'The Psychology of Human Misjudgment'
    'Psychology_of_Human_Misjudgment': 'https://janav.files.wordpress.com/2015/12/thepsychologyofhumanmisjudgment.pdf',
    'USC_Commencement_2007': 'https://worldlypartners.com/charlie-munger-archive/'
}
for name, url in speeches.items():
    if url.endswith('.pdf'):
        file_path = os.path.join(MUNGER_DIR, f'{name}.pdf')
        
        # Robustness check: Delete the corrupted file if it exists, to ensure a clean download attempt
        if os.path.exists(file_path):
            if os.path.getsize(file_path) < 100000 and name == 'Psychology_of_Human_Misjudgment':
                # Assuming the PDF should be large; if it's small, it's likely corrupted HTML
                print(f'   CLEANUP: Removing potentially corrupted {name}.pdf ({os.path.getsize(file_path)} bytes).')
                os.remove(file_path)
            elif os.path.getsize(file_path) > 1000:
                 # If file exists and is reasonably large (already downloaded correctly)
                 print(f'   EXISTS: PDF for {name} already downloaded and verified.')
                 continue
            
        if not os.path.exists(file_path): # Check again if it was removed or didn't exist
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    print(f'   SUCCESS: Downloaded PDF: {name} (using stable link)')
                else:
                    print(f'   SKIPPED: PDF for {name} (Status code {response.status_code})')
            except requests.exceptions.RequestException as e:
                print(f'   ERROR: Could not download PDF for {name}: {e}')
    else:
        file_path = os.path.join(MUNGER_DIR, f'{name}.txt')
        with open(file_path, 'w') as f:
            f.write(f'Transcript URL:\n{url}\n\nNote: The full content of this link should be scraped later if you want deep RAG analysis.')
        print(f'   ADDED: {name} link to {MUNGER_DIR}')
        
print("-" * 30)
print('Data download setup complete! Check your knowledge_base/docs folder.')