import requests
from bs4 import BeautifulSoup
import json

###########################################################################################################
##bitne klase divova:
## confession tag sadrzi sve bitno
## confession-text -> ovde nalazimo tekst ispovesti
## class="confession-value" id="approve-count-3229833" -> unutar ovog taga iscitavamo koliko ljudi odobrava
## class="confession-value" id="disapprove-count-3229833" -> unutar ovog taga iscitavamo koliko ljudi osudjuje
## confession-timestamp -> unutar ovog taga iscitavamo kada je post napravljen
############################################################################################################

headers = {"User-Agent": "Mozilla/5.0 (learning project)"}
BASE_URL = "https://ispovesti.com/sort/top"
confession_class_name = "confession"
confession_text_class_name = "confession-text"
confession_value_class_name = "confession-value" 
approve_count_id = "approve-count-"
disapprove_count_id = "disapprove-count-"
confession_timestamp_class_name = "confession-timestamp"
confessions_list = []
maxPageNum = 99

############################################################################################################

def scrape():

    print("\t##### BEGINNING SCRAPING PROCESS ####\t")
    for i in range (1 , maxPageNum):
        current_url = f'{BASE_URL}/{i}'

        response = requests.get(current_url)

        response.encoding = "utf-8"

        html = response.text

        soup = BeautifulSoup(html, "lxml")

        confessions_divs = soup.find_all("div", class_= confession_class_name)

        print(f"-SCRAPED PAGE {i}")
        print(f"current url : {current_url}")

        for conf in confessions_divs:

            confession_text_div = conf.find("div", class_ = confession_text_class_name)

            confession_text = confession_text_div.get_text(strip=True) if confession_text_div else ""


            approve_div = conf.find("div", class_ = confession_value_class_name, id = lambda x : x and x.startswith(approve_count_id) )

            approve_count = approve_div.get_text(strip=True) if approve_div else ""


            disapprove_div = conf.find("div", class_ = confession_value_class_name, id = lambda x : x and x.startswith(disapprove_count_id))

            disapprove_count = disapprove_div.get_text(strip=True) if disapprove_div else ""


            timestamp_div = conf.find("div", class_ = confession_timestamp_class_name)

            timestamp = timestamp_div.get_text(strip = True) if timestamp_div else None

            confession_dict = {
                "text" : confession_text,
                "approve_count" : approve_count,
                "disapprove_count" : disapprove_count,
                "timestamp" : timestamp
            }

            confessions_list.append(confession_dict)

    return confessions_list    

def saveToJson(confessions_list):
    with open("confessions_page.json", "w", encoding="utf-8") as f:
        json.dump(confessions_list, f, ensure_ascii=False, indent=4)  

confessions_list = scrape()
saveToJson(confessions_list)
