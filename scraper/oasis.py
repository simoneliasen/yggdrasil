import string
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from time import sleep
import pyautogui as pyg

def s(seconds):
    sleep(seconds)

class ImportBot():
    """Kør: python -i scraper/chadOasisScraper.py
    Kræver Chrome v105"""
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("--start-maximized")
        self.driver = webdriver.Chrome(options=chrome_options)
        s(1)
        self.driver.get("http://oasis.caiso.com/mrioasis/logon.do")
        s(4)
    
    def goToFMM(self):
        pyg.moveTo(x=272, y=185)
        s(0.5)
        pyg.moveTo(x=284, y=209)
        s(0.5)
        pyg.moveTo(x=483, y=319)
        s(0.5)
        pyg.click()
        s(5)

    def selectAllNodes(self):
        pyg.moveTo(x=477, y=217)
        s(0.5)
        pyg.click()
        s(0.5)
        pyg.moveTo(x=465, y=250)
        s(0.5)
        pyg.click()
        s(0.5)
        

    def selectDate(self, date: string):
        """Selects and applies a date in oasis.
        IMPORTANT!: Input should be string in: mm/dd/yyyy"""

        #Date from:
        pyg.moveTo(x=143, y=219)
        s(0.5)
        pyg.click()
        s(0.5)
        pyg.hotkey('ctrl', 'a')
        s(0.5)
        pyg.typewrite(date)
        s(1)
        pyg.hotkey('enter')
        s(0.5)

    def download_csv(self):
        pyg.moveTo(x=293, y=217)
        s(0.5)
        pyg.click()
        s(0.5)

    def selectHour(self, hour: int):
        #hej mor
        print('hej mor')



if __name__ == '__main__':
    #hent sidste id.
    print("Den er startet")
    
    chad = ImportBot()
    chad.goToFMM()
    chad.selectAllNodes()
    chad.selectDate("08/09/2022")
    chad.download_csv()

    #todo lav loop over brugers input.
    #bare husk at det er mm/dd/yyyy
    #Plan:
    # 1. check om end er noget
    # 2. hvis nej: decrement dd hvis den er > 1
    # 3. ellers: decrement mm