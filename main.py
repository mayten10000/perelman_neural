from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd

options = webdriver.ChromeOptions()
options.headless = False
driver = webdriver.Chrome(options=options)

url = "https://math100.ru/algebra7-9_7_1/"
driver.get(url)

input("Решите капчу и нажмите Enter...")

try:
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "math"))
    )
except Exception as e:
    print("Ошибка загрузки страницы:", e)
    driver.quit()
    exit()

soup = BeautifulSoup(driver.page_source, "html.parser")


equation_elements = soup.find_all("span", class_="math")

equations_list = []
for eq in equation_elements:
    equations_list.append(eq.get_text(strip=True))

driver.quit()

if equations_list:
    df = pd.DataFrame(equations_list, columns=["Equation"])
    df.to_csv("equations.csv", index=False, encoding="utf-8")
    print("✅ Уравнения сохранены в equations.csv!")
else:
    print("⚠️ Уравнения не найдены. Проверьте правильность селектора или структуру страницы.")