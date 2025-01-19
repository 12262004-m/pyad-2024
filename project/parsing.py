from bs4 import BeautifulSoup
import requests
base_url = "https://yandex.ru/maps/org/bar_restoran_doski/1392831762/reviews/?indoorLevel=3&ll=30.335801%2C59.922314&tab=reviews&z=14"
response = requests.get(base_url)
soup = BeautifulSoup(response.text, "html.parser", exclude_encodings="utf-8")
temp = soup.findAll('span', 'business-review-view__body-text')
with open(r"data.txt", "w") as file:
    for line in temp:
        file.write(line.text + '\n' + "---------" + '\n')
