import json
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.select import Select
from webdriver_manager.chrome import ChromeDriverManager

LOADING_TIME = 3
init_url = "https://glaw.scourt.go.kr/wsjo/panre/sjo050.do#//"

def setup_driver(url, view = False):
    chrome_options = webdriver.ChromeOptions()
    if not view :
        chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver.get(url)
    time.sleep(LOADING_TIME)

    return driver


def set_num_of_contents(driver, content_num : int = 80):
    select_option = Select(
        driver.find_element(By.XPATH, '//*[@id="tabwrap"]/div/div/div[2]/div[3]/div[2]/fieldset/select'))
    if content_num == 40:
        select_option.select_by_index(1)
    elif content_num == 80:
        select_option.select_by_index(2)
    apply_option_button = driver.find_element(By.XPATH, '//*[@id="tabwrap"]/div/div/div[2]/div[3]/div[2]/fieldset/a')
    apply_option_button.click()
    time.sleep(LOADING_TIME)

    return driver



def get_title_time(driver):
    title = driver.find_elements(By.CLASS_NAME, "list_title")
    titles = []
    content_items = []
    for t in title:
        content_item = t.find_element(By.TAG_NAME, "strong")
        content_items.append(content_item)

        titles.append(t.text)
    print("len(titles) : ", len(titles))

    return driver, titles, content_items



def get_result_of_page(driver, content_items, titles, results, id):
    idx = 0
    first_tab = driver.window_handles[0]
    for content_item in content_items:
        result_dict = {}
        content_item.click()
        time.sleep(LOADING_TIME)
        last_tab = driver.window_handles[-1]
        print(last_tab)
        driver.switch_to.window(window_name=last_tab)
        time.sleep(LOADING_TIME)

        src = driver.find_elements(By.XPATH, '//*[@id="areaDetail"]/div[2]/div/div')
        result_dict['id'] = id
        id += 1
        result_dict['url'] = str(driver.current_url)
        result_dict['title'] = titles[idx]
        idx += 1

        body = ''
        for doc in src:
            body += doc.text
        result_dict['doc'] = body
        results.append(result_dict)

        driver.close()
        driver.switch_to.window(window_name=first_tab)
        time.sleep(LOADING_TIME)

    return driver, results, id



def move_to_next_page(j, driver):
    if j == 0:
        next_button = driver.find_element(By.XPATH, '//*[@id="tabwrap"]/div/div/div[2]/div[3]/div[2]/div/fieldset/p/a[1]')
    else:
        next_button = driver.find_element(By.XPATH, '//*[@id="tabwrap"]/div/div/div[2]/div[3]/div[2]/div/fieldset/p/a[2]')

    print(next_button)
    next_button.click()
    time.sleep(LOADING_TIME)

    return driver


def results_to_json(results, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for content in results:
            print(content)
            f.write(json.dumps(content, ensure_ascii=False))
            f.write('\n')


def main():
    output_path = './data_panre.json'
    results = []
    id = 0

    try:
        for i in range(12):
            driver = setup_driver(init_url, view=False)
            driver = set_num_of_contents(driver, content_num=80)
            for j in range(i):
                driver = move_to_next_page(j, driver)
            driver, titles, content_items = get_title_time(driver)
            driver, results, id = get_result_of_page(driver, content_items, titles, results, id)
    except:
        pass
    results_to_json(results, output_path)


if __name__ == '__main__':
    main()


