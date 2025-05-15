# coding:utf-8
import requests
import pandas as pd
import time
from bs4 import BeautifulSoup

# 定义请求头，模拟浏览器请求
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}

# 代理IP池
proxies = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
}

# 定义一个函数来爬取药品的Description
def get_description(db_id):
    url = f"https://go.drugbank.com/drugs/{db_id}"
    try:
        # 发送GET请求，添加请求头和代理
        response = requests.get(url, headers=headers, proxies=proxies)
        response.raise_for_status()

        # 使用BeautifulSoup解析HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        # 查找Description标签
        description_tag = soup.find('dt', text='Description')
        if description_tag:
            description = description_tag.find_next('dd', class_='description').text.strip()
        else:
            description = 'Description Not Available'

        return description
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error fetching description for {db_id}: {e}")
        return "Error: HTTP Forbidden"
    except Exception as e:
        print(f"Error fetching description for {db_id}: {e}")
        return "Error"

# 主函数
def main():
    df = pd.read_excel('drug_info_with_mfmw.xlsx')

    for index, row in df.iterrows():
        # 检查当前行的Description
        if row['Description'] in ['Description Not Available', 'Not Found']:
            db_id = row['DB编号']
            print(f"Fetching description for DB ID: {db_id}")

            description = get_description(db_id)

            # 更新Description列
            df.at[index, 'Description'] = description

            time.sleep(2)  # 增加延时

    df.to_excel('drug_info_with_new_description.xlsx', index=False)
    print("爬取完成")

# 运行主函数
if __name__ == "__main__":
    main()