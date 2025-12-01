# build_wild_test.py
import os
from icrawler.builtin import BingImageCrawler

# 下载两类
categories = {
    "persian_cat": "Persian cat",
    "siamese_cat": "Siamese cat"
}

# 每类下载数量
num_images = 300

# 保存根目录
output_root = "./dataset/test"

def download_category(folder_name, keyword):
    save_dir = os.path.join(output_root, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    crawler = BingImageCrawler(storage={'root_dir': save_dir})

    print(f"📌 开始下载：{keyword} → {save_dir}")
    crawler.crawl(keyword=keyword, max_num=num_images)
    print(f"✔ 完成：{keyword}\n")

def main():
    os.makedirs(output_root, exist_ok=True)
    for folder_name, keyword in categories.items():
        download_category(folder_name, keyword)

if __name__ == "__main__":
    main()
