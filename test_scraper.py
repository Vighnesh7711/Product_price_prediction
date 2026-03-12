from scraper.product_scraper import ProductScraper
import json

s = ProductScraper()
url = "https://www.flipkart.com/samriddhi-alloy-blue-jewellery-set/p/itm53ea639239d2c"
res = s.scrape(url)
print(json.dumps(res, indent=2))
