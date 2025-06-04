from intelliscraper.web_scraper import WebScraper

wanted_list = ['Don't Waste Time']
scraper = WebScraper(wanted_list, url='https://fundmycap.com/campaigns/latest')
results = scraper.build()
for result in results:
    print(result)