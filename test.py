import requests
import json

itemID = 560

r = requests.get('http://services.runecape.com/m=itemdb_oldschool/api/catalogue/detail.json', params ={'item':itemID})
json_data = json.loads(r.text)
print(json_data)