import json
from pprint import pprint
with open('QA_dev.json') as data_file:
    data = json.load(data_file)
pprint(data[1]['sentences'])