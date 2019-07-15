import yaml

with open('conf/settings.yml', 'r', encoding='utf8') as fin:
  conf = yaml.load(fin, Loader=yaml.FullLoader)

print(conf)
