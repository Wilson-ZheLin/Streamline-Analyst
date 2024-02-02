import requests
import yaml

with open('config/config.yaml', 'r') as file:
    config_data = yaml.safe_load(file)

def load_lottie(url = config_data['lottie_url']):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()