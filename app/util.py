import streamlit as st
import requests
import yaml
import time
import random

with open('config/config.yaml', 'r') as file:
    config_data = yaml.safe_load(file)

def load_lottie(url = config_data['lottie_url']):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def stream_data(line):
    for word in line.split():
        yield word + " "
        time.sleep(random.uniform(0.03, 0.08))
    time.sleep(0.5)

def developer_info():
    st.write(stream_data(":grey[Streamline Analyst is developed by Zhe Lin. You can reach out to me via] :blue[wilson.linzhe@gmail.com] :grey[or] :blue[[GitHub](https://github.com/Wilson-ZheLin)]"))