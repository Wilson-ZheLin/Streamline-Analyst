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

# write a stream of words
def stream_data(line):
    for word in line.split():
        yield word + " "
        time.sleep(random.uniform(0.02, 0.05))

# Show developer info at the bottom
def developer_info():
    time.sleep(2)
    st.write(stream_data(":grey[Streamline Analyst is developed by *Zhe Lin*. You can reach out to me via] :blue[wilson.linzhe@gmail.com] :grey[or] :blue[[GitHub](https://github.com/Wilson-ZheLin)]"))

def developer_info_static():
    st.write(":grey[Streamline Analyst is developed by *Zhe Lin*. You can reach out to me via] :blue[wilson.linzhe@gmail.com] :grey[or] :blue[[GitHub](https://github.com/Wilson-ZheLin)]")