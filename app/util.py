import streamlit as st
import requests
import yaml
import time
import random
import os

config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
with open(config_path, 'r') as file:
    config_data = yaml.safe_load(file)

def load_lottie():
    r1, r2 = requests.get(config_data['lottie_url1']), requests.get(config_data['lottie_url2'])
    if r1.status_code != 200 or r2.status_code != 200:
        return None
    return r1.json(), r2.json()

# write a stream of words
def stream_data(line):
    for word in line.split():
        yield word + " "
        time.sleep(random.uniform(0.02, 0.05))

# Store the welcome message and introduction
def welcome_message():
    return config_data['welcome_template']

def introduction_message():
    return config_data['introduction_template1'], config_data['introduction_template2']

# Show developer info at the bottom
def developer_info():
    time.sleep(2)
    st.write(stream_data(":grey[Streamline Analyst is developed by *Zhe Lin*. You can reach out to me via] :blue[wilson.linzhe@gmail.com] :grey[or] :blue[[GitHub](https://github.com/Wilson-ZheLin)]"))

def developer_info_static():
    st.write(":grey[Streamline Analyst is developed by *Zhe Lin*. You can reach out to me via] :blue[wilson.linzhe@gmail.com] :grey[or] :blue[[GitHub](https://github.com/Wilson-ZheLin)]")