import dotenv

dotenv.load_dotenv(dotenv.find_dotenv())

import sys

sys.path.append("./../app")
sys.path.append("./")

import os

os.chdir("./../app")
print(os.getcwd())
