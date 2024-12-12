from pathlib import Path
from utils.paths import CREDENTIALS_PATH
import yaml


def get_credentials(name: str) -> dict:
    with open(CREDENTIALS_PATH, 'r') as file:
        credentials = yaml.safe_load(file)
    return credentials[name]
