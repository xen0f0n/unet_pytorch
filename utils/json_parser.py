import json

def parse_json(config_file):
    with open(config_file, 'r') as f:
        config_params = json.load(f)
    return config_params
