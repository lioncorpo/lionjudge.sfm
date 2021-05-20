import yaml
default_config_yaml = '''
# Metadata
extract_features: False          # False = load from disk

refine_with_local_map: True
tracker_lk: False
'''


def default_config():
    """Return default configuration"""
    return yaml.safe_load(default_config_yaml)
