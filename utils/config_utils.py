import yaml
import gin

def load_yaml_config(config_file_path):
    with open(config_file_path, 'r') as file:
        return yaml.safe_load(file)

def parse_gin_config(gin_config_paths, gin_bindings=None):
    for gin_file in gin_config_paths:
        gin.parse_config_file(gin_file)
    if gin_bindings:
        for binding in gin_bindings:
            gin.parse_config(binding)

def get_gin_parameter(parameter_name):
    return gin.query_parameter(parameter_name)
