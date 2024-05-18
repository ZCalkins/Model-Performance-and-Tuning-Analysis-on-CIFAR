import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from utils.config_utils import load_yaml_config

def create_writer(experiment_name: str, model_name: str, extra: str=None) -> SummaryWriter:
    config_file_path = 'CIFAR100-Multi-Model-Ablation-Analysis/configurations/yaml/logging_config.yaml'
    config = load_yaml_config(config_file_path)
    base_log_dir = config['logging']['base_log_dir']
    
    timestamp = datetime.now().strftime("%Y-%m-%d")
    if extra:
        log_dir = os.path.join(base_log_dir, timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join(base_log_dir, timestamp, experiment_name, model_name)

    print(f"Created SummaryWriter, saving to: {log_dir}")
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)
