import os
import yaml
import appdirs
from dotwiz import DotWiz
from glob import glob

def clear_configs_dir() -> None:
    config_dir = os.path.join(appdirs.user_config_dir('kiss'), 'configs')
    
    for filename in os.listdir(config_dir):
        if os.path.isfile(os.path.join(config_dir, filename)):
            os.remove(os.path.join(config_dir, filename))


def get_default_configs() -> DotWiz:
    """
    Get the default configurations.

    Returns:
        DotWiz: A DotWiz instance containing default configurations.

    Example:
        default_configs = get_default_configs()
    """
    configs = {
        'torch': {
            'device': 'mps'
        }
    }
    
    return DotWiz(configs)


def get_configs() -> DotWiz:
    """
    Get configurations from YAML files in the user's configuration directory.

    Returns:
        DotWiz: A DotWiz instance containing user configurations.

    Example:
        user_configs = get_configs()
    """
    config_dir = os.path.join(appdirs.user_config_dir('kiss'), 'configs')
    yamls = glob(os.path.join(config_dir, '*.yaml'))

    configs = {}

    for file in yamls:
        key = os.path.splitext(os.path.basename(file))[0]
        with open(file, 'r') as f:
            configs[key] = yaml.safe_load(f)

    return DotWiz(configs)


def save_configs(configs: DotWiz) -> None:
    """
    Save configurations to YAML files in the user's configuration directory.

    Args:
        configs (DotWiz): A DotWiz instance containing configurations to be saved.

    Example:
        save_configs(user_configs)
    """
    config_dir = os.path.join(appdirs.user_config_dir('kiss'), 'configs')
    

    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    for key, value in configs.items():
        with open(os.path.join(config_dir, f'{key}.yaml'), 'w+') as f:
            yaml.dump(value.to_dict(), f, sort_keys=False)


def check_configs() -> bool:
    """
    Check if configuration files exist in the user's configuration directory.

    Returns:
        bool: True if configuration files exist, False otherwise.

    Example:
        has_configs = check_configs()
    """
    config_dir = os.path.join(appdirs.user_config_dir('kiss'), 'configs')
    return os.path.exists(config_dir)

if check_configs():
    CONFIGS: DotWiz = get_configs()
else:
    CONFIGS: DotWiz = get_default_configs()
    save_configs(CONFIGS)
