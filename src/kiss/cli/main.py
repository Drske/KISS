import click
from kiss.utils.configs import CONFIGS, save_configs, get_default_configs
from kiss.utils.strings import print_key_value, Format

@click.group()
def cli():
    """
    Top-level click command group for the CLI.

    Example:
        $ kiss config display
    """
    pass

@cli.command()
def hello():
    """
    Display hello message.

    Example:
        $ kiss hello
    """
    with Format(Format.BOLD, Format.GREEN):
        print("Everything seems to work fine! MWAH \U0001F618\U0001F609")

@cli.group()
def config():
    """
    Sub-command group for configuration-related operations.

    Example:
        $ kiss config reset
    """
    pass

@config.command()
def reset():
    """
    Reset configurations to default values.

    Example:
        $ kiss config reset
    """
    save_configs(get_default_configs())

@config.command()
def display():
    """
    Display current configurations.

    Example:
        $ kiss config display
    """
    print_key_value(CONFIGS)

@config.command()
@click.option('--key', '-k', required=True, type=str, help='Key to be set. Use colon to access nested keys.')
@click.option('--value', '-v', required=True, type=str, help='Value of the key to be set')
def set(key: str, value: str):
    """
    Set a configuration key to a specified value.

    Args:
        key (str): The configuration key to be set. Use colon to access nested keys.
        value (str): The value to set for the specified key.

    Example:
        $ kiss config set --key api:numista:key --value my_api_key
    """
    from functools import reduce
    from operator import getitem

    keys = key.split(':')

    reduce(getitem, keys[:-1], CONFIGS)[keys[-1]] = value
    save_configs(CONFIGS)
