import click
from kiss.utils.configs import CONFIGS, save_configs, get_default_configs, clear_configs_dir
from kiss.utils.strings import print_key_value, Format

@click.group()
def cli():
    """
    KISS package command line interface.
    
    Provides necessary commands to handle configuration-related operations and experiments. 
    """
    pass

@cli.command()
def hello():
    """
    Displays hello message.
    
    Hello message printed without warnings nor errors usually mean successful installation.
    """
    with Format(Format.BOLD, Format.GREEN):
        print("Everything seems to work fine! MWAH \U0001F618\U0001F609")

@cli.group()
def config():
    """
    Group of configuration-related operations.
    """
    pass

@config.command()
def reset():
    """
    Resets configurations to default values.
    """
    click.confirm("Are you sure you want to reset configurations to default values?", default=False, abort=True)

    clear_configs_dir()
    save_configs(get_default_configs())
    click.echo("Configurations reset to default values.")

@config.command()
def display():
    """
    Displays current configurations.
    """
    print_key_value(CONFIGS)

@config.command()
@click.option('--key', '-k', required=True, type=str, help='Key to be set.')
@click.option('--value', '-v', required=True, type=str, help='Value of the key to be set')
def set(key: str, value: str):
    """
    Sets a configuration key to a specified value.
    
    \b
    Warning:
        Use colons to access nested keys.
    
    \b
    Example:
        kiss config set --key torch:device --value mps
    """
    from functools import reduce
    from operator import getitem

    keys = key.split(':')

    reduce(getitem, keys[:-1], CONFIGS)[keys[-1]] = value
    save_configs(CONFIGS)
