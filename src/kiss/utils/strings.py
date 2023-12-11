from dotwiz import DotWiz
from typing import List, Dict, Union


class Format:
    """
    Utility class for terminal text formatting.

    Attributes:
        BLACK (int): ANSI escape code for black text.
        RED (int): ANSI escape code for red text.
        GREEN (int): ANSI escape code for green text.
        ...
        CROSSED_OUT (int): ANSI escape code for crossed-out text.

    Methods:
        __init__: Initializes the Format object with specified color codes.
        __enter__: Enters the formatted text context.
        __exit__: Exits the formatted text context.

    Example:
        Usage of Format to format text:
        with Format(Format.RED, Format.BOLD):
            print("This is bold red text.")
        print("This is regular text.")
    """
    BLACK = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36
    WHITE = 37
    BRIGHT_BLACK = 90
    BRIGHT_RED = 91
    BRIGHT_GREEN = 92
    BRIGHT_YELLOW = 93
    BRIGHT_BLUE = 94
    BRIGHT_MAGENTA = 95
    BRIGHT_CYAN = 96
    BRIGHT_WHITE = 97
    BOLD = 1
    UNDERLINE = 4
    CROSSED_OUT = 9

    def __init__(self, *args: int) -> None:
        """
        Initialize the Format object with specified color codes.

        Args:
            *args (int): Variable number of color codes to apply.

        Example:
            format_instance = Format(Format.RED, Format.BOLD)
        """
        self.colors = args

    def __enter__(self) -> "Format":
        """
        Enter the formatted text context.

        Returns:
            Format: The Format object.

        Example:
            with Format(Format.RED, Format.BOLD):
                print("This is bold red text.")
        """
        for color in self.colors:
            print(f"\033[{color}m", end='', sep='')

        return self

    def __exit__(self, exc_type, value, traceback) -> None:
        """
        Exit the formatted text context.

        Args:
            exc_type: Exception type.
            value: Exception value.
            traceback: Exception traceback.

        Example:
            with Format(Format.RED, Format.BOLD):
                print("This is bold red text.")
            # Exiting the context restores the default text formatting.
        """

        print("\033[0m", end='', sep='')


def print_key_value(container: Union[List, Dict, "DotWiz"], indent: str = '') -> None:
    """
    Recursively prints the key-value pairs of a nested container.

    Args:
        container (list | dict | DotWiz): The container to print.
        indent (str): The indentation string.

    Example:
        Usage of print_key_value with a dictionary:
        data = {'name': 'John', 'age': 30, 'address': {'city': 'New York', 'zip': '10001'}}
        print_key_value(data)
    """
    if hasattr(container, '__dict__'):
        container = vars(container)

    if not isinstance(container, (list, dict, DotWiz)):
        with Format(Format.MAGENTA):
            print(f'{indent}{container}')

    if isinstance(container, list):
        for idx, element in enumerate(container):
            with Format(Format.GREEN, Format.BOLD):
                print(f'{indent}- Element no. {idx}:')
            print_key_value(element, f'{indent}  ')

    if isinstance(container, (dict, DotWiz)):
        for key, value in container.items():
            if isinstance(value, (list, dict, DotWiz)):
                with Format(Format.BOLD, Format.BRIGHT_BLUE):
                    print(f'{indent}{key}:')
                print_key_value(value, indent=indent + '  ')
            else:
                with Format(Format.BOLD, Format.BRIGHT_BLUE):
                    print(f'{indent}{key}: ', end='')
                with Format(Format.MAGENTA):
                    print(f'{value}')
