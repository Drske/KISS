import time
from typing import Callable, Any

def TimeIt(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator that measures and prints the execution time of a function.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The decorated function.

    Example:
        Usage of TimeIt decorator:
        @TimeIt
        def example_function():
            # Your code here
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """
        Wrapper function that calculates and prints the execution time of the decorated function.

        Args:
            *args (Any): Positional arguments to pass to the decorated function.
            **kwargs (Any): Keyword arguments to pass to the decorated function.

        Returns:
            Any: The result of the decorated function.

        Example:
            This function is automatically called when using TimeIt as a decorator.
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        func_name = func.__name__
        print(f"{func_name} took {execution_time:.6f} seconds to execute.")
        return result

    return wrapper
