from functools import wraps
import inspect


def catch_as_str(fmt="An exception has occurred: {e!r}"):
    """
    A decorator that wraps a function in a try/except block, handling exceptions based on the `fmt` parameter.
    - If `fmt` is a string, it uses this as a formatting string for all exceptions.
    - If `fmt` is a dictionary, it maps exceptions or sequences of exceptions to specific formatting strings or True/False.
        - True: Return the raw string of the exception.
        - False: Reraise the exception.
    - If `fmt` is True globally, it will return the raw string of any exception.
    - If `fmt` is False or not provided for a specific exception in a dictionary, it will re-raise the exception.
    - If `fmt` is a callable, it calls the callable with the exception as its argument.
    """
    # Process fmt if it maps sequences of exceptions to handlers
    if isinstance(fmt, dict):
        new_fmt = {}
        for key, value in fmt.items():
            if isinstance(key, (list, tuple)):  # Check if key is a sequence of exceptions
                for exc in key:
                    new_fmt[exc] = value
            else:
                new_fmt[key] = value
        fmt = new_fmt

    def format_exception(e):
        """ Formats the exception based on fmt rules. """
        if callable(fmt):
            return fmt(e)
        elif isinstance(fmt, dict):
            handler = fmt.get(type(e), fmt.get(True, fmt))
            if handler is True:
                return str(e)
            elif handler is False:
                raise
            else:
                return handler.format(e=str(e))
        elif fmt is True:
            return str(e)
        elif isinstance(fmt, str):
            return fmt.format(e=str(e))
        else:
            raise

    def decorator(func):
        coroutine = inspect.iscoroutinefunction(func)

        if not coroutine:
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    return format_exception(e)
        else:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    return format_exception(e)
        return wrapper

    return decorator


def add_docstring(docstring):
    def decorator(func):
        func.__doc__ = docstring
        return func
    return decorator
