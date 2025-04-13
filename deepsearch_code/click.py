import asyncio
from functools import wraps
from typing import Any, Callable

import click


def async_command(
    group: click.Group, **kws
) -> Callable[[Callable[..., Any]], click.Command]:
    def dec(f):
        @group.command(**kws)
        @wraps(f)
        def wrapper(*args, **kwargs):
            return asyncio.run(f(*args, **kwargs))

        return wrapper

    return dec
