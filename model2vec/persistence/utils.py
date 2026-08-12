from typing import Any

from tqdm.auto import tqdm


class SilentTqdm(tqdm):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Init a tqdm that's disabled by default."""
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)
