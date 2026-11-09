from typing import Any

from .io_utils import dump_file, load_file
from .pydantic_utils import BaseModelNoExtra, get_pydantic_hash


class DB(BaseModelNoExtra):
    @classmethod
    def load(cls, path: str) -> "DB":
        return cls.model_validate(load_file(path))

    def dump(self, path: str, exclude_defaults: bool = False, **kwargs: Any) -> None:
        dump_file(path, self.model_dump(exclude_defaults=exclude_defaults))

    def get_hash(self) -> str:
        return get_pydantic_hash(self)
