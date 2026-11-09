from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict

from .utils import deep_update, get_dict_hash

T = TypeVar("T", bound=BaseModel)


class BaseModelNoExtra(BaseModel):
    model_config = ConfigDict(extra="forbid")


def get_pydantic_hash(obj: BaseModel, exclude: dict[str, Any] | None = None) -> str:
    return get_dict_hash(obj.model_dump(exclude=exclude))


def update_pydantic_model_with_dict(model_instance: T, update_data: dict[str, Any]) -> T:
    merged = deep_update(model_instance.model_dump(), update_data)
    return type(model_instance).model_validate(merged)
