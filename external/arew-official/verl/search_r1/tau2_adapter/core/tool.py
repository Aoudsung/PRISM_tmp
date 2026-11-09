import inspect
import re
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    short_desc: str
    func: Callable[..., Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    @property
    def signature(self) -> inspect.Signature:
        return inspect.signature(self.func)

    @staticmethod
    def _compact_annotation(annotation: Any) -> str:
        if annotation is inspect._empty:
            return ""
        if isinstance(annotation, type):
            return annotation.__name__
        text = str(annotation)
        text = text.replace("typing.", "")
        text = re.sub(r"\b(?:[A-Za-z_]\w*\.)+([A-Za-z_]\w*)\b", r"\1", text)
        return text

    def _compact_signature(self) -> str:
        parts = []
        for param in self.signature.parameters.values():
            piece = param.name
            annotation = self._compact_annotation(param.annotation)
            if annotation:
                piece += f": {annotation}"
            if param.default is not inspect._empty:
                piece += f" = {param.default!r}"
            parts.append(piece)
        return f"({', '.join(parts)})"

    def to_str(self) -> str:
        return f"{self.name}{self._compact_signature()}: {self.short_desc}"


def as_tool(func: Callable[..., Any]) -> Tool:
    doc = inspect.getdoc(func) or ""
    short_desc = doc.splitlines()[0] if doc else func.__name__
    return Tool(name=func.__name__, short_desc=short_desc, func=func)
