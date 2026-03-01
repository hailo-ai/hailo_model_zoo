from typing import Any, Optional


class Factory:
    def __init__(self, name: str) -> None:
        self._name = name
        self._name_to_callable = {}

    def _register_impl(self, name: str, callable: Any) -> None:
        assert name not in self._name_to_callable, f"'{name}' already registered in '{self._name}' registry!"
        self._name_to_callable[name] = callable

    def register(self, callable: Any = None, *, name: Optional[str] = None) -> Any:
        if callable is None:
            # used as a decorator
            def wrapper(func_or_class: Any) -> Any:
                final_name = name or func_or_class.__name__
                self._register_impl(final_name, func_or_class)
                return func_or_class

            return wrapper

        # used as a function call
        final_name = name or callable.__name__
        self._register_impl(final_name, callable)
        return callable

    def get(self, name: str) -> Any:
        ret = self._name_to_callable.get(name)
        if ret is None:
            raise ValueError(f"'{name}' not recognized in '{self._name}' factory ")
        return ret
