from typing import Any, Protocol


class Regressor(Protocol):
    """Protocol for regression models"""

    def fit(self, X: Any, y: Any, sample_weight: Any | None = None) -> "Regressor": ...

    def predict(self, X: Any) -> Any: ...
