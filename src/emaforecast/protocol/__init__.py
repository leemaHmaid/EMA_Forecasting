"""The rolling-origin forecasting protocol (described in the README)."""
from emaforecast.protocol import tables
from emaforecast.protocol.rolling import rolling_origin

__all__ = ["rolling_origin", "tables"]
