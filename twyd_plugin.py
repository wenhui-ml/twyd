"""Standalone ms-swift external plugin entry (absolute imports).

Usage: --external_plugins ./twyd_plugin.py
Loading twyd/plugin.py directly breaks its relative imports; this shim
imports the installed twyd package instead (pip install -e .).
"""
import twyd.data_pt  # noqa: F401  (registers twyd_pt dataset)
import twyd.plugin  # noqa: F401  (registers dense_lrc/core_lrc/plain_lrc + lrc_distill)

_TUNERS = twyd.plugin._TUNERS
_LOSSES = twyd.plugin._LOSSES
TwydInputCacheCallback = twyd.plugin.TwydInputCacheCallback

__all__ = ["_TUNERS", "_LOSSES", "TwydInputCacheCallback"]
