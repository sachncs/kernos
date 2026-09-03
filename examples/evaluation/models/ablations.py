"""Ablation presets for Kernos.

Each preset maps a display name to a dict of ``Plan`` flags.
"""

from __future__ import annotations

ABLATION_PRESETS: dict[str, dict[str, bool]] = {
    "K-NoRefresh": {"noref": True},
    "K-NoHysteresis": {"nohyst": True},
    "K-NoCooldown": {"nocool": True},
    "K-NoResidAnchors": {"noresid": True},
    "K-NoOrthog": {"noorth": True},
    "K-NoDivPenalty": {"nodiv": True},
    "K-NoFreeze": {"nofreeze": True},
}
"""Mapping from ablation name to ``Plan`` flag overrides."""
