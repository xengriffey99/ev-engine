"""NBA EV Engine v2 - Phase 0 market layer.

Close capture, de-vig, direction-aware price selection, CLV and the opinion log.
Spec: NBA_CONTEXT.md sections 4-6, 9 and the section 14 amendments (2026-10-21).
"""

__all__ = [
    "config",
    "odds",
    "devig",
    "quotes",
    "selection",
    "clv",
    "opinion_log",
    "odds_api",
    "scheduler",
    "feed_check",
    "store",
]
