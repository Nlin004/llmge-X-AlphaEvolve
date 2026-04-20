"""Active benchmark configuration.

This module is imported from two contexts:
- as ``src.cfg.constants`` from the project root
- as ``cfg.constants`` when scripts under ``src/`` are executed directly

Keep the imports dual-path so both entry styles continue to work.
"""

try:
    from .constants_C880 import *  # noqa: F401,F403
except ImportError:
    from cfg.constants_C880 import *  # type: ignore # noqa: F401,F403
