"""kami's mascot: a small floating spirit drawn with block characters.

Each pod state has its own pose (eyes and a spark). Disabled with KAMI_NO_MASCOT=1,
and skipped automatically when the console cannot render the characters.
"""

from __future__ import annotations

import os
import sys

from . import ui

HEAD = ["   ▄▄▄▄▄", "  ▟█████▙"]
BODY = "  ▐█████▌"
TAIL = "   ▀▄▀▄▀"

# eyes and spark per state
POSES = {
    "ready": ("  ▐█▀█▀█▌", " ✦"),    # eyes open, spark lit
    "wait":  ("  ▐█▄█▄█▌", ""),      # half-closed eyes: gathering form
    "fail":  ("  ▐█▙█▟█▌", " ·"),    # downcast, spark gone out
    "down":  ("  ▐█▀█▀█▌", ""),      # back to rest
    "rest":  ("  ▐█████▌", ""),      # closed eyes: no pod
}


def _art(pose: str) -> list[str]:
    face, spark = POSES[pose]
    return [*HEAD, face, BODY + spark, TAIL]


def _can_render() -> bool:
    if os.environ.get("KAMI_NO_MASCOT"):
        return False
    sample = "".join(HEAD + [BODY, TAIL] + [f + k for f, k in POSES.values()])
    try:
        sample.encode(sys.stdout.encoding or "ascii")
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def show(pose: str, lines: list[str], top: bool = False) -> None:
    """Show the mascot with text to its right (from its head, or from the top with
    top=True). If it cannot be drawn, print the text only.
    """
    lines = [ui.inline(line) for line in lines]
    if not _can_render():
        for line in lines:
            print(f"  {line}" if line else "")
        return
    art = _art(pose)
    # same text column for every pose (the widest spark wins)
    width = max(len(row) for p in POSES for row in _art(p))
    offset = 0 if top else 1
    for i, row in enumerate(art):
        j = i - offset
        text = lines[j] if 0 <= j < len(lines) else ""
        print(f"  {row:<{width}}  {text}".rstrip())
    for extra in lines[len(art) - offset:]:
        print(f"  {'':<{width}}  {extra}".rstrip())
