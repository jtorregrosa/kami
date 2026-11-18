"""Console output formatting: symbols, subtle colours and a live progress line.

Everything goes through here so the CLI has one consistent style. It honours
NO_COLOR, skips colour when the output is not a terminal, and falls back to ASCII
when the console cannot render the symbols.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import time


def _encodable(text: str) -> bool:
    try:
        text.encode(sys.stdout.encoding or "ascii")
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def _enable_vt() -> bool:
    """Enable ANSI sequences on the Windows console (no-op elsewhere)."""
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


TTY = sys.stdout.isatty()
# Redirected output on Windows uses the ANSI code page (cp1252), which cannot encode
# some characters in messages (≈, —): replace them instead of crashing mid-command.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
COLOR = TTY and not os.environ.get("NO_COLOR") and _enable_vt()
_UNICODE = _encodable("›✓✗…⠋")
SYM = ({"step": "›", "ok": "✓", "warn": "!", "err": "✗", "dots": "…"} if _UNICODE
       else {"step": ">", "ok": "+", "warn": "!", "err": "x", "dots": "..."})
SYM["pick"] = "→" if _encodable("→") else ">"
FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏" if _UNICODE else "|/-\\"
BAR = ("━", "─") if _encodable("━─") else ("#", "-")
BAR_WIDTH = 20


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if COLOR else text


def dim(text: str) -> str:
    return _c(text, "2")


def bold(text: str) -> str:
    return _c(text, "1")


def green(text: str) -> str:
    return _c(text, "32")


def yellow(text: str) -> str:
    return _c(text, "33")


def red(text: str) -> str:
    return _c(text, "31")


def cyan(text: str) -> str:
    return _c(text, "36")


_CODE = re.compile(r"`([^`]+)`")


def inline(text: str) -> str:
    """Highlight text wrapped in `backticks` (commands) in cyan."""
    return _CODE.sub(lambda m: cyan(m.group(1)), text)


def step(msg: str) -> None:
    print(f"  {dim(SYM['step'])} {inline(msg)}")


def ok(msg: str) -> None:
    print(f"  {green(SYM['ok'])} {inline(msg)}")


def warn(msg: str) -> None:
    print(f"  {yellow(SYM['warn'])} {inline(msg)}")


def hint(msg: str) -> None:
    print(f"    {dim(inline(msg))}")


def error(message: str) -> None:
    """First line as a headline; the rest as indented detail (to stderr)."""
    sys.stdout.flush()  # keep it in order with the output printed before it
    head, *rest = str(message).splitlines() or [""]
    print(f"  {red(SYM['err'])} {bold(inline(head))}", file=sys.stderr)
    for line in rest:
        print(f"    {inline(line.strip())}", file=sys.stderr)


def field(label: str, value: str, width: int = 8) -> str:
    """Aligned label/value pair (padded before colouring)."""
    return f"{dim(label.ljust(width))} {value}"


def duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"


class Progress:
    """A progress line with a phase label, elapsed time and, optionally, a bar.

    In a terminal it is rewritten in place with a spinner; when the output is not
    a terminal it prints one line per phase change.
    """

    def __init__(self, text: str):
        self.text = text
        self.fraction: float | None = None
        self.detail = ""
        self.start = time.monotonic()
        self._frame = 0
        self._width = 0
        if not TTY:
            step(text + SYM["dots"])

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start

    def phase(self, text: str) -> None:
        if text == self.text:
            return
        self.text = text
        self.fraction, self.detail = None, ""
        if not TTY:
            step(text + SYM["dots"])

    def bar(self, fraction: float | None, detail: str = "") -> None:
        """A bar (0..1; None for none) and a detail after the current phase, until the
        next phase change."""
        self.fraction = None if fraction is None else min(1.0, max(0.0, fraction))
        self.detail = detail

    def tick(self) -> None:
        if not TTY:
            return
        frame = FRAMES[self._frame % len(FRAMES)]
        self._frame += 1
        plain = f"  {frame} {self.text}{SYM['dots']} {duration(self.elapsed)}"
        line = f"  {cyan(frame)} {self.text}{SYM['dots']} {dim(duration(self.elapsed))}"
        # what fits the terminal: a wrapped line would not be rewritten in place
        room = shutil.get_terminal_size().columns - 1 - len(plain)
        if self.fraction is not None and room > BAR_WIDTH + 7:
            filled = round(BAR_WIDTH * self.fraction)
            pct = f" {self.fraction * 100:3.0f}%"
            plain += f"  {BAR[0] * BAR_WIDTH}{pct}"
            line += f"  {cyan(BAR[0] * filled)}{dim(BAR[1] * (BAR_WIDTH - filled))}{pct}"
            room -= BAR_WIDTH + 7
        if self.detail and room > len(self.detail) + 2:
            plain += f"  {self.detail}"
            line += f"  {dim(self.detail)}"
        pad = " " * max(0, self._width - len(plain))
        self._width = len(plain)
        sys.stdout.write("\r" + line + pad)
        sys.stdout.flush()

    def stop(self) -> None:
        """Clear the progress line (in a terminal) so later output stays clean."""
        if TTY and self._width:
            sys.stdout.write("\r" + " " * self._width + "\r")
            sys.stdout.flush()
            self._width = 0
