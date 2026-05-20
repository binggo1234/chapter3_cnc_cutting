from __future__ import annotations


class TerminalProgressBar:
    def __init__(self, total: int, *, enabled: bool = True, width: int = 28) -> None:
        self.total = max(total, 0)
        self.enabled = enabled
        self.width = max(width, 8)
        self.current = 0

    def start(self, label: str = "") -> None:
        if self.enabled:
            print(
                format_progress_line(0, self.total, "started", label, self.width),
                flush=True,
            )

    def advance(self, status: str, label: str = "") -> None:
        self.current += 1
        if self.enabled:
            print(
                format_progress_line(
                    self.current,
                    self.total,
                    status,
                    label,
                    self.width,
                ),
                flush=True,
            )


def format_progress_line(
    current: int,
    total: int,
    status: str,
    label: str = "",
    width: int = 28,
) -> str:
    safe_total = max(total, 0)
    safe_current = max(current, 0)
    if safe_total:
        shown_current = min(safe_current, safe_total)
        ratio = shown_current / safe_total
        percent = ratio * 100.0
        filled = round(width * ratio)
        count_text = f"{shown_current}/{safe_total}"
    else:
        percent = 100.0
        filled = width
        count_text = f"{safe_current}/0"

    bar = "#" * filled + "-" * (width - filled)
    suffix = f" {label}" if label else ""
    return f"progress [{bar}] {count_text:>9} {percent:5.1f}% {status}{suffix}"
