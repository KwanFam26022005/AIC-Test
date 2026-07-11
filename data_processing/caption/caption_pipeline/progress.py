"""Small progress helpers for long caption pipeline stages."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def progress_iter(
    iterable: Iterable[T],
    *,
    total: int,
    desc: str,
    logger: logging.Logger,
    log_every: int | None = None,
    item_label: Callable[[T], str] | None = None,
) -> Iterator[T]:
    """Yield items with a terminal progress bar or periodic log progress."""
    if total <= 0:
        yield from iterable
        return

    if sys.stderr.isatty():
        try:
            from tqdm.auto import tqdm

            bar = tqdm(
                total=total,
                desc=desc,
                unit="item",
                dynamic_ncols=True,
                leave=True,
            )
            try:
                for item in iterable:
                    label = _safe_item_label(item, item_label)
                    if label:
                        bar.set_postfix_str(label, refresh=True)
                    yield item
                    bar.update(1)
            finally:
                bar.close()
            return
        except Exception:  # noqa: BLE001 - progress must never break the job.
            pass

    interval = log_every or max(1, min(10, total // 20 or 1))
    started = time.monotonic()
    logger.info("%s progress: 0/%d", desc, total)
    for idx, item in enumerate(iterable, start=1):
        label = _safe_item_label(item, item_label)
        if label and (idx == 1 or idx % interval == 0):
            logger.info("%s processing: %d/%d %s", desc, idx, total, label)
        yield item
        if idx == total or idx % interval == 0:
            elapsed = time.monotonic() - started
            rate = idx / elapsed if elapsed > 0 else 0.0
            remaining = (total - idx) / rate if rate > 0 else 0.0
            logger.info(
                "%s progress: %d/%d (%.1f%%), elapsed=%s, eta=%s",
                desc,
                idx,
                total,
                idx * 100.0 / total,
                _format_duration(elapsed),
                _format_duration(remaining),
            )


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes:d}m{sec:02d}s"
    return f"{sec:d}s"


def _safe_item_label(item: T, item_label: Callable[[T], str] | None) -> str:
    if item_label is None:
        return ""
    try:
        return str(item_label(item))[:80]
    except Exception:  # noqa: BLE001 - progress labels are best effort only.
        return ""
