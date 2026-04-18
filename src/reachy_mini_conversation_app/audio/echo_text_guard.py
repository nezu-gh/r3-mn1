"""Text-level echo guard.

Belt-and-suspenders for residual speaker→mic leakage that slips past the AEC:
if an ASR transcription has high word overlap with something the robot just
said, drop it. Pattern cribbed from algal/sparky's ``_looks_like_echo_of_tts``
(``state_machine.py:1577``).

Kept deliberately simple — word set containment plus a short time window.
False positives (user legitimately repeating a word the bot said ~seconds ago)
are possible but rare; the window cap keeps them to cases where it's genuinely
ambiguous.
"""

from __future__ import annotations

import re
import time
from collections import deque


_WORD_RE = re.compile(r"[a-zA-ZäöüÄÖÜß0-9]+")


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text)}


class RecentTTSTextGuard:
    """Rolling set of recent TTS sentences with an echo-match check."""

    def __init__(self, window_secs: float = 3.0, threshold: float = 0.6, max_entries: int = 16) -> None:
        """Initialise the guard.

        Args:
            window_secs: How long a TTS sentence is considered a possible
                echo source. Short enough that legitimate repetition of
                earlier content isn't swallowed.
            threshold: Containment score in [0, 1] at or above which a
                transcript is treated as an echo. 0.6 means ~60% of the
                user's words overlap with the TTS.
            max_entries: Cap on the buffer so it doesn't grow unbounded
                in long sessions.
        """
        self._window_secs = window_secs
        self._threshold = threshold
        self._entries: deque[tuple[float, set[str]]] = deque(maxlen=max_entries)

    def note(self, text: str) -> None:
        """Record a TTS sentence so future transcripts can be checked against it."""
        tokens = _tokens(text)
        if not tokens:
            return
        self._entries.append((time.monotonic(), tokens))

    def looks_like_echo(self, text: str) -> bool:
        """Return True if *text*'s words are mostly contained in recent TTS."""
        user_tokens = _tokens(text)
        if not user_tokens:
            return False
        cutoff = time.monotonic() - self._window_secs
        for ts, tts_tokens in self._entries:
            if ts < cutoff:
                continue
            if not tts_tokens:
                continue
            overlap = len(user_tokens & tts_tokens) / len(user_tokens)
            if overlap >= self._threshold:
                return True
        return False

    def clear(self) -> None:
        """Drop all recorded TTS sentences (e.g. on session reset)."""
        self._entries.clear()
