"""Text-level echo guard.

Belt-and-suspenders for residual speaker→mic leakage that slips past the AEC:
if an ASR transcription has high word overlap with something the robot just
said, drop it. Algorithm lifted from algal/sparky's `_looks_like_echo_of_tts`
(``sparky_mvp/core/state_machine.py:1577-1619``). Kept close to the original
so behaviour is directly comparable.
"""

from __future__ import annotations

import re
import time
from collections import deque


# Non-alphanumeric -> single space. Lowercases first.
_ECHO_RE = re.compile(r"[^a-z0-9]+")


def normalize_text_for_echo(text: str) -> str:
    t = (text or "").lower().strip()
    t = _ECHO_RE.sub(" ", t)
    return " ".join(t.split())


def _words_for_echo(norm_text: str) -> set[str]:
    # Skip very short tokens which contribute mostly noise.
    return {w for w in norm_text.split() if len(w) >= 3}


class RecentTTSTextGuard:
    """Rolling buffer of recent TTS sentences with a sparky-style echo check.

    Each entry stores (monotonic_timestamp, normalized_text). The guard tests
    an ASR transcript against every buffered entry within ``window_secs`` and
    returns True if any of the following hold:

      - the transcript is a substring of the TTS sentence (≥ ``min_chars``);
      - word-set Jaccard ≥ ``threshold``;
      - user-containment (|∩| / |user|) ≥ ``threshold``;
      - tts-containment (|∩| / |tts|)  ≥ ``threshold``;

    subject to a minimum overlap size that scales down for short utterances.
    """

    def __init__(
        self,
        window_secs: float = 12.0,
        threshold: float = 0.78,
        min_overlap: int = 6,
        min_chars: int = 8,
        max_entries: int = 200,
    ) -> None:
        self._window_secs = window_secs
        self._threshold = threshold
        self._min_overlap = min_overlap
        self._min_chars = min_chars
        self._entries: deque[tuple[float, str]] = deque(maxlen=max_entries)

    def note(self, text: str) -> None:
        """Record a TTS sentence so future transcripts can be checked against it."""
        norm = normalize_text_for_echo(text)
        if not norm:
            return
        now = time.monotonic()
        self._entries.append((now, norm))
        # Prune by time window
        cutoff = now - self._window_secs
        while self._entries and self._entries[0][0] < cutoff:
            self._entries.popleft()

    def looks_like_echo(self, text: str) -> bool:
        """Return True if *text* looks like an echo of recent TTS output."""
        user_norm = normalize_text_for_echo(text)
        if not user_norm:
            return False
        user_words = _words_for_echo(user_norm)
        if not self._entries:
            return False

        now = time.monotonic()
        for ts, tts_norm in reversed(self._entries):
            if now - ts > self._window_secs:
                break
            tts_words = _words_for_echo(tts_norm)
            if not tts_words:
                continue

            # Short-utterance echo (e.g., "oh wonderful") should still be
            # blocked if the transcript is literally a substring of the
            # TTS sentence.
            if len(user_norm) >= self._min_chars and user_norm in tts_norm:
                return True

            inter = user_words & tts_words
            # Scale the overlap requirement down for short utterances so
            # "wonderful" vs "oh wonderful" can still match.
            required_overlap = min(
                self._min_overlap,
                max(1, min(len(user_words), len(tts_words))),
            )
            if len(inter) < required_overlap:
                continue

            union = user_words | tts_words
            jaccard = (len(inter) / len(union)) if union else 0.0
            contain_user = len(inter) / max(1, len(user_words))
            contain_tts = len(inter) / max(1, len(tts_words))

            if (
                jaccard >= self._threshold
                or contain_user >= self._threshold
                or contain_tts >= self._threshold
            ):
                return True

        return False

    def clear(self) -> None:
        """Drop all recorded TTS sentences (e.g. on session reset)."""
        self._entries.clear()
