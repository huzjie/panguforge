"""Text preprocessing utilities."""
import re

_WS = re.compile(r"\s+")


def clean_text(s):
    return _WS.sub(" ", s).strip()


def chunk_text(s, chunk_size=512):
    s = clean_text(s)
    return [s[i:i + chunk_size] for i in range(0, len(s), chunk_size)] or [""]


def dedupe(lines):
    seen = set()
    out = []
    for l in lines:
        if l not in seen:
            seen.add(l)
            out.append(l)
    return out
