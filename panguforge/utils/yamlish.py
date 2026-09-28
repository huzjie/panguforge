"""Minimal YAML-subset parser (zero dependency).

Supports the common YAML subset used by panguforge config files: nested mappings,
block lists, inline lists, scalars (str/int/float/bool/null), and comments.
This fallback lets the CLI read .yaml configs even when PyYAML is not installed.
"""
import re

_SCALAR_INT = re.compile(r"^-?\d+$")
_SCALAR_FLOAT = re.compile(r"^-?\d+\.\d+$")


def _scalar(v):
    v = v.strip()
    if v in ("null", "~"):
        return None
    if v in ("true", "True"):
        return True
    if v in ("false", "False"):
        return False
    if _SCALAR_INT.match(v):
        return int(v)
    if _SCALAR_FLOAT.match(v):
        return float(v)
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        return v[1:-1]
    return v


def _inline_list(v):
    v = v.strip()
    if not (v.startswith("[") and v.endswith("]")):
        return None
    inner = v[1:-1].strip()
    if not inner:
        return []
    parts = []
    cur = ""
    depth = 0
    for ch in inner:
        if ch in ("[", "{"):
            depth += 1
        elif ch in ("]", "}"):
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return [_scalar(p) for p in parts]


def loads(text):
    lines = text.splitlines()
    root = {}
    # stack of (indent, container, key)
    stack = [( -1, root, None)]
    for raw in lines:
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        if content.startswith("- "):
            item = _scalar(content[2:].strip())
            # find the list container on top
            _, cont, key = stack[-1]
            if isinstance(cont, dict) and key is not None:
                # parent mapping key -> list
                lst = cont.setdefault(key, [])
                lst.append(item)
            elif isinstance(cont, list):
                cont.append(item)
            continue
        if ":" in content:
            k, _, v = content.partition(":")
            k = k.strip()
            v = v.strip()
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if v == "":
                node = {}
                if isinstance(parent, list):
                    node = {}
                    parent.append(node)
                else:
                    parent[k] = node
                stack.append((indent, node, k))
            else:
                il = _inline_list(v)
                val = il if il is not None else _scalar(v)
                if isinstance(parent, list):
                    parent.append({k: val})
                else:
                    parent[k] = val
    return root


def load_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return loads(f.read())
