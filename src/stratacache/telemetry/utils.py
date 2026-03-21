from __future__ import annotations

import math

def human_readable_size(size_bytes, base=1024):
    if size_bytes <= 0:
        return "0 B"
    size_names = ("B", "KB", "MB", "GB", "TB", "PB")
    i = int(math.floor(math.log(size_bytes, base)))
    p = math.pow(base, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"
