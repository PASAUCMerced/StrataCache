from __future__ import annotations

import threading
import math

_shared_telemetry_lock = threading.Lock()

def thread_safe(func):
    def wrapper(*args, **kwargs):
        with _shared_telemetry_lock:
            result = func(*args, **kwargs)
        return result

    return wrapper

def human_readable_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"
