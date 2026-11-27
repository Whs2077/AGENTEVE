"""Read and write JSON, and hash files. No scoring."""

import hashlib
import json
from pathlib import Path


def readJson(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def writeJson(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fileSha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
