"""Replacing a file in one step, so a crash never leaves a half-written one.

Its own module because it is not part of the checkpoint format: an exported
deployment artefact is written the same way (`ttrl.deploy.export`), and the
format modules beside this one are about what a checkpoint MEANS.

Write a temp file beside the destination, fsync it, then `os.replace` onto the
canonical path. A reader therefore sees either the previous file or the whole
new one, never a prefix of the new one, which matters most for the file a
`latest` symlink points at.
"""

from __future__ import annotations

import os
import uuid

__all__ = ["atomic_write_bytes"]


def atomic_write_bytes(path: str, data: bytes) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = os.path.join(directory, f".{os.path.basename(path)}.tmp-{uuid.uuid4().hex}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
