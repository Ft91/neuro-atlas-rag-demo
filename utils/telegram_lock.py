"""OS-held per-token lock across local atlas checkouts; no token is persisted."""
from hashlib import sha256
import os
from pathlib import Path
import tempfile


class TelegramTokenLock:
    def __init__(self, token, directory=None):
        if not token:
            raise ValueError("Telegram token is missing")
        self.directory = Path(directory) if directory else Path(tempfile.gettempdir()) / "codex-atlas-telegram-locks"
        self.name = sha256(token.encode()).hexdigest() + ".lock"
        self.handle = None

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        handle = (self.directory / self.name).open("a+b")
        try:
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            raise RuntimeError("Бот с этим токеном уже запущен локально") from None
        self.handle = handle
        return self

    def __exit__(self, *args):
        if self.handle:
            self.handle.close()  # OS also releases this lock after a crash.
            self.handle = None
