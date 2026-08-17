"""Cho phép chạy ``python -m services.chunking`` như debug CLI."""

from .debug import main


if __name__ == "__main__":
    raise SystemExit(main())
