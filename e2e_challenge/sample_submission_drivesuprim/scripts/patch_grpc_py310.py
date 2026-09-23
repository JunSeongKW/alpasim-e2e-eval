"""Patch AlpaSim's temporary gRPC build tree for Python 3.10."""

from pathlib import Path

path = Path("/tmp/alpasim_grpc/scripts/compile_protos.py")
source = path.read_text()
source = source.replace(
    "from contextlib import chdir",
    """from contextlib import contextmanager

@contextmanager
def chdir(path):
    import os
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)""",
)
path.write_text(source)

pyproject = Path("/tmp/alpasim_grpc/pyproject.toml")
source = pyproject.read_text()
source = source.replace('requires-python = ">=3.11,<3.13"', 'requires-python = ">=3.10,<3.13"')
pyproject.write_text(source)
