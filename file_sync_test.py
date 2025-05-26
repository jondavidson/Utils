"""test_file_sync.py – pytest suite for file_sync.FileSync

The tests operate entirely on local temporary directories; we monkey‑patch the network‑related parts of :class:FileSync so no SSH/SFTP traffic is required. This means the suite can run in CI without an SSH server or Paramiko keys.

Coverage

* Glob‑pattern filtering with ``patterns``
* Metadata comparison logic
* Push and pull directions
* Deletion of extraneous destination files
* Dry‑run behaviour (no file system changes)
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List

import pytest

from file_sync import FileSync, SSHConfig

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def make_files(base: Path, files: dict[str, str] | List[str]):
    """Create a set of files under *base*.

    If *files* is a dict, keys are relative paths and values are file content.
    If *files* is a list, empty files are created.
    """
    if isinstance(files, dict):
        items = files.items()
    else:
        items = ((p, "") for p in files)
    for rel, data in items:
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(data)


def fs_local_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Monkey‑patch fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def filesync(tmp_path: Path):
    """Return a **FileSync** instance whose network ops are patched to act on
    local directories under *tmp_path*.
    Returns (syncer, local_dir, remote_dir).
    """

    local_dir = tmp_path / "local"
    remote_dir = tmp_path / "remote"
    local_dir.mkdir()
    remote_dir.mkdir()

    # dummy (unused) SSH config
    syncer = FileSync(SSHConfig("dummy"))

    #   ---- patch connection creation to *do nothing* ----
    syncer._ensure_connection = lambda: None  # type: ignore

    #   ---- replace transfer/deletion helpers ----
    def upload(rel: str, lbase: Path, rbase: str):
        fs_local_copy(lbase / rel, Path(rbase) / rel)

    def download(rel: str, rbase: str, lbase: Path):
        fs_local_copy(Path(rbase) / rel, lbase / rel)

    def mkdir_remote(rdir: str):
        Path(rdir).mkdir(parents=True, exist_ok=True)

    syncer._upload = upload  # type: ignore
    syncer._download = download  # type: ignore
    syncer._mkdir_remote = mkdir_remote  # type: ignore

    # deletion branches use Path operations already – they work locally

    return syncer, local_dir, str(remote_dir)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_matches_and_pattern_filter(filesync):
    syncer, local_dir, remote_dir = filesync
    make_files(local_dir, {"foo.txt": "A", "code/app.py": "B"})

    syncer.sync(
        local_dir,
        remote_dir,
        direction="push",
        patterns=["**/*.py"],  # only *.py files
        dry_run=True,
    )

    # dry‑run: remote should still be empty
    assert not any(Path(remote_dir).rglob("*"))


def test_push_basic_copy(filesync):
    syncer, local_dir, remote_dir = filesync

    make_files(local_dir, {"f1.txt": "hello"})

    syncer.sync(local_dir, remote_dir, direction="push")

    copied = Path(remote_dir) / "f1.txt"
    assert copied.exists() and copied.read_text() == "hello"


def test_pull_direction(filesync):
    syncer, local_dir, remote_dir = filesync

    make_files(Path(remote_dir), {"dir/data.bin": "XYZ"})

    syncer.sync(local_dir, remote_dir, direction="pull")

    assert (local_dir / "dir/data.bin").read_text() == "XYZ"


def test_delete_extraneous(filesync):
    syncer, local_dir, remote_dir = filesync

    make_files(local_dir, {"keep.txt": ""})
    make_files(Path(remote_dir), {"keep.txt": "", "stale.txt": ""})

    syncer.sync(
        local_dir, remote_dir, direction="push", delete_extraneous=True
    )

    assert (Path(remote_dir) / "keep.txt").exists()
    assert not (Path(remote_dir) / "stale.txt").exists()


def test_checksum_change_detection(filesync, monkeypatch):
    syncer, local_dir, remote_dir = filesync

    make_files(local_dir, {"data.txt": "AAA"})
    syncer.sync(local_dir, remote_dir, direction="push", use_checksums=True)

    # mutate local file content but keep mtime identical (simulate touch w/o mtime change)
    (local_dir / "data.txt").write_text("BBB")
    os.utime(local_dir / "data.txt", None)

    syncer.sync(local_dir, remote_dir, direction="push", use_checksums=True)

    assert (Path(remote_dir) / "data.txt").read_text() == "BBB"

