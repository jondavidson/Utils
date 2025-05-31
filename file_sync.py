"""file_sync.py

A class‑based utility for synchronising directories between a local machine
and one or more remote hosts over SSH/SFTP, inspired by **rsync** but written
purely in Python.

Key features
------------
* Incremental sync – transfers only files that are new or have changed
  (by mtime/size or optional SHA‑256 checksum).
* Push or pull direction (local→remote or remote→local).
* Recursive directory traversal.
* Optional deletion of extraneous files on the destination.
* **Selective date‑range sync** – limit transfer to YYYYMMDD sub‑directories
  between *start_date* and *end_date* (inclusive).
* Extensible transport layer (currently Paramiko SFTP; swap in SCP/HTTP, etc.).
* Verbose progress display with **tqdm**.

Example
-------
```python
from datetime import date
from file_sync import FileSync, SSHConfig

cfg = SSHConfig(host="my.server.com", username="alice")
syncer = FileSync(cfg)

# Push only backups created between 31 May and 2 June 2025, located in
# sub‑directories named YYYYMMDD under /home/alice/backups
syncer.sync(
    local_dir="/home/alice/backups", 
    remote_dir="/data/backups",
    direction="push",
    start_date=date(2025, 5, 31),
    end_date="20250602",           # str or datetime.date are accepted
    include_non_dated=False        # skip everything outside dated dirs
)
```
"""

from __future__ import annotations

import hashlib
import os
import stat
import time
from dataclasses import dataclass
from datetime import date as _date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import paramiko
from tqdm import tqdm

__all__ = [
    "SSHConfig",
    "FileSync",
]

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path, bufsize: int = 131_072) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_remote(
    sftp: paramiko.SFTPClient, remote_path: str, bufsize: int = 131_072
) -> str:
    h = hashlib.sha256()
    with sftp.open(remote_path, "rb") as f:
        while True:
            chunk = f.read(bufsize)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _to_date(val: str | _date | None) -> _date | None:
    """Convert *val* to a :class:`datetime.date` if possible.

    Accepts:
    * ``datetime.date`` – returned unchanged
    * ``YYYYMMDD`` or ``YYYY-MM-DD`` strings
    * ``None`` – returned unchanged
    """
    if val is None or isinstance(val, _date):
        return val  # type: ignore[return-value]

    if not isinstance(val, str):
        raise TypeError("start_date/end_date must be str | datetime.date | None")

    txt = val.replace("-", "").replace("/", "")
    try:
        return datetime.strptime(txt, "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError(
            f"Invalid date string '{val}'. Expected YYYYMMDD or YYYY-MM-DD"
        ) from exc


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------

@dataclass
class SSHConfig:
    """Minimal SSH configuration needed to open a Paramiko connection."""

    host: str
    port: int = 22
    username: str | None = None
    password: str | None = None
    key_filename: str | None = None
    timeout: float | None = 15.0

    def connect(self) -> paramiko.SSHClient:
        """Open and return an SSHClient with an SFTP subsystem."""
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
            key_filename=self.key_filename,
            timeout=self.timeout,
        )
        return client


# ---------------------------------------------------------------------------
# Metadata container
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FileMeta:
    """Cheap serialisable metadata for a single file or directory."""

    path: str  # relative to base
    size: int
    mtime: float
    sha256: str | None = None

    # -------------- constructors --------------

    @classmethod
    def from_stat(
        cls, base: Path, file_path: Path, use_checksums: bool = False
    ) -> "FileMeta":
        st = file_path.stat()
        sha = None
        if use_checksums and file_path.is_file():
            sha = _sha256(file_path)
        rel = str(file_path.relative_to(base))
        return cls(rel, st.st_size, st.st_mtime, sha)

    @classmethod
    def from_sftp_attr(
        cls,
        base: str,
        attr: paramiko.SFTPAttributes,
        use_checksums: bool = False,
        sftp: paramiko.SFTPClient | None = None,
    ) -> "FileMeta":
        rel = os.path.relpath(attr.filename, base)
        sha = None
        if use_checksums and sftp and stat.S_ISREG(attr.st_mode):
            sha = _sha256_remote(sftp, attr.filename)
        return cls(rel, attr.st_size, attr.st_mtime, sha)

    # -------------- comparison --------------

    def is_different(self, other: "FileMeta", compare_checksums: bool) -> bool:
        """Return True if two FileMeta objects differ in relevant aspects."""
        if self.size != other.size or abs(self.mtime - other.mtime) > 1:
            return True
        if compare_checksums:
            return self.sha256 != other.sha256
        return False


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FileSync:
    """Synchronise a directory tree between local and remote hosts."""

    # ------------------------------------------------------------
    # Construction / context‑management
    # ------------------------------------------------------------

    def __init__(self, ssh_config: SSHConfig) -> None:
        self.cfg = ssh_config
        self.ssh: paramiko.SSHClient | None = None
        self.sftp: paramiko.SFTPClient | None = None

    def close(self) -> None:
        if self.sftp:
            self.sftp.close()
        if self.ssh:
            self.ssh.close()

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def sync(
        self,
        local_dir: str | os.PathLike,
        remote_dir: str,
        *,
        direction: str = "push",
        use_checksums: bool = False,
        delete_extraneous: bool = False,
        dry_run: bool = False,
        # ---- new selective date‑range options ----
        start_date: str | _date | None = None,
        end_date: str | _date | None = None,
        date_component_index: int = 0,
        date_format: str = "%Y%m%d",
        include_non_dated: bool = True,
    ) -> None:
        """Synchronise *local_dir* <-> *remote_dir*.

        Parameters
        ----------
        direction
            "push" to copy local changes to remote, "pull" for the opposite.
        use_checksums
            If *True*, calculate SHA‑256 digests to detect changes.
        delete_extraneous
            Mirror behaviour – delete files that exist only in the destination.
        dry_run
            Compute the plan but do not transfer or delete anything.
        start_date / end_date
            If provided, restrict synchronisation to files whose **ancestor
            component** at *date_component_index* is a valid date in
            *date_format* and falls within the inclusive range
            ``[start_date, end_date]``. Accepts ``datetime.date`` or strings
            like ``"20250601"`` or ``"2025-06-01"``. ``None`` means open end.
        date_component_index
            Which path component (0‑based) should be interpreted as the date.
            For a layout like ``base/YYYYMMDD/hh/file``, you might set it to 0
            (default). If the date appears deeper (e.g. ``snapshots/DAILY/
            YYYYMMDD/…``) adjust accordingly.
        date_format
            ``strptime`` format string used to parse the component – change if
            you use a different naming convention (e.g. ``"%Y-%m-%d"``).
        include_non_dated
            *True* – keep files whose component **cannot** be parsed as a date
            (i.e. treat them as always in range). *False* – exclude them.
        """
        if direction not in {"push", "pull"}:
            raise ValueError("direction must be 'push' or 'pull'")
        local_base = Path(local_dir).expanduser().resolve()

        # Prepare date filter -------------------------------------------------
        sd = _to_date(start_date)
        ed = _to_date(end_date)
        if sd and ed and sd > ed:
            raise ValueError("start_date must be <= end_date")

        def in_date_range(rel: str) -> bool:  # closure captures sd/ed & args
            comps = rel.split("/")
            if len(comps) <= date_component_index:
                # Path does not have the component.
                return include_non_dated
            comp = comps[date_component_index]
            try:
                d = datetime.strptime(comp, date_format).date()
            except ValueError:
                return include_non_dated
            if sd and d < sd:
                return False
            if ed and d > ed:
                return False
            return True

        # Establish connection only once --------------------------------------
        self._ensure_connection()

        # Gather metadata ------------------------------------------------------
        local_meta = self._scan_local(local_base, use_checksums)
        remote_meta = self._scan_remote(remote_dir, use_checksums)

        if sd or ed or not include_non_dated:
            local_meta = {
                k: v for k, v in local_meta.items() if in_date_range(k)
            }
            remote_meta = {
                k: v for k, v in remote_meta.items() if in_date_range(k)
            }

        # Plan changes ---------------------------------------------------------
        if direction == "push":
            src_meta, dst_meta = local_meta, remote_meta
            src_base, dst_base = local_base, remote_dir
            transfer = self._upload
        else:
            src_meta, dst_meta = remote_meta, local_meta
            src_base, dst_base = remote_dir, local_base
            transfer = self._download

        to_copy, to_delete = self._compare_trees(
            src_meta, dst_meta, use_checksums, delete_extraneous
        )

        self._log_plan(
            to_copy,
            to_delete,
            direction,
            dst_base,
            sd,
            ed,
            date_component_index,
            date_format,
            include_non_dated,
        )

        if dry_run:
            print("[DRY‑RUN] No changes performed.")
            return

        # Execute plan ---------------------------------------------------------
        self._copy_files(to_copy, src_base, dst_base, transfer)
        if delete_extraneous:
            self._delete_files(to_delete, dst_base, direction)

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------

    # ------ connection ------

    def _ensure_connection(self) -> None:
        if self.ssh and self.sftp:
            return
        self.ssh = self.cfg.connect()
        self.sftp = self.ssh.open_sftp()

    # ------ scanning ------

    def _scan_local(
        self, base: Path, use_checksums: bool
    ) -> Dict[str, FileMeta]:
        meta: Dict[str, FileMeta] = {}
        for file_path in base.rglob("*"):
            if file_path.is_symlink():
                continue  # ignore symlinks
            meta[file_path.relative_to(base).as_posix()] = FileMeta.from_stat(
                base, file_path, use_checksums
            )
        return meta

    def _scan_remote(
        self, base: str, use_checksums: bool
    ) -> Dict[str, FileMeta]:
        assert self.sftp
        meta: Dict[str, FileMeta] = {}
        pending: List[str] = [base]
        while pending:
            current = pending.pop()
            for attr in self.sftp.listdir_attr(current):
                full = f"{current}/{attr.filename}"
                if stat.S_ISLNK(attr.st_mode):
                    continue
                if stat.S_ISDIR(attr.st_mode):
                    pending.append(full)
                meta_val = FileMeta.from_sftp_attr(
                    base, attr, use_checksums, self.sftp
                )
                meta[meta_val.path] = meta_val
        return meta

    # ------ comparison ------

    def _compare_trees(
        self,
        src_meta: Dict[str, FileMeta],
        dst_meta: Dict[str, FileMeta],
        compare_checksums: bool,
        delete_extraneous: bool,
    ) -> Tuple[List[str], List[str]]:
        to_copy: List[str] = []
        to_delete: List[str] = []

        # changed + new files
        for rel, sm in src_meta.items():
            dm = dst_meta.get(rel)
            if dm is None or sm.is_different(dm, compare_checksums):
                to_copy.append(rel)

        if delete_extraneous:
            for rel in dst_meta:
                if rel not in src_meta:
                    to_delete.append(rel)

        return to_copy, to_delete

    # ------ transfer ------

    def _copy_files(
        self,
        files: List[str],
        src_base: str | Path,
        dst_base: str | Path,
        transfer_func,
    ) -> None:
        for rel in tqdm(files, desc="transferring", unit="file"):
            transfer_func(rel, src_base, dst_base)

    def _upload(
        self, rel: str, local_base: Path, remote_base: str
    ) -> None:
        assert self.sftp
        lpath = local_base / rel
        rpath = f"{remote_base}/{rel}"
        # ensure remote dir exists
        self._mkdir_remote(os.path.dirname(rpath))
        self.sftp.put(str(lpath), rpath)

        # preserve mtime
        st = lpath.stat()
        self.sftp.utime(rpath, (st.st_atime, st.st_mtime))

    def _download(
        self, rel: str, remote_base: str, local_base: Path
    ) -> None:
        assert self.sftp
        rpath = f"{remote_base}/{rel}"
        lpath = local_base / rel
        # ensure local dir exists
        lpath.parent.mkdir(parents=True, exist_ok=True)
        self.sftp.get(rpath, str(lpath))

        # preserve mtime
        attr = self.sftp.stat(rpath)
        os.utime(lpath, (attr.st_atime, attr.st_mtime))

    # ------ deletion ------

    def _delete_files(
        self, rel_paths: List[str], base: str | Path, direction: str
    ) -> None:
        prefix = "remote" if direction == "push" else "local"
        for rel in tqdm(rel_paths, desc=f"deleting {prefix}", unit="file"):
            full = f"{base}/{rel}" if isinstance(base, str) else (base / rel)
            try:
                if isinstance(full, Path):
                    if full.is_dir():
                        full.rmdir()
                    else:
                        full.unlink()
                else:
                    assert self.sftp
                    try:
                        self.sftp.remove(full)
                    except IOError:
                        self.sftp.rmdir(full)
            except Exception as exc:
                print(f"[WARN] Failed to delete {full}: {exc}")

    # ------ misc ------

    def _mkdir_remote(self, remote_dir: str) -> None:
        """Create remote directories recursively (mkdir -p)."""
        assert self.sftp
        dirs: List[str] = []
        while remote_dir not in {"", "/"}:
            try:
                self.sftp.stat(remote_dir)
                break
            except IOError:
                dirs.append(remote_dir)
                remote_dir = os.path.dirname(remote_dir)
        for d in reversed(dirs):
            self.sftp.mkdir(d)

    @staticmethod
    def _log_plan(
        to_copy: List[str],
        to_delete: List[str],
        direction: str,
        dst_base: str | Path,
        sd: _date | None,
        ed: _date | None,
        comp_idx: int,
        date_fmt: str,
        inc_non_dated: bool,
    ) -> None:
        print("==== Sync plan ====")
        print(f"Direction      : {direction}")
        print(f"Destination    : {dst_base}")
        print(f"Copy           : {len(to_copy)} files")
        print(f"Delete         : {len(to_delete)} files")
        if sd or ed or not inc_non_dated:
            r = (
                f"[{sd.isoformat() if sd else '-∞'} – {ed.isoformat() if ed else '∞'}]"
            )
            print(
                "Date filter    : component #{}, format '{}', range {} ({} non‑dated)".format(
                    comp_idx,
                    date_fmt,
                    r,
                    "include" if inc_non_dated else "exclude",
                )
            )
        print("===================")
        time.sleep(0.1)  # allow tqdm to start cleanly
