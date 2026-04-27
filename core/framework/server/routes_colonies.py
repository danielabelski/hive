"""HTTP routes for colony import/export — moving a colony spec between hosts.

Today, just the import side: accept a `tar.gz` of a colony directory and
unpack it into ``HIVE_HOME/colonies/<name>/`` so a desktop client (or any
external mover) can hand a colony to a remote runtime to run.

  POST /api/colonies/import   -- multipart/form-data
    file              required  -- .tar / .tar.gz / .tar.bz2 / .tar.xz
    name              optional  -- override the colony name; defaults to the
                                   archive's single top-level directory
    replace_existing  optional  -- "true" to overwrite an existing colony,
                                   else 409 if the target dir exists
"""

from __future__ import annotations

import io
import logging
import re
import shutil
import tarfile
from pathlib import Path

from aiohttp import web

from framework.config import COLONIES_DIR

logger = logging.getLogger(__name__)

# Matches the convention used elsewhere in the codebase (see
# routes_colony_workers and queen_lifecycle_tools): lowercase alphanumerics
# and underscores only. No dots, no slashes — names are filesystem segments.
_COLONY_NAME_RE = re.compile(r"^[a-z0-9_]+$")

# 50 MB cap on upload size. Colonies bundle scripts, prompts, memories,
# and small data files; anything bigger usually shouldn't be in version
# control to begin with. Bump if a real use-case lands here.
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def _validate_colony_name(name: str) -> str | None:
    """Return an error message if name isn't a valid colony name, else None."""
    if not name:
        return "colony name is required"
    if len(name) > 64:
        return "colony name too long (max 64 chars)"
    if not _COLONY_NAME_RE.match(name):
        return "colony name must match [a-z0-9_]+"
    return None


def _archive_top_level(tf: tarfile.TarFile) -> tuple[str | None, str | None]:
    """Find the archive's single top-level directory, if it has one.

    Returns ``(name, error)``. Allows the archive to optionally include a
    leading ``./`` prefix on every member (some tar implementations emit this).
    """
    tops: set[str] = set()
    for member in tf.getmembers():
        # Reject empty / absolute / parent-traversal names early; the deeper
        # walker rejects them again, but failing fast here gives a cleaner
        # error message back to the caller.
        if not member.name or member.name.startswith("/"):
            return None, f"invalid member path: {member.name!r}"
        parts = Path(member.name).parts
        if not parts or parts[0] == "..":
            return None, f"invalid member path: {member.name!r}"
        # Skip the archive's own root entry if present (`tar` emits "./").
        first = parts[0] if parts[0] != "." else (parts[1] if len(parts) > 1 else "")
        if first:
            tops.add(first)
    if len(tops) != 1:
        return None, "archive must contain exactly one top-level directory"
    return next(iter(tops)), None


def _safe_extract_tar(tf: tarfile.TarFile, dest: Path, *, strip_prefix: str) -> str | None:
    """Extract every member of ``tf`` into ``dest``, stripping ``strip_prefix``.

    Each member's resolved path must stay under ``dest``; symlinks, hardlinks,
    and device/fifo entries are rejected. Returns an error string on failure.

    Python's ``tarfile.extractall(filter='data')`` does similar checks but
    only landed in 3.12; we run on 3.11+, so do the validation explicitly.
    """
    base = dest.resolve()
    base.mkdir(parents=True, exist_ok=True)

    for member in tf.getmembers():
        # Compute the relative target name after stripping the top-level dir.
        # Both "<prefix>/foo" and "./<prefix>/foo" map to "foo".
        name = member.name
        if name.startswith("./"):
            name = name[2:]
        if name == strip_prefix:
            # The top-level dir entry itself; nothing to extract beyond
            # making sure dest exists (handled above).
            continue
        prefix_with_sep = f"{strip_prefix}/"
        if not name.startswith(prefix_with_sep):
            return f"member outside top-level dir: {member.name!r}"
        rel = name[len(prefix_with_sep):]
        if not rel:
            continue
        # Reject any "..", absolute paths, or weird member types.
        if ".." in Path(rel).parts:
            return f"path traversal in member: {member.name!r}"
        if member.issym() or member.islnk():
            return f"symlinks/hardlinks not supported: {member.name!r}"
        if member.isdev() or member.isfifo():
            return f"device/fifo not supported: {member.name!r}"

        target = (base / rel).resolve()
        try:
            target.relative_to(base)
        except ValueError:
            return f"member escapes destination: {member.name!r}"

        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
            continue

        # Regular file. Extract via stream copy so we don't trust tarfile's
        # built-in path handling — we already resolved the destination.
        target.parent.mkdir(parents=True, exist_ok=True)
        src = tf.extractfile(member)
        if src is None:
            # Unknown member type that slipped past the checks above.
            return f"unsupported member: {member.name!r}"
        with target.open("wb") as out:
            shutil.copyfileobj(src, out)
        # Best-effort mode bits — masked to user-rwx + group/other-rx so we
        # don't accidentally honour world-writable bits from a tampered tar.
        target.chmod(member.mode & 0o755 if member.mode else 0o644)

    return None


async def handle_import_colony(request: web.Request) -> web.Response:
    """POST /api/colonies/import — unpack a colony tarball into HIVE_HOME."""
    if not request.content_type.startswith("multipart/"):
        return web.json_response(
            {"error": "expected multipart/form-data"}, status=400
        )

    reader = await request.multipart()
    upload: bytes | None = None
    upload_filename: str | None = None
    form: dict[str, str] = {}

    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name == "file":
            buf = io.BytesIO()
            while True:
                chunk = await part.read_chunk(size=65536)
                if not chunk:
                    break
                buf.write(chunk)
                if buf.tell() > _MAX_UPLOAD_BYTES:
                    return web.json_response(
                        {"error": f"upload exceeds {_MAX_UPLOAD_BYTES} bytes"},
                        status=413,
                    )
            upload = buf.getvalue()
            upload_filename = part.filename or ""
        else:
            form[part.name or ""] = (await part.text()).strip()

    if upload is None:
        return web.json_response({"error": "missing 'file' part"}, status=400)

    replace_existing = form.get("replace_existing", "false").lower() == "true"
    name_override = form.get("name", "").strip() or None

    # Open the archive — tarfile auto-detects compression with mode='r:*'.
    try:
        tf = tarfile.open(fileobj=io.BytesIO(upload), mode="r:*")
    except tarfile.TarError as err:
        return web.json_response(
            {"error": f"invalid tar archive: {err}"}, status=400
        )

    try:
        top, top_err = _archive_top_level(tf)
        if top_err or top is None:
            return web.json_response({"error": top_err}, status=400)

        colony_name = name_override or top
        name_err = _validate_colony_name(colony_name)
        if name_err:
            return web.json_response({"error": name_err}, status=400)

        target = COLONIES_DIR / colony_name
        if target.exists():
            if not replace_existing:
                return web.json_response(
                    {
                        "error": "colony already exists",
                        "name": colony_name,
                        "hint": "set replace_existing=true to overwrite",
                    },
                    status=409,
                )
            shutil.rmtree(target)

        extract_err = _safe_extract_tar(tf, target, strip_prefix=top)
        if extract_err:
            # Best-effort cleanup so a partial extract doesn't get left behind.
            shutil.rmtree(target, ignore_errors=True)
            return web.json_response({"error": extract_err}, status=400)
    finally:
        tf.close()

    files_imported = sum(1 for _ in target.rglob("*") if _.is_file())
    logger.info(
        "Imported colony %s (%d files) from upload %s (%d bytes)",
        colony_name,
        files_imported,
        upload_filename or "<unnamed>",
        len(upload),
    )

    return web.json_response(
        {
            "name": colony_name,
            "path": str(target),
            "files_imported": files_imported,
            "replaced": replace_existing and target.exists(),
        },
        status=201,
    )


def register_routes(app: web.Application) -> None:
    app.router.add_post("/api/colonies/import", handle_import_colony)
