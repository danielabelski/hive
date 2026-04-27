"""Tests for POST /api/colonies/import — tar-based colony onboarding.

The handler resolves writes against ``framework.config.COLONIES_DIR``;
every test redirects that into a ``tmp_path`` so we never touch the real
``~/.hive/colonies`` tree.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer

from framework.server import routes_colonies


def _build_tar(layout: dict[str, bytes | None], *, gzip: bool = True) -> bytes:
    """Build an in-memory tar with the given paths.

    ``layout`` maps archive member names to file contents; passing ``None``
    creates a directory entry instead of a regular file.
    """
    buf = io.BytesIO()
    mode = "w:gz" if gzip else "w"
    with tarfile.open(fileobj=buf, mode=mode) as tf:
        for name, content in layout.items():
            if content is None:
                info = tarfile.TarInfo(name=name)
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                tf.addfile(info)
            else:
                info = tarfile.TarInfo(name=name)
                info.size = len(content)
                info.mode = 0o644
                tf.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def _build_tar_with_symlink(top: str, link_name: str, link_target: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name=top)
        info.type = tarfile.DIRTYPE
        info.mode = 0o755
        tf.addfile(info)
        sym = tarfile.TarInfo(name=f"{top}/{link_name}")
        sym.type = tarfile.SYMTYPE
        sym.linkname = link_target
        tf.addfile(sym)
    return buf.getvalue()


@pytest.fixture
def colonies_dir(tmp_path, monkeypatch):
    """Redirect COLONIES_DIR into a tmp tree."""
    colonies = tmp_path / "colonies"
    colonies.mkdir()
    monkeypatch.setattr(routes_colonies, "COLONIES_DIR", colonies)
    return colonies


async def _client(app: web.Application) -> TestClient:
    return TestClient(TestServer(app))


def _app() -> web.Application:
    app = web.Application()
    routes_colonies.register_routes(app)
    return app


def _form(file_bytes: bytes, *, filename: str = "colony.tar.gz", **fields: str) -> FormData:
    fd = FormData()
    fd.add_field("file", file_bytes, filename=filename, content_type="application/gzip")
    for k, v in fields.items():
        fd.add_field(k, v)
    return fd


@pytest.mark.asyncio
async def test_happy_path_imports_colony(colonies_dir: Path) -> None:
    archive = _build_tar(
        {
            "x_daily/": None,
            "x_daily/metadata.json": b'{"colony_name":"x_daily"}',
            "x_daily/scripts/run.sh": b"#!/bin/sh\necho hi\n",
        }
    )
    async with await _client(_app()) as c:
        resp = await c.post("/api/colonies/import", data=_form(archive))
        assert resp.status == 201, await resp.text()
        body = await resp.json()
    assert body["name"] == "x_daily"
    assert body["files_imported"] == 2
    assert (colonies_dir / "x_daily" / "metadata.json").read_bytes() == b'{"colony_name":"x_daily"}'
    assert (colonies_dir / "x_daily" / "scripts" / "run.sh").exists()


@pytest.mark.asyncio
async def test_name_override(colonies_dir: Path) -> None:
    archive = _build_tar({"x_daily/": None, "x_daily/file.txt": b"hi"})
    async with await _client(_app()) as c:
        resp = await c.post(
            "/api/colonies/import", data=_form(archive, name="other_name")
        )
        assert resp.status == 201
        body = await resp.json()
    assert body["name"] == "other_name"
    assert (colonies_dir / "other_name" / "file.txt").read_bytes() == b"hi"
    assert not (colonies_dir / "x_daily").exists()


@pytest.mark.asyncio
async def test_rejects_existing_without_replace_flag(colonies_dir: Path) -> None:
    (colonies_dir / "x_daily").mkdir()
    (colonies_dir / "x_daily" / "old.txt").write_text("preserved")
    archive = _build_tar({"x_daily/": None, "x_daily/new.txt": b"new"})
    async with await _client(_app()) as c:
        resp = await c.post("/api/colonies/import", data=_form(archive))
        assert resp.status == 409
    # Original content untouched
    assert (colonies_dir / "x_daily" / "old.txt").read_text() == "preserved"


@pytest.mark.asyncio
async def test_replace_existing_overwrites(colonies_dir: Path) -> None:
    (colonies_dir / "x_daily").mkdir()
    (colonies_dir / "x_daily" / "old.txt").write_text("preserved")
    archive = _build_tar({"x_daily/": None, "x_daily/new.txt": b"new"})
    async with await _client(_app()) as c:
        resp = await c.post(
            "/api/colonies/import",
            data=_form(archive, replace_existing="true"),
        )
        assert resp.status == 201, await resp.text()
    assert not (colonies_dir / "x_daily" / "old.txt").exists()
    assert (colonies_dir / "x_daily" / "new.txt").read_text() == "new"


@pytest.mark.asyncio
async def test_rejects_path_traversal(colonies_dir: Path) -> None:
    archive = _build_tar(
        {
            "x_daily/": None,
            "x_daily/../escape.txt": b"oops",
        }
    )
    async with await _client(_app()) as c:
        resp = await c.post("/api/colonies/import", data=_form(archive))
        assert resp.status == 400
        assert "traversal" in (await resp.json())["error"].lower() or "outside" in (await resp.json())["error"].lower()
    assert not (colonies_dir / "x_daily").exists()
    assert not (colonies_dir.parent / "escape.txt").exists()


@pytest.mark.asyncio
async def test_rejects_absolute_member(colonies_dir: Path) -> None:
    archive = _build_tar({"x_daily/": None, "/etc/passwd": b"oops"})
    async with await _client(_app()) as c:
        resp = await c.post("/api/colonies/import", data=_form(archive))
        assert resp.status == 400


@pytest.mark.asyncio
async def test_rejects_symlinks(colonies_dir: Path) -> None:
    archive = _build_tar_with_symlink("x_daily", "evil", "/etc/passwd")
    async with await _client(_app()) as c:
        resp = await c.post("/api/colonies/import", data=_form(archive))
        assert resp.status == 400
        assert "symlink" in (await resp.json())["error"].lower()


@pytest.mark.asyncio
async def test_rejects_multiple_top_level_dirs(colonies_dir: Path) -> None:
    archive = _build_tar(
        {
            "a/": None,
            "a/x.txt": b"a",
            "b/": None,
            "b/y.txt": b"b",
        }
    )
    async with await _client(_app()) as c:
        resp = await c.post("/api/colonies/import", data=_form(archive))
        assert resp.status == 400
        assert "top-level" in (await resp.json())["error"].lower()


@pytest.mark.asyncio
async def test_rejects_invalid_colony_name(colonies_dir: Path) -> None:
    archive = _build_tar({"Bad-Name/": None, "Bad-Name/x.txt": b"x"})
    async with await _client(_app()) as c:
        resp = await c.post("/api/colonies/import", data=_form(archive))
        assert resp.status == 400


@pytest.mark.asyncio
async def test_rejects_non_multipart(colonies_dir: Path) -> None:
    async with await _client(_app()) as c:
        resp = await c.post("/api/colonies/import", data=b"not multipart", headers={"Content-Type": "application/octet-stream"})
        assert resp.status == 400


@pytest.mark.asyncio
async def test_rejects_corrupt_tar(colonies_dir: Path) -> None:
    async with await _client(_app()) as c:
        resp = await c.post("/api/colonies/import", data=_form(b"not a real tar"))
        assert resp.status == 400


@pytest.mark.asyncio
async def test_rejects_missing_file_part(colonies_dir: Path) -> None:
    fd = FormData()
    fd.add_field("name", "anything")
    async with await _client(_app()) as c:
        resp = await c.post("/api/colonies/import", data=fd)
        assert resp.status == 400


@pytest.mark.asyncio
async def test_accepts_uncompressed_tar(colonies_dir: Path) -> None:
    archive = _build_tar({"x_daily/": None, "x_daily/file.txt": b"plain"}, gzip=False)
    async with await _client(_app()) as c:
        resp = await c.post(
            "/api/colonies/import",
            data=_form(archive, filename="colony.tar"),
        )
        assert resp.status == 201
    assert (colonies_dir / "x_daily" / "file.txt").read_text() == "plain"
