"""Tests for the fs server's binary support: read_file's hexdump mode,
create_file's hex/base64 encodings and edit_file's byte-span edits. The fs
module is imported directly (no subprocess) and its tools called as plain
functions, chdir'd into tmp_path so safe_path's jail is the test dir."""
import os
import importlib.util

import pytest

from cai.environment import builtin_mcp_dir


def _load_fs():
    path = os.path.join(builtin_mcp_dir(), "fs.py")
    spec = importlib.util.spec_from_file_location("fs_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fs = _load_fs()

ELF = b"\x7fELF\x02\x01\x01\x00" + bytes(range(256))


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CAI_SCRATCH", raising=False)


# --- read_file ---

def test_read_text_still_works():
    with open("a.txt", "w") as f:
        f.write("line one\nline two\n")
    out = fs.read_file("a.txt")
    assert out == "line one\nline two\n"
    out = fs.read_file("a.txt", line_start=2, line_end=2)
    assert out == "line two\n"


def test_read_binary_returns_xxd_style_hexdump():
    with open("blob.bin", "wb") as f:
        f.write(b"\x7fELF\x00ABCDEFGHIJK")
    out = fs.read_file("blob.bin")
    first = out.splitlines()[0]
    assert first.startswith("00000000: 7f45 4c46 0041 4243")
    assert first.endswith(".ELF.ABCDEFGHIJK")


def test_read_binary_offsets_are_absolute_and_end_exclusive():
    with open("blob.bin", "wb") as f:
        f.write(ELF)
    out = fs.read_file("blob.bin", offset_start=16, offset_end=20)
    lines = out.splitlines()
    assert lines[0].startswith("00000010: ")
    # 4 bytes shown, footer points at the next offset
    assert "0809 0a0b" in lines[0]
    assert f"call again with offset_start=20" in out


def test_read_binary_default_window_and_footer():
    with open("big.bin", "wb") as f:
        f.write(b"\x00" * 5000)
    out = fs.read_file("big.bin")
    assert "[bytes 0-3200 of 5000; call again with offset_start=3200]" in out


def test_read_lines_on_binary_is_an_error():
    with open("blob.bin", "wb") as f:
        f.write(b"\x00\x01")
    out = fs.read_file("blob.bin", line_start=1)
    assert out.startswith("Error:")
    assert "offset_start" in out


def test_read_offsets_on_text_is_an_error():
    with open("a.txt", "w") as f:
        f.write("plain\n")
    out = fs.read_file("a.txt", offset_start=0)
    assert out.startswith("Error:")
    assert "line_start" in out


def test_read_both_range_kinds_is_an_error():
    with open("a.txt", "w") as f:
        f.write("plain\n")
    out = fs.read_file("a.txt", line_start=1, offset_start=0)
    assert out.startswith("Error:")


def test_read_binary_past_eof():
    with open("blob.bin", "wb") as f:
        f.write(b"\x00\x01\x02")
    out = fs.read_file("blob.bin", offset_start=100, offset_end=200)
    assert "[no bytes at offset 100; file is 3 bytes]" == out


# --- create_file ---

def test_create_hex_writes_bytes_and_tolerates_whitespace():
    out = fs.create_file("out.bin", "7f45 4c46\n0001", encoding="hex")
    assert "6 bytes" in out
    with open("out.bin", "rb") as f:
        assert f.read() == b"\x7fELF\x00\x01"


def test_create_base64_writes_bytes():
    out = fs.create_file("out.bin", "f0VMRgAB", encoding="base64")
    assert "6 bytes" in out
    with open("out.bin", "rb") as f:
        assert f.read() == b"\x7fELF\x00\x01"


def test_create_text_default_unchanged():
    fs.create_file("t.txt", "héllo\n")
    with open("t.txt", "rb") as f:
        assert f.read() == "héllo\n".encode("utf-8")


def test_create_rejects_bad_payloads():
    assert fs.create_file("x.bin", "zz-not-hex", encoding="hex").startswith("Error:")
    assert fs.create_file("x.bin", "@@@", encoding="base64").startswith("Error:")
    assert fs.create_file("x.bin", "hi", encoding="rot13").startswith("Error:")
    assert not os.path.exists("x.bin")


# --- edit_file ---

def test_edit_hex_replaces_a_byte_span():
    with open("blob.bin", "wb") as f:
        f.write(b"\x7fELF\x00\xde\xad\xbe\xef")
    out = fs.edit_file("blob.bin", "dead beef", "cafe babe", encoding="hex")
    assert "Replaced 1 occurrence" in out
    with open("blob.bin", "rb") as f:
        assert f.read() == b"\x7fELF\x00\xca\xfe\xba\xbe"


def test_edit_hex_ambiguity_and_replace_all():
    with open("blob.bin", "wb") as f:
        f.write(b"\x00\xaa\xbb\xaa\xbb")
    out = fs.edit_file("blob.bin", "aabb", "cccc", encoding="hex")
    assert "ambiguous" in out
    out = fs.edit_file("blob.bin", "aabb", "cccc", encoding="hex", replace_all=True)
    assert "Replaced 2 occurrences" in out
    with open("blob.bin", "rb") as f:
        assert f.read() == b"\x00\xcc\xcc\xcc\xcc"


def test_edit_text_on_binary_is_an_error():
    with open("blob.bin", "wb") as f:
        f.write(b"\x00abc")
    out = fs.edit_file("blob.bin", "abc", "xyz")
    assert out.startswith("Error:")
    assert "encoding='hex'" in out


def test_edit_hex_on_text_is_an_error():
    with open("a.txt", "w") as f:
        f.write("plain\n")
    out = fs.edit_file("a.txt", "706c", "5050", encoding="hex")
    assert out.startswith("Error:")


def test_edit_text_still_works():
    with open("a.txt", "w") as f:
        f.write("hello world\n")
    out = fs.edit_file("a.txt", "world", "there")
    assert "Replaced 1 occurrence" in out
    with open("a.txt") as f:
        assert f.read() == "hello there\n"


def test_edit_empty_old_is_an_error():
    with open("a.txt", "w") as f:
        f.write("x\n")
    assert fs.edit_file("a.txt", "", "y").startswith("Error:")


# --- copy_bytes ---

def _write(name, data):
    with open(name, "wb") as f:
        f.write(data)


def _bytes_of(name):
    with open(name, "rb") as f:
        return f.read()


def test_copy_bytes_extracts_a_head_to_a_new_file():
    _write("mv", bytes(range(256)))
    out = fs.copy_bytes("mv", "mv_head", src_offset_end=100)
    assert out == "Created mv_head (100 bytes from mv[0:100])"
    assert _bytes_of("mv_head") == bytes(range(100))


def test_copy_bytes_appends_to_an_existing_file():
    _write("a.bin", b"\x00AAA")
    _write("b.bin", b"\x00BBB")
    out = fs.copy_bytes("a.bin", "b.bin", src_offset_start=1)
    assert out == "Appended 3 bytes to b.bin (from a.bin[1:4])"
    assert _bytes_of("b.bin") == b"\x00BBBAAA"


def test_copy_bytes_patch_within_the_same_file():
    # replace the 8 bytes at offset 100 with the 8 bytes at offset 300
    _write("f", bytes(400))
    _write("f", bytes(300) + b"ABCDEFGH" + bytes(92))
    out = fs.copy_bytes("f", "f",
                        src_offset_start=300, src_offset_end=308,
                        dst_offset_start=100)
    assert out == "Replaced f[100:108] (8 bytes) with 8 bytes from f[300:308]"
    data = _bytes_of("f")
    assert data[100:108] == b"ABCDEFGH"
    assert data[300:308] == b"ABCDEFGH"
    assert len(data) == 400


def test_copy_bytes_dst_end_only_anchors_backwards():
    _write("src.bin", b"\x00NEW!")
    _write("dst.bin", b"\x00" + b"x" * 9)
    out = fs.copy_bytes("src.bin", "dst.bin",
                        src_offset_start=1,
                        dst_offset_end=10)
    assert out == "Replaced dst.bin[6:10] (4 bytes) with 4 bytes from src.bin[1:5]"
    assert _bytes_of("dst.bin") == b"\x00xxxxxNEW!"


def test_copy_bytes_explicit_range_may_change_size():
    _write("src.bin", b"\x00LONGER")
    _write("dst.bin", b"\x00abcdef")
    out = fs.copy_bytes("src.bin", "dst.bin",
                        src_offset_start=1,
                        dst_offset_start=1, dst_offset_end=3)
    assert "Replaced dst.bin[1:3] (2 bytes) with 6 bytes" in out
    assert _bytes_of("dst.bin") == b"\x00LONGERcdef"


def test_copy_bytes_insertion_via_empty_dst_range():
    _write("src.bin", b"\x00XY")
    _write("dst.bin", b"\x00ab")
    fs.copy_bytes("src.bin", "dst.bin",
                  src_offset_start=1,
                  dst_offset_start=2, dst_offset_end=2)
    assert _bytes_of("dst.bin") == b"\x00aXYb"


def test_copy_bytes_errors():
    _write("s.bin", b"\x00\x01\x02")
    out = fs.copy_bytes("s.bin", "missing.bin", dst_offset_start=0)
    assert out.startswith("Error:")
    assert "omit the dst offsets" in out
    _write("d.bin", b"\x00\x01")
    out = fs.copy_bytes("s.bin", "d.bin", dst_offset_start=1, dst_offset_end=99)
    assert out.startswith("Error: replace range")
    out = fs.copy_bytes("s.bin", "d.bin", src_offset_start=50)
    assert out.startswith("Error: empty source range")
    # dst_offset_end smaller than the incoming range anchors before byte 0
    out = fs.copy_bytes("s.bin", "d.bin", dst_offset_end=1)
    assert out.startswith("Error: replace range")


# --- search ---

def test_search_text_still_works():
    with open("a.txt", "w") as f:
        f.write("alpha\nbeta\n")
    out = fs.search("beta")
    assert "a.txt:2:1:beta" in out


def test_search_finds_a_string_inside_a_binary_file():
    _write("app.bin", b"\x00" * 32 + b"needle" + b"\x00" * 32)
    out = fs.search("needle")
    assert "app.bin:32: 6-byte match" in out
    assert "6e65 6564 6c65" in out
    assert "needle" in out


def test_search_binary_match_shows_32_bytes_of_context_each_side():
    _write("app.bin", b"\x00" * 100 + b"MAGIC" + b"\x00" * 100)
    out = fs.search("MAGIC")
    dump = out.split("\n")
    assert dump[0].endswith("app.bin:100: 5-byte match")
    # context starts at 100-32=68 aligned down to 64, ends past 100+5+32=137
    assert dump[1].startswith("00000040:")
    assert dump[-1].startswith("00000080:")


def test_search_reports_text_and_binary_matches_together():
    with open("a.txt", "w") as f:
        f.write("needle\n")
    _write("b.bin", b"\x00needle")
    out = fs.search("needle")
    assert "a.txt:1:1:needle" in out
    assert "b.bin:1: 6-byte match" in out


def test_search_binary_offset_plugs_into_read_file():
    _write("app.bin", b"\x00" * 100 + b"MAGIC" + b"\x00" * 100)
    out = fs.search("MAGIC", path="app.bin")
    offset = int(out.split("\n")[0].split(":")[1])
    assert offset == 100
    dump = fs.read_file("app.bin", offset_start=offset, offset_end=offset + 5)
    assert "MAGIC" in dump


def test_search_regex_matches_bytes_in_binary():
    _write("app.bin", b"\x00abc123xyz")
    out = fs.search("abc[0-9]+", path="app.bin")
    assert "app.bin:1: 6-byte match" in out


def test_search_byte_escapes_match_raw_bytes():
    _write("app.bin", ELF)
    out = fs.search(r"(?-u:\x7f\x45\x4c\x46)")
    assert "app.bin:0: 4-byte match" in out
    assert "7f45 4c46" in out


def test_search_file_glob_applies_to_binary_scan():
    _write("keep.bin", b"\x00needle")
    _write("skip.dat", b"\x00needle")
    out = fs.search("needle", file_glob="*.bin")
    assert "keep.bin" in out
    assert "skip.dat" not in out


def test_search_errors():
    assert fs.search("") == "Error: empty pattern"
    assert fs.search("[unclosed").startswith("Error:")
    # line-oriented like text search: a pattern matching \n is rejected
    assert fs.search(r"(?-u:\x42\x0a\x43)").startswith("Error:")
    _write("a.bin", b"\x00\x01")
    assert fs.search("nothing-here") == "No matches found."
