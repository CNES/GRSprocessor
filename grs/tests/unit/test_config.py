import hashlib
import io
import json
from unittest.mock import MagicMock

import pytest
import yaml

from grs import config as cfg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config(tmp_path, monkeypatch):
    """Provide an isolated packaged configuration."""
    path = tmp_path / "config.yml"
    path.write_text(
        "# Packaged GRS configuration\n\n"
        "path:\n"
        "  grsdata: null\n"
        "  toa_lut: toa.bin\n"
        "  trans_lut: trans.bin\n"
        "zenodo:\n"
        "  record: '12345'\n"
        "  url: https://zenodo.org\n"
        "  auto_download: true\n"
        "other:\n"
        "  nested:\n"
        "    value: 1\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(cfg, "default_config_file", lambda: path)
    monkeypatch.setattr(
        cfg, "default_data_root", lambda: tmp_path / "default-lut"
    )

    # The cache must not leak configuration between tests.
    cfg.load_config.cache_clear()

    yield path

    cfg.load_config.cache_clear()


@pytest.fixture
def clean_environment(monkeypatch):
    """Remove configuration-related environment variables."""
    for name in ("GRS_CONFIG", "GRS_DATA_ROOT", "XDG_CONFIG_HOME",
                 "GRS_NO_DOWNLOAD"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def isolated_config(default_config, clean_environment):
    """A clean configuration environment with a temporary default file."""
    return default_config


def write_user_config(path, data):
    """Write a YAML user configuration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def mock_http_response(payload):
    """Create an HTTP response that behaves like a real byte stream."""
    response = MagicMock()
    response.__enter__.return_value = response
    response.__exit__.return_value = False

    def read(size=-1):
        if not hasattr(read, "consumed"):
            read.consumed = False

        if read.consumed:
            return b""

        read.consumed = True
        return payload

    response.read.side_effect = read
    return response



# ---------------------------------------------------------------------------
# default_data_root / default_config_file / user_config_file
# ---------------------------------------------------------------------------

def test_default_data_root_uses_platformdirs(monkeypatch, tmp_path):
    expected = tmp_path / "user-data"
    monkeypatch.setattr(cfg.platformdirs, "user_data_path",
                        lambda name: expected)

    assert cfg.default_data_root() == expected / "lut"


def test_default_config_file_returns_packaged_config(isolated_config):
    assert cfg.default_config_file() == isolated_config


def test_user_config_file_uses_grs_config_environment_variable(
    monkeypatch, tmp_path, clean_environment
):
    custom = tmp_path / "custom.yml"
    monkeypatch.setenv("GRS_CONFIG", str(custom))

    assert cfg.user_config_file() == custom


def test_user_config_file_expands_home_directory(
    monkeypatch, tmp_path, clean_environment
):
    monkeypatch.setenv("GRS_CONFIG", "~/custom.yml")

    original_expanduser = cfg.Path.expanduser

    def fake_expanduser(path):
        if str(path).startswith("~/"):
            return tmp_path / str(path)[2:]
        return original_expanduser(path)

    monkeypatch.setattr(cfg.Path, "expanduser", fake_expanduser)

    assert cfg.user_config_file() == tmp_path / "custom.yml"



def test_user_config_file_uses_xdg_config_home(
    monkeypatch, tmp_path, clean_environment
):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    assert cfg.user_config_file() == tmp_path / "grs" / "config.yml"


def test_user_config_file_defaults_to_home_config_directory(
    monkeypatch, tmp_path, clean_environment
):
    monkeypatch.setattr(cfg.Path, "home", lambda: tmp_path)

    assert cfg.user_config_file() == (
        tmp_path / ".config" / "grs" / "config.yml"
    )


# ---------------------------------------------------------------------------
# read_user_config
# ---------------------------------------------------------------------------

def test_read_user_config_returns_empty_dict_when_file_is_missing(
    monkeypatch, tmp_path, clean_environment
):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    assert cfg.read_user_config() == {}


def test_read_user_config_reads_yaml(
    monkeypatch, tmp_path, clean_environment
):
    user_file = tmp_path / "config.yml"
    write_user_config(user_file, {"path": {"grsdata": "/custom/lut"}})
    monkeypatch.setenv("GRS_CONFIG", str(user_file))

    assert cfg.read_user_config() == {
        "path": {"grsdata": "/custom/lut"}
    }


def test_read_user_config_returns_empty_dict_for_empty_file(
    monkeypatch, tmp_path, clean_environment
):
    user_file = tmp_path / "config.yml"
    user_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("GRS_CONFIG", str(user_file))

    assert cfg.read_user_config() == {}


def test_read_user_config_raises_for_missing_explicit_file(
    monkeypatch, tmp_path, clean_environment
):
    missing = tmp_path / "missing.yml"
    monkeypatch.setenv("GRS_CONFIG", str(missing))

    with pytest.raises(FileNotFoundError, match="GRS_CONFIG"):
        cfg.read_user_config()


# ---------------------------------------------------------------------------
# _merge
# ---------------------------------------------------------------------------

def test_merge_recursively_merges_dictionaries():
    base = {
        "path": {"grsdata": None, "toa_lut": "toa.bin"},
        "other": {"a": 1, "b": 2},
    }
    update = {
        "path": {"grsdata": "/custom/lut"},
        "other": {"b": 3},
    }

    result = cfg._merge(base, update)

    assert result == {
        "path": {"grsdata": "/custom/lut", "toa_lut": "toa.bin"},
        "other": {"a": 1, "b": 3},
    }

    # The original dictionaries must remain unchanged.
    assert base["path"]["grsdata"] is None
    assert base["other"]["b"] == 2


def test_merge_replaces_non_dictionary_values():
    assert cfg._merge(
        {"value": {"nested": 1}},
        {"value": 42},
    ) == {"value": 42}


def test_merge_adds_new_keys():
    assert cfg._merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

def test_load_config_uses_packaged_defaults(isolated_config):
    config = cfg.load_config()

    assert config["path"]["toa_lut"] == "toa.bin"
    assert config["path"]["trans_lut"] == "trans.bin"
    assert config["path"]["grsdata"] == str(
        isolated_config.parent / "default-lut"
    )
    assert config["zenodo"]["record"] == "12345"


def test_load_config_merges_user_configuration(
    isolated_config, monkeypatch, tmp_path
):
    user_file = tmp_path / "user.yml"
    write_user_config(
        user_file,
        {
            "path": {"grsdata": "/user/lut"},
            "other": {"nested": {"value": 2}},
        },
    )
    monkeypatch.setenv("GRS_CONFIG", str(user_file))
    cfg.load_config.cache_clear()

    config = cfg.load_config()

    assert config["path"]["grsdata"] == "/user/lut"
    assert config["path"]["toa_lut"] == "toa.bin"
    assert config["other"]["nested"]["value"] == 2


def test_load_config_environment_overrides_user_file(
    isolated_config, monkeypatch, tmp_path
):
    user_file = tmp_path / "user.yml"
    write_user_config(user_file, {"path": {"grsdata": "/user/lut"}})
    monkeypatch.setenv("GRS_CONFIG", str(user_file))
    monkeypatch.setenv("GRS_DATA_ROOT", "/environment/lut")
    cfg.load_config.cache_clear()

    assert cfg.load_config()["path"]["grsdata"] == "/environment/lut"


def test_load_config_expands_user_directory(
    isolated_config, monkeypatch, tmp_path
):
    user_file = tmp_path / "user.yml"
    write_user_config(user_file, {"path": {"grsdata": "~/my-lut"}})
    monkeypatch.setenv("GRS_CONFIG", str(user_file))
    monkeypatch.setattr(cfg.os.path, "expanduser",
                        lambda path: path.replace("~", str(tmp_path)))
    cfg.load_config.cache_clear()

    assert cfg.load_config()["path"]["grsdata"] == str(
        tmp_path / "my-lut"
    )


def test_load_config_cache_can_be_cleared(
    isolated_config, monkeypatch, tmp_path
):
    user_file = tmp_path / "user.yml"
    write_user_config(user_file, {"path": {"grsdata": "/first"}})
    monkeypatch.setenv("GRS_CONFIG", str(user_file))

    first = cfg.load_config()
    assert first["path"]["grsdata"] == "/first"

    write_user_config(user_file, {"path": {"grsdata": "/second"}})

    # The cached value remains in effect until the cache is cleared.
    assert cfg.load_config()["path"]["grsdata"] == "/first"

    cfg.load_config.cache_clear()
    assert cfg.load_config()["path"]["grsdata"] == "/second"


# ---------------------------------------------------------------------------
# data_root_source / lut_path / lut_status / check_luts
# ---------------------------------------------------------------------------

def test_data_root_source_reports_environment(
    isolated_config, monkeypatch
):
    monkeypatch.setenv("GRS_DATA_ROOT", "/environment/lut")

    assert cfg.data_root_source() == "GRS_DATA_ROOT"


def test_data_root_source_reports_user_configuration(
    isolated_config, monkeypatch, tmp_path
):
    user_file = tmp_path / "user.yml"
    write_user_config(user_file, {"path": {"grsdata": "/user/lut"}})
    monkeypatch.setenv("GRS_CONFIG", str(user_file))

    assert cfg.data_root_source() == "user configuration"


def test_data_root_source_reports_default_folder(isolated_config):
    assert cfg.data_root_source() == "default folder"


@pytest.mark.parametrize(
    ("name", "filename"),
    [
        ("toa", "toa.bin"),
        ("transmittance", "trans.bin"),
    ],
)
def test_lut_path_returns_expected_path(
    isolated_config, name, filename
):
    path = cfg.lut_path(name)

    assert path == isolated_config.parent / "default-lut" / filename


def test_lut_path_rejects_unknown_name(isolated_config):
    with pytest.raises(KeyError):
        cfg.lut_path("unknown")


def test_lut_status_reports_missing_files(isolated_config):
    status = cfg.lut_status()

    assert len(status) == 2
    assert all(exists is False for _, _, exists in status)
    assert {name for name, _, _ in status} == {
        "toa", "transmittance"
    }


def test_lut_status_reports_existing_files(isolated_config):
    root = isolated_config.parent / "default-lut"
    root.mkdir()
    (root / "toa.bin").write_bytes(b"toa")
    (root / "trans.bin").write_bytes(b"trans")

    status = cfg.lut_status()

    assert all(exists for _, _, exists in status)


def test_check_luts_accepts_existing_files(tmp_path):
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    cfg.check_luts([first, second])


def test_check_luts_raises_for_missing_files(
    isolated_config, tmp_path
):
    missing = tmp_path / "missing.bin"

    with pytest.raises(FileNotFoundError, match="missing.bin"):
        cfg.check_luts([missing])


def test_check_luts_error_contains_download_instructions(
    isolated_config, tmp_path
):
    missing = tmp_path / "missing.bin"

    with pytest.raises(FileNotFoundError) as exc:
        cfg.check_luts([missing])

    message = str(exc.value)
    assert "grs_config --download" in message
    assert "GRS_DATA_ROOT" in message


# ---------------------------------------------------------------------------
# auto_download_enabled
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", True),
        ("0", True),
        ("1", False),
        ("true", False),
        ("false", False),
    ],
)
def test_auto_download_environment_variable(
    isolated_config, monkeypatch, value, expected
):
    monkeypatch.setenv("GRS_NO_DOWNLOAD", value)

    assert cfg.auto_download_enabled() is expected


def test_auto_download_respects_configuration(
    isolated_config, monkeypatch, tmp_path
):
    user_file = tmp_path / "user.yml"
    write_user_config(
        user_file, {"zenodo": {"auto_download": False}}
    )
    monkeypatch.setenv("GRS_CONFIG", str(user_file))
    cfg.load_config.cache_clear()

    assert cfg.auto_download_enabled() is False


def test_auto_download_environment_overrides_configuration(
    isolated_config, monkeypatch, tmp_path
):
    user_file = tmp_path / "user.yml"
    write_user_config(
        user_file, {"zenodo": {"auto_download": True}}
    )
    monkeypatch.setenv("GRS_CONFIG", str(user_file))
    monkeypatch.setenv("GRS_NO_DOWNLOAD", "1")
    cfg.load_config.cache_clear()

    assert cfg.auto_download_enabled() is False


def test_auto_download_defaults_to_enabled(
    isolated_config, monkeypatch
):
    monkeypatch.delenv("GRS_NO_DOWNLOAD", raising=False)

    # The configuration has no zenodo section.
    isolated_config.write_text("path:\n  grsdata: null\n", encoding="utf-8")
    cfg.load_config.cache_clear()

    assert cfg.auto_download_enabled() is True


# ---------------------------------------------------------------------------
# ensure_luts
# ---------------------------------------------------------------------------

def test_ensure_luts_does_not_download_when_all_files_exist(
    isolated_config, monkeypatch
):
    paths = [isolated_config.parent / "a", isolated_config.parent / "b"]
    status = [
        ("toa", paths[0], True),
        ("transmittance", paths[1], True),
    ]

    monkeypatch.setattr(cfg, "lut_status", lambda: status)

    def unexpected_download(*args, **kwargs):
        pytest.fail("download_luts should not be called")

    monkeypatch.setattr(cfg, "download_luts", unexpected_download)

    assert cfg.ensure_luts() == paths

def test_ensure_luts_downloads_when_files_are_missing(
    isolated_config, monkeypatch
):
    paths = [
        isolated_config.parent / "default-lut" / "toa.bin",
        isolated_config.parent / "default-lut" / "trans.bin",
    ]

    calls = []

    def fake_download():
        calls.append(True)

        # Simulate the actual download by creating the LUT files.
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake LUT content")

    monkeypatch.setattr(cfg, "auto_download_enabled", lambda: True)
    monkeypatch.setattr(cfg, "download_luts", fake_download)

    result = cfg.ensure_luts()

    assert result == paths
    assert calls == [True]
    assert all(path.is_file() for path in paths)


def test_ensure_luts_raises_when_download_is_disabled(
    isolated_config, monkeypatch
):
    paths = [isolated_config.parent / "toa.bin",
             isolated_config.parent / "trans.bin"]

    monkeypatch.setattr(
        cfg,
        "lut_status",
        lambda: [
            ("toa", paths[0], False),
            ("transmittance", paths[1], False),
        ],
    )
    monkeypatch.setattr(cfg, "auto_download_enabled", lambda: False)

    with pytest.raises(FileNotFoundError):
        cfg.ensure_luts()


def test_ensure_luts_checks_files_after_download_failure(
    isolated_config, monkeypatch
):
    paths = [isolated_config.parent / "toa.bin",
             isolated_config.parent / "trans.bin"]

    monkeypatch.setattr(
        cfg,
        "lut_status",
        lambda: [
            ("toa", paths[0], False),
            ("transmittance", paths[1], False),
        ],
    )
    monkeypatch.setattr(cfg, "auto_download_enabled", lambda: True)

    def failed_download():
        raise OSError("network unavailable")

    monkeypatch.setattr(cfg, "download_luts", failed_download)

    with pytest.raises(FileNotFoundError):
        cfg.ensure_luts()


# ---------------------------------------------------------------------------
# init_user_config
# ---------------------------------------------------------------------------

def test_init_user_config_creates_config_file(
    isolated_config, monkeypatch, tmp_path
):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    data_root = tmp_path / "my-lut"

    result = cfg.init_user_config(data_root)

    assert result.is_file()

    content = result.read_text(encoding="utf-8")
    assert str(data_root.absolute()) in content
    assert "Packaged defaults" in content
    assert "grsdata:" in content


def test_init_user_config_refuses_to_overwrite_existing_file(
    isolated_config, monkeypatch, tmp_path
):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    user_file = cfg.user_config_file()
    user_file.parent.mkdir(parents=True)
    user_file.write_text("existing: true\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        cfg.init_user_config(tmp_path / "lut")


def test_init_user_config_force_overwrites_file(
    isolated_config, monkeypatch, tmp_path
):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    user_file = cfg.user_config_file()
    user_file.parent.mkdir(parents=True)
    user_file.write_text("existing: true\n", encoding="utf-8")

    result = cfg.init_user_config(tmp_path / "new-lut", force=True)

    assert result == user_file
    assert "existing: true" not in result.read_text(encoding="utf-8")
    assert str((tmp_path / "new-lut").absolute()) in result.read_text(
        encoding="utf-8"
    )


def test_init_user_config_creates_default_data_folder(
    isolated_config, monkeypatch, tmp_path
):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    default_root = tmp_path / "default-lut"
    monkeypatch.setattr(cfg, "default_data_root", lambda: default_root)

    cfg.init_user_config()

    assert default_root.is_dir()


def test_init_user_config_invalidates_config_cache(
    isolated_config, monkeypatch, tmp_path
):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    cfg.load_config()

    cfg.init_user_config(tmp_path / "new-lut")

    assert cfg.load_config()["path"]["grsdata"] == str(
        (tmp_path / "new-lut").absolute()
    )


# ---------------------------------------------------------------------------
# zenodo_record_url
# ---------------------------------------------------------------------------

def test_zenodo_record_url_uses_default_url(isolated_config):
    assert cfg.zenodo_record_url() == (
        "https://zenodo.org/api/records/12345"
    )


def test_zenodo_record_url_removes_trailing_slash(
    isolated_config, monkeypatch, tmp_path
):
    user_file = tmp_path / "user.yml"
    write_user_config(
        user_file,
        {
            "zenodo": {
                "url": "https://example.org/",
                "record": "abc",
            }
        },
    )
    monkeypatch.setenv("GRS_CONFIG", str(user_file))
    cfg.load_config.cache_clear()

    assert cfg.zenodo_record_url() == (
        "https://example.org/api/records/abc"
    )


def test_zenodo_record_url_raises_without_record(
    isolated_config
):
    isolated_config.write_text(
        "path:\n  grsdata: null\nzenodo:\n  url: https://zenodo.org\n",
        encoding="utf-8",
    )
    cfg.load_config.cache_clear()

    with pytest.raises(ValueError, match="no Zenodo record"):
        cfg.zenodo_record_url()


# ---------------------------------------------------------------------------
# download_luts
# ---------------------------------------------------------------------------

def test_download_luts_downloads_and_verifies_files(
    isolated_config, monkeypatch, tmp_path
):
    root = tmp_path / "lut"
    root.mkdir()

    paths = [root / "toa.bin", root / "trans.bin"]
    contents = [b"TOA LUT contents", b"transmittance LUT contents"]

    monkeypatch.setattr(
        cfg,
        "lut_status",
        lambda: [
            ("toa", paths[0], False),
            ("transmittance", paths[1], False),
        ],
    )
    monkeypatch.setattr(cfg, "zenodo_record_url", lambda: "https://example.org/record")

    files = []
    for path, content in zip(paths, contents):
        files.append(
            {
                "key": path.name,
                "checksum": "md5:" + hashlib.md5(content).hexdigest(),
                "size": len(content),
                "links": {"self": f"https://example.org/{path.name}"},
            }
        )

    record_response = mock_http_response(b"")
    record_response.__enter__.return_value = io.BytesIO(
        json.dumps({"files": files, "links": {}}).encode()
    )

    file_responses = [
        mock_http_response(content) for content in contents
    ]

    responses = [record_response, *file_responses]
    monkeypatch.setattr(
        cfg.urllib.request,
        "urlopen",
        lambda url: responses.pop(0),
    )

    result = cfg.download_luts(progress=False)

    assert result == paths
    assert paths[0].read_bytes() == contents[0]
    assert paths[1].read_bytes() == contents[1]
    assert not (root / "toa.bin.part").exists()
    assert not (root / "trans.bin.part").exists()


def test_download_luts_skips_existing_files_unless_forced(
    isolated_config, monkeypatch, tmp_path
):
    path = tmp_path / "toa.bin"
    path.write_bytes(b"already downloaded")

    monkeypatch.setattr(
        cfg,
        "lut_status",
        lambda: [
            ("toa", path, True),
            ("transmittance", tmp_path / "trans.bin", False),
        ],
    )
    monkeypatch.setattr(cfg, "zenodo_record_url", lambda: "https://example.org/record")

    record_response = mock_http_response(b"")
    record_response.__enter__.return_value = io.BytesIO(
        json.dumps(
            {
                "files": [],
                "links": {},
            }
        ).encode()
    )

    monkeypatch.setattr(
        cfg.urllib.request, "urlopen", lambda url: record_response
    )

    with pytest.raises(FileNotFoundError, match="trans.bin"):
        cfg.download_luts(progress=False)

    # The existing file remains untouched.
    assert path.read_bytes() == b"already downloaded"


def test_download_luts_raises_when_file_is_absent_from_record(
    isolated_config, monkeypatch, tmp_path
):
    path = tmp_path / "missing.bin"

    monkeypatch.setattr(
        cfg,
        "lut_status",
        lambda: [
            ("toa", path, False),
            ("transmittance", tmp_path / "trans.bin", False),
        ],
    )
    monkeypatch.setattr(cfg, "zenodo_record_url", lambda: "https://example.org/record")

    response = mock_http_response(b"")
    response.__enter__.return_value = io.BytesIO(
        json.dumps({"files": [], "links": {}}).encode()
    )
    monkeypatch.setattr(
        cfg.urllib.request, "urlopen", lambda url: response
    )

    with pytest.raises(FileNotFoundError, match="missing.bin"):
        cfg.download_luts(progress=False)


def test_download_luts_removes_part_file_on_checksum_mismatch(
    isolated_config, monkeypatch, tmp_path
):
    path = tmp_path / "toa.bin"
    content = b"downloaded data"

    monkeypatch.setattr(
        cfg,
        "lut_status",
        lambda: [
            ("toa", path, False),
            ("transmittance", tmp_path / "trans.bin", True),
        ],
    )
    monkeypatch.setattr(cfg, "zenodo_record_url", lambda: "https://example.org/record")

    record = {
        "files": [
            {
                "key": path.name,
                "checksum": "md5:incorrect-checksum",
                "size": len(content),
                "links": {"self": "https://example.org/toa.bin"},
            }
        ],
        "links": {},
    }

    record_response = mock_http_response(b"")
    record_response.__enter__.return_value = io.BytesIO(
        json.dumps(record).encode()
    )
    file_response = mock_http_response(content)

    responses = [record_response, file_response]
    monkeypatch.setattr(
        cfg.urllib.request,
        "urlopen",
        lambda url: responses.pop(0),
    )

    with pytest.raises(IOError, match="checksum mismatch"):
        cfg.download_luts(progress=False)

    assert not path.exists()
    assert not path.with_name(path.name + ".part").exists()


# ---------------------------------------------------------------------------
# main / CLI
# ---------------------------------------------------------------------------

def test_main_returns_zero_when_luts_exist(
    isolated_config, monkeypatch, tmp_path, capsys
):
    root = tmp_path / "default-lut"
    root.mkdir()
    (root / "toa.bin").write_bytes(b"toa")
    (root / "trans.bin").write_bytes(b"trans")

    assert cfg.main([]) == 0

    output = capsys.readouterr().out
    assert "LUT folder:" in output
    assert "LUT toa" in output
    assert "LUT transmittance" in output


def test_main_returns_one_when_luts_are_missing(
    isolated_config, capsys
):
    result = cfg.main([])

    captured = capsys.readouterr()

    assert result == 1
    assert "MISSING" in captured.out
    assert "LUT file(s) missing" in captured.err


def test_main_init_refuses_to_overwrite_existing_config(
    isolated_config, monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    user_file = cfg.user_config_file()
    user_file.parent.mkdir(parents=True)
    user_file.write_text("existing: true\n", encoding="utf-8")

    result = cfg.main(["--init", str(tmp_path / "lut")])

    captured = capsys.readouterr()

    assert result == 1
    assert "already exists" in captured.err


def test_main_download_returns_one_when_download_fails(
    isolated_config, monkeypatch, capsys
):
    def failed_download(force=False, progress=True):
        raise OSError("network unavailable")

    monkeypatch.setattr(cfg, "download_luts", failed_download)

    result = cfg.main(["--download"])

    captured = capsys.readouterr()

    assert result == 1
    assert "network unavailable" in captured.err