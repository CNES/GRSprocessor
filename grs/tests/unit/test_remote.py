"""Remote inputs and outputs (grs.remote, issue #11): URLs in the configuration, the grs command and the launcher.

The tests use the in-memory file system of fsspec (``memory://``): the same code paths as ``s3://`` or
``https://`` (fsspec URLs), without network nor credentials.
"""

import sys

import fsspec
import pytest
import yaml

import grs.grs_process
from grs import config, remote, run
from grs.grs_exceptions import GRS_IO_Exception
from grs.run import EXIT_OK, EXIT_INPUT, EXIT_SKIPPED

SAFE_NAME = 'S2A_MSIL1C_20210211T104151_N0500_R008_T31TFJ_20230608T124831.SAFE'
MTD = ('<?xml version="1.0"?><n1:Level-1C_User_Product xmlns:n1="https://psd-14.sentinel2.eo.esa.int/PSD/'
       'User_Product_Level-1C.xsd"><n1:Quality_Indicators_Info><Cloud_Coverage_Assessment>{cc}'
       '</Cloud_Coverage_Assessment></n1:Quality_Indicators_Info></n1:Level-1C_User_Product>')


@pytest.fixture
def memfs(tmp_path, monkeypatch):
    """Empty in-memory file system, and a local cache folder for the remote copies."""
    fs = fsspec.filesystem('memory')
    fs.store.clear()
    fs.pseudo_dirs.clear()
    fs.pseudo_dirs.append('')
    monkeypatch.setenv('GRS_CACHE_DIR', str(tmp_path / 'cache'))
    yield fs
    fs.store.clear()


@pytest.fixture
def bucket(memfs):
    """Level-1 product (cloud cover 10 %) and CAMS file in the 'bucket'."""
    safe = f'memory://bucket/l1c/{SAFE_NAME}'
    remote.write_text(f'{safe}/MTD_MSIL1C.xml', MTD.format(cc=10.0))
    cams = 'memory://bucket/cams/cams_forecast_2021-02.nc'
    remote.write_text(cams, 'cams')
    return safe, cams


# ---------------------------------------------------------------------------------------------------- grs.remote

def test_is_url():
    assert remote.is_url('s3://bucket/key') and remote.is_url('https://host/file.yml')
    assert not remote.is_url('/data/file') and not remote.is_url('file:///data/file')


def test_join_and_basename(tmp_path):
    assert remote.join('s3://bucket/l2a/', 'name', 'name.zarr') == 's3://bucket/l2a/name/name.zarr'
    assert remote.join(tmp_path, 'a', 'b') == tmp_path / 'a' / 'b'
    assert remote.basename('s3://bucket/l1c/scene.SAFE/') == 'scene.SAFE'
    assert remote.basename(tmp_path / 'scene.SAFE') == 'scene.SAFE'


def test_read_write_exists(memfs):
    url = 'memory://bucket/dir/file.txt'
    assert not remote.exists(url)
    remote.write_text(url, 'hello')
    assert remote.exists(url) and remote.exists('memory://bucket/dir')
    assert remote.read_text(url) == 'hello'


def test_local_file(memfs, tmp_path):
    url = 'memory://bucket/cams.nc'
    remote.write_text(url, 'cams data')
    local = remote.local_file(url)
    assert not remote.is_url(local) and open(local).read() == 'cams data'
    assert str(tmp_path / 'cache') in local
    # a local path is returned as is
    assert remote.local_file(tmp_path / 'x.nc') == str(tmp_path / 'x.nc')


def test_upload(memfs, tmp_path):
    (tmp_path / 'log.txt').write_text('log')
    remote.upload(tmp_path / 'log.txt', 'memory://bucket/out/log.txt')
    assert remote.read_text('memory://bucket/out/log.txt') == 'log'


# ---------------------------------------------------------------------------------------------------- configuration

def test_config_from_url(memfs, monkeypatch):
    url = 'memory://git/grs/config.yml'
    remote.write_text(url, yaml.safe_dump({'path': {'grsdata': '/lut/from/url'}}))
    monkeypatch.setenv('GRS_CONFIG', url)
    monkeypatch.delenv('GRS_DATA_ROOT', raising=False)
    config.load_config.cache_clear()
    try:
        assert config.user_config_file() == url
        assert config.load_config()['path']['grsdata'] == '/lut/from/url'
    finally:
        config.load_config.cache_clear()


def test_config_missing_url(memfs, monkeypatch):
    monkeypatch.setenv('GRS_CONFIG', 'memory://git/grs/missing.yml')
    with pytest.raises(FileNotFoundError):
        config.read_user_config()


# ---------------------------------------------------------------------------------------------------- processing

def test_cloud_cover_from_url(bucket):
    safe, _ = bucket
    assert run.s2_cloud_cover(safe) == pytest.approx(0.10)


def test_no_water_marker_url():
    assert grs.grs_process.no_water_marker('s3://b/l2a/scene/scene.zarr') == 's3://b/l2a/scene/scene_no_water.txt'
    assert grs.grs_process.no_water_marker('s3://b/l2a/scene') == 's3://b/l2a/scene/scene_no_water.txt'


def test_remote_input_needs_grsdriver_remote(monkeypatch):
    import importlib.util
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, 'find_spec',
                        lambda name, *a: None if name == 'GRSdriver.remote' else find_spec(name, *a))
    with pytest.raises(GRS_IO_Exception):
        grs.grs_process._check_remote_driver('s3://bucket/l1c/scene.SAFE')


class FakeProcess:
    """Process recording its arguments; the processing itself is not run."""
    calls = []

    def __init__(self):
        self.no_water = False

    def execute(self, l1c_prod, **kwargs):
        FakeProcess.calls.append((l1c_prod, kwargs))

    def write_output(self):
        pass


@pytest.fixture
def fake_process(monkeypatch):
    FakeProcess.calls = []
    monkeypatch.setattr(grs.grs_process, 'Process', FakeProcess)
    return FakeProcess


def output_names(safe):
    name = run.set_ofile(remote.basename(safe), level_name='L2AGRS', suffix=f'_V{grs.__version__}').name
    outdir = f'memory://bucket/l2a/{name}'
    return name, outdir, f'{outdir}/{name}.zarr'


def test_grs_command_with_urls(bucket, fake_process, tmp_path):
    safe, cams = bucket
    name, outdir, store = output_names(safe)
    status = run.main([safe, '--cams_file', cams, '--odir', 'memory://bucket/l2a', '--log_dir', str(tmp_path / 'log')])
    assert status == EXIT_OK
    l1c_prod, kwargs = fake_process.calls[0]
    # the URLs are passed as strings (not Path, which would turn memory:// into memory:/)
    assert l1c_prod == safe and kwargs['cams_file'] == cams and kwargs['odir'] == store
    # log files written locally, then copied next to the product
    assert (tmp_path / 'log' / 'log_file.log').is_file()
    assert remote.exists(f'{outdir}/log_file.log') and remote.exists(f'{outdir}/error.log')


def test_grs_command_missing_url(bucket, fake_process):
    _, cams = bucket
    assert run.main(['memory://bucket/l1c/missing.SAFE', '--cams_file', cams]) == EXIT_INPUT
    safe, _ = bucket
    assert run.main([safe, '--cams_file', 'memory://bucket/cams/missing.nc']) == EXIT_INPUT


def test_grs_command_no_clobber_url(bucket, fake_process, tmp_path):
    safe, cams = bucket
    name, outdir, store = output_names(safe)
    args = [safe, '--cams_file', cams, '--odir', 'memory://bucket/l2a', '--no_clobber', '--log_dir', str(tmp_path)]
    remote.write_text(f'{outdir}/{name}_no_water.txt', 'no water')
    assert run.main(args) == EXIT_SKIPPED
    remote.write_text(f'{store}/zarr.json', '{}')
    assert run.main(args) == EXIT_OK
    assert not fake_process.calls


def test_grs_command_cloud_cover_url(bucket, fake_process, tmp_path):
    safe, cams = bucket
    assert run.main([safe, '--cams_file', cams, '--max_cloud_cover', '0.05']) == EXIT_SKIPPED


def test_launcher_with_urls(bucket, fake_process, monkeypatch, tmp_path):
    from exe import launcher
    monkeypatch.setattr(launcher, 'Process', FakeProcess)
    safe, cams = bucket
    name, outdir, store = output_names(safe)
    hymotep = dict(input_file=safe, cams_folder=cams, output_dir='memory://bucket/l2a', surfwater_file=None,
                   dem_file=None, suffix=None, noclobber=False, max_cloud_cover=1., resolution=60, scale_aot=1.,
                   opac_model=None, allpixels=False, snap_compliant=False)
    (tmp_path / 'hymotep.yml').write_text(yaml.safe_dump(hymotep))
    (tmp_path / 'global.yml').write_text(yaml.safe_dump(dict(logfile=str(tmp_path / 'log' / 'grs.log'), level='INFO',
                                                             hymotep_config=str(tmp_path / 'hymotep.yml'))))
    monkeypatch.setattr(sys, 'argv', ['launcher.py', str(tmp_path / 'global.yml')])
    assert launcher.main() == EXIT_OK
    l1c_prod, kwargs = fake_process.calls[0]
    assert l1c_prod == safe and kwargs['cams_file'] == cams and kwargs['odir'] == store


# ---------------------------------------------------------------------------------------------------- end to end

@pytest.mark.skipif(not __import__('grs.tests.unit.conftest', fromlist=['luts_available']).luts_available(),
                    reason='look-up tables not installed (grs_config --download)')
def test_processing_with_remote_cams_and_output(memfs, l1c_path, cams_path):
    """Full processing of the test subset with the CAMS file and the output in the 'bucket'."""
    memfs.put_file(cams_path, '/bucket/cams/cams.nc')
    process = grs.grs_process.Process()
    process.execute(l1c_path, cams_file='memory://bucket/cams/cams.nc', resolution=60,
                    odir='memory://bucket/l2a/test/test.zarr')
    process.write_output()
    written = memfs.ls('/bucket/l2a/test', detail=False)
    assert '/bucket/l2a/test/test.zarr' in written
    assert '/bucket/l2a/test/test_ql_l2a_rgb.png' in written
    assert remote.exists('memory://bucket/l2a/test/test.zarr/zarr.json')
