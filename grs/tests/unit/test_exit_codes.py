"""Exit status of the grs command (grs.run.main) and of the CNES launcher (exe/launcher.py), issue #13.

The processing is replaced by a fake Process: these tests only check how each outcome is turned into an exit
status (0 product, 1 processing error, 2 input error, 3 skipped on purpose).
"""

import sys

import pytest
import yaml

import grs.grs_process
from grs import run
from grs.grs_exceptions import GRS_IO_Exception
from grs.run import EXIT_OK, EXIT_ERROR, EXIT_INPUT, EXIT_SKIPPED

SAFE_NAME = 'S2C_MSIL1C_20260707T112111_N0512_R037_T28PEC_20260707T145811.SAFE'
MTD = ('<?xml version="1.0"?><n1:Level-1C_User_Product xmlns:n1="https://psd-14.sentinel2.eo.esa.int/PSD/'
       'User_Product_Level-1C.xsd"><n1:Quality_Indicators_Info><Cloud_Coverage_Assessment>{cc}'
       '</Cloud_Coverage_Assessment></n1:Quality_Indicators_Info></n1:Level-1C_User_Product>')


class FakeProcess:
    """Process whose execute ends as set by `outcome`."""
    outcome = 'ok'

    def __init__(self):
        self.no_water = False

    def execute(self, l1c_prod, **kwargs):
        if self.outcome == 'error':
            # the error of issue #13 (GRS 2.1.9, image T28PEC of 2026-07-07)
            raise ValueError('arange: cannot compute length')
        if self.outcome == 'io':
            raise GRS_IO_Exception(l1c_prod, OSError('cannot read'))
        if self.outcome == 'no_water':
            self.no_water = True

    def write_output(self):
        pass


@pytest.fixture
def fake_process(monkeypatch):
    monkeypatch.setattr(grs.grs_process, 'Process', FakeProcess)
    yield FakeProcess
    FakeProcess.outcome = 'ok'


@pytest.fixture
def safe(tmp_path):
    """Level-1 product folder with a cloud cover of 10 %."""
    path = tmp_path / SAFE_NAME
    path.mkdir()
    (path / 'MTD_MSIL1C.xml').write_text(MTD.format(cc=10.0))
    cams = tmp_path / 'cams.nc'
    cams.write_text('')
    return path, cams


# ---------------------------------------------------------------------------------------------------- grs command

def grs_main(safe, *options):
    path, cams = safe
    return run.main([str(path), '--cams_file', str(cams), '--odir', str(path.parent / 'out'), *options])


@pytest.mark.parametrize('outcome, status', [('ok', EXIT_OK), ('error', EXIT_ERROR), ('io', EXIT_INPUT),
                                             ('no_water', EXIT_SKIPPED)])
def test_grs_processing_outcome(fake_process, safe, outcome, status):
    fake_process.outcome = outcome
    assert grs_main(safe) == status


def test_grs_cloud_cover_skipped(fake_process, safe):
    assert grs_main(safe, '--max_cloud_cover', '0.05') == EXIT_SKIPPED


def test_grs_missing_input(fake_process, safe, tmp_path):
    _, cams = safe
    assert run.main([str(tmp_path / 'missing.SAFE'), '--cams_file', str(cams)]) == EXIT_INPUT


def test_grs_missing_cams(fake_process, safe, tmp_path):
    path, _ = safe
    assert run.main([str(path), '--cams_file', str(tmp_path / 'missing.nc')]) == EXIT_INPUT


def test_grs_invalid_arguments(fake_process):
    assert run.main(['image.SAFE', '--unknown_option']) == EXIT_INPUT


def test_grs_no_clobber(fake_process, safe):
    path, _ = safe
    outdir = run.set_ofile(path.name, odir=path.parent / 'out', level_name='L2AGRS', suffix=f'_V{grs.__version__}')
    outdir.mkdir(parents=True)
    # no-water marker of a previous run: skipped, no product
    marker = grs.grs_process.no_water_marker(outdir)
    marker.write_text('')
    assert grs_main(safe, '--no_clobber') == EXIT_SKIPPED
    # product of a previous run: available for the next steps
    (outdir / (outdir.name + '.zarr')).mkdir()
    assert grs_main(safe, '--no_clobber') == EXIT_OK


def test_grs_version(capsys):
    with pytest.raises(SystemExit) as exc:
        run.main(['--version'])
    assert exc.value.code in (None, 0)


# ---------------------------------------------------------------------------------------------------- CNES launcher

@pytest.fixture
def launcher(monkeypatch, tmp_path, safe):
    """exe.launcher.main with configuration files in tmp_path; returns a function (settings) -> exit status."""
    from exe import launcher as launcher_module
    monkeypatch.setattr(launcher_module, 'Process', FakeProcess)
    path, cams = safe

    def call(**settings):
        hymotep = dict(input_file=str(path), cams_folder=str(cams), output_dir=str(tmp_path / 'out'),
                       surfwater_file=None, dem_file=None, suffix=None, noclobber=False, max_cloud_cover=1.,
                       resolution=60, scale_aot=1., opac_model=None, allpixels=False, snap_compliant=False)
        hymotep.update(settings)
        (tmp_path / 'hymotep.yml').write_text(yaml.safe_dump(hymotep))
        (tmp_path / 'global.yml').write_text(yaml.safe_dump(dict(logfile=str(tmp_path / 'log' / 'grs.log'),
                                                                 level='INFO',
                                                                 hymotep_config=str(tmp_path / 'hymotep.yml'))))
        monkeypatch.setattr(sys, 'argv', ['launcher.py', str(tmp_path / 'global.yml')])
        return launcher_module.main()

    yield call
    FakeProcess.outcome = 'ok'


@pytest.mark.parametrize('outcome, status', [('ok', EXIT_OK), ('error', EXIT_ERROR), ('io', EXIT_INPUT),
                                             ('no_water', EXIT_SKIPPED)])
def test_launcher_processing_outcome(launcher, outcome, status):
    FakeProcess.outcome = outcome
    assert launcher() == status


def test_launcher_missing_input(launcher, tmp_path):
    assert launcher(input_file=str(tmp_path / 'missing.SAFE')) == EXIT_INPUT


def test_launcher_missing_cams(launcher, tmp_path):
    assert launcher(cams_folder=str(tmp_path / 'missing.nc')) == EXIT_INPUT


def test_launcher_cloud_cover_skipped(launcher):
    assert launcher(max_cloud_cover=0.05) == EXIT_SKIPPED
