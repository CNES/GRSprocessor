# limit the malloc arenas before the heavy imports and the threads (see grs.run.limit_malloc_arenas)
from grs.run import limit_malloc_arenas
limit_malloc_arenas()

from grs import class_logger
from grs import __version__
from grs.grs_process import Process, no_water_marker
from grs.grs_exceptions import GRS_IO_Exception
from grs import remote
from grs.run import s2_cloud_cover, EXIT_OK, EXIT_ERROR, EXIT_INPUT, EXIT_SKIPPED
from pathlib import Path
import yaml
import sys
import logging
import os
from datetime import datetime

from exe.procutils import misc

misc = misc()


def main():
    '''
    Process the image of the configuration files (see `_main`) and close the log files whatever the outcome;
    returns the exit status (``grs.run.EXIT_*``).
    '''
    try:
        return _main()
    finally:
        # Close logger and get stats (if it was set up)
        logger = getattr(class_logger.THIS, 'klass', None)
        if logger is not None:
            logger.close()


def _main():
    '''
    Process the image of the configuration files; returns the exit status (``grs.run.EXIT_*``): 0 product written
    (or already there with noclobber), 1 processing error, 2 missing or unreadable input, 3 image skipped on
    purpose (cloud cover, no water): no product.
    '''
    # read config and prepare environment
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "/home/grs2/exe/global_config.yml"

    with open(config_file, 'r') as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)

    # file handle
    logfile = Path(data['logfile'])
    log_folder = logfile.parent
    logfile.parent.mkdir(parents=True, exist_ok=True)
    class_logger.ServiceLogger(log_file=logfile, error_log=Path(log_folder, "error.log"), log_level=data['level'],
                               log_console=True)

    # get all config
    config_path = os.path.expandvars(data['hymotep_config'])
    with open(config_path, 'r') as config_file:
        data.update(yaml.load(config_file, Loader=yaml.FullLoader))

    for key, value in data.items():
        if value is not None and value != '':
            data[key] = value
        else:
            data[key] = None
    # local paths or URLs (s3://, https://..., see grs.remote); a URL is kept as a string
    file = os.path.expandvars(data['input_file'])
    file = file if remote.is_url(file) else Path(file)

    if not remote.exists(file):
        logging.error("Missing input file. Process stopped")
        return EXIT_INPUT
    if not data["cams_folder"]:
        logging.error("Missing CAMS folder. Process stopped")
        return EXIT_INPUT

    input_filename = remote.basename(file)

    # Get CAMS file
    cams_folder = os.path.expandvars(str(data['cams_folder']))
    if remote.is_url(cams_folder) and cams_folder.rstrip('/').endswith('.nc'):
        cams_file = cams_folder
    elif not remote.is_url(cams_folder) and Path(cams_folder).is_file():
        cams_file = Path(cams_folder)
    else:
        input_date = datetime.strptime(input_filename.split("_")[2], '%Y%m%dT%H%M%S').date()
        year = input_date.strftime('%Y')
        month = input_date.strftime('%m')
        day = input_date.strftime('%d')
        logging.info('Search for the daily CAMS file')
        cams_file = remote.join(cams_folder, year, month, day,
                                input_date.strftime('%Y-%m-%d') + '-cams-global-atmospheric-composition-forecasts.nc')
        if not remote.exists(cams_file):
            logging.info('No daily CAMS file found. Search for the monthly one')
            cams_file = remote.join(cams_folder, year,
                                    input_date.strftime('%Y-%m') + '_month_cams-global-atmospheric-composition-forecasts.nc')
    logging.info(f'CAMS file : {cams_file}')

    # Verify existence of inputs
    if not remote.is_url(file) and not file.is_dir():
        logging.error("Input file doesn't exist. Process stopped")
        return EXIT_INPUT

    if not remote.exists(cams_file):
        logging.error("CAMS file doesn't exist. Process stopped")
        return EXIT_INPUT

    if data["surfwater_file"] and not remote.exists(data["surfwater_file"]):
        logging.error("SurfWater file doesn't exist. Process stopped")
        return EXIT_INPUT

    # prepare outfile
    suffix = data["suffix"]
    if not suffix:
        suffix = f'_V{__version__}'

    # output folder <output_dir>/<name> (local path or URL) and its Zarr store <name>.zarr
    name = misc.set_ofile(input_filename, odir='', level_name='L2AGRS', suffix=suffix).name
    outdir = remote.join(data['output_dir'], name)
    remote_output = remote.is_url(outdir)
    if not remote_output:
        outdir.mkdir(parents=True, exist_ok=True)

    # skip if already processed (Zarr store, the default, or NetCDF file), or already found without water
    if data["noclobber"]:
        for outfile in (remote.join(outdir, f"{name}.zarr"), remote.join(outdir, f"{name}.nc")):
            if remote.exists(outfile):
                logging.info(f'File {outfile} already processed; skip!')
                return EXIT_OK
        if remote.exists(no_water_marker(outdir)):
            logging.info(f'File {no_water_marker(outdir)} found (image without water); skip!')
            return EXIT_SKIPPED

    logging.info(f'call grs_process for the following parameters. '
                 f'File: {file} , '
                 f'output idirectory:  {outdir} , '
                 f'cams_file:  {cams_file} , '
                 f'surfwater_file: {data["surfwater_file"]} , '
                 f'resolution: {data["resolution"]} , '
                 f'allpixels: {data["allpixels"]} , '
                 f'snap_compliant: {data["snap_compliant"]})')

    # first check cloud cover (for S2, not implemented for Landsat)
    if 'MSIL1C' in input_filename:
        max_cc = data["max_cloud_cover"]
        cc = s2_cloud_cover(file)
        if cc >= max_cc:
            logging.info('input file not processed since cloud cover {:.3f} is greater than {:.3f}'.format(cc, max_cc))
            return EXIT_SKIPPED

    try:
        process_ = Process()
        process_.execute(file,
                         # the Zarr store itself for a URL (an object storage has no folders to create)
                         odir=remote.join(outdir, f"{name}.zarr") if remote_output else outdir,
                         cams_file=cams_file,
                         resolution=data["resolution"],
                         scale_aot=data["scale_aot"],
                         opac_model=data["opac_model"],
                         dem_file=data["dem_file"],
                         allpixels=data["allpixels"],
                         surfwater_file=data["surfwater_file"],
                         snap_compliant=data["snap_compliant"])
        # for an image without water: marker file <outdir>/<name>_no_water.txt instead of the product
        process_.write_output()
        return EXIT_SKIPPED if getattr(process_, 'no_water', False) else EXIT_OK

    except Exception as inst:
        logging.error('-------------------------------')
        message = 'error for file  ' + str(inst) + ' skip'
        logging.error(message)
        logging.error('-------------------------------')
        logging.error('error during grs', exc_info=True)
        return EXIT_INPUT if isinstance(inst, GRS_IO_Exception) else EXIT_ERROR


if __name__ == '__main__':
    sys.exit(main())
