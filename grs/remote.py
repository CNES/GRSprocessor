'''
Paths that can be local or remote (object storage, web server), for the inputs and outputs of the processing.

A remote path is an `fsspec <https://filesystem-spec.readthedocs.io>`_ URL: ``s3://bucket/key`` (S3 or any
S3-compatible storage), ``gs://``, ``az://``, ``https://``... Anything else (including ``file://``) is a local path.

Credentials and endpoint of an S3-compatible storage are those of s3fs/botocore: the environment
(``AWS_PROFILE``, ``AWS_ACCESS_KEY_ID``/``AWS_SECRET_ACCESS_KEY``, ``AWS_ENDPOINT_URL``...) and the AWS
configuration files (``~/.aws/config``, ``~/.aws/credentials``), or ``storage_options`` given to the functions
below (passed to the fsspec file system, e.g. ``{'key': ..., 'secret': ..., 'endpoint_url': ...}``).

Example (the ``grs`` command, inputs and outputs on an S3-compatible storage)::

    export AWS_PROFILE=my-profile AWS_ENDPOINT_URL=https://s3.example.org
    grs s3://bucket/l1c/S2A_MSIL1C_..._.SAFE --cams_file s3://bucket/cams/2023-06-14-cams.nc \\
        --odir s3://bucket/l2a --resolution 20
'''

import os
from pathlib import Path, PurePosixPath


def is_url(path):
    '''True if `path` is a remote URL (``s3://``, ``https://``...), False for a local path (or ``file://``).'''
    path = str(path)
    return '://' in path and not path.startswith('file://')


def join(base, *parts):
    '''
    Join path components: a URL string for a URL `base` (``/`` separators, ``Path`` would turn ``s3://`` into
    ``s3:/``), else a ``Path``.
    '''
    if is_url(base):
        return '/'.join([str(base).rstrip('/')] + [str(p).strip('/') for p in parts])
    return Path(base, *parts)


def basename(path):
    '''Last component of a local path or of a URL (trailing ``/`` ignored).'''
    path = str(path).rstrip('/')
    return PurePosixPath(path.split('://', 1)[-1]).name if is_url(path) else Path(path).name


def filesystem(url, storage_options=None):
    '''fsspec file system and path in it of `url`.'''
    import fsspec
    return fsspec.core.url_to_fs(str(url), **(storage_options or {}))


def exists(path, storage_options=None):
    '''True if the local path or the URL (file, folder or prefix) exists.'''
    if not is_url(path):
        return Path(path).exists()
    fs, fs_path = filesystem(path, storage_options)
    return fs.exists(fs_path.rstrip('/'))


def read_bytes(path, storage_options=None):
    '''Content of a local file or of a URL.'''
    if not is_url(path):
        return Path(path).read_bytes()
    fs, fs_path = filesystem(path, storage_options)
    with fs.open(fs_path, 'rb') as file:
        return file.read()


def read_text(path, storage_options=None, encoding='utf-8'):
    '''Text of a local file or of a URL.'''
    return read_bytes(path, storage_options).decode(encoding)


def write_text(path, text, storage_options=None, encoding='utf-8'):
    '''Write `text` to a local file (parent folders created) or to a URL.'''
    if not is_url(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding=encoding)
        return
    fs, fs_path = filesystem(path, storage_options)
    with fs.open(fs_path, 'wb') as file:
        file.write(text.encode(encoding))


def upload(local_file, url, storage_options=None):
    '''Copy a local file to a URL.'''
    fs, fs_path = filesystem(url, storage_options)
    fs.put_file(str(local_file), fs_path)


def cache_dir():
    '''
    Local folder of the copies of remote input files (``GRS_CACHE_DIR``, by default ``grs/remote`` in the
    user cache folder).
    '''
    if os.environ.get('GRS_CACHE_DIR'):
        return Path(os.environ['GRS_CACHE_DIR']).expanduser()
    from platformdirs import user_cache_dir
    return Path(user_cache_dir('grs'), 'remote')


def local_file(path, storage_options=None):
    '''
    Local path of a file: the path itself if it is local, else a copy of the URL in `cache_dir` (downloaded once,
    reused while the remote file does not change).

    Used for the inputs read as a whole by libraries that need a local file (e.g. the CAMS NetCDF files).
    '''
    if not is_url(path):
        return str(path)
    import fsspec
    protocol = str(path).split('://', 1)[0]
    cache = cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    return fsspec.open_local(f'filecache::{path}',
                             filecache={'cache_storage': str(cache), 'check_files': True},
                             **{protocol: storage_options or {}})
