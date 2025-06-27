# Minimal sysconfigdata for Windows Python on WSL build
# This allows pyo3-ffi to find required configuration

build_time_vars = {
    'EXT_SUFFIX': '.pyd',
    'SHLIB_SUFFIX': '.pyd',
    'SO': '.pyd',
    'SOABI': 'cp311-win_amd64',
    'Py_ENABLE_SHARED': 1,
    'LIBDIR': '',
    'BINDIR': '',
    'INCLUDEPY': '',
    'VERSION': '3.11',
    'prefix': '',
    'exec_prefix': '',
}
