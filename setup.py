#!/usr/bin/env python

try:
    # If possible, use setuptools.Extension so we get setup_requires
    from setuptools import Extension, setup
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
from distutils.cmd import Command
from distutils.dist import Distribution
from distutils.version import LooseVersion
import warnings
import sys, os
import os.path as op
from functools import reduce

import configure   # Sticky-options configuration and version auto-detect

VERSION = '2.2.1'


# --- Encapsulate NumPy imports in a specialized Extension type ---------------

# https://mail.python.org/pipermail/distutils-sig/2007-September/008253.html
class NumpyExtension(Extension, object):
    """Extension type that adds the NumPy include directory to include_dirs."""

    def __init__(self, *args, **kwargs):
        super(NumpyExtension, self).__init__(*args, **kwargs)

    @property
    def include_dirs(self):
        from numpy import get_include
        return self._include_dirs + [get_include()]

    @include_dirs.setter
    def include_dirs(self, include_dirs):
        self._include_dirs = include_dirs


# --- Autodetect Cython -------------------------------------------------------

HAVE_CYTHON = False
try:
    import Cython.Compiler.Version
    s = LooseVersion(Cython.Compiler.Version.version)
    if s.version[0:2] < [0, 13]:
        warnings.warn("Cython version %s too old; not used" % s.vstring)
        raise ImportError
    from Cython.Distutils import build_ext
    SUFFIX = '.pyx'
    HAVE_CYTHON = True
except ImportError:
    from distutils.command.build_ext import build_ext
    SUFFIX = '.c'

if sys.version_info[0] >= 3:
    # Shamelessly stolen from Cython 0.14
    import lib2to3.refactor
    from distutils.command.build_py \
         import build_py_2to3 as build_py
else:
    from distutils.command.build_py import build_py


# --- Support functions and "super-option" configuring ------------------------

def localpath(*args):
    return op.abspath(reduce(op.join, (op.dirname(__file__),)+args))

def configure_cython(settings, modules):
    """ Set up the Cython build environment.

    If configuration settings have changed since the last time Cython was
    run, re-write the file "config.pxi".  Also bump the utime of all the
    Cython modules to trigger a rebuild.
    """

    try:
        f = open(localpath('h5py/config.pxi'),'rb')
        oldcontents = f.read()
    except IOError:
        oldcontents = b""
    else:
        f.close()

    newcontents = """\
# This file is automatically generated by the h5py setup script.  Don't modify.

DEF MPI = %(mpi)s
DEF HDF5_VERSION = %(hdf5_version)s
DEF EFF = %(eff)s
"""
    newcontents %= settings
    newcontents = newcontents.encode('utf-8')

    # Only reconfigure and rebuild if settings have actually changed.
    if newcontents != oldcontents:
        with open(localpath('h5py/config.pxi'),'wb') as f:
            f.write(newcontents)
        for m in MODULES:
            os.utime(localpath('h5py',m+'.pyx'),None)


# --- Pre-compiling API generation --------------------------------------------

if not op.isfile(localpath('h5py','defs.pyx')):
    if not HAVE_CYTHON:
        raise ValueError("A modern version of Cython is required to build from source")
    import api_gen
    api_gen.run()


# --- Determine configuration settings ----------------------------------------

settings = configure.scrape_eargs()          # lowest priority
settings.update(configure.scrape_cargs())    # highest priority

HDF5 = settings.get('hdf5')
HDF5_VERSION = settings.get('hdf5_version')

EFF = settings.setdefault('eff', False)
MPI = settings.setdefault('mpi', False)
if MPI:
    if not HAVE_CYTHON:
        raise ValueError("Cython is required to compile h5py in MPI mode")
    try:
        import mpi4py
    except ImportError:
        raise ImportError("mpi4py is required to compile h5py in MPI mode")


# --- Configure Cython and create extensions ----------------------------------

if sys.platform.startswith('win'):
    COMPILER_SETTINGS = {
        'libraries'     : ['hdf5dll18'],
        'include_dirs'  : [localpath('lzf'),
                           localpath('win_include')],
        'library_dirs'  : [],
        'define_macros' : [('H5_USE_16_API', None), ('_HDF5USEDLL_', None)]
    }
    if HDF5 is not None:
        COMPILER_SETTINGS['include_dirs'] += [op.join(HDF5, 'include')]
        COMPILER_SETTINGS['library_dirs'] += [op.join(HDF5, 'dll')]
else:
    COMPILER_SETTINGS = {
       'libraries'      : ['hdf5'],
       'include_dirs'   : [localpath('lzf')],
       'library_dirs'   : [],
       'define_macros'  : [('H5_USE_16_API', None)]
    }
    if HDF5 is not None:
        COMPILER_SETTINGS['include_dirs'] += [op.join(HDF5, 'include')]
        COMPILER_SETTINGS['library_dirs'] += [op.join(HDF5, 'lib'), op.join(HDF5, 'lib64')]
    elif sys.platform == 'darwin':
        # putting here both macports and homebrew paths will generate
        # "ld: warning: dir not found" at the linking phase
        COMPILER_SETTINGS['include_dirs'] += ['/opt/local/include'] # macports
        COMPILER_SETTINGS['library_dirs'] += ['/opt/local/lib']     # macports
        COMPILER_SETTINGS['include_dirs'] += ['/usr/local/include'] # homebrew
        COMPILER_SETTINGS['library_dirs'] += ['/usr/local/lib']     # homebrew
    elif sys.platform.startswith('freebsd'):
        COMPILER_SETTINGS['include_dirs'] += ['/usr/local/include'] # homebrew
        COMPILER_SETTINGS['library_dirs'] += ['/usr/local/lib']     # homebrew
    if MPI:
        COMPILER_SETTINGS['include_dirs'] += [mpi4py.get_include()]
    COMPILER_SETTINGS['runtime_library_dirs'] = [op.abspath(x) for x in COMPILER_SETTINGS['library_dirs']]

MODULES =  ['defs','_errors','_objects','_proxy', 'h5fd', 'h5z',
            'h5','h5i','h5r','utils', '_conv', 'h5t','h5s',
            'h5p', 'h5d', 'h5a', 'h5f', 'h5g', 'h5l', 'h5o',
            'h5ac']  #, 'h5ds']

# Exascale FastForward low-level modules
if EFF:
    MODULES += ['h5es', 'eff_control', 'h5rc', 'h5tr',
                'h5m', 'h5q', 'h5v'] #+ ['h5x']

# Exascale FastForward low-level modules
if EFF:
    MODULES += ['h5es', 'eff_control', 'h5rc', 'h5tr',
                'h5m', 'h5q', 'h5v'] #+ ['h5x']

# No Cython, no point in configuring
if HAVE_CYTHON:

    # Don't autodetect if version is manually given
    if HDF5_VERSION is None:
        HDF5_VERSION = configure.autodetect(COMPILER_SETTINGS['library_dirs'])

    if HDF5_VERSION is None:
        HDF5_VERSION = (1, 8, 4)
        configure.printerr("HDF5 autodetection failed; building for 1.8.4+")

    settings['hdf5_version'] = HDF5_VERSION
    configure_cython(settings, MODULES)

else:
    configure.printerr("Cython not present; building for HDF5 1.8.4+")

EXTRA_SRC = {'h5z': [ localpath("lzf/lzf_filter.c"),
                      localpath("lzf/lzf/lzf_c.c"),
                      localpath("lzf/lzf/lzf_d.c")]}

def make_extension(module):
    sources = [op.join('h5py', module+SUFFIX)] + EXTRA_SRC.get(module, [])
    return NumpyExtension('h5py.'+module, sources, **COMPILER_SETTINGS)

EXTENSIONS = [make_extension(m) for m in MODULES]


# --- Custom distutils commands -----------------------------------------------

class test(Command):

    """Run the Exascale FastForward test suite."""

    description = "Run the Exascale FastForward test suite"

    user_options = [('verbosity=', 'V', 'set test report verbosity')]

    def initialize_options(self):
        self.verbosity = 0

    def finalize_options(self):
        try:
            self.verbosity = int(self.verbosity)
        except ValueError:
            raise ValueError('verbosity must be an integer.')

    def run(self):
        import sys
        py_version = sys.version_info[:2]
        if py_version == (2,7) or py_version >= (3,2):
            import unittest
        else:
            try:
                import unittest2 as unittest
            except ImportError:
                raise ImportError(
                    "unittest2 is required to run tests with python-%d.%d"
                    % py_version
                    )
        buildobj = self.distribution.get_command_obj('build')
        buildobj.run()
        oldpath = sys.path
        try:
            # Make sure newly built h5py is found first...
            sys.path = [op.abspath(buildobj.build_lib)] + oldpath

            # Import build options...
            from configure import loadpickle
            settings = loadpickle('h5py_config.pickle')
            if settings is None:
                raise RuntimeError("Failed to load build options from %s"
                                   % 'h5py_config.pickle')
            EFF = settings.setdefault('eff', False)
            if EFF:
                # Discover only FastForward tests...
                suite = unittest.TestLoader().discover(op.join(buildobj.build_lib,'h5py'),
                                                       pattern='test_ff*.py')
            else:
                # Discover only standard tests...
                suite = unittest.TestLoader().discover(op.join(buildobj.build_lib,'h5py'))

            # Run found tests...
            result = unittest.TextTestRunner(verbosity=self.verbosity+1).run(suite)

            if not result.wasSuccessful():
                sys.exit(1)
        finally:
            sys.path = oldpath



# --- Distutils setup and metadata --------------------------------------------

cls_txt = \
"""
Development Status :: 5 - Production/Stable
Intended Audience :: Developers
Intended Audience :: Information Technology
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Programming Language :: Python
Topic :: Scientific/Engineering
Topic :: Database
Topic :: Software Development :: Libraries :: Python Modules
Operating System :: Unix
Operating System :: POSIX :: Linux
Operating System :: MacOS :: MacOS X
Operating System :: Microsoft :: Windows
"""

short_desc = "Read and write HDF5 files from Python"

long_desc = \
"""
The h5py package provides both a high- and low-level interface to the HDF5
library from Python. The low-level interface is intended to be a complete
wrapping of the HDF5 API, while the high-level component supports  access to
HDF5 files, datasets and groups using established Python and NumPy concepts.

A strong emphasis on automatic conversion between Python (Numpy) datatypes and
data structures and their HDF5 equivalents vastly simplifies the process of
reading and writing data from Python.

Supports HDF5 versions 1.8.3 and higher.  On Windows, HDF5 is included with
the installer.
"""

if os.name == 'nt':
    package_data = {'h5py': ['*.pyx', '*.dll']}
else:
    package_data = {'h5py': ['*.pyx']}

# Avoid going off and installing NumPy if the user only queries for information
if any('--' + opt in sys.argv for opt in Distribution.display_option_names + ['help']):
    setup_requires = []
else:
    setup_requires = ['numpy >=1.0.1']

setup(
  name = 'h5py',
  version = VERSION,
  description = short_desc,
  long_description = long_desc,
  classifiers = [x for x in cls_txt.split("\n") if x],
  author = 'Andrew Collette',
  author_email = 'andrew dot collette at gmail dot com',
  maintainer = 'Andrew Collette',
  maintainer_email = 'andrew dot collette at gmail dot com',
  url = 'http://www.h5py.org',
  download_url = 'http://code.google.com/p/h5py/downloads/list',
  packages = ['h5py', 'h5py._hl', 'h5py._hl.tests', 'h5py.lowtest'],
  package_data = package_data,
  ext_modules = EXTENSIONS,
  requires = ['numpy (>=1.0.1)'],
  setup_requires = setup_requires,
  cmdclass = {'build_ext': build_ext, 'test': test, 'build_py':build_py}
)
