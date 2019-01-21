#!/usr/bin/env python3
"""
Print storage information for every HDF5 dataset in a file.

Run "store_info.py --help" for information.
"""

from __future__ import print_function
from os import SEEK_SET
from os.path import basename
import argparse
import json
from functools import partial
from hashlib import sha3_256
from uuid import uuid4
import h5py


def get_cksum(fobj, offset, blen):
    if fobj:
        fobj.seek(offset, SEEK_SET)
        byte_stream = fobj.read(blen)
        if len(byte_stream) != blen:
            raise IOError(
                'Read %d bytes instead of %d bytes at byte %s from %s' %
                (len(byte_stream), blen, offset, fobj.name))
        return sha3_256(byte_stream).hexdigest()
    else:
        return None


def plain_fmt(name, h5obj, fobj=None):
    """Print storage information for each dataset in the file."""
    if isinstance(h5obj, h5py.Dataset):
        try:
            stinfo = h5obj.store
        except Exception:
            print('Caught exception for {}'.format(h5obj.name))
            return

        if len(stinfo) == 0:
            print('Dataset: {} has no data'.format(h5obj.name))
            return

        for si in stinfo:
            cksum = get_cksum(fobj, si.file_offset, si.size)
            if cksum is None:
                cksum_str = ''
            else:
                cksum_str = '; SHA-3-256: ' + cksum
            print ('Dataset {}: index: {}; array offset: {}; '
                   'at byte {} of size {} bytes{}'
                   .format(h5obj.name, si.index, si.chunk_offset,
                           si.file_offset, si.size, cksum_str))


def json_fmt(name, h5obj, fobj=None, dict_=None):
    if isinstance(h5obj, h5py.Dataset):
        try:
            stinfo = h5obj.store
        except Exception:
            print('Caught exception for {}'.format(h5obj.name))
            return

        byte_streams = list()
        for si in stinfo:
            byte_streams.append({'file_offset': si.file_offset,
                                 'size': si.size,
                                 'index': si.index,
                                 'array_offset': si.chunk_offset,
                                 'uuid': str(uuid4())})
            cksum = get_cksum(fobj, si.file_offset, si.size)
            if cksum:
                byte_streams[-1].update(
                    {'cksum': {'type': 'SHA-3-256', 'value': cksum}})
        dict_.update({h5obj.name: {'byteStreams': byte_streams}})


parser = argparse.ArgumentParser()
parser.add_argument('h5file', help='HDF5 file path')
parser.add_argument('-j', help='Produce dataset storage info in JSON format',
                    action='store_true')
parser.add_argument('-c', help='Add SHA-3-256 checksum for each byte stream',
                    action='store_true')
args = parser.parse_args()

with h5py.File(args.h5file, 'r') as h5f:
    if args.c:
        f = open(args.h5file, 'rb')
        if not f.seekable():
            raise OSError('Byte stream for %s not seekable' % args.file)
        f.seek(0, SEEK_SET)
    else:
        f = None

    if args.j:
        fname = basename(args.h5file)
        stinfo = {fname: {}}
        h5f.visititems(partial(json_fmt, fobj=f, dict_=stinfo[fname]))
        print(json.dumps(stinfo))
    else:
        h5f.visititems(partial(plain_fmt, fobj=f))
