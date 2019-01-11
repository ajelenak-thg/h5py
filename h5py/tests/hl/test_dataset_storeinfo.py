from __future__ import absolute_import

from os import SEEK_SET
import numpy as np
import h5py

from ..common import ut, TestCase


@ut.skipIf(h5py.version.hdf5_version_tuple < (1, 10, 5),
           'Requires HDF5 library version 1.10.5 or later')
class TestStoreInfo(TestCase):
    """Test chunk query functionality"""

    def setUp(self):
        TestCase.setUp(self)
        self.data = dict()
        chunked = self.f.create_dataset('chunked', shape=(24, 16),
                                        dtype='i4',
                                        chunks=(6, 4),
                                        maxshape=(None, None))
        self.data['cont'] = np.random.randint(10, size=(7, 11, 4))
        self.f.create_dataset('cont', data=self.data['cont'])
        self.f.create_dataset('chunked_nodata', shape=(10, 20), chunks=(5, 4),
                              dtype='i4')
        self.f.create_dataset('empty', dtype=h5py.Empty('f'))

        # Store data into specific chunks only...
        chnk_dim0, chnk_dim1 = chunked.chunks
        self.chunk_offsets = list()
        self.data['chunked'] = list()
        for i in range(2):
            for j in range(2, 4):
                slice_dim0 = slice(i * chnk_dim0, (i + 1) * chnk_dim0)
                slice_dim1 = slice(j * chnk_dim1, (j + 1) * chnk_dim1)
                region = (slice_dim0, slice_dim1)
                data = np.random.randint(10, size=chunked.chunks,
                                         dtype=chunked.dtype)
                chunked[region] = data
                self.chunk_offsets.append((slice_dim0.start, slice_dim1.start))
                self.data['chunked'].append(data)

        self.f.flush()

    def get_bytes(self, boffset, size):
        """Read specified number of bytes from a location within an HDF5 file"""
        with open(self.f.filename, 'rb') as h5:
            h5.seek(boffset, SEEK_SET)
            buf = h5.read(size)
            if len(buf) != size:
                raise IOError(
                    'Read %d bytes instead of %d bytes at location %s from %s' %
                    (len(buf), size, boffset, self.f.filename))
        return buf

    def test_chunked(self):
        """ Chunk store information """
        dset = self.f['/chunked']
        self.assertIsNotNone(dset.chunks)
        self.assertEqual(len(self.chunk_offsets), dset.id.get_num_chunks())
        idx = len(self.chunk_offsets) - 1
        s = dset.id.get_chunk_info(idx)
        self.assertIsInstance(s, h5py.h5d.StoreInfo)
        self.assertIsNotNone(s.file_offset)
        self.assertNotEqual(s.file_offset, 0)
        chunk = dset.chunks
        self.assertEqual(s.size, chunk[0] * chunk[1] * dset.dtype.itemsize)
        self.assertEqual(s.chunk_offset, self.chunk_offsets[idx])

    def test_chunked_out_index(self):
        """ Store information of an unwritten chunk using its offset """
        dset = self.f['/chunked']
        idx = len(self.chunk_offsets) + 1
        s = dset.id.get_chunk_info(idx)
        self.assertIsInstance(s, h5py.h5d.StoreInfo)
        self.assertEqual(idx, s.index)
        self.assertIsNone(s.file_offset)
        self.assertEqual(s.size, 0)
        self.assertEqual(0, s.filter_mask)

    def test_chunked_store(self):
        """ Store information of all written chunks """
        dset = self.f['/chunked']
        store = dset.store
        self.assertIsInstance(store, list)
        self.assertEqual(len(store), len(self.chunk_offsets))
        chunk_size_bytes = np.prod(dset.chunks) * dset.dtype.itemsize
        for idx, s in enumerate(store):
            self.assertEqual(idx, s.index)
            self.assertEqual(chunk_size_bytes, s.size)
            self.assertEqual(self.chunk_offsets[idx], s.chunk_offset)
            self.assertIsNotNone(s.file_offset)
            self.assertEqual(0, s.filter_mask)
            buf = self.get_bytes(s.file_offset, s.size)
            data = np.frombuffer(buf, dset.dtype)
            self.assertArrayEqual(data.reshape(dset.chunks),
                                  self.data['chunked'][idx])

    def test_chunked_coord(self):
        """ Store information of a written chunk using its offset """
        dset = self.f['/chunked']
        idx = len(self.chunk_offsets) - 1
        s = dset.id.get_chunk_info_by_coord(self.chunk_offsets[idx])
        self.assertIsInstance(s, h5py.h5d.StoreInfo)
        self.assertIsNotNone(s.file_offset)
        chunk_size_bytes = np.prod(dset.chunks) * dset.dtype.itemsize
        self.assertEqual(s.size, chunk_size_bytes)
        self.assertEqual(s.chunk_offset, self.chunk_offsets[idx])

    def test_contiguous_store(self):
        """Contiguous dataset store information"""
        dset = self.f['/cont']
        self.assertIsNone(dset.chunks)
        s = dset.store
        self.assertIsInstance(s, list)
        self.assertEqual(len(s), 1)
        self.assertIsInstance(s[0], h5py.h5d.StoreInfo)
        self.assertIsNotNone(s[0].file_offset)
        self.assertEqual(s[0].size, dset.size * dset.dtype.itemsize)
        self.assertEqual(s[0].filter_mask, 0)
        self.assertEqual(s[0].chunk_offset, (0,) * dset.ndim)
        buf = self.get_bytes(s[0].file_offset, s[0].size)
        data = np.frombuffer(buf, dset.dtype)
        self.assertArrayEqual(data.reshape(dset.shape), self.data['cont'])

    def test_chunked_nodata(self):
        """Chunked dataset without data"""
        dset = self.f['/chunked_nodata']
        self.assertEqual(0, dset.id.get_num_chunks())
        s = dset.id.get_chunk_info(0)
        self.assertIsInstance(s, h5py.h5d.StoreInfo)
        self.assertIsNone(s.file_offset)
        self.assertEqual(s.size, 0)
        self.assertEqual(s.index, 0)
        self.assertIsNone(s.chunk_offset)

    def test_chunked_nodata_coord(self):
        """ Store information for unwritten chunks """
        dset = self.f['/chunked_nodata']
        s = dset.id.get_chunk_info_by_coord((0, 0))
        self.assertIsInstance(s, h5py.h5d.StoreInfo)
        self.assertIsNone(s.file_offset)
        self.assertEqual(s.size, 0)
        self.assertIsNone(s.chunk_offset)
        self.assertIsNone(s.index)

    def test_empty(self):
        """ Store information for an empty dataset """
        dset = self.f['/empty']
        self.assertIsNone(dset.shape)
        s = dset.store
        self.assertIsInstance(s, list)
        self.assertEqual(len(s), 0)
