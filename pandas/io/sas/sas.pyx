# cython: profile=False
# cython: boundscheck=False, initializedcheck=False
from cpython.mem cimport (
    PyMem_Free,
    PyMem_Malloc,
)
from libc.stdint cimport (
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)
from libc.string cimport (
    memcmp,
    memcpy,
    memset,
)

import struct

import numpy as np

import pandas.io.sas.sas_constants as const

# Typed const aliases
assert len(const.page_mix_types) == 2
cdef:
    int page_meta_type = const.page_meta_type
    int page_mix_types_0 = const.page_mix_types[0]
    int page_mix_types_1 = const.page_mix_types[1]
    int page_data_type = const.page_data_type

    int subheader_pointers_offset = const.subheader_pointers_offset
    int truncated_subheader_id = const.truncated_subheader_id
    int compressed_subheader_id = const.compressed_subheader_id
    int compressed_subheader_type = const.compressed_subheader_type

    int data_subheader_index = const.SASIndex.data_subheader_index
    int row_size_index = const.SASIndex.row_size_index
    int column_size_index = const.SASIndex.column_size_index
    int column_text_index = const.SASIndex.column_text_index
    int column_name_index = const.SASIndex.column_name_index
    int column_attributes_index = const.SASIndex.column_attributes_index
    int format_and_label_index = const.SASIndex.format_and_label_index
    int column_list_index = const.SASIndex.column_list_index
    int subheader_counts_index = const.SASIndex.subheader_counts_index

# Typed const aliases: subheader_signature_to_index
assert all(len(k) in (4, 8) for k in const.subheader_signature_to_index)
_sigs32 = {k: v for k, v in const.subheader_signature_to_index.items() if len(k) == 4}
_sigs64 = {k: v for k, v in const.subheader_signature_to_index.items() if len(k) == 8}
cdef:
    _subheader_signature_to_index_keys32 = b"".join(_sigs32.keys())
    const uint32_t *subheader_signature_to_index_keys32 = <const uint32_t *><const uint8_t *>_subheader_signature_to_index_keys32
    Py_ssize_t[:] subheader_signature_to_index_values32 = np.asarray(list(_sigs32.values()))

    _subheader_signature_to_index_keys64 = b"".join(_sigs64.keys())
    const uint64_t *subheader_signature_to_index_keys64 = <const uint64_t *><const uint8_t *>_subheader_signature_to_index_keys64
    Py_ssize_t[:] subheader_signature_to_index_values64 = np.asarray(list(_sigs64.values()))


cdef class _SubheaderPointer:
    cdef:
        Py_ssize_t offset, length

    def __init__(self, Py_ssize_t offset, Py_ssize_t length):
        self.offset = offset
        self.length = length


cdef class BasePage:
    cdef:
        object sas7bdatreader
        readonly bytes data
        const uint8_t *data_array
        Py_ssize_t data_len

    def __init__(self, sas7bdatreader, data):
        self.sas7bdatreader = sas7bdatreader
        self.data = data
        self.data_array = self.data
        self.data_len = len(data)

    def __len__(self):
        return self.data_len

    def read_bytes(self, Py_ssize_t offset, Py_ssize_t width):
        self.check_read(offset, width)
        return self.data_array[offset:offset+width]

    cpdef check_read(self, Py_ssize_t offset, Py_ssize_t width):
        if offset + width > self.data_len:
            self.sas7bdatreader.close()
            raise ValueError("The cached page is too small.")


cdef class Page(BasePage):
    cdef bint is_little_endian

    def __init__(self, sas7bdatreader, data, is_little_endian):
        super().__init__(sas7bdatreader, data)
        self.is_little_endian = is_little_endian

    def process_page_metadata(self):
        cdef:
            Py_ssize_t int_length = self.sas7bdatreader._int_length
            Py_ssize_t i, total_offset, subheader_offset, subheader_length, subheader_compression, subheader_type
            Py_ssize_t page_bit_offset = self.sas7bdatreader._page_bit_offset
            Py_ssize_t current_page_subheaders_count = self.sas7bdatreader._current_page_subheaders_count
            Py_ssize_t subheader_pointer_length = self.sas7bdatreader._subheader_pointer_length
            list current_page_data_subheader_pointers = self.sas7bdatreader._current_page_data_subheader_pointers

        for i in range(current_page_subheaders_count):
            total_offset = subheader_pointers_offset + page_bit_offset + subheader_pointer_length * i

            subheader_offset = self.read_int(total_offset, int_length)
            total_offset += int_length

            subheader_length = self.read_int(total_offset, int_length)
            total_offset += int_length

            subheader_compression = self.read_int(total_offset, 1)
            total_offset += 1

            subheader_type = self.read_int(total_offset, 1)

            if subheader_length == 0 or subheader_compression == truncated_subheader_id:
                continue

            subheader_index = self._get_subheader_index(subheader_offset, int_length, subheader_compression, subheader_type)
            processor = self._get_subheader_processor(subheader_index)
            if processor is None:
                current_page_data_subheader_pointers.append(
                    _SubheaderPointer(subheader_offset, subheader_length)
                )
            else:
                processor(subheader_offset, subheader_length)

    cdef Py_ssize_t _get_subheader_index(self, Py_ssize_t signature_offset, Py_ssize_t signature_length, Py_ssize_t compression, Py_ssize_t ptype):
        # TODO: return here could be made an enum
        cdef Py_ssize_t i

        self.check_read(signature_offset, signature_length)

        if signature_length == 4:
            for i in range(len(subheader_signature_to_index_values32)):
                if not memcmp(&subheader_signature_to_index_keys32[i], &self.data_array[signature_offset], 4):
                    return subheader_signature_to_index_values32[i]
        else:
            for i in range(len(subheader_signature_to_index_values64)):
                if not memcmp(&subheader_signature_to_index_keys64[i], &self.data_array[signature_offset], 8):
                    return subheader_signature_to_index_values64[i]

        if self.sas7bdatreader.compression and (compression in (compressed_subheader_id, 0)) and ptype == compressed_subheader_type:
            return data_subheader_index
        else:
            self.sas7bdatreader.close()
            raise ValueError(f"Unknown subheader signature {self.data_array[signature_offset:signature_offset+signature_length]}")

    cdef object _get_subheader_processor(self, Py_ssize_t index):
        if index == data_subheader_index:
            return None
        elif index == row_size_index:
            return self.sas7bdatreader._process_rowsize_subheader
        elif index == column_size_index:
            return self.sas7bdatreader._process_columnsize_subheader
        elif index == column_text_index:
            return self.sas7bdatreader._process_columntext_subheader
        elif index == column_name_index:
            return self.sas7bdatreader._process_columnname_subheader
        elif index == column_attributes_index:
            return self.sas7bdatreader._process_columnattributes_subheader
        elif index == format_and_label_index:
            return self.sas7bdatreader._process_format_subheader
        elif index == column_list_index:
            return self.sas7bdatreader._process_columnlist_subheader
        elif index == subheader_counts_index:
            return self.sas7bdatreader._process_subheader_counts
        else:
            raise ValueError(f"unknown subheader index {index}")

    cpdef double read_float(self, Py_ssize_t offset, Py_ssize_t width):
        self.check_read(offset, width)
        return (float_unpack4 if width == 4 else float_unpack8)(
            &self.data_array[offset],
            self.is_little_endian,
        )

    cpdef Py_ssize_t read_int(self, Py_ssize_t offset, Py_ssize_t width):
        self.check_read(offset, width)
        cdef:
            const uint8_t *d = &self.data_array[offset]
        if width == 1:
            return d[0]
        elif width == 2:
            return (le16toh_ if self.is_little_endian else be16toh_)((<uint16_t *>d)[0])
        elif width == 4:
            return (le32toh_ if self.is_little_endian else be32toh_)((<uint32_t *>d)[0])
        else:
            return (le64toh_ if self.is_little_endian else be64toh_)((<uint64_t *>d)[0])


cdef class SAS7BDATCythonReader:
    cdef:
        # Static
        object sas7bdatreader
        uint8_t[:, :] byte_chunk
        object[:, :] string_chunk
        int row_length
        int page_bit_offset
        int subheader_pointer_length
        int row_count
        int mix_page_row_count
        # Synced Cython <-> Python, see _update_{c,p}ython_row_indices()
        public int current_row_in_chunk_index
        public int current_row_in_file_index
        # Synced Python -> Cython, see _update_cython_page_info()
        public int current_row_on_page_index
        public int current_page_type
        public int current_page_block_count
        public list current_page_data_subheader_pointers
        public int current_page_subheaders_count
        public Page cached_page

        Py_ssize_t (*decompress)(const uint8_t *, Py_ssize_t, uint8_t *)

        Py_ssize_t[:] column_data_offsets, column_data_lengths
        char[:] column_types

    def __init__(self,
                sas7bdatreader,
                byte_chunk,
                string_chunk,
                row_length,
                page_bit_offset,
                subheader_pointer_length,
                row_count,
                mix_page_row_count,
                column_data_offsets,
                column_data_lengths,
                column_types,
                compression
     ):
        self.sas7bdatreader = sas7bdatreader
        self.byte_chunk = byte_chunk
        self.string_chunk = string_chunk
        self.row_length = row_length
        self.page_bit_offset = page_bit_offset
        self.subheader_pointer_length = subheader_pointer_length
        self.row_count = row_count
        self.mix_page_row_count = mix_page_row_count
        self.column_data_offsets = column_data_offsets
        self.column_data_lengths = column_data_lengths
        self.column_types = column_types

        # Compression
        if compression == const.rle_compression:
            self.decompress = _rle_decompress
        elif compression == const.rdc_compression:
            self.decompress = _rdc_decompress
        else:
            self.decompress = NULL

    def read(self, int nrows):
        cdef bint done

        for _ in range(nrows):
            done = self._readline()
            if done:
                break

    cdef bint _readline(self):
        cdef:
            bint done
            _SubheaderPointer current_subheader_pointer

        # Loop until a data row is read
        while True:
            if self.current_page_type == page_meta_type:
                if self.current_row_on_page_index >= len(self.current_page_data_subheader_pointers):
                    done = self.sas7bdatreader._read_next_page()
                    if done:
                        return True
                    else:
                        continue
                else:
                    current_subheader_pointer = self.current_page_data_subheader_pointers[self.current_row_on_page_index]
                    self.process_byte_array_with_data(current_subheader_pointer.offset, current_subheader_pointer.length)
                    return False
            elif self.current_page_type in (page_mix_types_0, page_mix_types_1):
                return self._readline_mix_page()
            else:
                return self._readline_data_page()

    cdef bint _readline_mix_page(self):
        cdef:
            Py_ssize_t align_correction, offset
        align_correction = (
            self.page_bit_offset
            + subheader_pointers_offset
            + self.current_page_subheaders_count * self.subheader_pointer_length
        )
        align_correction = align_correction % 8
        offset = self.page_bit_offset + align_correction
        offset += subheader_pointers_offset
        offset += self.current_page_subheaders_count * self.subheader_pointer_length
        offset += self.current_row_on_page_index * self.row_length
        self.process_byte_array_with_data(offset, self.row_length)
        if self.current_row_on_page_index == min(self.row_count, self.mix_page_row_count):
            return self.sas7bdatreader._read_next_page()
        else:
            return False

    cdef bint _readline_data_page(self):
        self.process_byte_array_with_data(
            self.page_bit_offset
            + subheader_pointers_offset
            + self.current_row_on_page_index * self.row_length,
            self.row_length,
        )
        if self.current_row_on_page_index == self.current_page_block_count:
            return self.sas7bdatreader._read_next_page()
        else:
            return False

    cpdef void process_byte_array_with_data(self, int offset, int length):
        cdef:
            char column_type
            Py_ssize_t data_length, data_offset, pos
            const uint8_t *source
            Py_ssize_t j, outpos, m, jb = 0, js = 0
            uint8_t *decompressed_source = NULL

        source = &self.cached_page.data_array[offset]
        if self.decompress != NULL and length < self.row_length:
            decompressed_source = <uint8_t *>PyMem_Malloc(self.row_length * sizeof(uint8_t))
            outpos = self.decompress(source, length, decompressed_source)
            memset(&decompressed_source[outpos], 0, self.row_length - outpos)
            source = decompressed_source

        for j in range(len(self.column_data_offsets)):
            column_type = self.column_types[j]
            data_length = self.column_data_lengths[j]
            data_offset = self.column_data_offsets[j]
            if data_length == 0:
                break
            if column_type == b"d":
                # decimal
                m = 8 * self.current_row_in_chunk_index
                if self.cached_page.is_little_endian:
                    m += 8 - data_length
                memcpy(&self.byte_chunk[jb, m], &source[data_offset], data_length)
                jb += 1
            elif column_type == b"s":
                # string
                while False and data_length > 8 \
                    and source[data_offset+data_length-8] in b"\x00 " \
                    and source[data_offset+data_length-7] in b"\x00 " \
                    and source[data_offset+data_length-6] in b"\x00 " \
                    and source[data_offset+data_length-5] in b"\x00 " \
                    and source[data_offset+data_length-4] in b"\x00 " \
                    and source[data_offset+data_length-3] in b"\x00 " \
                    and source[data_offset+data_length-2] in b"\x00 " \
                    and source[data_offset+data_length-1] in b"\x00 ":
                        data_length -= 8
                while data_length > 0 and source[data_offset+data_length-1] in b"\x00 ":
                    data_length -= 1
                self.string_chunk[js, self.current_row_in_chunk_index] = source[data_offset:data_offset+data_length]
                js += 1
            else:
                raise ValueError(f"unknown column type {column_type!r}")

        self.current_row_in_chunk_index += 1
        self.current_row_in_file_index += 1
        self.current_row_on_page_index += 1

        if decompressed_source != NULL:
            PyMem_Free(decompressed_source)


# Decompression

# _rle_decompress decompresses data using a Run Length Encoding
# algorithm.  It is partially documented here:
#
# https://cran.r-project.org/package=sas7bdat/vignettes/sas7bdat.pdf
cdef Py_ssize_t _rle_decompress(const uint8_t *inbuff, Py_ssize_t length, uint8_t *outbuff):

    cdef:
        uint8_t control_byte, x
        Py_ssize_t rpos = 0
        Py_ssize_t i, nbytes, end_of_first_byte
        Py_ssize_t ipos = 0

    while ipos < length:
        control_byte = inbuff[ipos] & 0xF0
        end_of_first_byte = <Py_ssize_t>(inbuff[ipos] & 0x0F)
        ipos += 1

        if control_byte == 0x00:
            if end_of_first_byte != 0:
                raise ValueError("Unexpected non-zero end_of_first_byte")
            nbytes = <Py_ssize_t>(inbuff[ipos]) + 64
            ipos += 1
            for _ in range(nbytes):
                outbuff[rpos] = inbuff[ipos]
                rpos += 1
                ipos += 1
        elif control_byte == 0x40:
            # not documented
            nbytes = end_of_first_byte * 16
            nbytes += <Py_ssize_t>(inbuff[ipos])
            ipos += 1
            for _ in range(nbytes):
                outbuff[rpos] = inbuff[ipos]
                rpos += 1
            ipos += 1
        elif control_byte == 0x60:
            nbytes = end_of_first_byte * 256 + <Py_ssize_t>(inbuff[ipos]) + 17
            ipos += 1
            for _ in range(nbytes):
                outbuff[rpos] = 0x20
                rpos += 1
        elif control_byte == 0x70:
            nbytes = end_of_first_byte * 256 + <Py_ssize_t>(inbuff[ipos]) + 17
            ipos += 1
            for _ in range(nbytes):
                outbuff[rpos] = 0x00
                rpos += 1
        elif control_byte == 0x80:
            nbytes = end_of_first_byte + 1
            for i in range(nbytes):
                outbuff[rpos] = inbuff[ipos + i]
                rpos += 1
            ipos += nbytes
        elif control_byte == 0x90:
            nbytes = end_of_first_byte + 17
            for i in range(nbytes):
                outbuff[rpos] = inbuff[ipos + i]
                rpos += 1
            ipos += nbytes
        elif control_byte == 0xA0:
            nbytes = end_of_first_byte + 33
            for i in range(nbytes):
                outbuff[rpos] = inbuff[ipos + i]
                rpos += 1
            ipos += nbytes
        elif control_byte == 0xB0:
            nbytes = end_of_first_byte + 49
            for i in range(nbytes):
                outbuff[rpos] = inbuff[ipos + i]
                rpos += 1
            ipos += nbytes
        elif control_byte == 0xC0:
            nbytes = end_of_first_byte + 3
            x = inbuff[ipos]
            ipos += 1
            for _ in range(nbytes):
                outbuff[rpos] = x
                rpos += 1
        elif control_byte == 0xD0:
            nbytes = end_of_first_byte + 2
            for _ in range(nbytes):
                outbuff[rpos] = 0x40
                rpos += 1
        elif control_byte == 0xE0:
            nbytes = end_of_first_byte + 2
            for _ in range(nbytes):
                outbuff[rpos] = 0x20
                rpos += 1
        elif control_byte == 0xF0:
            nbytes = end_of_first_byte + 2
            for _ in range(nbytes):
                outbuff[rpos] = 0x00
                rpos += 1
        else:
            raise ValueError(f"unknown control byte: {control_byte}")

    return rpos

# _rdc_decompress decompresses data using the Ross Data Compression algorithm:
#
# http://collaboration.cmc.ec.gc.ca/science/rpn/biblio/ddj/Website/articles/CUJ/1992/9210/ross/ross.htm
cdef Py_ssize_t _rdc_decompress(const uint8_t *inbuff, Py_ssize_t length, uint8_t *outbuff):

    cdef:
        uint8_t cmd
        uint16_t ctrl_bits = 0, ctrl_mask = 0, ofs, cnt
        Py_ssize_t rpos = 0, k
        Py_ssize_t ipos = 0

    ii = -1

    while ipos < length:
        ii += 1
        ctrl_mask = ctrl_mask >> 1
        if ctrl_mask == 0:
            ctrl_bits = ((<uint16_t>inbuff[ipos] << 8) +
                         <uint16_t>inbuff[ipos + 1])
            ipos += 2
            ctrl_mask = 0x8000

        if ctrl_bits & ctrl_mask == 0:
            outbuff[rpos] = inbuff[ipos]
            ipos += 1
            rpos += 1
            continue

        cmd = (inbuff[ipos] >> 4) & 0x0F
        cnt = <uint16_t>(inbuff[ipos] & 0x0F)
        ipos += 1

        # short RLE
        if cmd == 0:
            cnt += 3
            for k in range(cnt):
                outbuff[rpos + k] = inbuff[ipos]
            rpos += cnt
            ipos += 1

        # long RLE
        elif cmd == 1:
            cnt += <uint16_t>inbuff[ipos] << 4
            cnt += 19
            ipos += 1
            for k in range(cnt):
                outbuff[rpos + k] = inbuff[ipos]
            rpos += cnt
            ipos += 1

        # long pattern
        elif cmd == 2:
            ofs = cnt + 3
            ofs += <uint16_t>inbuff[ipos] << 4
            ipos += 1
            cnt = <uint16_t>inbuff[ipos]
            ipos += 1
            cnt += 16
            for k in range(cnt):
                outbuff[rpos + k] = outbuff[rpos - <Py_ssize_t>ofs + k]
            rpos += cnt

        # short pattern
        elif (cmd >= 3) & (cmd <= 15):
            ofs = cnt + 3
            ofs += <uint16_t>inbuff[ipos] << 4
            ipos += 1
            for k in range(cmd):
                outbuff[rpos + k] = outbuff[rpos - <Py_ssize_t>ofs + k]
            rpos += cmd

        else:
            raise ValueError("unknown RDC command")

    return rpos


# Float unpacking

# cdef extern from "Python.h":
#     double PyFloat_Unpack4(const char *data, bint le)
#     double PyFloat_Unpack8(const char *data, bint le)

cdef _float_unpack4_le = struct.Struct("<f").unpack
cdef _float_unpack4_be = struct.Struct(">f").unpack
cdef _float_unpack8_le = struct.Struct("<d").unpack
cdef _float_unpack8_be = struct.Struct(">d").unpack


cdef double float_unpack4(const uint8_t *data, bint le):
    # if PY_MAJOR_VERSION >= 3 and PY_MINOR_VERSION >= 11:
    #     return PyFloat_Unpack4(<const char *>data, le)
    # else:
    return (_float_unpack4_le if le else _float_unpack4_be)(data[:4])[0]


cdef double float_unpack8(const uint8_t *data, bint le):
    # if PY_MAJOR_VERSION >= 3 and PY_MINOR_VERSION >= 11:
    #     return PyFloat_Unpack8(<const char *>data, le)
    # else:
    return (_float_unpack8_le if le else _float_unpack8_be)(data[:8])[0]


# Integer unpacking

cdef extern from "portable_endian.h":
    uint16_t le16toh(uint16_t)
    uint32_t le32toh(uint32_t)
    uint64_t le64toh(uint64_t)
    uint16_t be16toh(uint16_t)
    uint32_t be32toh(uint32_t)
    uint64_t be64toh(uint64_t)
# Aliases for the macros
cdef inline uint16_t le16toh_(uint16_t x): return le16toh(x)
cdef inline uint32_t le32toh_(uint32_t x): return le32toh(x)
cdef inline uint64_t le64toh_(uint64_t x): return le64toh(x)
cdef inline uint16_t be16toh_(uint16_t x): return be16toh(x)
cdef inline uint32_t be32toh_(uint32_t x): return be32toh(x)
cdef inline uint64_t be64toh_(uint64_t x): return be64toh(x)
