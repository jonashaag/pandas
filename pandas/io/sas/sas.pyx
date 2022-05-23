# cython: language_level=3
# cython: profile=False
# cython: boundscheck=False, initializedcheck=False
from libc.stdint cimport (
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)
from libc.stdlib cimport (
    calloc,
    free,
    malloc,
)
from libc.string cimport (
    memcmp,
    memcpy,
    memset,
)

import numpy as np

import pandas.io.sas.sas_constants as const


cdef object _np_nan = np.nan
# Buffer for decompressing short rows.
cdef uint8_t *_process_byte_array_with_data_buf = <uint8_t *>malloc(1024 * sizeof(uint8_t))

# Typed const aliases for quick access.
assert len(const.page_mix_types) == 2
cdef:
    int page_meta_type = const.page_meta_type
    int page_mix_types_0 = const.page_mix_types[0]
    int page_mix_types_1 = const.page_mix_types[1]

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

# Typed const aliases: subheader_signature_to_index.
# Flatten the const.subheader_signature_to_index dictionary to lists of raw keys and values.
# Since the dictionary is small it is much faster to have an O(n) loop through the raw keys
# rather than use a Python dictionary lookup.
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
    cdef Py_ssize_t offset, length

    def __init__(self, Py_ssize_t offset, Py_ssize_t length):
        self.offset = offset
        self.length = length


cdef class BasePage:
    """A page (= bunch of bytes) with with unknown endianness.

    Supports reading raw bytes."""
    cdef:
        object sas7bdatreader
        readonly bytes data
        const uint8_t *data_raw
        Py_ssize_t data_len

    def __init__(self, sas7bdatreader, data):
        self.sas7bdatreader = sas7bdatreader
        self.data = data
        self.data_raw = self.data
        self.data_len = len(data)

    def __len__(self):
        return self.data_len

    def read_bytes(self, Py_ssize_t offset, Py_ssize_t width):
        self.check_read(offset, width)
        return self.data_raw[offset:offset+width]

    cpdef bint check_read(self, Py_ssize_t offset, Py_ssize_t width) except -1:
        if offset + width > self.data_len:
            self.sas7bdatreader.close()
            raise ValueError("The cached page is too small.")
        return True


cdef class Page(BasePage):
    """A page with known endianness.

    Supports reading raw bytes, integers and floats."""
    cdef bint file_is_little_endian, need_byteswap

    def __init__(self, sas7bdatreader, data, file_is_little_endian):
        super().__init__(sas7bdatreader, data)
        self.file_is_little_endian = file_is_little_endian
        self.need_byteswap = file_is_little_endian != _machine_is_little_endian()

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

            subheader_index = self._get_subheader_index(
                subheader_offset,
                int_length,
                subheader_compression,
                subheader_type,
            )
            processor = self._get_subheader_processor(subheader_index)
            if processor is None:
                current_page_data_subheader_pointers.append(
                    _SubheaderPointer(subheader_offset, subheader_length)
                )
            else:
                processor(subheader_offset, subheader_length)

    cdef int _get_subheader_index(
        self,
        Py_ssize_t signature_offset,
        Py_ssize_t signature_length,
        Py_ssize_t compression,
        Py_ssize_t ptype,
    ) except -1:
        cdef Py_ssize_t i

        self.check_read(signature_offset, signature_length)

        if signature_length == 4:
            for i in range(len(subheader_signature_to_index_values32)):
                if not memcmp(&subheader_signature_to_index_keys32[i], &self.data_raw[signature_offset], 4):
                    return subheader_signature_to_index_values32[i]
        else:
            for i in range(len(subheader_signature_to_index_values64)):
                if not memcmp(&subheader_signature_to_index_keys64[i], &self.data_raw[signature_offset], 8):
                    return subheader_signature_to_index_values64[i]

        if self.sas7bdatreader.compression and (compression in (compressed_subheader_id, 0)) and ptype == compressed_subheader_type:
            return data_subheader_index
        else:
            self.sas7bdatreader.close()
            raise ValueError(f"Unknown subheader signature {self.data_raw[signature_offset:signature_offset+signature_length]}")

    cdef _get_subheader_processor(self, Py_ssize_t index):
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

    cpdef double read_float(self, Py_ssize_t offset, Py_ssize_t width) except? 1337:
        self.check_read(offset, width)
        cdef const uint8_t *d = &self.data_raw[offset]
        if width == 4:
            return _read_float_with_byteswap(d, self.need_byteswap)
        else:
            return _read_double_with_byteswap(d, self.need_byteswap)

    cpdef uint64_t read_int(self, Py_ssize_t offset, Py_ssize_t width) except? 1337:
        self.check_read(offset, width)
        cdef const uint8_t *d = &self.data_raw[offset]
        if width == 1:
            return d[0]
        elif width == 2:
            return _read_uint16_with_byteswap(d, self.need_byteswap)
        elif width == 4:
            return _read_uint32_with_byteswap(d, self.need_byteswap)
        else:
            return _read_uint64_with_byteswap(d, self.need_byteswap)


cdef class SAS7BDATCythonReader:
    """Fast extensions to SAS7BDATCythonReader."""
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
        bint blank_missing
        bytes encoding
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

        Py_ssize_t (*decompress)(const uint8_t *, Py_ssize_t, uint8_t *, Py_ssize_t) except -1

        Py_ssize_t[:] column_data_offsets, column_data_lengths
        char[:] column_types

    def __init__(
        self,
        sas7bdatreader,
        byte_chunk,
        string_chunk,
        row_length,
        page_bit_offset,
        subheader_pointer_length,
        row_count,
        mix_page_row_count,
        blank_missing,
        encoding,
        column_data_offsets,
        column_data_lengths,
        column_types,
        compression,
     ):
        self.sas7bdatreader = sas7bdatreader
        self.byte_chunk = byte_chunk
        self.string_chunk = string_chunk
        self.row_length = row_length
        self.page_bit_offset = page_bit_offset
        self.subheader_pointer_length = subheader_pointer_length
        self.row_count = row_count
        self.mix_page_row_count = mix_page_row_count
        self.blank_missing = blank_missing
        self.encoding = None if encoding is None else encoding.encode("ascii")
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

    def read(self, Py_ssize_t nrows):
        cdef bint done

        for _ in range(nrows):
            done = self._readline()
            if done:
                break

    cdef bint _readline(self) except -1:
        # Loop until a data row is read
        while self.current_page_type == page_meta_type and self.current_row_on_page_index >= len(self.current_page_data_subheader_pointers):
            if self.sas7bdatreader._read_next_page():
                return True

        if self.current_page_type == page_meta_type:
            return self._readline_meta_page()
        elif self.current_page_type in (page_mix_types_0, page_mix_types_1):
            return self._readline_mix_page()
        else:
            return self._readline_data_page()

    cdef bint _readline_meta_page(self) except -1:
        cdef _SubheaderPointer current_subheader_pointer = self.current_page_data_subheader_pointers[self.current_row_on_page_index]
        self.process_byte_array_with_data(current_subheader_pointer.offset, current_subheader_pointer.length)
        return False

    cdef bint _readline_mix_page(self) except -1:
        cdef Py_ssize_t align_correction, offset
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

    cdef bint _readline_data_page(self) except -1:
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

    cpdef bint process_byte_array_with_data(self, int offset, int length) except -1:
        cdef:
            char column_type
            Py_ssize_t data_length, data_offset
            const uint8_t *source
            Py_ssize_t j, rpos, m, jb = 0, js = 0
            uint8_t *decompress_dynamic_buf = NULL
            object tmp

        source = &self.cached_page.data_raw[offset]
        if self.decompress != NULL and length < self.row_length:
            if self.row_length <= 1024:
                memset(_process_byte_array_with_data_buf, 0, length)
                rpos = self.decompress(source, length, _process_byte_array_with_data_buf, self.row_length)
                source = _process_byte_array_with_data_buf
            else:
                decompress_dynamic_buf = <uint8_t *>calloc(self.row_length, sizeof(uint8_t))
                if decompress_dynamic_buf == NULL:
                    nbytes = self.row_length * sizeof(uint8_t)
                    raise MemoryError(f"Failed to allocate {nbytes} bytes")
                rpos = self.decompress(source, length, decompress_dynamic_buf, self.row_length)
                source = decompress_dynamic_buf
            if rpos != self.row_length:
                raise ValueError(f"Expected decompressed line of length {self.row_length} bytes but decompressed {rpos} bytes")

        for j in range(len(self.column_data_offsets)):
            column_type = self.column_types[j]
            data_length = self.column_data_lengths[j]
            data_offset = self.column_data_offsets[j]
            if data_length == 0:
                break
            if column_type == b"d":
                # decimal
                m = 8 * self.current_row_in_chunk_index
                if self.cached_page.file_is_little_endian:
                    m += 8 - data_length
                memcpy(&self.byte_chunk[jb, m], &source[data_offset], data_length)
                jb += 1
            elif column_type == b"s":
                # string
                # Quickly skip 8-byte blocks of trailing whitespace.
                while (
                    data_length > 8
                    and source[data_offset+data_length-8] in b"\x00 "
                    and source[data_offset+data_length-7] in b"\x00 "
                    and source[data_offset+data_length-6] in b"\x00 "
                    and source[data_offset+data_length-5] in b"\x00 "
                    and source[data_offset+data_length-4] in b"\x00 "
                    and source[data_offset+data_length-3] in b"\x00 "
                    and source[data_offset+data_length-2] in b"\x00 "
                    and source[data_offset+data_length-1] in b"\x00 "
                ):
                    data_length -= 8
                # Skip the rest of the trailing whitespace.
                while data_length > 0 and source[data_offset+data_length-1] in b"\x00 ":
                    data_length -= 1
                if self.blank_missing and not data_length:
                    self.string_chunk[js, self.current_row_in_chunk_index] = _np_nan
                else:
                    self.string_chunk[js, self.current_row_in_chunk_index] = (
                        source[data_offset:data_offset+data_length]
                        if self.encoding is None else
                        source[data_offset:data_offset+data_length].decode(self.encoding)
                    )
                js += 1
            else:
                raise ValueError(f"unknown column type {column_type!r}")

        self.current_row_in_chunk_index += 1
        self.current_row_in_file_index += 1
        self.current_row_on_page_index += 1

        if decompress_dynamic_buf != NULL:
            free(decompress_dynamic_buf)

        return True


cdef inline float _read_float_with_byteswap(const uint8_t *data, bint byteswap):
    cdef float res = (<float*>data)[0]
    if byteswap:
        res = _byteswap_float(res)
    return res


cdef inline double _read_double_with_byteswap(const uint8_t *data, bint byteswap):
    cdef double res = (<double*>data)[0]
    if byteswap:
        res = _byteswap_double(res)
    return res


cdef inline uint16_t _read_uint16_with_byteswap(const uint8_t *data, bint byteswap):
    cdef uint16_t res = (<uint16_t *>data)[0]
    if byteswap:
        res = _byteswap2(res)
    return res


cdef inline uint32_t _read_uint32_with_byteswap(const uint8_t *data, bint byteswap):
    cdef uint32_t res = (<uint32_t *>data)[0]
    if byteswap:
        res = _byteswap4(res)
    return res


cdef inline uint64_t _read_uint64_with_byteswap(const uint8_t *data, bint byteswap):
    cdef uint64_t res = (<uint64_t *>data)[0]
    if byteswap:
        res = _byteswap8(res)
    return res


# Byteswapping
# From https://github.com/WizardMac/ReadStat/blob/master/src/readstat_bits.
# Copyright (c) 2013-2016 Evan Miller, Apache 2 License

cdef inline bint _machine_is_little_endian():
    cdef int test_byte_order = 1;
    return (<char*>&test_byte_order)[0]


cdef inline uint16_t _byteswap2(uint16_t num):
    return ((num & 0xFF00) >> 8) | ((num & 0x00FF) << 8)


cdef inline uint32_t _byteswap4(uint32_t num):
    num = ((num & <uint32_t>0xFFFF0000) >> 16) | ((num & <uint32_t>0x0000FFFF) << 16)
    return ((num & <uint32_t>0xFF00FF00) >> 8) | ((num & <uint32_t>0x00FF00FF) << 8)


cdef inline uint64_t _byteswap8(uint64_t num):
    num = ((num & <uint64_t>0xFFFFFFFF00000000) >> 32) | ((num & <uint64_t>0x00000000FFFFFFFF) << 32)
    num = ((num & <uint64_t>0xFFFF0000FFFF0000) >> 16) | ((num & <uint64_t>0x0000FFFF0000FFFF) << 16)
    return ((num & <uint64_t>0xFF00FF00FF00FF00) >> 8) | ((num & <uint64_t>0x00FF00FF00FF00FF) << 8)


cdef inline float _byteswap_float(float num):
    cdef uint32_t answer = 0
    memcpy(&answer, &num, 4)
    answer = _byteswap4(answer)
    memcpy(&num, &answer, 4)
    return num


cdef inline double _byteswap_double(double num):
    cdef uint64_t answer = 0
    memcpy(&answer, &num, 8)
    answer = _byteswap8(answer)
    memcpy(&num, &answer, 8)
    return num


# Decompression

# _rle_decompress decompresses data using a Run Length Encoding
# algorithm.  It is partially documented here:
#
# https://cran.r-project.org/package=sas7bdat/vignettes/sas7bdat.pdf
cdef Py_ssize_t _rle_decompress(const uint8_t *inbuff, Py_ssize_t input_length, uint8_t *outbuff, Py_ssize_t row_length) except -1:

    cdef:
        Py_ssize_t rpos = 0, ipos = 0, nbytes, control_byte, end_of_first_byte

    while ipos < input_length:
        if rpos >= row_length:
            raise ValueError(f"Invalid RLE out of bounds write at position {rpos} of {row_length}")

        control_byte = inbuff[ipos] & 0xF0
        end_of_first_byte = inbuff[ipos] & 0x0F
        ipos += 1

        if control_byte == 0x00:
            if end_of_first_byte != 0:
                raise ValueError("Unexpected non-zero end_of_first_byte")
            nbytes = <Py_ssize_t>inbuff[ipos] + 64
            ipos += 1
            memcpy(&outbuff[rpos], &inbuff[ipos], nbytes)
            ipos += nbytes
            rpos += nbytes
        elif control_byte == 0x40:
            # not documented
            nbytes = <Py_ssize_t>inbuff[ipos] + 18 + end_of_first_byte * 256
            ipos += 1
            memset(&outbuff[rpos], inbuff[ipos], nbytes)
            rpos += nbytes
            ipos += 1
        elif control_byte == 0x60:
            nbytes = <Py_ssize_t>inbuff[ipos] + 17 + end_of_first_byte * 256
            ipos += 1
            memset(&outbuff[rpos], 0x20, nbytes)
            rpos += nbytes
        elif control_byte == 0x70:
            nbytes = <Py_ssize_t>inbuff[ipos] + 17 + end_of_first_byte * 256
            ipos += 1
            memset(&outbuff[rpos], 0x00, nbytes)
            rpos += nbytes
        elif control_byte == 0x80:
            nbytes = end_of_first_byte + 1
            memcpy(&outbuff[rpos], &inbuff[ipos], nbytes)
            rpos += nbytes
            ipos += nbytes
        elif control_byte == 0x90:
            nbytes = end_of_first_byte + 17
            memcpy(&outbuff[rpos], &inbuff[ipos], nbytes)
            rpos += nbytes
            ipos += nbytes
        elif control_byte == 0xA0:
            nbytes = end_of_first_byte + 33
            memcpy(&outbuff[rpos], &inbuff[ipos], nbytes)
            rpos += nbytes
            ipos += nbytes
        elif control_byte == 0xB0:
            nbytes = end_of_first_byte + 49
            memcpy(&outbuff[rpos], &inbuff[ipos], nbytes)
            rpos += nbytes
            ipos += nbytes
        elif control_byte == 0xC0:
            nbytes = end_of_first_byte + 3
            memset(&outbuff[rpos], inbuff[ipos], nbytes)
            ipos += 1
            rpos += nbytes
        elif control_byte == 0xD0:
            nbytes = end_of_first_byte + 2
            memset(&outbuff[rpos], 0x40, nbytes)
            rpos += nbytes
        elif control_byte == 0xE0:
            nbytes = end_of_first_byte + 2
            memset(&outbuff[rpos], 0x20, nbytes)
            rpos += nbytes
        elif control_byte == 0xF0:
            nbytes = end_of_first_byte + 2
            memset(&outbuff[rpos], 0x00, nbytes)
            rpos += nbytes
        else:
            raise ValueError(f"unknown control byte: {control_byte}")

    return rpos

# _rdc_decompress decompresses data using the Ross Data Compression algorithm:
#
# http://collaboration.cmc.ec.gc.ca/science/rpn/biblio/ddj/Website/articles/CUJ/1992/9210/ross/ross.htm
cdef Py_ssize_t _rdc_decompress(const uint8_t *inbuff, Py_ssize_t input_length, uint8_t *outbuff, Py_ssize_t row_length) except -1:

    cdef:
        uint8_t cmd
        uint16_t ctrl_bits = 0, ctrl_mask = 0, ofs, cnt
        Py_ssize_t rpos = 0, ipos = 0, ii = -1

    while ipos < input_length:
        if rpos >= row_length:
            raise ValueError(f"Invalid RDC out of bounds write at position {rpos} of {row_length}")
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
            memset(&outbuff[rpos], inbuff[ipos], cnt)
            rpos += cnt
            ipos += 1

        # long RLE
        elif cmd == 1:
            cnt += <uint16_t>inbuff[ipos] << 4
            cnt += 19
            ipos += 1
            memset(&outbuff[rpos], inbuff[ipos], cnt)
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
            memcpy(&outbuff[rpos], &outbuff[rpos - ofs], cnt)
            rpos += cnt

        # short pattern
        else:
            ofs = cnt + 3
            ofs += <uint16_t>inbuff[ipos] << 4
            ipos += 1
            memcpy(&outbuff[rpos], &outbuff[rpos - ofs], cmd)
            rpos += cmd

    return rpos
