"""TIFF LZW decoding without imagecodecs.

Ported from ImageJ 1.54 ij/io/ImageReader.java ``lzwUncompress`` (public domain, NIH;
W. Rasband et al.). tifffile decodes LZW only through the optional imagecodecs package (13-20
MB), so :func:`install` registers :func:`lzw_decode` as tifffile's decoder for compression 5;
tifffile still applies the horizontal predictor itself.
"""
from __future__ import annotations

CLEAR_CODE, EOI_CODE = 256, 257


def lzw_decode(data: bytes, out=None) -> bytes:
    """Decode a TIFF LZW strip (MSB-first codes, 9-12 bits, early change)."""
    if not data:
        return b""
    expected = out if isinstance(out, int) and out > 0 else 1 << 62
    src = bytes(data)
    nbits_total = len(src) * 8
    table: list[bytes] = [bytes([i]) for i in range(256)] + [b"", b""]
    output = bytearray()
    bit_pos = 0
    bits = 9
    old: bytes | None = None
    n = len(src)
    while len(output) < expected:
        if bit_pos + bits > nbits_total:
            break
        # read `bits` bits MSB-first starting at bit_pos
        byte_i = bit_pos >> 3
        chunk = int.from_bytes(src[byte_i:byte_i + 3].ljust(3, b"\0"), "big") if byte_i < n \
            else 0
        code = (chunk >> (24 - (bit_pos & 7) - bits)) & ((1 << bits) - 1)
        bit_pos += bits
        if code == EOI_CODE:
            break
        if code == CLEAR_CODE:
            del table[258:]
            bits = 9
            if bit_pos + bits > nbits_total:
                break
            byte_i = bit_pos >> 3
            chunk = int.from_bytes(src[byte_i:byte_i + 3].ljust(3, b"\0"), "big")
            code = (chunk >> (24 - (bit_pos & 7) - bits)) & ((1 << bits) - 1)
            bit_pos += bits
            if code == EOI_CODE:
                break
            entry = table[code] if code < len(table) else b"\0"
            output += entry
            old = entry
            continue
        if old is None:
            old = table[0]
        if code < len(table):
            entry = table[code]
            output += entry
            table.append(old + entry[:1])
        else:
            entry = old + old[:1]
            output += entry
            table.append(entry)
        old = entry
        size = len(table)
        if size == 511:
            bits = 10
        elif size == 1023:
            bits = 11
        elif size == 2047:
            bits = 12
        if size >= 4096:      # corrupt stream without CLEAR: keep 12 bits, stop growing
            del table[4095:]
    if isinstance(out, int) and out > 0:
        return bytes(output[:out])
    return bytes(output)


def lzw_encode(data: bytes) -> bytes:
    """Encode bytes as TIFF LZW (used by tests and the optional LZW TIFF writer)."""
    out_bits = 0
    nbits = 0
    result = bytearray()

    def emit(code: int, width: int) -> None:
        nonlocal out_bits, nbits
        out_bits = (out_bits << width) | code
        nbits += width
        while nbits >= 8:
            nbits -= 8
            result.append((out_bits >> nbits) & 0xFF)
        out_bits &= (1 << nbits) - 1 if nbits else 0

    table = {bytes([i]): i for i in range(256)}
    nxt = 258
    width = 9
    emit(CLEAR_CODE, width)
    w = b""
    for b in data:
        wc = w + bytes([b])
        if wc in table:
            w = wc
            continue
        emit(table[w], width)
        table[wc] = nxt
        nxt += 1
        if nxt == 512:
            width = 10
        elif nxt == 1024:
            width = 11
        elif nxt == 2048:
            width = 12
        if nxt >= 4094:
            emit(CLEAR_CODE, width)
            table = {bytes([i]): i for i in range(256)}
            nxt = 258
            width = 9
        w = bytes([b])
    if w:
        emit(table[w], width)
        # the decoder grows its table after this code as well
        if nxt == 511:
            width = 10
        elif nxt == 1023:
            width = 11
        elif nxt == 2047:
            width = 12
    emit(EOI_CODE, width)
    if nbits:
        result.append((out_bits << (8 - nbits)) & 0xFF)
    return bytes(result)


_INSTALLED = [False]


def install() -> None:
    """Make tifffile decode LZW strips with :func:`lzw_decode` (no-op with imagecodecs)."""
    if _INSTALLED[0]:
        return
    import tifffile
    codecs = tifffile.TIFF.DECOMPRESSORS
    try:
        codecs[5]
    except (KeyError, ImportError, AttributeError, NotImplementedError):
        codecs._codecs[5] = lzw_decode  # noqa: SLF001 - documented fallback hook
    _INSTALLED[0] = True
