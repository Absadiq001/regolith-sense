"""Reader for Geosoft / Oasis montaj binary grids (.grd), plain and compressed."""
import struct, zlib
import numpy as np

DUMMIES = {("b", 1): -127, ("B", 1): 255, ("h", 2): -32767, ("H", 2): 65535,
           ("i", 4): -2147483647, ("I", 4): 4294967295, ("f", 4): -1e32, ("d", 8): -1e32}

def read_grd(path_or_bytes):
    raw = path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray)) else open(path_or_bytes, "rb").read()
    if len(raw) < 512:
        raise ValueError("File is too small to be a Geosoft grid.")
    es, sign, ne, nv, kx = struct.unpack("<5i", raw[0:20])
    de, dv, x0, y0, rot, base, mult = struct.unpack("<7d", raw[20:76])
    compressed = es > 1024
    if compressed:
        es -= 1024
    if es not in (1, 2, 4, 8) or sign not in (0, 1, 2) or ne <= 0 or nv <= 0 or abs(kx) != 1:
        raise ValueError("Not a recognised Geosoft grid header.")
    code = {(0, 1): "B", (1, 1): "b", (0, 2): "H", (1, 2): "h", (0, 4): "I", (1, 4): "i", (2, 4): "f", (2, 8): "d"}[(sign, es)]
    body = raw[512:]
    if compressed:
        nblocks, vecs = struct.unpack("<2i", body[8:16])
        offs = struct.unpack("<%dq" % nblocks, body[16:16 + 8 * nblocks])
        sizes = struct.unpack("<%di" % nblocks, body[16 + 8 * nblocks:16 + 12 * nblocks])
        parts = [zlib.decompress(body[o - 512 + 16:o - 512 + s]) for o, s in zip(offs, sizes)]
        body = b"".join(parts)
    arr = np.frombuffer(body, dtype="<" + code, count=ne * nv).astype(float)
    dummy = DUMMIES[(code, es)]
    arr[arr <= -1e31 if code in "fd" else arr == dummy] = np.nan
    if mult not in (0.0, 1.0):
        arr = arr / mult
    arr = arr + base
    grid = arr.reshape((nv, ne)) if kx == 1 else arr.reshape((ne, nv)).T
    return {"grid": grid, "nx": grid.shape[1], "ny": grid.shape[0], "dx": de, "dy": dv,
            "x0": x0, "y0": y0, "rotation": rot, "compressed": compressed}
