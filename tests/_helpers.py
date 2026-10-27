"""Shared test helpers: which raw data are present, data fingerprints, and the reader
for the Fisher et al. authors' own R workspaces (optional check against their R output)."""
import gzip
import hashlib
import zipfile

import numpy as np
import pytest

from emaforecast import config

# Optional: the authors' "R Data.zip" from https://osf.io/5ybxt/ (Fisher et al., 2017), placed
# in data/. Only the comparison with the authors' own R output needs it.
FISHER_RDATA_ZIP = config.DATA / "R Data.zip"


def requires_data(*datasets):
    """Skip a test unless the openESM files of these datasets are in data/."""
    missing = [d for d in datasets if not config.RAW[d].exists()]
    return pytest.mark.skipif(bool(missing),
                              reason=f"raw data not present: {', '.join(missing)} (see the README)")


requires_fisher_rdata = pytest.mark.skipif(
    not FISHER_RDATA_ZIP.exists(), reason="optional authors' file data/R Data.zip not present")


def fingerprint(a):
    """SHA-256 of an array's shape, missing-value pattern and values: identical
    fingerprints <=> identical arrays (bit for bit), without storing the data."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    nan = np.isnan(a)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(np.packbits(nan).tobytes())
    h.update(np.where(nan, 0.0, a).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- authors' R workspaces
def _tagname(node):
    """Variable name of a top-level R pairlist entry."""
    tag = node.tag
    if tag is None:
        return None
    t = tag.referenced_object if tag.info.type.name == 'REF' else tag
    try:
        return t.value.value.decode()
    except Exception:
        return None


def read_fisher_rdata(pid, names=('data',)):
    """Selected objects of P0xx_final.RData, read from R Data.zip in memory. The files
    are full workspace dumps; only the requested objects are converted, since some
    other objects cannot be converted."""
    import rdata
    with zipfile.ZipFile(FISHER_RDATA_ZIP) as z:
        member = [n for n in z.namelist() if n.split('/')[-1].lower() == f'{pid.lower()}_final.rdata']
        if not member:
            raise FileNotFoundError(f'{pid}_final.RData not found in R Data.zip')
        parsed = rdata.parser.parse_data(gzip.decompress(z.read(member[0])))
    found, node = {}, parsed.object
    while node is not None and node.info.type.name == 'LIST':
        name = _tagname(node)
        car, cdr = node.value
        if name in names and name not in found:
            found[name] = car
        node = cdr
    return {name: rdata.conversion.convert(obj) for name, obj in found.items()}
