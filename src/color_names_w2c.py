"""w2c color-naming — the van de Weijer "Color Names" model as a second opinion.

A 32768 x 11 lookup learned from real-world images that maps any sRGB pixel to a
probability over 11 basic color terms. Unlike our hue-anchor thresholds (which
mislabel washed olive/grey as green/purple — measured), this was trained on real
photos specifically so neutral/washed tones don't pick up false chromatic names.

We use it per hold: average the 11-way probability over the hold's selected
pixels (same `utils.select_region_pixels` the Lab path uses) and take the argmax.

Model: van de Weijer et al., "Learning Color Names for Real-World Applications"
(IEEE TIP 2009). w2c.mat order is fixed below. Index for an 8-bit sRGB pixel is
floor(R/8) + 32*floor(G/8) + 32*32*floor(B/8)  (0..32767).

NOT offline-pure: needs models/w2c.mat (scipy.io) on first call.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

# Make `import utils` work whether run as a module or via sys.path (same trick
# as the other src modules).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils  # noqa: E402

# Fixed column order of w2c.mat (the paper's basic color terms).
W2C_COLORS = [
    "black", "blue", "brown", "grey", "green",
    "orange", "pink", "purple", "red", "white", "yellow",
]

# Map w2c's 11 terms onto our palette. Our palette has no "brown" (this gym uses
# none) — brown holds read as tan/orange, so fold brown -> orange. w2c has no
# "cyan"; faded turquoise comes out blue/green here (see name_pixels_bgr caller,
# which can graft our hue-based cyan rescue back on top).
W2C_TO_PALETTE = {c: c for c in W2C_COLORS}
W2C_TO_PALETTE["brown"] = "orange"

_W2C: Optional[np.ndarray] = None


def load_w2c(path: str = "models/w2c.mat") -> np.ndarray:
    """Load (and cache) the 32768 x 11 w2c matrix as float32."""
    global _W2C
    if _W2C is None:
        import scipy.io  # lazy: keep module import offline-safe
        p = utils.resolve_path(path)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"w2c.mat not found at {p}. Download it, e.g.:\n"
                "  curl -sL -o models/w2c.mat https://raw.githubusercontent.com/"
                "tinghuiz/dataset_bias/master/feature_extract/color/ColorNaming/w2c.mat"
            )
        _W2C = scipy.io.loadmat(p)["w2c"].astype(np.float32)
    return _W2C


def _index_bgr(px_bgr: np.ndarray) -> np.ndarray:
    """w2c row index per pixel from an (N, 3) uint8 BGR array."""
    px = np.asarray(px_bgr).reshape(-1, 3).astype(np.int32)
    b, g, r = px[:, 0] // 8, px[:, 1] // 8, px[:, 2] // 8
    return r + 32 * g + 32 * 32 * b


def name_pixels_bgr(
    px_bgr: np.ndarray, w2c: Optional[np.ndarray] = None, map_to_palette: bool = True
) -> str:
    """Name a hold from its selected BGR pixels by mean w2c probability -> argmax.

    '' for no pixels. With `map_to_palette` the w2c term is folded onto our
    palette names (brown -> orange).
    """
    if px_bgr is None or len(px_bgr) == 0:
        return ""
    w2c = w2c if w2c is not None else load_w2c()
    probs = w2c[_index_bgr(px_bgr)].mean(axis=0)
    name = W2C_COLORS[int(probs.argmax())]
    return W2C_TO_PALETTE.get(name, name) if map_to_palette else name
