"""Doc anh hyperspectral cua HODC2026: PNG mosaic 16-bit -> cube (H, W, 16)."""
import numpy as np
from PIL import Image


def X2Cube(img, cellSize=4):
    """Demosaic spectral filter array 4x4. Ham chinh thuc cua ban to chuc, giu nguyen."""
    B = [cellSize, cellSize]
    skip = [cellSize, cellSize]
    M, N = img.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    out = np.transpose(out)
    return out.reshape(M // cellSize, N // cellSize, cellSize * cellSize)


def load_cube(png_path):
    """PNG mosaic -> cube uint16 (H, W, 16). H,W la kich thuoc SAU demosaic, khop voi bbox trong XML."""
    return X2Cube(np.array(Image.open(png_path)))
