import cv2
import numpy as np

from pipeline_sd_components.refinement import _build_no_bangs_hairline_halo_mask


def _synthetic_short_no_bangs_case(include_halo: bool = True):
    h, w = 160, 160
    face_bbox = (45, 40, 115, 125)
    composited = np.full((h, w, 3), (228, 206, 190), dtype=np.uint8)
    generated = composited.copy()

    composite_mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(composite_mask, (80, 45), (52, 24), 0, 0, 360, 1.0, -1)

    seed = np.zeros((h, w), dtype=np.float32)
    seed[58:76, 48:112] = 1.0
    release = cv2.GaussianBlur(seed, (0, 0), sigmaX=2.0, sigmaY=2.0)

    protect = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(protect, (80, 82), (38, 52), 0, 0, 360, 1.0, -1)

    if include_halo:
        composited[66:69, 52:108] = (248, 248, 244)
        generated[66:69, 52:108] = (92, 70, 58)

    return composited, generated, composite_mask, seed, release, protect, face_bbox


def test_no_bangs_hairline_halo_mask_detects_bright_boundary():
    args = _synthetic_short_no_bangs_case(include_halo=True)
    mask = _build_no_bangs_hairline_halo_mask(
        *args,
        hair_length="short",
        subject_gender="female",
    )

    assert mask.dtype == np.float32
    assert mask.shape == args[0].shape[:2]
    assert int((mask > 0.04).sum()) >= 40
    assert float(mask[:30].sum()) == 0.0
    assert float(mask[95:].sum()) == 0.0


def test_no_bangs_hairline_halo_mask_ignores_clean_boundary():
    args = _synthetic_short_no_bangs_case(include_halo=False)
    mask = _build_no_bangs_hairline_halo_mask(
        *args,
        hair_length="short",
        subject_gender="female",
    )

    assert int((mask > 0.04).sum()) == 0
