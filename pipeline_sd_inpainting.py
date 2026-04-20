"""
MirrAI SD Inpainting Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

완전 생성형(Text→Hair) 파이프라인.

아키텍처:
  ┌─────────────────────────────────────────────────────────────┐
  │  입력: 사진 + hairstyle_text + color_text                    │
  ├─────────────────────────────────────────────────────────────┤
  │  [1] MediaPipe FaceDetection → 얼굴 bbox + landmarks        │
  │  [2] SegFace → base hair/face/cloth mask (원본 해상도)      │
  │  [3] SAM2   → 정밀 hair mask 보정 (point + text prompt)     │
  │  [4] Canny edge → ControlNet conditioning (얼굴 구조 보존)  │
  │  [5] face crop → IP-Adapter conditioning (얼굴 identity)    │
  │  [6] SD 1.5 Inpainting + ControlNet → hair 영역 생성        │
  │  [7] Composite → 원본 얼굴 유지 + 생성 헤어 합성             │
  └─────────────────────────────────────────────────────────────┘
  출력: top-k 결과 이미지 (각기 다른 seed)

모델:
  - SegFace(custom): siik/segface_hair_khairstyle
  - SegFace(base):   kartiknarayan/SegFace
  - SAM2:    pretrained_models/sam2.pt  (기존 모델 재사용)
  - SD Inpaint: runwayml/stable-diffusion-inpainting (HF Hub)
  - ControlNet: lllyasviel/control_v11p_sd15_canny   (HF Hub)
  - IP-Adapter: h94/IP-Adapter / ip-adapter-plus-face_sd15.bin (HF Hub)
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

from pipeline_sd_components.output import (
    build_length_aware_output_crop_box,
    crop_with_padding,
)
from pipeline_sd_components.config import (
    PROJECT_ROOT,
    SD_INPAINT_MODEL_ID,
    CONTROLNET_MODEL_ID,
    IP_ADAPTER_REPO_ID,
    IP_ADAPTER_WEIGHT,
    DEFAULT_RUNTIME_LORA_HF_REPO_ID,
    DEFAULT_RUNTIME_LORA_HF_FILENAME,
    DEFAULT_SEGFACE_HF_REPO_ID,
    DEFAULT_SEGFACE_HF_SUBFOLDER,
    DEFAULT_SEGFACE_HF_FILENAME,
    DEFAULT_SEGFACE_BASE_HF_REPO_ID,
    DEFAULT_SEGFACE_BASE_HF_SUBFOLDER,
    DEFAULT_SEGFACE_BASE_HF_FILENAME,
    DEFAULT_SEGFACE_MODEL_VARIANT,
    DEFAULT_SEGFACE_INPUT_RES,
    HAIR_CLASS_IDX,
    GLASS_CLASS_IDX,
    EARRING_CLASS_IDX,
    NECKLACE_CLASS_IDX,
    FACE_CLASS_IDXS,
    CLOTH_CLASS_IDX,
    SD_SIZE,
    _NEGATIVE_BASE,
    _COMMON_STYLE_BLOCK_NEGATIVE,
    _SHORT_HAIR_KEYWORDS,
    _MEDIUM_HAIR_KEYWORDS,
    _MALE_SUBJECT_HINTS,
    _FEMALE_SUBJECT_HINTS,
    _MALE_STYLE_HINTS,
    _FEMALE_STYLE_HINTS,
    _NO_COLOR_HINTS,
    _HAIR_COLOR_TARGET_RGB,
    SDInpaintConfig,
    SDInpaintResult,
)
from pipeline_sd_components.generation_backends import (
    get_generation_backend_spec,
    resolve_generation_canvas_size,
    resolve_generation_steps,
)

try:
    from utils.env_loader import load_project_dotenv
except Exception:
    load_project_dotenv = None

if load_project_dotenv is not None:
    load_project_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────


class MirrAISDPipeline:
    """
    SAM2 + SD Inpainting + ControlNet(Canny) + IP-Adapter 기반 헤어 변환 파이프라인.

    - hair segmentation: SegFace + SAM2 refinement
    - 생성:              SD 1.5 Inpainting + ControlNet canny + IP-Adapter face
    """

    def __init__(self, config: Optional[SDInpaintConfig] = None) -> None:
        self.config = config or SDInpaintConfig()
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        self.dtype = torch.float16 if self.config.dtype == "float16" else torch.bfloat16

        self._segface = None  # SegFace (Swin-B) custom hair parsing
        self._segface_base = None  # SegFace (Swin-B) base parsing for face/cloth
        self._sam2_factory = None  # SAM2 predictor factory (callable)
        self._sd_pipe = None  # StableDiffusionControlNetInpaintPipeline
        self._generation_backend_spec = None
        self._mp_face = None  # MediaPipe FaceDetection
        self._mp_face_mesh = None  # MediaPipe FaceMesh
        self._lama = None  # LaMa large mask inpainting
        self._lama_device: Optional[str] = None
        self._lama_error: Optional[str] = None
        self._segface_load_info: Dict[str, Any] = {}
        self._segface_base_load_info: Dict[str, Any] = {}
        self._segface_custom_binary_hair = False
        self._segface_custom_hair_threshold = 0.5
        self._last_segface_mask_debug: Optional[Dict[str, Any]] = None
        self._active_lora_path: Optional[str] = None
        self._active_lora_scale: float = 1.0
        self._active_lora_adapter_name: Optional[str] = None
        self._loaded = False

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """모델 로드 (cold start). 이미 로드된 경우 no-op."""
        if self._loaded:
            return
        logger.info("[SDPipeline] 모델 로딩 시작...")
        stage_started = time.time()
        self._load_segface()
        logger.info(
            "[SDPipeline] load stage complete: segface (%.2fs)",
            time.time() - stage_started,
        )
        stage_started = time.time()
        self._load_sam2()
        logger.info(
            "[SDPipeline] load stage complete: sam2 (%.2fs)",
            time.time() - stage_started,
        )
        stage_started = time.time()
        self._load_mediapipe()
        logger.info(
            "[SDPipeline] load stage complete: mediapipe (%.2fs)",
            time.time() - stage_started,
        )
        stage_started = time.time()
        self._load_sd_pipeline()
        logger.info(
            "[SDPipeline] load stage complete: sd_pipeline (%.2fs)",
            time.time() - stage_started,
        )
        stage_started = time.time()
        self._load_lama()
        logger.info(
            "[SDPipeline] load stage complete: lama (%.2fs)",
            time.time() - stage_started,
        )
        self._loaded = True
        logger.info("[SDPipeline] 모든 모델 로드 완료")

    def run(
        self,
        image: np.ndarray,  # BGR, any resolution
        hairstyle_text: str,
        color_text: str,
        top_k: int = 3,
        return_intermediates: bool = False,
        mask_refine_mode: Optional[str] = None,
        subject_gender: Optional[str] = None,
        lora_path: Optional[str] = None,
        lora_scale: Optional[float] = None,
        sd_prompt_data: Optional[Dict[str, Any]] = None,
        prompt_context: Optional[Dict[str, Any]] = None,
        white_tshirt_experiment: bool = False,
    ) -> List[SDInpaintResult]:
        """
        헤어 스타일 변환 실행.

        Args:
            image:          입력 이미지 (BGR numpy)
            hairstyle_text: 사용자/백엔드에서 전달한 헤어스타일 텍스트
            color_text:     헤어 컬러 텍스트
            top_k:          반환 결과 수 (기본 3)
            return_intermediates: 중간 산출물 디버그 이미지 포함 여부
            sd_prompt_data: 백엔드/상위 레이어에서 전달한 SD 프롬프트 데이터
                            {"sd_positive", "sd_negative", "sd_guidance"}

        Returns:
            SDInpaintResult 리스트 (rank 0이 first)
        """
        if not self._loaded:
            self.load()
        self._apply_runtime_lora(lora_path=lora_path, lora_scale=lora_scale)

        requested_top_k = top_k

        # 시드 결정: config에 고정값 있으면 사용, 없으면 매 요청마다 랜덤 생성
        if self.config.seeds:
            seeds = self.config.seeds[:top_k]
        else:
            seeds = [random.randint(0, 2**31 - 1) for _ in range(top_k)]
        logger.info(f"[SDPipeline] seeds={seeds}")

        image_bgr = image.copy()
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        requested_hairstyle_text = " ".join(str(hairstyle_text or "").strip().split())
        requested_color_text = self._normalize_color_text(color_text)
        effective_hairstyle_text = requested_hairstyle_text
        effective_color_text = requested_color_text
        normalized_prompt_context = self._normalize_prompt_context(prompt_context)
        target_hair_length = self._resolve_requested_hair_length(
            effective_hairstyle_text,
            normalized_prompt_context,
        )
        normalized_color_text = self._normalize_color_text(effective_color_text)
        explicit_branch = normalized_prompt_context.get("gender_branch")
        subject_gender_mode = explicit_branch or self._infer_subject_gender(
            effective_hairstyle_text,
            subject_gender=subject_gender,
        )
        subject_profile = self._resolve_subject_pipeline_profile(subject_gender_mode)
        has_color_request = bool(normalized_color_text)
        target_hair_lab = (
            self._resolve_target_hair_lab(normalized_color_text)
            if has_color_request
            else None
        )
        if not has_color_request:
            logger.info("[SDPipeline] color_text 미지정 → 원본 머리 톤 유지 모드")
        elif target_hair_lab is None:
            logger.info("[SDPipeline] color_text 파싱 실패 → 색상 재정렬은 스킵")

        generation_spec = get_generation_backend_spec(self.config)
        generation_canvas_size = resolve_generation_canvas_size(self.config)
        generation_steps = resolve_generation_steps(self.config)
        logger.info(
            "[SDPipeline] generation backend=%s model=%s canvas=%d steps=%d",
            generation_spec.key,
            generation_spec.model_id,
            generation_canvas_size,
            generation_steps,
        )

        H, W = image_bgr.shape[:2]
        debug_images_common: Optional[Dict[str, np.ndarray]] = (
            {} if return_intermediates else None
        )
        debug_data_common: Optional[Dict[str, Any]] = (
            {} if return_intermediates else None
        )
        if debug_data_common is not None:
            debug_data_common["generation_backend"] = {
                "key": generation_spec.key,
                "label": generation_spec.label,
                "model_id": generation_spec.model_id,
                "canvas_size": int(generation_canvas_size),
                "steps": int(generation_steps),
            }
            debug_data_common["subject_gender"] = subject_gender_mode
            debug_data_common["subject_pipeline_branch"] = subject_profile.key
            debug_data_common["prompt_input"] = {
                "hairstyle_text": requested_hairstyle_text,
                "color_text": requested_color_text,
                "sd_prompt_data_provided": bool(
                    sd_prompt_data and sd_prompt_data.get("sd_positive")
                ),
                "prompt_context_provided": bool(
                    normalized_prompt_context.get("structured_payload_present")
                ),
                "white_tshirt_experiment": bool(white_tshirt_experiment),
            }
            debug_data_common["request_resolution"] = {
                "resolved_gender_branch": subject_gender_mode,
                "resolved_canonical_preferences": normalized_prompt_context.get(
                    "canonical_preferences", {}
                ),
                "fallback_mode": bool(normalized_prompt_context.get("fallback_mode")),
                "structured_payload_used": bool(
                    normalized_prompt_context.get("structured_payload_present")
                ),
            }
        logger.info(
            "[SDPipeline] subject pipeline branch=%s (subject_gender=%s, structured=%s, canonical=%s, fallback=%s)",
            subject_profile.key,
            subject_gender_mode,
            bool(normalized_prompt_context.get("structured_payload_present")),
            normalized_prompt_context.get("canonical_preferences", {}),
            bool(normalized_prompt_context.get("fallback_mode")),
        )

        def _store_mask(name: str, mask: Optional[np.ndarray]) -> None:
            if debug_images_common is None or mask is None:
                return
            m = np.clip(mask, 0.0, 1.0)
            m_u8 = (m * 255).astype(np.uint8)
            bgr = cv2.cvtColor(m_u8, cv2.COLOR_GRAY2BGR)

            # Very aggressive resize for payload reliability
            max_dim = 320
            h, w = bgr.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                bgr = cv2.resize(
                    bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
                )
            debug_images_common[name] = bgr

        def _store_rgb(name: str, rgb_img: np.ndarray) -> None:
            if debug_images_common is None:
                return
            bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

            # Very aggressive resize for payload reliability
            max_dim = 320
            h, w = bgr.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                bgr = cv2.resize(
                    bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
                )
            debug_images_common[name] = bgr

        def _make_rect(
            x1: float,
            y1: float,
            x2: float,
            y2: float,
            *,
            image_shape: Optional[Tuple[int, int]] = None,
        ) -> Optional[Tuple[int, int, int, int]]:
            target_h, target_w = image_shape if image_shape is not None else (H, W)
            rx1 = max(0, min(target_w, int(round(x1))))
            ry1 = max(0, min(target_h, int(round(y1))))
            rx2 = max(0, min(target_w, int(round(x2))))
            ry2 = max(0, min(target_h, int(round(y2))))
            if rx2 <= rx1 or ry2 <= ry1:
                return None
            return (rx1, ry1, rx2, ry2)

        def _rect_area(rect: Optional[Tuple[int, int, int, int]]) -> int:
            if rect is None:
                return 0
            x1, y1, x2, y2 = rect
            return max(0, x2 - x1) * max(0, y2 - y1)

        def _build_diagnostic_rois(
            bbox: Tuple[int, int, int, int],
            *,
            image_shape: Optional[Tuple[int, int]] = None,
        ) -> Dict[str, Optional[Tuple[int, int, int, int]]]:
            target_h, target_w = image_shape if image_shape is not None else (H, W)
            x1, y1, x2, y2 = [int(v) for v in bbox]
            face_w = max(x2 - x1, 1)
            face_h = max(y2 - y1, 1)
            center_x = 0.5 * (x1 + x2)
            rois: Dict[str, Optional[Tuple[int, int, int, int]]] = {
                "torso_front": _make_rect(
                    x1 - face_w * 0.98,
                    y2 - face_h * 0.04,
                    x2 + face_w * 0.98,
                    y2 + face_h * 1.92,
                    image_shape=(target_h, target_w),
                ),
                "chest_center": _make_rect(
                    center_x - face_w * 0.34,
                    y2 + face_h * 0.18,
                    center_x + face_w * 0.34,
                    y2 + face_h * 1.06,
                    image_shape=(target_h, target_w),
                ),
                "left_side": _make_rect(
                    x1 - face_w * 0.92,
                    y2 + face_h * 0.08,
                    x1 + face_w * 0.20,
                    y2 + face_h * 1.24,
                    image_shape=(target_h, target_w),
                ),
                "right_side": _make_rect(
                    x2 - face_w * 0.20,
                    y2 + face_h * 0.08,
                    x2 + face_w * 0.92,
                    y2 + face_h * 1.24,
                    image_shape=(target_h, target_w),
                ),
                "neckline": _make_rect(
                    center_x - face_w * 0.46,
                    y2 - face_h * 0.10,
                    center_x + face_w * 0.46,
                    y2 + face_h * 0.44,
                    image_shape=(target_h, target_w),
                ),
            }
            rois["full_image"] = _make_rect(
                0, 0, target_w, target_h, image_shape=(target_h, target_w)
            )
            return rois

        def _extract_bbox_from_mask(
            mask: Optional[np.ndarray], threshold: float = 0.08
        ) -> Optional[List[int]]:
            if mask is None:
                return None
            mask_rs = self._resize_mask_to_shape(mask, (H, W))
            if mask_rs is None:
                return None
            mask_bin = np.clip(mask_rs.astype(np.float32), 0.0, 1.0) > threshold
            if not bool(mask_bin.any()):
                return None
            ys, xs = np.where(mask_bin)
            return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

        def _mask_stats(
            name: str,
            mask: Optional[np.ndarray],
            torso_rect: Optional[Tuple[int, int, int, int]],
            *,
            bucket: str = "mask_stats",
        ) -> None:
            if debug_data_common is None or mask is None:
                return
            mask_rs = self._resize_mask_to_shape(mask, (H, W))
            if mask_rs is None:
                return
            mask_f = np.clip(mask_rs.astype(np.float32), 0.0, 1.0)
            mask_bin = mask_f > 0.08
            nonzero_count = int(mask_bin.sum())
            image_area = max(float(H * W), 1.0)
            torso_nonzero = 0
            torso_ratio = 0.0
            torso_area = _rect_area(torso_rect)
            if torso_rect is not None and torso_area > 0:
                tx1, ty1, tx2, ty2 = torso_rect
                torso_region = mask_bin[ty1:ty2, tx1:tx2]
                torso_nonzero = int(torso_region.sum())
                torso_ratio = float(torso_nonzero) / max(float(torso_area), 1.0)
            roi_occupancy: Dict[str, Dict[str, float]] = {}
            for roi_name, rect in diagnostic_rois.items():
                if roi_name == "full_image" or rect is None:
                    continue
                rx1, ry1, rx2, ry2 = rect
                roi_area = _rect_area(rect)
                roi_nonzero = 0
                roi_ratio = 0.0
                if roi_area > 0:
                    roi_nonzero = int(mask_bin[ry1:ry2, rx1:rx2].sum())
                    roi_ratio = float(roi_nonzero) / max(float(roi_area), 1.0)
                roi_occupancy[roi_name] = {
                    "nonzero_pixel_count": roi_nonzero,
                    "occupancy": roi_ratio,
                    "area": roi_area,
                }
            stats = {
                "sum": float(mask_f.sum()),
                "nonzero_pixel_count": nonzero_count,
                "bbox": _extract_bbox_from_mask(mask_f),
                "torso_roi_nonzero_pixel_count": torso_nonzero,
                "torso_roi_occupancy": torso_ratio,
                "torso_roi_area": torso_area,
                "image_area_ratio": float(nonzero_count) / image_area,
                "roi_occupancy": roi_occupancy,
            }
            diag = debug_data_common.setdefault("diagnostics", {})
            mask_stats = diag.setdefault(bucket, {})
            mask_stats[name] = stats
            logger.info(
                "[SDPipeline][diag][mask][%s] %s sum=%.2f nonzero=%d bbox=%s torso_occ=%.4f chest_occ=%.4f left_occ=%.4f right_occ=%.4f area_ratio=%.4f",
                bucket,
                name,
                stats["sum"],
                stats["nonzero_pixel_count"],
                stats["bbox"],
                stats["torso_roi_occupancy"],
                stats["roi_occupancy"].get("chest_center", {}).get("occupancy", 0.0),
                stats["roi_occupancy"].get("left_side", {}).get("occupancy", 0.0),
                stats["roi_occupancy"].get("right_side", {}).get("occupancy", 0.0),
                stats["image_area_ratio"],
            )

        def _overlay_mask_rgb(
            base_rgb: np.ndarray,
            mask: Optional[np.ndarray],
            color: Tuple[int, int, int],
            *,
            alpha: float = 0.46,
        ) -> np.ndarray:
            out = base_rgb.astype(np.float32).copy()
            if mask is None:
                return base_rgb.copy()
            mask_rs = self._resize_mask_to_shape(mask, base_rgb.shape[:2])
            if mask_rs is None:
                return base_rgb.copy()
            mask_f = np.clip(mask_rs.astype(np.float32), 0.0, 1.0)[..., np.newaxis]
            color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
            out = out * (1.0 - mask_f * alpha) + color_arr * (mask_f * alpha)
            return np.clip(out, 0, 255).astype(np.uint8)

        def _overlay_edges_rgb(
            base_rgb: np.ndarray,
            edge_img: Optional[np.ndarray],
            color: Tuple[int, int, int],
            *,
            alpha: float = 0.74,
        ) -> np.ndarray:
            if edge_img is None:
                return base_rgb.copy()
            if edge_img.ndim == 3:
                gray = cv2.cvtColor(edge_img, cv2.COLOR_RGB2GRAY)
            else:
                gray = edge_img
            edge_f = (gray.astype(np.float32) / 255.0)[..., np.newaxis]
            out = base_rgb.astype(np.float32).copy()
            color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
            out = out * (1.0 - edge_f * alpha) + color_arr * (edge_f * alpha)
            return np.clip(out, 0, 255).astype(np.uint8)

        def _store_roi_crops(stage_name: str, rgb_img: Optional[np.ndarray]) -> None:
            if debug_images_common is None or rgb_img is None:
                return
            image_h, image_w = rgb_img.shape[:2]
            roi_rects = _build_diagnostic_rois(
                face_bbox, image_shape=(image_h, image_w)
            )
            for roi_name, rect in roi_rects.items():
                if roi_name == "full_image" or rect is None:
                    continue
                x1, y1, x2, y2 = rect
                crop = rgb_img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                debug_images_common[f"diagnostic_{stage_name}_{roi_name}"] = (
                    cv2.cvtColor(
                        crop,
                        cv2.COLOR_RGB2BGR,
                    )
                )

        def _project_generated_to_original(
            gen_pil: Image.Image,
            current_scale: float,
            current_pad: Tuple[int, int],
            original_size: Tuple[int, int],
        ) -> np.ndarray:
            out_w, out_h = original_size
            pad_l, pad_t = current_pad
            new_w = int(out_w * current_scale)
            new_h = int(out_h * current_scale)
            gen_np = np.array(gen_pil)
            gen_cropped = gen_np[pad_t : pad_t + new_h, pad_l : pad_l + new_w]
            return cv2.resize(
                gen_cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4
            )

        def _mean_abs_diff(
            rgb_a: Optional[np.ndarray],
            rgb_b: Optional[np.ndarray],
            rect: Optional[Tuple[int, int, int, int]] = None,
        ) -> float:
            if rgb_a is None or rgb_b is None:
                return 0.0
            if rgb_a.shape != rgb_b.shape:
                return 0.0
            arr_a = rgb_a.astype(np.float32)
            arr_b = rgb_b.astype(np.float32)
            if rect is not None:
                x1, y1, x2, y2 = rect
                arr_a = arr_a[y1:y2, x1:x2]
                arr_b = arr_b[y1:y2, x1:x2]
            if arr_a.size == 0 or arr_b.size == 0:
                return 0.0
            return float(np.abs(arr_a - arr_b).mean())

        def _build_abs_diff_heatmap_rgb(
            rgb_a: Optional[np.ndarray],
            rgb_b: Optional[np.ndarray],
        ) -> Optional[np.ndarray]:
            if rgb_a is None or rgb_b is None:
                return None
            if rgb_a.shape != rgb_b.shape:
                return None
            diff = np.abs(rgb_a.astype(np.float32) - rgb_b.astype(np.float32)).mean(
                axis=2
            )
            diff_u8 = np.clip(diff * 3.0, 0.0, 255.0).astype(np.uint8)
            return cv2.cvtColor(
                cv2.applyColorMap(diff_u8, cv2.COLORMAP_TURBO),
                cv2.COLOR_BGR2RGB,
            )

        def _source_similarity_ratio(
            source_rgb: Optional[np.ndarray],
            target_rgb: Optional[np.ndarray],
            rect: Optional[Tuple[int, int, int, int]] = None,
            *,
            threshold: float = 12.0,
        ) -> float:
            if source_rgb is None or target_rgb is None:
                return 0.0
            if source_rgb.shape != target_rgb.shape:
                return 0.0
            src = source_rgb.astype(np.float32)
            tgt = target_rgb.astype(np.float32)
            if rect is not None:
                x1, y1, x2, y2 = rect
                src = src[y1:y2, x1:x2]
                tgt = tgt[y1:y2, x1:x2]
            if src.size == 0 or tgt.size == 0:
                return 0.0
            per_pixel_diff = np.abs(src - tgt).mean(axis=2)
            return float((per_pixel_diff <= threshold).mean())

        def _build_boundary_band(
            mask: Optional[np.ndarray], shape: Tuple[int, int]
        ) -> np.ndarray:
            mask_rs = self._resize_mask_to_shape(mask, shape)
            if mask_rs is None:
                return np.zeros(shape, dtype=np.uint8)
            mask_bin = (np.clip(mask_rs.astype(np.float32), 0.0, 1.0) > 0.08).astype(
                np.uint8
            )
            if int(mask_bin.sum()) == 0:
                return np.zeros(shape, dtype=np.uint8)
            band_outer = cv2.dilate(
                mask_bin,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
                iterations=1,
            )
            band_inner = cv2.erode(
                mask_bin,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                iterations=1,
            )
            return cv2.subtract(band_outer, band_inner).astype(np.uint8) * 255

        def _edge_stats(
            name: str,
            edge_img: Optional[np.ndarray],
            hair_roi_mask: Optional[np.ndarray],
            roi_rects: Dict[str, Optional[Tuple[int, int, int, int]]],
        ) -> None:
            if debug_data_common is None or edge_img is None:
                return
            if edge_img.ndim == 3:
                gray = cv2.cvtColor(edge_img, cv2.COLOR_RGB2GRAY)
            else:
                gray = edge_img
            edge_f = gray.astype(np.float32)
            edge_nonzero = gray > 0
            hair_mask_rs = self._resize_mask_to_shape(hair_roi_mask, gray.shape[:2])
            hair_bin = (
                np.clip(hair_mask_rs.astype(np.float32), 0.0, 1.0) > 0.08
                if hair_mask_rs is not None
                else np.zeros(gray.shape[:2], dtype=bool)
            )
            stats: Dict[str, Any] = {
                "total_edge_sum": float(edge_f.sum()),
                "total_edge_nonzero_count": int(edge_nonzero.sum()),
                "hair_roi_edge_sum": (
                    float(edge_f[hair_bin].sum()) if bool(hair_bin.any()) else 0.0
                ),
                "hair_roi_edge_nonzero_count": (
                    int(edge_nonzero[hair_bin].sum()) if bool(hair_bin.any()) else 0
                ),
            }
            for roi_name, rect in roi_rects.items():
                if roi_name == "full_image" or rect is None:
                    continue
                x1, y1, x2, y2 = rect
                roi = edge_f[y1:y2, x1:x2]
                roi_nz = edge_nonzero[y1:y2, x1:x2]
                stats[f"{roi_name}_edge_sum"] = float(roi.sum()) if roi.size else 0.0
                stats[f"{roi_name}_edge_nonzero_count"] = (
                    int(roi_nz.sum()) if roi_nz.size else 0
                )
            diag = debug_data_common.setdefault("diagnostics", {})
            conditioning_stats = diag.setdefault("conditioning_stats", {})
            conditioning_stats[name] = stats
            logger.info(
                "[SDPipeline][diag][edge] %s hair=%.2f chest=%.2f left=%.2f right=%.2f",
                name,
                stats.get("hair_roi_edge_sum", 0.0),
                stats.get("chest_center_edge_sum", 0.0),
                stats.get("left_side_edge_sum", 0.0),
                stats.get("right_side_edge_sum", 0.0),
            )

        if debug_images_common is not None:
            debug_images_common["pipeline_input_image"] = image_bgr.copy()

        # ── Step 1: 얼굴 검출 ────────────────────────────────────────────────
        face_obs = self._detect_face(img_rgb)
        if face_obs is None:
            raise ValueError("얼굴을 검출할 수 없습니다.")
        face_bbox = face_obs  # (x1, y1, x2, y2)
        standardized_meta = self._maybe_standardize_input_portrait(
            img_rgb,
            face_bbox,
            target_hair_length=target_hair_length,
        )
        if standardized_meta.get("applied"):
            std_rgb = standardized_meta.get("image_rgb")
            if isinstance(std_rgb, np.ndarray):
                img_rgb = std_rgb
                image_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                H, W = image_bgr.shape[:2]
                face_obs = self._detect_face(img_rgb)
                if face_obs is None:
                    raise ValueError("표준화 후 얼굴을 다시 검출할 수 없습니다.")
                face_bbox = face_obs
                if debug_images_common is not None:
                    debug_images_common["pipeline_standardized_input_image"] = (
                        image_bgr.copy()
                    )
                if debug_data_common is not None:
                    debug_data_common["input_standardization"] = {
                        k: v for k, v in standardized_meta.items() if k != "image_rgb"
                    }
        logger.info(f"[SDPipeline] 얼굴 검출: {face_bbox}")
        diagnostic_rois = _build_diagnostic_rois(face_bbox)
        if debug_data_common is not None:
            debug_data_common.setdefault("diagnostics", {})
            debug_data_common["diagnostics"]["rois"] = {
                key: list(value) if value is not None else None
                for key, value in diagnostic_rois.items()
                if key != "full_image"
            }
        landmark_obs = self._detect_landmark_data(img_rgb, face_bbox)
        landmark_face_mask = landmark_obs.get("face_mask")

        if landmark_obs.get("detected"):
            logger.info(
                "[SDPipeline] mediapipe landmarks, points=%s",
                landmark_obs.get("landmarks_count"),
            )
            landmark_debug_images = landmark_obs.get("debug_images")
            if debug_images_common is not None and isinstance(
                landmark_debug_images, dict
            ):
                debug_images_common.update(landmark_debug_images)

            landmark_debug_data = landmark_obs.get("debug_data")
            if debug_data_common is not None and isinstance(landmark_debug_data, dict):
                debug_data_common["mediapipe_face_mesh"] = landmark_debug_data

            if debug_images_common is not None and isinstance(
                landmark_face_mask, np.ndarray
            ):
                _store_mask("mediapipe_face_protect_mask", landmark_face_mask)
        elif debug_data_common is not None:
            debug_data_common["mediapipe_face_mesh"] = {"detected": False}

        # ── Step 2: SegFace base hair mask + 얼굴 픽셀 마스크 + 옷 마스크 ──────
        hair_mask_base, face_region_mask, cloth_mask = self._segface_hair_mask(
            img_rgb, face_bbox
        )
        face_region_mask = self._sanitize_face_region_mask(
            face_region_mask,
            face_bbox,
            landmark_face_mask=landmark_face_mask,
        )
        cloth_mask = self._sanitize_cloth_mask(
            img_rgb, cloth_mask, hair_mask_base, face_bbox
        )
        _store_mask("segface_hair_mask", hair_mask_base)
        _store_mask("segface_face_region_mask", face_region_mask)
        _store_mask("segface_cloth_mask", cloth_mask)
        segface_debug = self._last_segface_mask_debug or {}
        raw_cloth_mask = segface_debug.get("raw_cloth_mask")
        custom_hair_mask = segface_debug.get("custom_hair_mask")
        base_hair_mask = segface_debug.get("base_hair_mask")
        base_hair_support_mask = segface_debug.get("base_hair_support_mask")
        glasses_mask = segface_debug.get("glasses_mask")
        earring_mask = segface_debug.get("earring_mask")
        necklace_mask = segface_debug.get("necklace_mask")
        sparse_dark_cloth_support_mask = segface_debug.get(
            "sparse_dark_cloth_support_mask"
        )
        subject_cloth_anchor_mask = segface_debug.get("subject_cloth_anchor_mask")
        subject_cloth_filtered_mask = segface_debug.get("subject_cloth_filtered_mask")
        subject_shoulder_bridge_mask = segface_debug.get("subject_shoulder_bridge_mask")
        if isinstance(custom_hair_mask, np.ndarray):
            _store_mask("segface_custom_hair_mask", custom_hair_mask)
        if isinstance(base_hair_mask, np.ndarray):
            _store_mask("segface_base_hair_mask", base_hair_mask)
        if isinstance(base_hair_support_mask, np.ndarray):
            _store_mask("segface_base_hair_support_mask", base_hair_support_mask)
        if isinstance(raw_cloth_mask, np.ndarray):
            _store_mask("segface_raw_cloth_mask", raw_cloth_mask)
        if isinstance(glasses_mask, np.ndarray):
            _store_mask("segface_glasses_mask", glasses_mask)
        if isinstance(earring_mask, np.ndarray):
            _store_mask("segface_earring_mask", earring_mask)
        if isinstance(necklace_mask, np.ndarray):
            _store_mask("segface_necklace_mask", necklace_mask)
        if isinstance(sparse_dark_cloth_support_mask, np.ndarray):
            _store_mask(
                "segface_sparse_dark_cloth_support_mask", sparse_dark_cloth_support_mask
            )
        if isinstance(subject_cloth_anchor_mask, np.ndarray):
            _store_mask("segface_subject_cloth_anchor_mask", subject_cloth_anchor_mask)
        if isinstance(subject_cloth_filtered_mask, np.ndarray):
            _store_mask(
                "segface_subject_cloth_filtered_mask", subject_cloth_filtered_mask
            )
        if isinstance(subject_shoulder_bridge_mask, np.ndarray):
            _store_mask(
                "segface_subject_shoulder_bridge_mask", subject_shoulder_bridge_mask
            )
        subject_torso_anchor_mask = segface_debug.get("subject_torso_anchor_mask")
        subject_torso_candidate_mask = segface_debug.get("subject_torso_candidate_mask")
        subject_torso_filtered_mask = segface_debug.get("subject_torso_filtered_mask")
        if isinstance(subject_torso_anchor_mask, np.ndarray):
            _store_mask("segface_subject_torso_anchor_mask", subject_torso_anchor_mask)
        if isinstance(subject_torso_candidate_mask, np.ndarray):
            _store_mask(
                "segface_subject_torso_candidate_mask", subject_torso_candidate_mask
            )
        if isinstance(subject_torso_filtered_mask, np.ndarray):
            _store_mask(
                "segface_subject_torso_filtered_mask", subject_torso_filtered_mask
            )
        if debug_data_common is not None and segface_debug.get("meta"):
            debug_data_common["segface_mask_debug"] = segface_debug["meta"]

        source_accessory_profile = self._build_accessory_profile(
            img_rgb,
            face_bbox,
            hair_mask=hair_mask_base,
            face_mask=face_region_mask,
            glasses_mask=glasses_mask,
            earring_mask=earring_mask,
            necklace_mask=necklace_mask,
        )
        if debug_data_common is not None:
            debug_data_common.setdefault("diagnostics", {})
            debug_data_common["diagnostics"]["source_accessory_profile"] = {
                key: float(value) for key, value in source_accessory_profile.items()
            }

        # ── Step 3: SAM2 refinement ───────────────────────────────────────────
        hair_mask, mask_source, mask_refine_mode_used = self._refine_with_sam2(
            img_rgb,
            hair_mask_base,
            face_bbox,
            effective_hairstyle_text,
            mask_refine_mode=mask_refine_mode,
        )
        logger.info(
            f"[SDPipeline] hair mask source={mask_source}, refine_mode={mask_refine_mode_used}, "
            f"pixels={hair_mask.sum():.0f}"
        )
        _store_mask(f"{mask_source}_refined_hair_mask", hair_mask)
        if debug_data_common is not None:
            debug_data_common["mask_refine_mode"] = {
                "requested": self._resolve_mask_refine_mode(mask_refine_mode),
                "used": mask_refine_mode_used,
                "mask_used": mask_source,
            }

        if hair_mask.sum() < 300:
            raise ValueError("머리카락 영역이 너무 작습니다.")

        # ── Step 3-b: 헤어 길이 분류 ─────────────────────────────────────────
        hair_length = target_hair_length
        logger.info(
            f"[SDPipeline] 헤어 길이 분류: {hair_length}, subject_gender={subject_gender_mode}"
        )
        bangs_requested = self._resolve_requested_bangs_state(
            effective_hairstyle_text,
            normalized_prompt_context,
            subject_gender=subject_gender_mode,
        )
        explicit_no_bangs_requested = self._resolve_requested_no_bangs_state(
            effective_hairstyle_text,
            normalized_prompt_context,
            subject_gender=subject_gender_mode,
        )
        short_no_bangs_target = bool(hair_length == "short" and not bangs_requested)
        logger.info(
            "[SDPipeline] front coverage resolution: bangs_requested=%s explicit_no_bangs_requested=%s "
            "short_no_bangs_target=%s style_axes=%s",
            bangs_requested,
            explicit_no_bangs_requested,
            short_no_bangs_target,
            normalized_prompt_context.get("style_axes", {}),
        )
        source_cloth_preclean_analysis = self._analyze_source_cloth_preclean_need(
            source_hair_mask=hair_mask_base,
            cloth_mask=cloth_mask,
            face_bbox=face_bbox,
            subject_gender=subject_gender_mode,
        )
        skip_source_cloth_preclean = bool(
            source_cloth_preclean_analysis.get("skip_preclean")
        )
        source_hair_length_estimate = str(
            source_cloth_preclean_analysis.get("source_hair_length", "unknown")
        )
        logger.info(
            "[SDPipeline] source cloth preclean: source_length=%s skip=%s reason=%s "
            "bottom_ratio=%.4f torso_ratio=%.4f cloth_overlap_ratio=%.4f",
            source_hair_length_estimate,
            skip_source_cloth_preclean,
            source_cloth_preclean_analysis.get("skip_reason", ""),
            float(source_cloth_preclean_analysis.get("hair_bottom_ratio", 0.0)),
            float(source_cloth_preclean_analysis.get("torso_hair_ratio", 0.0)),
            float(source_cloth_preclean_analysis.get("cloth_overlap_ratio", 0.0)),
        )
        source_torso_hair_mask = source_cloth_preclean_analysis.get("torso_hair_mask")
        if isinstance(source_torso_hair_mask, np.ndarray):
            _store_mask("pipeline_source_torso_hair_mask", source_torso_hair_mask)
        source_cloth_overlap_mask = source_cloth_preclean_analysis.get(
            "cloth_overlap_mask"
        )
        if isinstance(source_cloth_overlap_mask, np.ndarray):
            _store_mask("pipeline_source_cloth_overlap_mask", source_cloth_overlap_mask)
        if debug_data_common is not None:
            debug_data_common["source_cloth_preclean"] = {
                "source_hair_length": source_hair_length_estimate,
                "needs_preclean": bool(
                    source_cloth_preclean_analysis.get("needs_preclean")
                ),
                "skip_preclean": skip_source_cloth_preclean,
                "skip_reason": str(
                    source_cloth_preclean_analysis.get("skip_reason", "")
                ),
                "hair_bottom_ratio": float(
                    source_cloth_preclean_analysis.get("hair_bottom_ratio", 0.0)
                ),
                "torso_hair_ratio": float(
                    source_cloth_preclean_analysis.get("torso_hair_ratio", 0.0)
                ),
                "cloth_overlap_ratio": float(
                    source_cloth_preclean_analysis.get("cloth_overlap_ratio", 0.0)
                ),
                "bangs_requested": bool(bangs_requested),
                "explicit_no_bangs_requested": bool(explicit_no_bangs_requested),
                "short_no_bangs_target": bool(short_no_bangs_target),
            }
        source_garment_prepass_mask: Optional[np.ndarray] = None
        source_garment_prepass_enabled = False
        source_garment_prepass_applied = False
        source_garment_prepass_apply_mode = ""
        source_garment_prepass_error = ""
        source_garment_prepass_stabilized = False
        disable_short_postprocess_experiment = False
        if disable_short_postprocess_experiment:
            logger.info("[SDPipeline] short postprocess disabled for experiment")
        if debug_data_common is not None:
            debug_data_common.setdefault("short_postprocess", {})
            debug_data_common["short_postprocess"]["disabled_for_experiment"] = bool(
                disable_short_postprocess_experiment
            )
        if hair_length == "short":
            short_internal_target = max(
                int(requested_top_k),
                int(self.config.short_internal_candidate_count),
                int(subject_profile.short_internal_candidate_min),
            )
            if len(seeds) < short_internal_target:
                extra = short_internal_target - len(seeds)
                seeds.extend(random.randint(0, 2**31 - 1) for _ in range(extra))
            logger.info(
                f"[SDPipeline] short internal candidate count: requested={requested_top_k}, internal={len(seeds)}"
            )
        elif hair_length == "medium" and int(
            subject_profile.medium_internal_candidate_min
        ) > len(seeds):
            extra = int(subject_profile.medium_internal_candidate_min) - len(seeds)
            seeds.extend(random.randint(0, 2**31 - 1) for _ in range(extra))
            logger.info(
                "[SDPipeline] subject-branch internal candidate expansion: requested=%d, internal=%d, branch=%s",
                requested_top_k,
                len(seeds),
                subject_profile.key,
            )
        landmark_debug_data = landmark_obs.get("debug_data")
        if not isinstance(landmark_debug_data, dict):
            landmark_debug_data = None

        lower_tail_support_mask = self._build_lower_hair_tail_support_mask(
            img_rgb=img_rgb,
            hair_mask=hair_mask,
            cloth_mask=cloth_mask,
            face_bbox=face_bbox,
            hair_length=hair_length,
        )
        if float(lower_tail_support_mask.sum()) > 0.0:
            hair_mask = np.maximum(hair_mask, lower_tail_support_mask).astype(
                np.float32
            )
            logger.info(
                "[SDPipeline] lower tail support added: pixels=%.0f merged_pixels=%.0f",
                lower_tail_support_mask.sum(),
                hair_mask.sum(),
            )
        _store_mask("pipeline_lower_tail_support_mask", lower_tail_support_mask)

        protect_mask_for_sd = self._build_generation_protect_mask(
            face_region_mask,
            face_bbox=face_bbox,
            hair_length=hair_length,
            subject_gender=subject_gender_mode,
        )
        protect_mask_for_removal = self._build_removal_protect_mask(
            face_region_mask,
            face_bbox=face_bbox,
            hair_length=hair_length,
            subject_gender=subject_gender_mode,
        )
        accessory_protect_mask = self._build_accessory_protect_mask(
            face_bbox=face_bbox,
            earring_mask=earring_mask,
            necklace_mask=necklace_mask,
            hair_length=hair_length,
        )
        if float(accessory_protect_mask.sum()) > 0.0:
            protect_mask_for_sd = np.maximum(
                protect_mask_for_sd, accessory_protect_mask
            ).astype(np.float32)
            protect_mask_for_removal = np.maximum(
                protect_mask_for_removal, accessory_protect_mask
            ).astype(np.float32)
            _store_mask("pipeline_accessory_protect_mask", accessory_protect_mask)
        _store_mask("pipeline_generation_protect_mask", protect_mask_for_sd)
        _store_mask("pipeline_removal_protect_mask", protect_mask_for_removal)

        # ── Step 3-c: SegFace 얼굴 픽셀 제거 (bbox 직사각형 대신 픽셀 단위 보정) ─
        hair_mask_before_face_protect = hair_mask.copy()
        hair_mask_for_removal = np.clip(hair_mask - protect_mask_for_removal, 0.0, 1.0)
        hair_mask = np.clip(hair_mask - protect_mask_for_sd, 0.0, 1.0)
        bangs_restore_for_removal = self._build_bangs_recovery_mask(
            hair_mask_before_face_protect,
            protect_mask_for_removal,
            face_bbox,
            landmark_debug_data=landmark_debug_data,
            hair_length=hair_length,
            subject_gender=subject_gender_mode,
            bangs_requested=bangs_requested,
        )
        bangs_restore_for_sd = self._build_bangs_recovery_mask(
            hair_mask_before_face_protect,
            protect_mask_for_sd,
            face_bbox,
            landmark_debug_data=landmark_debug_data,
            hair_length=hair_length,
            subject_gender=subject_gender_mode,
            bangs_requested=bangs_requested,
        )
        # front=down 합성 시 원본 앞머리 보존 여부 판단용:
        # protect_mask가 잘라낸 앞머리 픽셀이 많으면 원본에 앞머리가 있는 것 → SD 재생성 대신 보존
        _original_bang_px_in_protect = float(bangs_restore_for_sd.sum())
        # [v183 fix] 원본 앞머리 마스크를 미리 저장: 나중에 removal_mask_for_post에서 제외하는 데 사용
        _saved_original_bangs_mask: np.ndarray = np.clip(
            bangs_restore_for_sd.astype(np.float32), 0.0, 1.0
        ).copy()
        requested_front_coverage_mask = self._build_requested_front_coverage_mask(
            (H, W),
            face_bbox,
            normalized_prompt_context,
            hair_length=hair_length,
            subject_gender=subject_gender_mode,
            fringe_requested=bangs_requested,
        )
        logger.info(
            "[SDPipeline] requested front coverage mask: px=%s bangs_requested=%s style_axes=%s",
            self._count_active_mask_px(requested_front_coverage_mask),
            bangs_requested,
            normalized_prompt_context.get("style_axes", {}),
        )
        no_bangs_forehead_lama_preclean_seed_mask = np.zeros((H, W), dtype=np.float32)
        if explicit_no_bangs_requested or (
            short_no_bangs_target
            and bool(getattr(self.config, "short_no_bangs_disable_bangs_recovery", True))
        ):
            no_bangs_forehead_lama_preclean_seed_mask = np.maximum(
                np.clip(bangs_restore_for_removal.astype(np.float32), 0.0, 1.0),
                np.clip(bangs_restore_for_sd.astype(np.float32), 0.0, 1.0),
            ).astype(np.float32)
            # explicit_no_bangs(front=lifted): 좁은 band seed만으로는 무거운 앞머리 하단이
            # 포함되지 않아 LaMA가 부분적으로만 제거함 → SD가 조건 이미지에서 앞머리를 보고
            # 다시 생성하는 문제. explicit_no_bangs 시에는 항상 넓은 이마 band와 합집합.
            # Fallback: seed가 비어있으면 동일 logic으로 LaMA 건너뜀 방지.
            if explicit_no_bangs_requested or float(no_bangs_forehead_lama_preclean_seed_mask.sum()) < 20.0:
                _x1, _y1, _x2, _y2 = face_bbox
                _face_h = max(int(_y2 - _y1), 1)
                _forehead_top = max(0, int(_y1 - _face_h * 0.16))
                # 남성 no-bangs: 짧게 치고 올리는 스타일 → 이마 아래까지 적극 제거 (0.50)
                # 여성 no-bangs: 가르마/자연스럽게 넘기는 스타일. short는
                # 원본 앞머리 경계 halo가 남지 않도록 hairline seed를 조금 더 포함한다.
                _forehead_ratio = (
                    0.50
                    if subject_gender_mode == "male"
                    else 0.38 if hair_length == "short" else 0.32
                )
                _forehead_bottom = min(H, int(_y1 + _face_h * _forehead_ratio))
                _forehead_band = np.zeros((H, W), dtype=np.float32)
                _forehead_band[_forehead_top:_forehead_bottom, :] = 1.0
                _forehead_hair = np.clip(
                    hair_mask_before_face_protect.astype(np.float32) * _forehead_band,
                    0.0,
                    1.0,
                )
                if float(_forehead_hair.sum()) >= 12.0:
                    no_bangs_forehead_lama_preclean_seed_mask = np.maximum(
                        no_bangs_forehead_lama_preclean_seed_mask,
                        _forehead_hair,
                    )
                    logger.info(
                        "[SDPipeline] no-bangs forehead preclean seed expanded (%s): "
                        "explicit_no_bangs=%s forehead_ratio=%.2f px=%.0f",
                        subject_gender_mode,
                        explicit_no_bangs_requested,
                        _forehead_ratio,
                        float(no_bangs_forehead_lama_preclean_seed_mask.sum()),
                    )
            bangs_restore_for_removal = np.zeros((H, W), dtype=np.float32)
            bangs_restore_for_sd = np.zeros((H, W), dtype=np.float32)
            requested_front_coverage_mask = np.zeros((H, W), dtype=np.float32)
        elif float(requested_front_coverage_mask.sum()) > 0.0:
            bangs_restore_for_sd = np.maximum(
                np.clip(bangs_restore_for_sd.astype(np.float32), 0.0, 1.0),
                np.clip(requested_front_coverage_mask.astype(np.float32), 0.0, 1.0),
            ).astype(np.float32)
        if float(bangs_restore_for_removal.sum()) > 0.0:
            hair_mask_for_removal = np.maximum(
                hair_mask_for_removal, bangs_restore_for_removal
            ).astype(np.float32)
        if float(bangs_restore_for_sd.sum()) > 0.0:
            hair_mask = np.maximum(hair_mask, bangs_restore_for_sd).astype(np.float32)
        # short/medium 긴머리 제거 단계에서는 "옷 위로 떨어진 머리카락"도 지워야 하므로
        # cloth 제거 전 마스크를 별도로 보관한다.
        _store_mask(
            "pipeline_no_bangs_forehead_lama_preclean_seed_mask",
            no_bangs_forehead_lama_preclean_seed_mask,
        )
        _store_mask("pipeline_bangs_recovery_mask_removal", bangs_restore_for_removal)
        _store_mask("pipeline_bangs_recovery_mask_generation", bangs_restore_for_sd)
        _store_mask(
            "pipeline_requested_front_coverage_mask", requested_front_coverage_mask
        )
        _store_mask("pipeline_hair_mask_face_protected", hair_mask_for_removal)
        if debug_data_common is not None:
            debug_data_common.setdefault("source_cloth_preclean", {})
            debug_data_common["source_cloth_preclean"][
                "requested_front_coverage_px"
            ] = int(self._count_active_mask_px(requested_front_coverage_mask))
            debug_data_common["source_cloth_preclean"][
                "explicit_no_bangs_requested"
            ] = bool(explicit_no_bangs_requested)
        logger.info(
            f"[SDPipeline] 얼굴 픽셀 제거 완료, gen_px={hair_mask.sum():.0f}, removal_px={hair_mask_for_removal.sum():.0f}"
        )

        # ── Step 3-d: SegFace 옷 픽셀 제거 (옷이 바뀌는 문제 방지) ────────────
        # expand 전에 먼저 제거해야 옷 영역이 마스크 확장에 영향받지 않음
        cloth_dilate_px = max(
            5,
            int(
                self.config.mask_dilate_px
                * (
                    0.20
                    if hair_length == "short"
                    else 0.28 if hair_length == "medium" else 0.40
                )
            ),
        )
        cloth_mask_dilated = self._dilate_mask_with_px(
            cloth_mask,
            cloth_dilate_px,
        )
        cloth_ratio_limit = (
            0.10
            if hair_length == "short"
            else 0.14 if hair_length == "medium" else 0.18
        )
        if self._mask_ratio(cloth_mask_dilated) > cloth_ratio_limit:
            logger.info(
                "[SDPipeline] dilated cloth mask disabled: ratio=%.4f",
                self._mask_ratio(cloth_mask_dilated),
            )
            cloth_mask_dilated = np.zeros_like(cloth_mask_dilated, dtype=np.float32)
        cloth_generation_guard = cloth_mask_dilated.copy().astype(np.float32)
        cloth_generation_guard_release_mask = np.zeros((H, W), dtype=np.float32)
        short_upper_body_repaint_seed_mask = np.zeros((H, W), dtype=np.float32)
        short_upper_body_repaint_mask = np.zeros((H, W), dtype=np.float32)
        short_upper_body_repaint_px = 0
        upper_clothes_overwrite_mask = np.zeros((H, W), dtype=np.float32)
        upper_clothes_overwrite_core_mask = np.zeros((H, W), dtype=np.float32)
        effective_upper_clothes_overwrite_mask = np.zeros((H, W), dtype=np.float32)
        effective_upper_clothes_overwrite_core_mask = np.zeros((H, W), dtype=np.float32)
        overwrite_core_restore_mask_for_debug = np.zeros((H, W), dtype=np.float32)
        short_torso_generation_freeze_mask = np.zeros((H, W), dtype=np.float32)
        short_torso_generation_freeze_px = 0
        short_torso_generation_freeze_active = False
        gen_mask_before_upper_overwrite = np.zeros((H, W), dtype=np.float32)
        gen_mask_after_upper_overwrite = np.zeros((H, W), dtype=np.float32)
        gen_mask_after_initial_core = np.zeros((H, W), dtype=np.float32)
        gen_mask_before_final_core = np.zeros((H, W), dtype=np.float32)
        gen_mask_after_final_core = np.zeros((H, W), dtype=np.float32)
        hair_mask_for_sd_before_core = np.zeros((H, W), dtype=np.float32)
        upper_clothes_overwrite_anchor_mask = np.zeros((H, W), dtype=np.float32)
        upper_clothes_overwrite_px = 0
        use_upper_clothes_overwrite = bool(self.config.enable_upper_clothes_overwrite)
        overwrite_alpha = float(
            np.clip(self.config.upper_clothes_overwrite_alpha, 0.0, 1.0)
        )
        if hair_length == "short":
            overwrite_alpha = min(
                overwrite_alpha,
                float(
                    np.clip(self.config.short_upper_clothes_overwrite_alpha, 0.0, 1.0)
                ),
            )
        elif hair_length == "medium":
            overwrite_alpha = min(
                overwrite_alpha,
                float(
                    np.clip(self.config.medium_upper_clothes_overwrite_alpha, 0.0, 1.0)
                ),
            )
        else:
            overwrite_alpha = min(
                overwrite_alpha,
                float(
                    np.clip(self.config.long_upper_clothes_overwrite_alpha, 0.0, 1.0)
                ),
            )
        overwrite_core_alpha = float(
            np.clip(min(0.90, overwrite_alpha + 0.10), 0.0, 1.0)
        )
        overwrite_prepass_alpha = float(
            np.clip(min(0.88, overwrite_alpha + 0.08), 0.0, 1.0)
        )
        completed_torso_fill_mask = self._build_completed_subject_torso_fill_mask(
            torso_candidate_mask=subject_torso_candidate_mask,
            source_torso_hair_mask=source_torso_hair_mask,
            shoulder_bridge_mask=subject_shoulder_bridge_mask,
            protect_mask=protect_mask_for_sd,
            face_bbox=face_bbox,
        )
        source_garment_prepass_mask = self._build_source_garment_prepass_mask(
            hair_length=hair_length,
            source_torso_hair_mask=source_torso_hair_mask,
            source_cloth_overlap_mask=source_cloth_overlap_mask,
            cloth_mask=cloth_mask_dilated,
            protect_mask=protect_mask_for_sd,
            face_bbox=face_bbox,
            torso_candidate_mask=subject_torso_candidate_mask,
            completed_torso_fill_mask=completed_torso_fill_mask,
        )
        source_shoulder_contour_anchor_mask = (
            self._build_source_shoulder_contour_anchor_mask(
                source_torso_hair_mask=source_torso_hair_mask,
                source_cloth_overlap_mask=source_cloth_overlap_mask,
                cloth_mask=cloth_mask_dilated,
                shoulder_bridge_mask=subject_shoulder_bridge_mask,
                protect_mask=protect_mask_for_sd,
                face_bbox=face_bbox,
            )
        )
        if isinstance(
            subject_torso_anchor_mask, np.ndarray
        ) and subject_torso_anchor_mask.shape == (H, W):
            upper_clothes_overwrite_anchor_mask = np.maximum(
                upper_clothes_overwrite_anchor_mask,
                np.clip(subject_torso_anchor_mask.astype(np.float32), 0.0, 1.0),
            ).astype(np.float32)
        if isinstance(
            source_shoulder_contour_anchor_mask, np.ndarray
        ) and source_shoulder_contour_anchor_mask.shape == (H, W):
            upper_clothes_overwrite_anchor_mask = np.maximum(
                upper_clothes_overwrite_anchor_mask,
                np.clip(
                    source_shoulder_contour_anchor_mask.astype(np.float32), 0.0, 1.0
                ),
            ).astype(np.float32)
        if use_upper_clothes_overwrite:
            upper_clothes_overwrite_mask = self._build_upper_clothes_overwrite_mask(
                cloth_mask=cloth_mask_dilated,
                amodal_torso_mask=completed_torso_fill_mask,
                shoulder_anchor_mask=upper_clothes_overwrite_anchor_mask,
                torso_candidate_mask=subject_torso_candidate_mask,
                completed_torso_fill_mask=completed_torso_fill_mask,
                source_torso_hair_mask=source_torso_hair_mask,
                source_cloth_overlap_mask=source_cloth_overlap_mask,
                protect_mask=protect_mask_for_sd,
                face_bbox=face_bbox,
            )
            upper_clothes_overwrite_u8 = (
                np.clip(upper_clothes_overwrite_mask.astype(np.float32), 0.0, 1.0)
                > 0.08
            ).astype(np.uint8) * 255
            upper_clothes_overwrite_px = int((upper_clothes_overwrite_u8 > 0).sum())
            if upper_clothes_overwrite_px >= int(
                self.config.upper_clothes_overwrite_min_px
            ):
                if hair_length != "short":
                    cloth_generation_guard_release_mask = np.maximum(
                        cloth_generation_guard_release_mask.astype(np.float32),
                        np.clip(
                            upper_clothes_overwrite_mask.astype(np.float32), 0.0, 1.0
                        ),
                    ).astype(np.float32)
                source_garment_prepass_mask = np.maximum(
                    source_garment_prepass_mask.astype(np.float32),
                    np.clip(
                        upper_clothes_overwrite_mask.astype(np.float32)
                        * overwrite_prepass_alpha,
                        0.0,
                        1.0,
                    ),
                ).astype(np.float32)
                logger.info(
                    "[SDPipeline] upper clothes overwrite opened: overwrite_px=%d release_px=%.0f alpha=%.2f",
                    upper_clothes_overwrite_px,
                    float(cloth_generation_guard_release_mask.sum()),
                    overwrite_alpha,
                )
        if hair_length == "short":
            try:
                overwrite_core_seed_debug: Dict[str, Any] = {}
                overwrite_core_seed_debug_masks: Dict[str, np.ndarray] = {}
                _, _, _, _, short_repaint_cutoff_y = self._estimate_head_generation_box(
                    image_shape=(H, W),
                    face_bbox=face_bbox,
                    landmark_face_mask=landmark_face_mask,
                    landmark_debug_data=landmark_debug_data,
                    hair_length=hair_length,
                )
                short_upper_body_repaint_seed_mask = (
                    self._build_short_below_bob_torso_mask(
                        cloth_mask=cloth_mask_dilated,
                        face_bbox=face_bbox,
                        cutoff_y=short_repaint_cutoff_y,
                        hair_length=hair_length,
                        final_hair_mask=None,
                        torso_candidate_mask=subject_torso_candidate_mask,
                        completed_torso_fill_mask=completed_torso_fill_mask,
                        shoulder_anchor_mask=upper_clothes_overwrite_anchor_mask,
                        debug_info=overwrite_core_seed_debug,
                        debug_masks=overwrite_core_seed_debug_masks,
                    )
                )
                if short_upper_body_repaint_seed_mask.shape == (H, W):
                    upper_clothes_overwrite_core_mask = np.maximum(
                        upper_clothes_overwrite_core_mask,
                        np.clip(
                            short_upper_body_repaint_seed_mask.astype(np.float32)
                            - protect_mask_for_sd.astype(np.float32),
                            0.0,
                            1.0,
                        ),
                    ).astype(np.float32)
                short_upper_body_repaint_mask = (
                    self._build_short_torso_garment_repaint_mask(
                        cloth_mask=cloth_mask_dilated,
                        torso_mask=short_upper_body_repaint_seed_mask,
                        torso_anchor_mask=subject_torso_anchor_mask,
                        torso_candidate_mask=subject_torso_candidate_mask,
                        shoulder_bridge_mask=subject_shoulder_bridge_mask,
                        sam2_hair_mask=hair_mask_for_removal,
                        face_mask=face_region_mask,
                        face_bbox=face_bbox,
                        cutoff_y=short_repaint_cutoff_y,
                        hair_length=hair_length,
                        final_hair_mask=None,
                        protect_mask=protect_mask_for_sd,
                    )
                )
                short_upper_body_repaint_u8 = (
                    np.clip(short_upper_body_repaint_mask.astype(np.float32), 0.0, 1.0)
                    > 0.08
                ).astype(np.uint8) * 255
                short_upper_body_repaint_px = int(
                    (short_upper_body_repaint_u8 > 0).sum()
                )
                if (
                    use_upper_clothes_overwrite
                    and upper_clothes_overwrite_core_mask.shape == (H, W)
                ):
                    upper_clothes_overwrite_mask = np.maximum(
                        upper_clothes_overwrite_mask.astype(np.float32),
                        np.clip(
                            upper_clothes_overwrite_core_mask.astype(np.float32)
                            * overwrite_core_alpha,
                            0.0,
                            1.0,
                        ),
                    ).astype(np.float32)
                if short_upper_body_repaint_px >= 16:
                    if hair_length != "short":
                        repaint_release_u8 = cv2.dilate(
                            short_upper_body_repaint_u8,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 27)),
                            iterations=1,
                        )
                        repaint_release_f = cv2.GaussianBlur(
                            repaint_release_u8.astype(np.float32) / 255.0,
                            (0, 0),
                            sigmaX=4.0,
                            sigmaY=6.0,
                        )
                        cloth_generation_guard_release_mask = np.maximum(
                            cloth_generation_guard_release_mask.astype(np.float32),
                            np.clip(repaint_release_f.astype(np.float32), 0.0, 1.0),
                        ).astype(np.float32)
                    source_garment_prepass_mask = np.maximum(
                        source_garment_prepass_mask.astype(np.float32),
                        np.clip(
                            short_upper_body_repaint_mask.astype(np.float32)
                            * overwrite_prepass_alpha,
                            0.0,
                            1.0,
                        ),
                    ).astype(np.float32)
                    logger.info(
                        "[SDPipeline] short upper-body repaint opened: repaint_px=%d release_px=%.0f alpha=%.2f",
                        short_upper_body_repaint_px,
                        float(cloth_generation_guard_release_mask.sum()),
                        overwrite_prepass_alpha,
                    )
                if debug_data_common is not None:
                    diag = debug_data_common.setdefault("diagnostics", {})
                    diag["overwrite_core_seed_debug"] = overwrite_core_seed_debug
                if debug_images_common is not None:
                    for name, mask in overwrite_core_seed_debug_masks.items():
                        _store_mask(f"pipeline_overwrite_core_seed_{name}_mask", mask)
            except Exception as e:
                logger.warning(
                    f"[SDPipeline] short upper-body repaint preparation failed (ignored): {e}"
                )
        if use_upper_clothes_overwrite:
            effective_upper_clothes_overwrite_mask = np.clip(
                upper_clothes_overwrite_mask.astype(np.float32) * overwrite_alpha,
                0.0,
                1.0,
            ).astype(np.float32)
            effective_upper_clothes_overwrite_core_mask = np.clip(
                upper_clothes_overwrite_core_mask.astype(np.float32)
                * overwrite_core_alpha,
                0.0,
                1.0,
            ).astype(np.float32)
            if hair_length != "short":
                cloth_generation_guard_release_mask = np.maximum(
                    cloth_generation_guard_release_mask.astype(np.float32),
                    effective_upper_clothes_overwrite_mask.astype(np.float32),
                ).astype(np.float32)
                cloth_generation_guard_release_mask = np.maximum(
                    cloth_generation_guard_release_mask.astype(np.float32),
                    effective_upper_clothes_overwrite_core_mask.astype(np.float32),
                ).astype(np.float32)
                cloth_generation_guard_release_mask = np.maximum(
                    cloth_generation_guard_release_mask.astype(np.float32),
                    np.clip(
                        completed_torso_fill_mask.astype(np.float32) * 0.92, 0.0, 1.0
                    ),
                ).astype(np.float32)
                cloth_generation_guard_release_mask = np.maximum(
                    cloth_generation_guard_release_mask.astype(np.float32),
                    np.clip(
                        source_garment_prepass_mask.astype(np.float32) * 0.96, 0.0, 1.0
                    ),
                ).astype(np.float32)
                if float(cloth_generation_guard_release_mask.sum()) > 0.0:
                    release_expand_px = max(
                        8,
                        int(
                            getattr(self.config, "overwrite_cloth_guard_expand_px", 24)
                        ),
                    )
                    release_u8 = (
                        np.clip(
                            cloth_generation_guard_release_mask.astype(np.float32),
                            0.0,
                            1.0,
                        )
                        > 0.04
                    ).astype(np.uint8) * 255
                    release_u8 = cv2.dilate(
                        release_u8,
                        cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE, (release_expand_px, release_expand_px)
                        ),
                        iterations=1,
                    )
                    release_sigma_x = max(2.2, float(release_expand_px) * 0.28)
                    release_sigma_y = max(2.8, float(release_expand_px) * 0.36)
                    release_f = cv2.GaussianBlur(
                        release_u8.astype(np.float32) / 255.0,
                        (0, 0),
                        sigmaX=release_sigma_x,
                        sigmaY=release_sigma_y,
                    ).astype(np.float32)
                    guard_residual = float(
                        np.clip(
                            getattr(
                                self.config, "overwrite_cloth_guard_residual", 0.12
                            ),
                            0.0,
                            0.70,
                        )
                    )
                    guard_keep_factor = np.clip(
                        1.0 - np.clip(release_f, 0.0, 1.0) * (1.0 - guard_residual),
                        0.0,
                        1.0,
                    ).astype(np.float32)
                    cloth_generation_guard = np.clip(
                        cloth_mask_dilated.astype(np.float32) * guard_keep_factor,
                        0.0,
                        1.0,
                    ).astype(np.float32)
        hair_mask = np.clip(hair_mask - cloth_generation_guard, 0.0, 1.0)
        _store_mask("segface_cloth_mask_dilated", cloth_mask_dilated)
        _store_mask(
            "pipeline_cloth_generation_guard_release_mask",
            cloth_generation_guard_release_mask,
        )
        _store_mask("pipeline_cloth_generation_guard_mask", cloth_generation_guard)
        _store_mask(
            "pipeline_short_upper_body_repaint_seed_mask",
            short_upper_body_repaint_seed_mask,
        )
        _store_mask(
            "pipeline_short_upper_body_repaint_mask", short_upper_body_repaint_mask
        )
        for name, mask in (
            locals().get("overwrite_core_seed_debug_masks") or {}
        ).items():
            _store_mask(f"pipeline_overwrite_core_seed_{name}_mask", mask)
        _store_mask(
            "pipeline_upper_clothes_overwrite_mask", upper_clothes_overwrite_mask
        )
        _store_mask(
            "pipeline_upper_clothes_overwrite_core_mask",
            upper_clothes_overwrite_core_mask,
        )
        _store_mask(
            "pipeline_upper_clothes_overwrite_effective_mask",
            effective_upper_clothes_overwrite_mask,
        )
        _store_mask(
            "pipeline_upper_clothes_overwrite_core_effective_mask",
            effective_upper_clothes_overwrite_core_mask,
        )
        _store_mask("pipeline_hair_mask_cloth_protected", hair_mask)
        logger.info(f"[SDPipeline] 옷 픽셀 제거 완료, pixels={hair_mask.sum():.0f}")
        completed_torso_fill_px = int(
            (
                np.clip(completed_torso_fill_mask.astype(np.float32), 0.0, 1.0) > 0.08
            ).sum()
        )
        source_shoulder_contour_anchor_px = int(
            (
                np.clip(
                    source_shoulder_contour_anchor_mask.astype(np.float32), 0.0, 1.0
                )
                > 0.08
            ).sum()
        )
        source_garment_prepass_state = self._finalize_source_garment_prepass(
            source_garment_prepass_mask=source_garment_prepass_mask,
            face_bbox=face_bbox,
            hair_length=hair_length,
            skip_source_cloth_preclean=skip_source_cloth_preclean,
            hair_mask_for_removal=hair_mask_for_removal,
            source_torso_hair_mask=source_torso_hair_mask,
        )
        source_garment_prepass_mask = source_garment_prepass_state["mask"]
        source_garment_prepass_px = int(source_garment_prepass_state["px"])
        source_garment_prepass_enabled = bool(source_garment_prepass_state["enabled"])
        source_garment_prepass_min_px = int(source_garment_prepass_state["min_px"])
        source_garment_prepass_bridge_px = int(
            source_garment_prepass_state["bridge_px"]
        )
        source_garment_prepass_bridge_applied = bool(
            source_garment_prepass_state["bridge_applied"]
        )
        source_garment_prepass_bridge_min_px = int(
            source_garment_prepass_state["bridge_min_px"]
        )
        if hair_length == "short" and source_garment_prepass_enabled:
            short_torso_generation_freeze_mask = np.maximum(
                short_torso_generation_freeze_mask.astype(np.float32),
                np.clip(source_garment_prepass_mask.astype(np.float32), 0.0, 1.0),
            ).astype(np.float32)
            if effective_upper_clothes_overwrite_mask.shape == (H, W):
                short_torso_generation_freeze_mask = np.maximum(
                    short_torso_generation_freeze_mask.astype(np.float32),
                    np.clip(
                        effective_upper_clothes_overwrite_mask.astype(np.float32),
                        0.0,
                        1.0,
                    ),
                ).astype(np.float32)
            if effective_upper_clothes_overwrite_core_mask.shape == (H, W):
                short_torso_generation_freeze_mask = np.maximum(
                    short_torso_generation_freeze_mask.astype(np.float32),
                    np.clip(
                        effective_upper_clothes_overwrite_core_mask.astype(np.float32),
                        0.0,
                        1.0,
                    ),
                ).astype(np.float32)
            if short_upper_body_repaint_seed_mask.shape == (H, W):
                short_torso_generation_freeze_mask = np.maximum(
                    short_torso_generation_freeze_mask.astype(np.float32),
                    np.clip(
                        short_upper_body_repaint_seed_mask.astype(np.float32) * 0.92,
                        0.0,
                        1.0,
                    ),
                ).astype(np.float32)
            if cloth_mask_dilated.shape == (H, W):
                freeze_cloth_gate = self._dilate_mask_with_px(
                    cloth_mask_dilated.astype(np.float32),
                    13,
                )
                freeze_cloth_gate = np.maximum(
                    freeze_cloth_gate.astype(np.float32),
                    np.clip(source_garment_prepass_mask.astype(np.float32), 0.0, 1.0),
                ).astype(np.float32)
                short_torso_generation_freeze_mask = np.clip(
                    short_torso_generation_freeze_mask.astype(np.float32)
                    * np.clip(freeze_cloth_gate.astype(np.float32), 0.0, 1.0),
                    0.0,
                    1.0,
                ).astype(np.float32)
            short_torso_generation_freeze_px = self._count_active_mask_px(
                short_torso_generation_freeze_mask
            )
            short_torso_generation_freeze_active = (
                short_torso_generation_freeze_px >= 120
            )

        _store_mask("pipeline_completed_torso_fill_mask", completed_torso_fill_mask)
        _store_mask("pipeline_source_garment_prepass_mask", source_garment_prepass_mask)
        _store_mask(
            "pipeline_short_torso_generation_freeze_mask",
            short_torso_generation_freeze_mask,
        )
        _store_mask(
            "pipeline_source_garment_prepass_bridge_mask",
            source_garment_prepass_state.get("bridge_mask"),
        )
        _store_mask(
            "pipeline_source_shoulder_contour_anchor_mask",
            source_shoulder_contour_anchor_mask,
        )
        if debug_data_common is not None:
            debug_data_common.setdefault("source_cloth_preclean", {})
            debug_data_common["source_cloth_preclean"][
                "completed_torso_fill_px"
            ] = completed_torso_fill_px
            debug_data_common["source_cloth_preclean"][
                "garment_prepass_px"
            ] = source_garment_prepass_px
            debug_data_common["source_cloth_preclean"][
                "garment_prepass_min_px"
            ] = source_garment_prepass_min_px
            debug_data_common["source_cloth_preclean"]["garment_prepass_enabled"] = (
                bool(source_garment_prepass_enabled)
            )
            debug_data_common["source_cloth_preclean"][
                "garment_prepass_bridge_px"
            ] = source_garment_prepass_bridge_px
            debug_data_common["source_cloth_preclean"][
                "garment_prepass_bridge_min_px"
            ] = source_garment_prepass_bridge_min_px
            debug_data_common["source_cloth_preclean"][
                "garment_prepass_bridge_applied"
            ] = bool(source_garment_prepass_bridge_applied)
            debug_data_common["source_cloth_preclean"][
                "torso_generation_freeze_px"
            ] = short_torso_generation_freeze_px
            debug_data_common["source_cloth_preclean"][
                "torso_generation_freeze_active"
            ] = bool(short_torso_generation_freeze_active)
            debug_data_common["source_cloth_preclean"][
                "shoulder_contour_anchor_px"
            ] = source_shoulder_contour_anchor_px
            debug_data_common["source_cloth_preclean"][
                "short_upper_body_repaint_px"
            ] = short_upper_body_repaint_px
            debug_data_common["source_cloth_preclean"][
                "upper_clothes_overwrite_px"
            ] = upper_clothes_overwrite_px

        # ── Step 3-e: 숏컷/중단발 — 전략 2 (Post-Inpainting) ────────────────
        # 단발/숏컷에서 기존 긴머리 prior가 강하면, 생성 후 잔여 long-hair만
        # 차집합 마스크 기반으로 후처리하는 쪽이 더 안정적이다.
        cutoff_y_for_post: Optional[int] = None
        removal_mask_for_post: Optional[np.ndarray] = None
        shoulder_protect_for_post: Optional[np.ndarray] = None
        neckline_preserve_for_post: Optional[np.ndarray] = None
        lateral_neck_preserve_for_post: Optional[np.ndarray] = None
        torso_cloth_preserve_for_post: Optional[np.ndarray] = None
        bright_cloth_preserve_for_post: Optional[np.ndarray] = None
        shoulder_cloth_release_for_post: Optional[np.ndarray] = None
        artifact_cleanup_mask_for_post: Optional[np.ndarray] = None
        lower_tail_support_for_post: Optional[np.ndarray] = None
        short_generation_tail_release_mask = np.zeros((H, W), dtype=np.float32)
        short_generation_tail_release_px = 0
        short_generation_conditioning_cleanup_mask = np.zeros((H, W), dtype=np.float32)
        short_generation_conditioning_cleanup_px = 0
        short_generation_conditioning_cleanup_applied = False
        short_generation_white_tshirt_conditioning_fill_mask = np.zeros(
            (H, W), dtype=np.float32
        )
        short_generation_white_tshirt_conditioning_fill_px = 0
        short_generation_white_tshirt_conditioning_fill_applied = False
        no_bangs_forehead_lama_preclean_mask = np.zeros((H, W), dtype=np.float32)
        no_bangs_forehead_lama_preclean_px = 0
        no_bangs_forehead_lama_preclean_applied = False
        shoulder_cloth_refine_skipped_for_freeze = False
        below_bob_generation_block_for_post: Optional[np.ndarray] = None
        below_bob_cloth_restore_for_post: Optional[np.ndarray] = None
        shoulder_hair_forbid_for_post: Optional[np.ndarray] = None
        residual_side_hair_lane_removal_for_post = np.zeros((H, W), dtype=np.float32)
        composite_bangs_release_mask = np.zeros((H, W), dtype=np.float32)
        center_chest_strand_mask = np.zeros((H, W), dtype=np.float32)
        center_chest_strand_removal_mask = np.zeros((H, W), dtype=np.float32)
        shoulder_cross_bridge_for_post = np.zeros((H, W), dtype=np.float32)
        lower_tail_post_debug_masks: Dict[str, np.ndarray] = {}
        if hair_length in ("short", "medium"):
            head_x1, head_y1, head_x2, head_y2, cutoff_y = (
                self._estimate_head_generation_box(
                    image_shape=(H, W),
                    face_bbox=face_bbox,
                    landmark_face_mask=landmark_face_mask,
                    landmark_debug_data=landmark_debug_data,
                    hair_length=hair_length,
                )
            )
            cutoff_y_for_post = cutoff_y
            shoulder_protect_for_post = self._build_shoulder_protect_mask(
                cloth_mask=cloth_mask_dilated,
                face_bbox=face_bbox,
                cutoff_y=cutoff_y,
            )
            neckline_preserve_for_post = self._build_neckline_preserve_mask(
                face_bbox=face_bbox,
                cloth_mask=cloth_mask_dilated,
                hair_length=hair_length,
            )
            lateral_neck_preserve_for_post = (
                self._build_short_lateral_neck_preserve_mask(
                    face_bbox=face_bbox,
                    cutoff_y=cutoff_y,
                    cloth_mask=cloth_mask_dilated,
                    hair_length=hair_length,
                )
            )
            torso_cloth_preserve_for_post = self._build_torso_cloth_preserve_mask(
                face_bbox=face_bbox,
                cloth_mask=cloth_mask_dilated,
                cutoff_y=cutoff_y,
                hair_length=hair_length,
            )
            lower_tail_support_for_post = self._build_lower_tail_post_support_mask(
                support_mask=lower_tail_support_mask,
                cloth_mask=cloth_mask_dilated,
                torso_hair_mask=source_torso_hair_mask,
                face_bbox=face_bbox,
                cutoff_y=cutoff_y,
                hair_length=hair_length,
                debug_outputs=(
                    lower_tail_post_debug_masks
                    if debug_images_common is not None
                    else None
                ),
            )
            center_chest_strand_mask = self._build_center_chest_strand_support_mask(
                img_rgb=img_rgb,
                cloth_mask=cloth_mask_dilated,
                support_mask=lower_tail_support_mask,
                face_bbox=face_bbox,
                cutoff_y=cutoff_y,
                hair_length=hair_length,
            )
            if shoulder_protect_for_post.sum() > 0:
                logger.info(
                    "[SDPipeline] 어깨 보호 마스크 적용: "
                    f"pixels={shoulder_protect_for_post.sum():.0f}"
                )
            _store_mask("pipeline_shoulder_protect_mask", shoulder_protect_for_post)
            _store_mask("pipeline_neckline_preserve_mask", neckline_preserve_for_post)
            _store_mask(
                "pipeline_lateral_neck_preserve_mask", lateral_neck_preserve_for_post
            )
            _store_mask(
                "pipeline_torso_cloth_preserve_mask", torso_cloth_preserve_for_post
            )
            _store_mask(
                "pipeline_lower_tail_support_post_mask", lower_tail_support_for_post
            )
            _store_mask("pipeline_center_chest_strand_mask", center_chest_strand_mask)
            if (
                hair_length == "short"
                and lower_tail_support_for_post is not None
                and lower_tail_support_for_post.shape == (H, W)
            ):
                short_generation_tail_release_mask = np.clip(
                    lower_tail_support_for_post.astype(np.float32),
                    0.0,
                    1.0,
                )
                if cloth_mask_dilated is not None and cloth_mask_dilated.shape == (
                    H,
                    W,
                ):
                    short_generation_tail_release_mask = np.clip(
                        short_generation_tail_release_mask.astype(np.float32)
                        * self._dilate_mask_with_px(
                            cloth_mask_dilated.astype(np.float32), 9
                        ),
                        0.0,
                        1.0,
                    ).astype(np.float32)
                if protect_mask_for_sd.shape == (H, W):
                    short_generation_tail_release_mask = np.clip(
                        short_generation_tail_release_mask
                        - protect_mask_for_sd.astype(np.float32) * 0.95,
                        0.0,
                        1.0,
                    )
                if face_region_mask.shape == (H, W):
                    short_generation_tail_release_mask = np.clip(
                        short_generation_tail_release_mask
                        - face_region_mask.astype(np.float32) * 1.20,
                        0.0,
                        1.0,
                    )
                short_generation_tail_release_px = self._count_active_mask_px(
                    short_generation_tail_release_mask
                )
            _store_mask(
                "pipeline_short_generation_tail_release_mask",
                short_generation_tail_release_mask,
            )
            for name, mask in lower_tail_post_debug_masks.items():
                _store_mask(f"pipeline_{name}", mask)
            if isinstance(
                subject_shoulder_bridge_mask, np.ndarray
            ) and subject_shoulder_bridge_mask.shape == (H, W):
                face_h = max(int(face_bbox[3] - face_bbox[1]), 1)
                shoulder_hair_forbid_for_post = np.clip(
                    subject_shoulder_bridge_mask.astype(np.float32),
                    0.0,
                    1.0,
                )
                shoulder_hair_forbid_for_post[
                    : max(0, int(cutoff_y + face_h * 0.02)), :
                ] = 0.0
                if protect_mask_for_sd.shape == (H, W):
                    shoulder_hair_forbid_for_post = np.clip(
                        shoulder_hair_forbid_for_post - protect_mask_for_sd * 0.92,
                        0.0,
                        1.0,
                    )
                if face_region_mask.shape == (H, W):
                    shoulder_hair_forbid_for_post = np.clip(
                        shoulder_hair_forbid_for_post - face_region_mask * 1.20,
                        0.0,
                        1.0,
                    )
                shoulder_hair_forbid_for_post = (
                    cv2.dilate(
                        (shoulder_hair_forbid_for_post > 0.08).astype(np.uint8) * 255,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 15)),
                        iterations=1,
                    ).astype(np.float32)
                    / 255.0
                )
            if (
                source_shoulder_contour_anchor_mask is not None
                and source_shoulder_contour_anchor_mask.shape == (H, W)
                and float(source_shoulder_contour_anchor_mask.sum()) > 0.0
            ):
                shoulder_anchor_forbid = (
                    cv2.dilate(
                        (
                            np.clip(
                                source_shoulder_contour_anchor_mask.astype(np.float32),
                                0.0,
                                1.0,
                            )
                            > 0.08
                        ).astype(np.uint8)
                        * 255,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 17)),
                        iterations=1,
                    ).astype(np.float32)
                    / 255.0
                )
                if shoulder_hair_forbid_for_post is None:
                    shoulder_hair_forbid_for_post = shoulder_anchor_forbid.astype(
                        np.float32
                    )
                else:
                    shoulder_hair_forbid_for_post = np.maximum(
                        shoulder_hair_forbid_for_post.astype(np.float32),
                        shoulder_anchor_forbid.astype(np.float32),
                    ).astype(np.float32)
            residual_side_hair_lane_removal_for_post = (
                self._build_residual_side_hair_lane_removal_mask(
                    source_torso_hair_mask=source_torso_hair_mask,
                    face_bbox=face_bbox,
                    cutoff_y=cutoff_y,
                    protect_mask=protect_mask_for_sd,
                )
            )
            _store_mask(
                "pipeline_shoulder_hair_forbid_mask", shoulder_hair_forbid_for_post
            )
            _store_mask(
                "pipeline_residual_side_hair_lane_removal_mask",
                residual_side_hair_lane_removal_for_post,
            )

            # v4 쪽이 더 안정적이었던 핵심:
            # 1) cutoff 아래 long hair를 먼저 실제 hair 기반으로 비운 뒤
            # 2) 생성은 head box 안에서만 제한적으로 수행한다.
            if hair_length == "short":
                base_prior = np.clip(
                    hair_mask_base - face_region_mask,
                    0.0,
                    1.0,
                ).astype(np.float32)
                prior_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                base_prior = cv2.dilate(
                    (base_prior > 0.5).astype(np.uint8) * 255,
                    prior_k,
                    iterations=1,
                )
                base_prior = (base_prior > 0).astype(np.float32)

                removal_seed = np.clip(hair_mask_for_removal * base_prior, 0.0, 1.0)
                tail_below_cutoff = hair_mask_for_removal.copy()
                tail_below_cutoff[:cutoff_y, :] = 0.0
                tail_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 7))
                tail_below_cutoff = (
                    cv2.dilate(
                        (tail_below_cutoff > 0.5).astype(np.uint8) * 255,
                        tail_k,
                        iterations=1,
                    ).astype(np.float32)
                    / 255.0
                )

                if removal_seed.sum() > 120:
                    removal_mask = np.clip(
                        np.maximum(removal_seed, tail_below_cutoff),
                        0.0,
                        1.0,
                    )
                else:
                    removal_mask = hair_mask_for_removal.copy()
            else:
                removal_mask = hair_mask_for_removal.copy()

            removal_mask[:cutoff_y, :] = 0.0
            bright_cloth_preserve_for_post = self._build_bright_cloth_preserve_mask(
                img_rgb=img_rgb,
                removal_mask=removal_mask,
                cloth_mask=cloth_mask_dilated,
                face_bbox=face_bbox,
                cutoff_y=cutoff_y,
                hair_length=hair_length,
            )
            _store_mask(
                "pipeline_bright_cloth_preserve_mask", bright_cloth_preserve_for_post
            )

            hair_below = (hair_mask_for_removal > 0.5).astype(np.uint8)
            hair_below[:cutoff_y, :] = 0
            if hair_length == "short":
                face_x1, _, face_x2, _ = [int(v) for v in face_bbox]
                face_w = max(int(face_x2 - face_x1), 1)
                hair_below_left = max(0, int(face_x1 - face_w * 1.08))
                hair_below_right = min(W, int(face_x2 + face_w * 1.08))
            else:
                hair_below_left = max(0, head_x1 - 20)
                hair_below_right = min(W, head_x2 + 20)
            hair_below[:, :hair_below_left] = 0
            hair_below[:, hair_below_right:] = 0
            if int((hair_below > 0).sum()) > 0:
                expand_k = (
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 7))
                    if hair_length == "short"
                    else cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 13))
                )
                hair_below_expanded = cv2.dilate(
                    hair_below,
                    expand_k,
                    iterations=1,
                ).astype(np.float32)
                removal_mask = np.maximum(removal_mask, hair_below_expanded)
                logger.info(
                    f"[SDPipeline] removal_mask hair기반 확장({hair_length}): "
                    f"pixels={removal_mask.sum():.0f}"
                )

            close_size = 5 if hair_length == "short" else 11
            remove_close_k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (close_size, close_size),
            )
            removal_mask = cv2.morphologyEx(
                removal_mask,
                cv2.MORPH_CLOSE,
                remove_close_k,
            )
            removal_mask = np.clip(removal_mask, 0.0, 1.0).astype(np.float32)

            if (
                shoulder_protect_for_post is not None
                and shoulder_protect_for_post.shape == (H, W)
            ):
                shoulder_protect_weight = 0.30 if hair_length == "short" else 0.85
                removal_mask = np.clip(
                    removal_mask - shoulder_protect_for_post * shoulder_protect_weight,
                    0.0,
                    1.0,
                )
            if (
                neckline_preserve_for_post is not None
                and neckline_preserve_for_post.shape == (H, W)
            ):
                removal_mask = np.clip(
                    removal_mask
                    - neckline_preserve_for_post
                    * (0.28 if hair_length == "short" else 0.26),
                    0.0,
                    1.0,
                )
            if (
                torso_cloth_preserve_for_post is not None
                and torso_cloth_preserve_for_post.shape == (H, W)
            ):
                removal_mask = np.clip(
                    removal_mask
                    - torso_cloth_preserve_for_post
                    * (0.48 if hair_length == "short" else 0.34),
                    0.0,
                    1.0,
                )
            if (
                bright_cloth_preserve_for_post is not None
                and bright_cloth_preserve_for_post.shape == (H, W)
            ):
                removal_mask = np.clip(
                    removal_mask
                    - bright_cloth_preserve_for_post
                    * (0.58 if hair_length == "short" else 0.44),
                    0.0,
                    1.0,
                )
            if hair_length == "short":
                removal_mask = self._filter_short_torso_box_mask(
                    img_rgb=img_rgb,
                    removal_mask=removal_mask,
                    cloth_mask=cloth_mask_dilated,
                    support_mask=lower_tail_support_for_post,
                    center_support_mask=center_chest_strand_mask,
                    face_bbox=face_bbox,
                    cutoff_y=cutoff_y,
                )
                shoulder_cloth_release_for_post = (
                    self._build_shoulder_cloth_restore_mask(
                        removal_mask=removal_mask,
                        cloth_mask=cloth_mask_dilated,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y,
                        hair_length=hair_length,
                        protect_mask=protect_mask_for_sd,
                    )
                )
                if (
                    shoulder_cloth_release_for_post is not None
                    and shoulder_cloth_release_for_post.shape == (H, W)
                ):
                    removal_mask = np.clip(
                        removal_mask - shoulder_cloth_release_for_post * 0.72,
                        0.0,
                        1.0,
                    )
            if float(center_chest_strand_mask.sum()) > 0.0:
                center_chest_strand_removal_mask = (
                    cv2.dilate(
                        (center_chest_strand_mask > 0.08).astype(np.uint8) * 255,
                        cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE,
                            (7, 21) if hair_length == "short" else (5, 15),
                        ),
                        iterations=1,
                    ).astype(np.float32)
                    / 255.0
                )
                removal_mask = np.maximum(
                    removal_mask, center_chest_strand_removal_mask
                ).astype(np.float32)
            lower_tail_removal_extension = np.zeros((H, W), dtype=np.float32)
            if (
                lower_tail_support_for_post is not None
                and lower_tail_support_for_post.shape == (H, W)
            ):
                lower_tail_removal_extension = (
                    self._build_lower_tail_removal_extension_mask(
                        support_mask=lower_tail_support_for_post,
                        removal_mask=removal_mask,
                        cloth_mask=cloth_mask_dilated,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y,
                        hair_length=hair_length,
                    )
                )
                if float(lower_tail_removal_extension.sum()) > 0.0:
                    removal_mask = np.maximum(
                        removal_mask, lower_tail_removal_extension
                    ).astype(np.float32)
                    logger.info(
                        "[SDPipeline] lower tail removal extension added: pixels=%.0f merged_pixels=%.0f",
                        lower_tail_removal_extension.sum(),
                        removal_mask.sum(),
                    )
            if hair_length == "short":
                shoulder_cross_bridge_for_post = self._build_short_shoulder_cross_bridge_mask(
                    source_shoulder_contour_anchor_mask=source_shoulder_contour_anchor_mask,
                    source_torso_hair_mask=source_torso_hair_mask,
                    cloth_mask=cloth_mask_dilated,
                    protect_mask=protect_mask_for_sd,
                    face_bbox=face_bbox,
                    cutoff_y=cutoff_y,
                    support_mask=lower_tail_support_for_post,
                    center_support_mask=center_chest_strand_mask,
                )
                if float(shoulder_cross_bridge_for_post.sum()) > 0.0:
                    removal_mask = np.maximum(
                        removal_mask.astype(np.float32),
                        shoulder_cross_bridge_for_post.astype(np.float32),
                    ).astype(np.float32)
                    logger.info(
                        "[SDPipeline] short shoulder cross bridge added: pixels=%.0f merged_pixels=%.0f",
                        shoulder_cross_bridge_for_post.sum(),
                        removal_mask.sum(),
                    )
            if hair_length == "short":
                removal_mask = self._filter_short_torso_box_mask(
                    img_rgb=img_rgb,
                    removal_mask=removal_mask,
                    cloth_mask=cloth_mask_dilated,
                    support_mask=lower_tail_support_for_post,
                    center_support_mask=center_chest_strand_mask,
                    face_bbox=face_bbox,
                    cutoff_y=cutoff_y,
                )
                shoulder_cloth_release_for_post = (
                    self._build_shoulder_cloth_restore_mask(
                        removal_mask=removal_mask,
                        cloth_mask=cloth_mask_dilated,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y,
                        hair_length=hair_length,
                        protect_mask=protect_mask_for_sd,
                    )
                )
                if (
                    shoulder_cloth_release_for_post is not None
                    and shoulder_cloth_release_for_post.shape == (H, W)
                ):
                    removal_mask = np.clip(
                        removal_mask - shoulder_cloth_release_for_post * 0.74,
                        0.0,
                        1.0,
                    )
                removal_mask = self._restrict_short_removal_to_tail_lanes(
                    removal_mask=removal_mask,
                    support_mask=lower_tail_support_for_post,
                    center_support_mask=center_chest_strand_mask,
                    face_bbox=face_bbox,
                    cutoff_y=cutoff_y,
                    hair_length=hair_length,
                )
                if (
                    lateral_neck_preserve_for_post is not None
                    and lateral_neck_preserve_for_post.shape == (H, W)
                ):
                    removal_mask = np.clip(
                        removal_mask - lateral_neck_preserve_for_post * 0.70,
                        0.0,
                        1.0,
                    )
                if float(removal_mask.sum()) > 0.0:
                    x1f, y1f, x2f, y2f = [int(v) for v in face_bbox]
                    face_w = max(int(x2f - x1f), 1)
                    face_h = max(int(y2f - y1f), 1)
                    face_cx = int(0.5 * (x1f + x2f))

                    removal_u8 = (
                        np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.08
                    ).astype(np.uint8) * 255
                    lane_gate_u8 = np.zeros((H, W), dtype=np.uint8)
                    gate_top = max(0, int(y2f + face_h * 0.02))
                    gate_bottom = min(H, int(y2f + face_h * 1.20))
                    left_outer = max(0, int(x1f - face_w * 0.38))
                    left_inner = max(left_outer + 1, int(face_cx - face_w * 0.14))
                    right_inner = min(W - 1, int(face_cx + face_w * 0.14))
                    right_outer = min(W, int(x2f + face_w * 0.38))
                    if gate_top < gate_bottom:
                        lane_gate_u8[gate_top:gate_bottom, left_outer:left_inner] = 255
                        lane_gate_u8[gate_top:gate_bottom, right_inner:right_outer] = (
                            255
                        )

                    center_anchor_u8 = np.zeros((H, W), dtype=np.uint8)
                    if float(center_chest_strand_removal_mask.sum()) > 0.0:
                        center_anchor_u8 = (
                            np.clip(
                                center_chest_strand_removal_mask.astype(np.float32),
                                0.0,
                                1.0,
                            )
                            > 0.08
                        ).astype(np.uint8) * 255
                        center_band_u8 = np.zeros((H, W), dtype=np.uint8)
                        center_left = max(0, int(face_cx - face_w * 0.10))
                        center_right = min(W, int(face_cx + face_w * 0.10))
                        center_bottom = min(H, int(y2f + face_h * 0.55))
                        if gate_top < center_bottom and center_left < center_right:
                            center_band_u8[
                                gate_top:center_bottom, center_left:center_right
                            ] = 255
                            lane_gate_u8 = cv2.bitwise_or(
                                lane_gate_u8,
                                cv2.bitwise_and(center_anchor_u8, center_band_u8),
                            )

                    shoulder_cross_bridge_u8 = np.zeros((H, W), dtype=np.uint8)
                    if float(shoulder_cross_bridge_for_post.sum()) > 0.0:
                        shoulder_cross_bridge_u8 = (
                            np.clip(
                                shoulder_cross_bridge_for_post.astype(np.float32),
                                0.0,
                                1.0,
                            )
                            > 0.08
                        ).astype(np.uint8) * 255
                        lane_gate_u8 = cv2.bitwise_or(
                            lane_gate_u8,
                            shoulder_cross_bridge_u8,
                        )

                    removal_u8 = cv2.bitwise_and(removal_u8, lane_gate_u8)
                    tail_rescue_u8 = np.zeros((H, W), dtype=np.uint8)
                    rescue_min_area = max(18, int(face_w * face_h * 0.00012))
                    rescue_max_area = max(4200, int(face_w * face_h * 0.11))
                    rescue_max_width = max(88, int(face_w * 0.36))
                    rescue_min_height = max(34, int(face_h * 0.22))
                    rescue_side_offset = max(18, int(face_w * 0.16))
                    rescue_top_max = min(H, int(y2f + face_h * 0.48))
                    rescue_bottom_min = min(H, int(y2f + face_h * 0.30))
                    num_rescue_labels, rescue_labels, rescue_stats, rescue_centroids = (
                        cv2.connectedComponentsWithStats(
                            (removal_u8 > 0).astype(np.uint8),
                            8,
                        )
                    )
                    for label in range(1, num_rescue_labels):
                        area = int(rescue_stats[label, cv2.CC_STAT_AREA])
                        width = int(rescue_stats[label, cv2.CC_STAT_WIDTH])
                        height = int(rescue_stats[label, cv2.CC_STAT_HEIGHT])
                        top = int(rescue_stats[label, cv2.CC_STAT_TOP])
                        bottom = top + height
                        comp_cx = float(rescue_centroids[label][0])
                        if area < rescue_min_area or area > rescue_max_area:
                            continue
                        if width > rescue_max_width or height < rescue_min_height:
                            continue
                        if top > rescue_top_max:
                            continue
                        if bottom < rescue_bottom_min:
                            continue

                        component_u8 = np.zeros((H, W), dtype=np.uint8)
                        component_u8[rescue_labels == label] = 255
                        lane_pixels = int(
                            np.logical_and(component_u8 > 0, lane_gate_u8 > 0).sum()
                        )
                        if lane_pixels < max(12, int(area * 0.24)):
                            continue

                        center_pixels = int(
                            np.logical_and(component_u8 > 0, center_anchor_u8 > 0).sum()
                        )
                        is_side_component = (
                            abs(comp_cx - float(face_cx)) >= rescue_side_offset
                        )
                        if is_side_component:
                            if height < max(rescue_min_height, int(width * 0.78)):
                                continue
                        else:
                            if center_pixels <= 0:
                                continue
                            if width > max(48, int(face_w * 0.18)):
                                continue
                        tail_rescue_u8 = cv2.bitwise_or(
                            tail_rescue_u8,
                            component_u8,
                        )

                    if (
                        shoulder_hair_forbid_for_post is not None
                        and shoulder_hair_forbid_for_post.shape == (H, W)
                    ):
                        shoulder_forbid_u8 = cv2.dilate(
                            (
                                np.clip(
                                    shoulder_hair_forbid_for_post.astype(np.float32),
                                    0.0,
                                    1.0,
                                )
                                > 0.08
                            ).astype(np.uint8)
                            * 255,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 17)),
                            iterations=1,
                        )
                        removal_u8 = cv2.bitwise_and(
                            removal_u8,
                            cv2.bitwise_not(shoulder_forbid_u8),
                        )
                    if int((tail_rescue_u8 > 0).sum()) > 0:
                        removal_u8 = cv2.bitwise_or(
                            removal_u8,
                            cv2.bitwise_and(tail_rescue_u8, lane_gate_u8),
                        )
                    if int((shoulder_cross_bridge_u8 > 0).sum()) > 0:
                        removal_u8 = cv2.bitwise_or(
                            removal_u8,
                            cv2.bitwise_and(shoulder_cross_bridge_u8, lane_gate_u8),
                        )

                    filtered_removal_u8 = np.zeros((H, W), dtype=np.uint8)
                    min_lane_area = max(28, int(face_w * face_h * 0.00018))
                    max_lane_width = max(136, int(face_w * 0.34))
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                        (removal_u8 > 0).astype(np.uint8),
                        8,
                    )
                    for label in range(1, num_labels):
                        width = int(stats[label, cv2.CC_STAT_WIDTH])
                        area = int(stats[label, cv2.CC_STAT_AREA])
                        if area < min_lane_area:
                            continue

                        component_u8 = np.zeros((H, W), dtype=np.uint8)
                        component_u8[labels == label] = 255
                        center_pixels = int(
                            np.logical_and(component_u8 > 0, center_anchor_u8 > 0).sum()
                        )
                        rescue_pixels = int(
                            np.logical_and(component_u8 > 0, tail_rescue_u8 > 0).sum()
                        )
                        bridge_pixels = int(
                            np.logical_and(
                                component_u8 > 0, shoulder_cross_bridge_u8 > 0
                            ).sum()
                        )
                        if (
                            width > max_lane_width
                            and center_pixels <= 0
                            and rescue_pixels < max(10, int(area * 0.10))
                            and bridge_pixels <= 0
                        ):
                            continue
                        filtered_removal_u8 = cv2.bitwise_or(
                            filtered_removal_u8,
                            component_u8,
                        )

                    removal_mask = filtered_removal_u8.astype(np.float32) / 255.0
            if (
                residual_side_hair_lane_removal_for_post.shape == (H, W)
                and float(residual_side_hair_lane_removal_for_post.sum()) > 0.0
            ):
                removal_mask = np.maximum(
                    removal_mask.astype(np.float32),
                    residual_side_hair_lane_removal_for_post.astype(np.float32),
                ).astype(np.float32)
            removal_mask_for_post = removal_mask.copy()

            _store_mask(
                "pipeline_lower_tail_removal_extension_mask",
                lower_tail_removal_extension,
            )
            _store_mask(
                "pipeline_center_chest_strand_removal_mask",
                center_chest_strand_removal_mask,
            )
            _store_mask(
                "pipeline_shoulder_cloth_release_mask", shoulder_cloth_release_for_post
            )
            _store_mask(
                "pipeline_short_shoulder_cross_bridge_mask",
                shoulder_cross_bridge_for_post,
            )
            gen_mask = np.zeros((H, W), dtype=np.float32)
            short_generation_seed_mask_for_debug: Optional[np.ndarray] = None
            if hair_length == "short":
                x1f, y1f, x2f, y2f = [int(v) for v in face_bbox]
                face_w = max(int(x2f - x1f), 1)
                face_h = max(int(y2f - y1f), 1)
                face_cx = int(0.5 * (x1f + x2f))
                seed_top = max(0, int(head_y1))
                seed_bottom = min(H, int(cutoff_y + face_h * 0.08))
                seed_left = max(0, int(x1f - face_w * 0.76))
                seed_right = min(W, int(x2f + face_w * 0.76))
                short_seed_u8 = np.zeros((H, W), dtype=np.uint8)

                if seed_top < seed_bottom and seed_left < seed_right:
                    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
                    corridor_u8[seed_top:seed_bottom, seed_left:seed_right] = 255

                    crown_center_y = int(
                        max(seed_top + 1, min(seed_bottom - 1, y1f + face_h * 0.09))
                    )
                    crown_axes_y = max(18, int((seed_bottom - seed_top) * 0.30))
                    crown_axes_x = max(24, int(face_w * 0.68))
                    cv2.ellipse(
                        short_seed_u8,
                        (face_cx, crown_center_y),
                        (crown_axes_x, crown_axes_y),
                        0,
                        0,
                        360,
                        255,
                        thickness=-1,
                    )

                    side_top = max(seed_top, int(y1f + face_h * 0.10))
                    side_bottom = min(seed_bottom, int(y2f + face_h * 0.08))
                    side_inner_gap = max(18, int(face_w * 0.22))
                    side_outer_span = max(22, int(face_w * 0.50))
                    left_outer = max(0, int(face_cx - side_outer_span))
                    left_inner = max(left_outer + 1, int(face_cx - side_inner_gap))
                    right_inner = min(W - 1, int(face_cx + side_inner_gap))
                    right_outer = min(W, int(face_cx + side_outer_span))
                    if side_top < side_bottom:
                        short_seed_u8[side_top:side_bottom, left_outer:left_inner] = 255
                        short_seed_u8[side_top:side_bottom, right_inner:right_outer] = (
                            255
                        )

                    upper_prior_u8 = cv2.dilate(
                        (
                            np.clip(base_prior.astype(np.float32), 0.0, 1.0) > 0.08
                        ).astype(np.uint8)
                        * 255,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                        iterations=1,
                    )
                    prior_cap_y = min(H, int(cutoff_y + face_h * 0.01))
                    if prior_cap_y < H:
                        upper_prior_u8[prior_cap_y:, :] = 0
                    short_seed_u8 = cv2.bitwise_or(short_seed_u8, upper_prior_u8)
                    short_seed_u8 = cv2.bitwise_and(short_seed_u8, corridor_u8)
                    short_seed_u8 = cv2.morphologyEx(
                        short_seed_u8,
                        cv2.MORPH_CLOSE,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 13)),
                    )
                    short_seed_u8 = cv2.dilate(
                        short_seed_u8,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7)),
                        iterations=1,
                    )

                if int((short_seed_u8 > 0).sum()) >= 120:
                    gen_mask = short_seed_u8.astype(np.float32) / 255.0
                else:
                    fallback_bottom = min(H, int(cutoff_y + face_h * 0.08))
                    fallback_left = max(0, int(x1f - face_w * 0.70))
                    fallback_right = min(W, int(x2f + face_w * 0.70))
                    if seed_top < fallback_bottom and fallback_left < fallback_right:
                        gen_mask[
                            seed_top:fallback_bottom, fallback_left:fallback_right
                        ] = 1.0
                short_generation_seed_mask_for_debug = gen_mask.copy()
            if hair_length == "short":
                # SHORT 변환 시 기존 긴머리 영역(SAM2)을 인페인팅 범위에 포함시켜야 지울 수 있음
                if (
                    hair_mask_for_removal is not None
                    and hair_mask_for_removal.shape == (H, W)
                ):
                    gen_mask = np.maximum(
                        gen_mask,
                        np.clip(
                            hair_mask_for_removal.astype(np.float32) * 0.92, 0.0, 1.0
                        ),
                    )
                # 리무벌 마스크(LaMa 타겟)도 포함
                if (
                    removal_mask_for_post is not None
                    and removal_mask_for_post.shape == (H, W)
                ):
                    gen_mask = np.maximum(
                        gen_mask,
                        np.clip(
                            removal_mask_for_post.astype(np.float32) * 0.85, 0.0, 1.0
                        ),
                    )

            gen_mask = np.clip(gen_mask - protect_mask_for_sd, 0.0, 1.0)
            gen_mask = np.clip(gen_mask - cloth_generation_guard, 0.0, 1.0)
            if (
                use_upper_clothes_overwrite
                and effective_upper_clothes_overwrite_mask.shape == (H, W)
                and not short_torso_generation_freeze_active
            ):
                gen_mask = np.maximum(
                    gen_mask, effective_upper_clothes_overwrite_mask
                ).astype(np.float32)

            if hair_length == "short":
                below_bob_generation_block_for_post = (
                    self._build_short_below_bob_generation_block_mask(
                        removal_mask=removal_mask_for_post,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y,
                        hair_length=hair_length,
                        support_mask=lower_tail_support_for_post,
                        force_keep_mask=upper_clothes_overwrite_core_mask,
                    )
                )
                below_bob_cloth_restore_for_post = (
                    self._build_short_below_bob_cloth_restore_mask(
                        removal_mask=removal_mask_for_post,
                        cloth_mask=cloth_mask_dilated,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y,
                        hair_length=hair_length,
                        support_mask=lower_tail_support_for_post,
                    )
                )
                # 억제 마스크 가중치 대폭 완화 (1.75 -> 0.42, 1.55 -> 0.38)
                if (
                    below_bob_generation_block_for_post is not None
                    and below_bob_generation_block_for_post.shape == (H, W)
                ):
                    gen_mask = np.clip(
                        gen_mask - below_bob_generation_block_for_post * 0.42, 0.0, 1.0
                    )
                if (
                    below_bob_cloth_restore_for_post is not None
                    and below_bob_cloth_restore_for_post.shape == (H, W)
                ):
                    gen_mask = np.clip(
                        gen_mask - below_bob_cloth_restore_for_post * 0.38, 0.0, 1.0
                    )

            if (
                shoulder_protect_for_post is not None
                and shoulder_protect_for_post.shape == (H, W)
            ):
                gen_mask = np.clip(
                    gen_mask
                    - (
                        shoulder_protect_for_post
                        * (0.12 if hair_length == "short" else 0.28)
                    ),
                    0.0,
                    1.0,
                )
            if (
                shoulder_hair_forbid_for_post is not None
                and shoulder_hair_forbid_for_post.shape == (H, W)
            ):
                gen_mask = np.clip(
                    gen_mask
                    - shoulder_hair_forbid_for_post
                    * (0.45 if hair_length == "short" else 1.10),
                    0.0,
                    1.0,
                )
            if (
                neckline_preserve_for_post is not None
                and neckline_preserve_for_post.shape == (H, W)
            ):
                gen_mask = np.clip(
                    gen_mask
                    - neckline_preserve_for_post
                    * (0.08 if hair_length == "short" else 0.18),
                    0.0,
                    1.0,
                )
            if hair_length == "short":
                gen_u8 = (np.clip(gen_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(
                    np.uint8
                ) * 255
                taper_top = max(0, int(y2f - face_h * 0.04))
                taper_bottom = min(H, int(y2f + face_h * 0.10))
                if taper_top < taper_bottom:
                    lower_taper_u8 = np.zeros((H, W), dtype=np.uint8)
                    for y in range(taper_top, taper_bottom):
                        progress = (
                            0.0
                            if taper_bottom <= taper_top + 1
                            else float(y - taper_top)
                            / float(taper_bottom - taper_top - 1)
                        )
                        half_width = max(
                            int(face_w * 0.30),
                            int(round(face_w * (0.54 - 0.12 * progress))),
                        )
                        left_x = max(0, face_cx - half_width)
                        right_x = min(W, face_cx + half_width)
                        lower_taper_u8[y, left_x:right_x] = 255
                    upper_keep_u8 = np.zeros((H, W), dtype=np.uint8)
                    upper_keep_u8[:taper_top, :] = 255
                    gen_u8 = cv2.bitwise_and(
                        gen_u8,
                        cv2.bitwise_or(upper_keep_u8, lower_taper_u8),
                    )
                    gen_u8 = cv2.morphologyEx(
                        gen_u8,
                        cv2.MORPH_CLOSE,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                    )
                    gen_mask = gen_u8.astype(np.float32) / 255.0
                gen_mask = cv2.erode(
                    gen_mask,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                    iterations=1,
                )
            gen_mask_before_upper_overwrite = gen_mask.astype(np.float32).copy()
            if (
                use_upper_clothes_overwrite
                and effective_upper_clothes_overwrite_mask.shape == (H, W)
                and not short_torso_generation_freeze_active
            ):
                overwrite_restore_mask = effective_upper_clothes_overwrite_mask.astype(
                    np.float32
                )
                if protect_mask_for_sd.shape == (H, W):
                    overwrite_restore_mask = np.clip(
                        overwrite_restore_mask - protect_mask_for_sd.astype(np.float32),
                        0.0,
                        1.0,
                    )
                gen_mask = np.maximum(
                    gen_mask.astype(np.float32), overwrite_restore_mask
                ).astype(np.float32)
            gen_mask_after_upper_overwrite = gen_mask.astype(np.float32).copy()
            if (
                use_upper_clothes_overwrite
                and effective_upper_clothes_overwrite_core_mask.shape == (H, W)
                and not short_torso_generation_freeze_active
            ):
                core_restore_mask = effective_upper_clothes_overwrite_core_mask.astype(
                    np.float32
                )
                if protect_mask_for_sd.shape == (H, W):
                    core_restore_mask = np.clip(
                        core_restore_mask - protect_mask_for_sd.astype(np.float32),
                        0.0,
                        1.0,
                    )
                overwrite_core_restore_mask_for_debug = np.maximum(
                    overwrite_core_restore_mask_for_debug.astype(np.float32),
                    core_restore_mask.astype(np.float32),
                ).astype(np.float32)
                gen_mask = np.maximum(
                    gen_mask.astype(np.float32), core_restore_mask
                ).astype(np.float32)
            gen_mask_after_initial_core = gen_mask.astype(np.float32).copy()
            composite_bangs_release_mask = np.zeros((H, W), dtype=np.float32)
            if float(bangs_restore_for_sd.sum()) > 0.0:
                soft_bangs_generation_mask = self._build_soft_bangs_generation_mask(
                    bangs_restore_for_sd,
                    face_bbox=face_bbox,
                    hair_length=hair_length,
                    subject_gender=subject_gender_mode,
                    fringe_requested=bangs_requested,
                )
                if float(soft_bangs_generation_mask.sum()) > 0.0:
                    preserve_fringe_detail = bool(
                        subject_gender_mode == "male"
                        and bangs_requested
                        and hair_length in ("short", "medium")
                    )
                    _bangs_dilate_k = (
                        (5, 5)
                        if preserve_fringe_detail and hair_length == "short"
                        else (
                            (5, 7)
                            if preserve_fringe_detail
                            else (5, 7) if hair_length == "short" else (7, 9)
                        )
                    )
                    _bangs_sx = (
                        1.8
                        if preserve_fringe_detail and hair_length == "short"
                        else (
                            2.0
                            if preserve_fringe_detail
                            else 2.2 if hair_length == "short" else 2.4
                        )
                    )
                    _bangs_sy = (
                        2.1
                        if preserve_fringe_detail and hair_length == "short"
                        else (
                            2.3
                            if preserve_fringe_detail
                            else 2.6 if hair_length == "short" else 2.8
                        )
                    )
                    _bangs_u8 = cv2.dilate(
                        (
                            np.clip(
                                soft_bangs_generation_mask.astype(np.float32), 0.0, 1.0
                            )
                            > 0.04
                        ).astype(np.uint8)
                        * 255,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, _bangs_dilate_k),
                        iterations=1,
                    )
                    soft_bangs_generation_mask = cv2.GaussianBlur(
                        _bangs_u8.astype(np.float32) / 255.0,
                        (0, 0),
                        sigmaX=_bangs_sx,
                        sigmaY=_bangs_sy,
                    ).astype(np.float32)
                    soft_bangs_generation_mask = np.clip(
                        soft_bangs_generation_mask * 1.10, 0.0, 1.0
                    )
                composite_bangs_release_mask = np.clip(
                    soft_bangs_generation_mask.astype(np.float32)
                    * (
                        1.24
                        if subject_gender_mode == "male"
                        and bangs_requested
                        and hair_length in ("short", "medium")
                        else 1.15
                    ),
                    0.0,
                    1.0,
                )
                # ── composite release mask 이마 이내로 cap ───────────────────
                # requested_front_coverage_mask 가 y1+0.56*face_h 까지 내려가고
                # 이게 soft_bangs_generation_mask → composite_bangs_release_mask로
                # 이어지면 보호 마스크가 눈/코 레벨까지 해제되어, 생성된 앞머리가
                # 얼굴 중간까지 찍히는 dark artifact 유발.
                # gen_mask(SD inpaint 범위)는 깊게 유지하고 release만 이마로 제한.
                if (
                    face_bbox is not None
                    and float(composite_bangs_release_mask.sum()) > 0.0
                ):
                    _rx1, _ry1, _rx2, _ry2 = face_bbox
                    _rface_h = max(int(_ry2 - _ry1), 1)
                    # 눈썹 위까지 해제. bangs_requested(앞머리 내림) 요청 시 이마 전체를
                    # 덮어야 하므로 눈썹 직전(≈0.34*face_h)까지 release를 확장한다.
                    # 기본값(0.25)은 중앙 center_lobe만 이마 깊이까지 가고 측면 이마는
                    # 원본 스킨이 그대로 노출되던 문제가 있었다.
                    _bangs_release_extended = bool(
                        bangs_requested and hair_length in ("short", "medium")
                    )
                    if subject_gender_mode == "male":
                        if _bangs_release_extended:
                            _rel_ratio = (
                                0.30 if hair_length == "short" else 0.27
                            )
                        else:
                            _rel_ratio = (
                                0.25
                                if hair_length == "short"
                                else 0.22 if hair_length == "medium" else 0.20
                            )
                    else:
                        if _bangs_release_extended:
                            _rel_ratio = (
                                0.33 if hair_length == "short" else 0.29
                            )
                        else:
                            _rel_ratio = (
                                0.30
                                if hair_length == "short"
                                else 0.26 if hair_length == "medium" else 0.22
                            )
                    _rel_y_limit = int(_ry1 + _rface_h * _rel_ratio)
                    _rel_cap = np.zeros_like(composite_bangs_release_mask)
                    _rel_cap[:_rel_y_limit, :] = 1.0
                    if _bangs_release_extended:
                        # soft vertical fade near y_limit so the composite doesn't
                        # show a hard horizontal band at the eyebrow cap.
                        # 기존 5% fade는 너무 짧아 갈색 사각형 띠 artifact가 남았음.
                        # 12%로 확장하여 release mask 강도가 충분히 점진적으로 떨어지게 함.
                        _rel_fade_h = max(20, int(_rface_h * 0.12))
                        _rel_fade_bot = min(
                            composite_bangs_release_mask.shape[0],
                            _rel_y_limit + _rel_fade_h,
                        )
                        if _rel_fade_bot > _rel_y_limit:
                            _rel_cap[_rel_y_limit:_rel_fade_bot, :] = np.linspace(
                                1.0,
                                0.0,
                                _rel_fade_bot - _rel_y_limit,
                                dtype=np.float32,
                            )[:, np.newaxis]
                    if (
                        subject_gender_mode == "male"
                        and bangs_requested
                        and hair_length in ("short", "medium")
                    ):
                        _center_lobe_u8 = np.zeros_like(
                            composite_bangs_release_mask, dtype=np.uint8
                        )
                        _rcx = int(0.5 * (_rx1 + _rx2))
                        _center = (
                            _rcx,
                            int(
                                _ry1
                                + _rface_h
                                * (0.24 if hair_length == "short" else 0.22)
                            ),
                        )
                        _axes = (
                            max(
                                14,
                                int(
                                    (_rx2 - _rx1)
                                    * (0.18 if hair_length == "short" else 0.16)
                                ),
                            ),
                            max(
                                10,
                                int(_rface_h * (0.11 if hair_length == "short" else 0.09)),
                            ),
                        )
                        cv2.ellipse(
                            _center_lobe_u8,
                            _center,
                            _axes,
                            0,
                            0,
                            360,
                            255,
                            -1,
                        )
                        _center_lobe = cv2.GaussianBlur(
                            _center_lobe_u8.astype(np.float32) / 255.0,
                            (0, 0),
                            sigmaX=8.0,
                            sigmaY=9.0,
                        ).astype(np.float32)
                        # center_lobe가 rel_cap fade 바깥으로 삐져나와 이마
                        # 중앙에 W자 모양(검은 얼룩) artifact를 만드는 문제 방지:
                        # rel_cap의 vertical envelope와 multiply하여 fade 영역
                        # 안쪽에서만 lobe가 효과를 발휘하도록 제한.
                        _center_lobe = _center_lobe * _rel_cap
                        _rel_cap = np.maximum(
                            _rel_cap,
                            np.clip(_center_lobe * 1.15, 0.0, 1.0),
                        )
                    composite_bangs_release_mask = (
                        composite_bangs_release_mask * _rel_cap
                    )
                    logger.debug(
                        "[SDPipeline] composite_bangs_release_mask capped at y=%d "
                        "(face_top=%d face_h=%d ratio=%.2f gender=%s)",
                        _rel_y_limit,
                        _ry1,
                        _rface_h,
                        _rel_ratio,
                        subject_gender_mode,
                    )
                gen_mask = np.maximum(
                    gen_mask,
                    np.clip(soft_bangs_generation_mask.astype(np.float32), 0.0, 1.0),
                )
                # soft_bangs_generation_mask는 상단(이마 머리선)이 vertical fade로
                # alpha 0.22까지 약해져 SD가 원본 이마 스킨을 거의 덮지 못한다.
                # front=down/bangs가 명시적으로 요청된 경우 requested_front_coverage_mask를
                # full strength로 병합해 이마 전체 영역을 강하게 inpaint한다.
                # 단, 눈썹/눈 아래로 내려가면 SD가 머리를 눈 위에 생성해 ghost eyebrow
                # artifact가 생기므로 눈썹 위(≈y1+0.30*face_h)로 수직 cap한다.
                if (
                    bangs_requested
                    and face_bbox is not None
                    and float(requested_front_coverage_mask.sum()) > 0.0
                ):
                    _gm_rx1, _gm_ry1, _gm_rx2, _gm_ry2 = face_bbox
                    _gm_face_h = max(int(_gm_ry2 - _gm_ry1), 1)
                    # rel_cap과 정렬: rel은 0.30에서 시작해 0.42까지 fade.
                    # gen_cap은 SD inpaint 범위 → 약간 더 위/아래로 넓혀 두 mask
                    # 경계가 동일 위치에서 갑자기 끝나지 않도록 함 (사각 띠 방지).
                    _gm_top_ratio = (
                        0.27 if hair_length == "short" else 0.24
                    )
                    _gm_bot_ratio = (
                        0.42 if hair_length == "short" else 0.38
                    )
                    _gm_y_top = int(_gm_ry1 + _gm_face_h * _gm_top_ratio)
                    _gm_y_bot = int(_gm_ry1 + _gm_face_h * _gm_bot_ratio)
                    _gm_cap = np.ones_like(requested_front_coverage_mask)
                    if _gm_y_bot > _gm_y_top:
                        _fade_h = _gm_y_bot - _gm_y_top
                        _gm_cap[_gm_y_top:_gm_y_bot, :] = np.linspace(
                            1.0, 0.0, _fade_h, dtype=np.float32
                        )[:, np.newaxis]
                    _gm_cap[_gm_y_bot:, :] = 0.0
                    _gm_merge = (
                        np.clip(
                            requested_front_coverage_mask.astype(np.float32),
                            0.0,
                            1.0,
                        )
                        * _gm_cap
                    )
                    gen_mask = np.maximum(gen_mask, _gm_merge)
                _store_mask(
                    "pipeline_bangs_generation_soft_mask", soft_bangs_generation_mask
                )
                _store_mask(
                    "pipeline_bangs_composite_release_mask",
                    composite_bangs_release_mask,
                )
            gen_mask_before_final_core = gen_mask.astype(np.float32).copy()
            if (
                use_upper_clothes_overwrite
                and effective_upper_clothes_overwrite_core_mask.shape == (H, W)
                and not short_torso_generation_freeze_active
            ):
                core_restore_mask = effective_upper_clothes_overwrite_core_mask.astype(
                    np.float32
                )
                if protect_mask_for_sd.shape == (H, W):
                    core_restore_mask = np.clip(
                        core_restore_mask - protect_mask_for_sd.astype(np.float32),
                        0.0,
                        1.0,
                    )
                overwrite_core_restore_mask_for_debug = np.maximum(
                    overwrite_core_restore_mask_for_debug.astype(np.float32),
                    core_restore_mask.astype(np.float32),
                ).astype(np.float32)
                gen_mask = np.maximum(
                    gen_mask.astype(np.float32), core_restore_mask
                ).astype(np.float32)
            if (
                short_torso_generation_freeze_active
                and short_torso_generation_freeze_mask.shape == (H, W)
            ):
                gen_mask = np.clip(
                    gen_mask.astype(np.float32)
                    - short_torso_generation_freeze_mask.astype(np.float32) * 1.45,
                    0.0,
                    1.0,
                ).astype(np.float32)
            if hair_length == "short" and short_generation_tail_release_mask.shape == (
                H,
                W,
            ):
                if short_generation_tail_release_px >= 80:
                    gen_mask = np.maximum(
                        gen_mask.astype(np.float32),
                        np.clip(
                            short_generation_tail_release_mask.astype(np.float32)
                            * 1.08,
                            0.0,
                            1.0,
                        ),
                    ).astype(np.float32)
            gen_mask_after_final_core = gen_mask.astype(np.float32).copy()
            _store_mask("pipeline_short_removal_mask", removal_mask_for_post)
            _store_mask(
                "pipeline_short_generation_seed_mask",
                short_generation_seed_mask_for_debug,
            )
            _store_mask("pipeline_short_generation_mask", gen_mask)
            _store_mask(
                "pipeline_short_generation_mask_before_upper_overwrite",
                gen_mask_before_upper_overwrite,
            )
            _store_mask(
                "pipeline_short_generation_mask_after_upper_overwrite",
                gen_mask_after_upper_overwrite,
            )
            _store_mask(
                "pipeline_short_generation_mask_after_initial_core",
                gen_mask_after_initial_core,
            )
            _store_mask(
                "pipeline_short_generation_mask_before_final_core",
                gen_mask_before_final_core,
            )
            _store_mask(
                "pipeline_short_generation_mask_after_final_core",
                gen_mask_after_final_core,
            )
            _store_mask(
                "pipeline_overwrite_core_restore_mask",
                overwrite_core_restore_mask_for_debug,
            )
            _store_mask(
                "pipeline_short_below_bob_generation_block_mask",
                below_bob_generation_block_for_post,
            )
            _store_mask(
                "pipeline_short_below_bob_cloth_restore_mask",
                below_bob_cloth_restore_for_post,
            )

            logger.info(
                f"[SDPipeline] 전략2: removal_px={removal_mask_for_post.sum():.0f}, "
                f"gen_px={gen_mask.sum():.0f}, cutoff_y={cutoff_y}, "
                f"head_box=({head_x1},{head_y1})-({head_x2},{head_y2})"
            )

            bg_mode = self.config.bg_fill_mode
            logger.info(f"[SDPipeline] bg_fill_mode={bg_mode}")
            tail_core_mask = np.zeros((H, W), dtype=np.float32)

            if removal_mask.sum() > 50:
                removal_u8 = (removal_mask > 0.5).astype(np.uint8) * 255
                removal_u8_dilated = cv2.dilate(
                    removal_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                    iterations=1,
                )

                inpaint_radius = 8 if hair_length == "short" else 10
                img_rgb_ns = cv2.inpaint(
                    img_rgb,
                    removal_u8_dilated,
                    inpaintRadius=inpaint_radius,
                    flags=cv2.INPAINT_NS,
                )
                img_rgb_telea = cv2.inpaint(
                    img_rgb,
                    removal_u8_dilated,
                    inpaintRadius=max(3, inpaint_radius),
                    flags=cv2.INPAINT_TELEA,
                )
                img_rgb_filled = cv2.addWeighted(
                    img_rgb_ns, 0.25, img_rgb_telea, 0.75, 0.0
                )

                removal_px = int((removal_u8 > 0).sum())
                max_clone_px = int(H * W * 0.18)
                if 50 <= removal_px <= max_clone_px:
                    ys, xs = np.where(removal_u8 > 0)
                    c_x = int((xs.min() + xs.max()) * 0.5)
                    c_y = int((ys.min() + ys.max()) * 0.5)
                    src_bgr = cv2.cvtColor(img_rgb_filled, cv2.COLOR_RGB2BGR)
                    dst_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    try:
                        cloned_bgr = cv2.seamlessClone(
                            src_bgr,
                            dst_bgr,
                            removal_u8,
                            (c_x, c_y),
                            cv2.NORMAL_CLONE,
                        )
                        img_rgb_cleaned = cv2.cvtColor(cloned_bgr, cv2.COLOR_BGR2RGB)
                    except Exception:
                        soft_alpha = removal_u8.astype(np.float32) / 255.0
                        soft_alpha = cv2.GaussianBlur(soft_alpha, (0, 0), sigmaX=1.2)[
                            ..., np.newaxis
                        ]
                        soft_alpha = np.clip((soft_alpha - 0.28) / 0.72, 0.0, 1.0)
                        img_rgb_cleaned = (
                            (
                                img_rgb_filled.astype(np.float32) * soft_alpha
                                + img_rgb.astype(np.float32) * (1.0 - soft_alpha)
                            )
                            .clip(0, 255)
                            .astype(np.uint8)
                        )
                else:
                    soft_alpha = removal_u8.astype(np.float32) / 255.0
                    soft_alpha = cv2.GaussianBlur(soft_alpha, (0, 0), sigmaX=1.2)[
                        ..., np.newaxis
                    ]
                    soft_alpha = np.clip((soft_alpha - 0.28) / 0.72, 0.0, 1.0)
                    img_rgb_cleaned = (
                        (
                            img_rgb_filled.astype(np.float32) * soft_alpha
                            + img_rgb.astype(np.float32) * (1.0 - soft_alpha)
                        )
                        .clip(0, 255)
                        .astype(np.uint8)
                    )

                cloth_u8 = (cloth_mask_dilated > 0.45).astype(np.uint8) * 255
                cloth_overlap = cv2.bitwise_and(removal_u8, cloth_u8)
                protect_u8 = (protect_mask_for_sd > 0.2).astype(np.uint8) * 255
                cloth_overlap = cv2.bitwise_and(
                    cloth_overlap, cv2.bitwise_not(protect_u8)
                )
                if (
                    not skip_source_cloth_preclean
                    and not source_garment_prepass_enabled
                    and int((cloth_overlap > 0).sum()) >= 80
                ):
                    cloth_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                    cloth_overlap = cv2.dilate(cloth_overlap, cloth_k, iterations=1)
                    cloth_ns = cv2.inpaint(
                        img_rgb, cloth_overlap, inpaintRadius=4, flags=cv2.INPAINT_NS
                    )
                    cloth_te = cv2.inpaint(
                        img_rgb, cloth_overlap, inpaintRadius=3, flags=cv2.INPAINT_TELEA
                    )
                    cloth_refill = cv2.addWeighted(cloth_ns, 0.40, cloth_te, 0.60, 0.0)
                    cloth_alpha = cloth_overlap.astype(np.float32) / 255.0
                    cloth_alpha = cv2.GaussianBlur(
                        cloth_alpha,
                        (0, 0),
                        sigmaX=2.2,
                        sigmaY=2.2,
                    )[..., np.newaxis]
                    cloth_alpha = np.clip(cloth_alpha * 0.92, 0.0, 1.0)
                    img_rgb_cleaned = (
                        (
                            cloth_refill.astype(np.float32) * cloth_alpha
                            + img_rgb_cleaned.astype(np.float32) * (1.0 - cloth_alpha)
                        )
                        .clip(0, 255)
                        .astype(np.uint8)
                    )
                    logger.info(
                        f"[SDPipeline] cloth overlap 복원 적용: pixels={int((cloth_overlap > 0).sum())}"
                    )

                if hair_length == "short" and not disable_short_postprocess_experiment:
                    try:
                        tail_core_mask = self._build_short_tail_core_mask(
                            removal_mask=removal_mask,
                            face_bbox=face_bbox,
                            cutoff_y=cutoff_y,
                            hair_length=hair_length,
                        )
                        if (
                            shoulder_protect_for_post is not None
                            and shoulder_protect_for_post.shape == (H, W)
                        ):
                            tail_core_mask = np.clip(
                                tail_core_mask - shoulder_protect_for_post * 0.16,
                                0.0,
                                1.0,
                            )
                        if (
                            neckline_preserve_for_post is not None
                            and neckline_preserve_for_post.shape == (H, W)
                        ):
                            tail_core_mask = np.clip(
                                tail_core_mask - neckline_preserve_for_post * 0.18,
                                0.0,
                                1.0,
                            )
                        tail_core_u8 = cv2.dilate(
                            (tail_core_mask > 0.10).astype(np.uint8) * 255,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 13)),
                            iterations=1,
                        )
                        if int((tail_core_u8 > 0).sum()) >= 60:
                            img_rgb_cleaned = self._lama_inpaint(
                                img_rgb_cleaned, tail_core_u8
                            )
                            logger.info(
                                f"[SDPipeline] short tail-core LaMa preclean applied: pixels={int((tail_core_u8 > 0).sum())}"
                            )
                        _store_mask("pipeline_short_tail_core_mask", tail_core_mask)

                        artifact_cleanup_mask = self._build_dark_tail_residual_mask(
                            img_rgb=img_rgb_cleaned,
                            removal_mask=removal_mask,
                            face_bbox=face_bbox,
                            cutoff_y=cutoff_y,
                            hair_length=hair_length,
                        )
                        # Use existing center_chest_strand_mask as base for front strand cleanup
                        front_strand_cleanup_mask = center_chest_strand_mask.copy()
                        front_strand_cleanup_u8 = (
                            np.clip(
                                front_strand_cleanup_mask.astype(np.float32), 0.0, 1.0
                            )
                            > 0.08
                        ).astype(np.uint8) * 255
                        artifact_cleanup_mask = np.maximum(
                            artifact_cleanup_mask,
                            np.clip(tail_core_mask * 0.55, 0.0, 1.0),
                        )
                        artifact_cleanup_mask = np.maximum(
                            artifact_cleanup_mask,
                            np.clip(front_strand_cleanup_mask * 0.46, 0.0, 1.0),
                        )
                        if float(center_chest_strand_removal_mask.sum()) > 0.0:
                            artifact_cleanup_mask = np.maximum(
                                artifact_cleanup_mask,
                                np.clip(
                                    center_chest_strand_removal_mask * 0.62, 0.0, 1.0
                                ),
                            )
                        if (
                            shoulder_protect_for_post is not None
                            and shoulder_protect_for_post.shape == (H, W)
                        ):
                            artifact_cleanup_mask = np.clip(
                                artifact_cleanup_mask
                                - shoulder_protect_for_post * 0.08,
                                0.0,
                                1.0,
                            )
                        if (
                            neckline_preserve_for_post is not None
                            and neckline_preserve_for_post.shape == (H, W)
                        ):
                            artifact_cleanup_mask = np.clip(
                                artifact_cleanup_mask
                                - neckline_preserve_for_post * 0.10,
                                0.0,
                                1.0,
                            )
                        if (
                            torso_cloth_preserve_for_post is not None
                            and torso_cloth_preserve_for_post.shape == (H, W)
                        ):
                            artifact_cleanup_mask = np.clip(
                                artifact_cleanup_mask
                                - torso_cloth_preserve_for_post * 0.12,
                                0.0,
                                1.0,
                            )
                        artifact_cleanup_mask = np.clip(
                            artifact_cleanup_mask - cloth_mask_dilated * 0.04,
                            0.0,
                            1.0,
                        )
                        artifact_anchor_u8 = np.zeros((H, W), dtype=np.uint8)
                        artifact_anchor_u8 = cv2.bitwise_or(
                            artifact_anchor_u8,
                            (
                                np.clip(tail_core_mask.astype(np.float32), 0.0, 1.0)
                                > 0.10
                            ).astype(np.uint8)
                            * 255,
                        )
                        artifact_anchor_u8 = cv2.bitwise_or(
                            artifact_anchor_u8,
                            (
                                np.clip(
                                    front_strand_cleanup_mask.astype(np.float32),
                                    0.0,
                                    1.0,
                                )
                                > 0.08
                            ).astype(np.uint8)
                            * 255,
                        )
                        if float(center_chest_strand_removal_mask.sum()) > 0.0:
                            artifact_anchor_u8 = cv2.bitwise_or(
                                artifact_anchor_u8,
                                (
                                    np.clip(
                                        center_chest_strand_removal_mask.astype(
                                            np.float32
                                        ),
                                        0.0,
                                        1.0,
                                    )
                                    > 0.08
                                ).astype(np.uint8)
                                * 255,
                            )
                        artifact_cleanup_mask_for_post = np.clip(
                            artifact_cleanup_mask,
                            0.0,
                            1.0,
                        ).astype(np.float32)
                        artifact_cleanup_u8 = (
                            artifact_cleanup_mask_for_post > 0.08
                        ).astype(np.uint8) * 255
                        artifact_cleanup_u8 = cv2.bitwise_and(
                            artifact_cleanup_u8,
                            (
                                np.clip(cloth_mask_dilated.astype(np.float32), 0.0, 1.0)
                                > 0.04
                            ).astype(np.uint8)
                            * 255,
                        )
                        artifact_cleanup_u8 = self._filter_short_center_cleanup_mask(
                            artifact_cleanup_u8,
                            face_bbox,
                            cutoff_y,
                            anchor_u8=artifact_anchor_u8,
                            top_scale=0.00,
                            bottom_scale=0.78,
                            half_w_scale=0.25,
                            shrink_half_w_scale=0.18,
                            max_area_scale=0.10,
                            max_width_scale=0.34,
                            min_height_scale=0.10,
                            center_allow_scale=0.18,
                            max_total_scale=0.05,
                        )
                        artifact_cleanup_mask_for_post = np.where(
                            artifact_cleanup_u8 > 0,
                            artifact_cleanup_mask_for_post,
                            0.0,
                        ).astype(np.float32)
                        artifact_cleanup_mask = artifact_cleanup_mask_for_post.copy()
                        artifact_cleanup_u8 = cv2.dilate(
                            artifact_cleanup_u8,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 13)),
                            iterations=1,
                        )
                        artifact_cleanup_u8 = self._filter_short_center_cleanup_mask(
                            artifact_cleanup_u8,
                            face_bbox,
                            cutoff_y,
                            anchor_u8=artifact_anchor_u8,
                            top_scale=0.00,
                            bottom_scale=0.80,
                            half_w_scale=0.26,
                            shrink_half_w_scale=0.19,
                            max_area_scale=0.11,
                            max_width_scale=0.36,
                            min_height_scale=0.10,
                            center_allow_scale=0.19,
                            max_total_scale=0.06,
                        )
                        front_strand_cleanup_u8 = (
                            front_strand_cleanup_mask > 0.08
                        ).astype(np.uint8) * 255
                        front_strand_cleanup_u8 = (
                            self._filter_short_center_cleanup_mask(
                                front_strand_cleanup_u8,
                                face_bbox,
                                cutoff_y,
                                anchor_u8=artifact_anchor_u8,
                                top_scale=0.00,
                                bottom_scale=0.82,
                                half_w_scale=0.28,
                                shrink_half_w_scale=0.20,
                                max_area_scale=0.40,
                                max_width_scale=0.80,
                                min_height_scale=0.10,
                                center_allow_scale=0.20,
                                max_total_scale=0.40,
                            )
                        )

                        # artifact_cleanup (side dark tail)
                        if int((artifact_cleanup_u8 > 0).sum()) >= 80:
                            img_rgb_cleaned = self._lama_inpaint(
                                img_rgb_cleaned, artifact_cleanup_u8
                            )
                            img_rgb_cleaned = self._cv2_cleanup_dark_tail_blob(
                                img_rgb_cleaned, artifact_cleanup_u8
                            )
                            logger.info(
                                f"[SDPipeline] short artifact LaMa/CV2 preclean applied: pixels={int((artifact_cleanup_u8 > 0).sum())}"
                            )

                        # front_strand_cleanup (chest center)
                        if int((front_strand_cleanup_u8 > 0).sum()) >= 20:
                            if use_upper_clothes_overwrite:
                                if debug_images_common is not None:
                                    _store_rgb("lama_before_base", img_rgb_cleaned)
                                img_rgb_cleaned = self._lama_inpaint(
                                    img_rgb_cleaned, front_strand_cleanup_u8
                                )
                                if debug_images_common is not None:
                                    _store_rgb("lama_after_base", img_rgb_cleaned)
                                if debug_data_common is not None:
                                    ys, xs = np.where(front_strand_cleanup_u8 > 0)
                                    bbox = (
                                        (
                                            int(np.min(xs)),
                                            int(np.min(ys)),
                                            int(np.max(xs)),
                                            int(np.max(ys)),
                                        )
                                        if len(xs) > 0
                                        else (0, 0, 0, 0)
                                    )
                                    debug_data_common["lama_front_strand_bbox"] = bbox
                                    debug_data_common["lama_front_strand_pixels"] = int(
                                        len(xs)
                                    )
                                logger.info(
                                    f"[SDPipeline] short front-strand LaMa preclean applied: pixels={int((front_strand_cleanup_u8 > 0).sum())}"
                                )
                            else:
                                img_rgb_cleaned = self._cv2_cleanup_dark_tail_blob(
                                    img_rgb_cleaned, front_strand_cleanup_u8
                                )

                        _store_mask(
                            "pipeline_front_strand_cleanup_mask",
                            front_strand_cleanup_mask,
                        )
                        _store_mask(
                            "pipeline_short_artifact_cleanup_mask",
                            artifact_cleanup_mask,
                        )
                    except Exception as e:
                        logger.warning(
                            f"[SDPipeline] short tail-core LaMa preclean 실패(무시): {e}"
                        )

                # ── v110: hair_length 무관 chest-center LaMa pre-clean ──────────────
                # short_postprocess 블록은 hair_length=="short" 케이스에서만 실행된다.
                # long/medium hair에서 chest-center 머리카락이 그대로 SD에 들어가는 문제를 수정.
                # short에서 이미 front_strand_cleanup을 처리한 경우와 중복되지 않도록
                # _needs_chest_preclean 조건으로 분기한다.
                _needs_chest_preclean = (
                    hair_length != "short" or disable_short_postprocess_experiment
                )
                if _needs_chest_preclean:
                    try:
                        chest_preclean_strand_mask = center_chest_strand_mask.copy()
                        # center_chest_strand_removal_mask도 통합
                        if float(center_chest_strand_removal_mask.sum()) > 0.0:
                            chest_preclean_strand_mask = np.maximum(
                                chest_preclean_strand_mask,
                                np.clip(
                                    center_chest_strand_removal_mask * 0.70, 0.0, 1.0
                                ),
                            ).astype(np.float32)
                        # 어깨/넥라인 보호
                        if (
                            shoulder_protect_for_post is not None
                            and shoulder_protect_for_post.shape == (H, W)
                        ):
                            chest_preclean_strand_mask = np.clip(
                                chest_preclean_strand_mask
                                - shoulder_protect_for_post * 0.60,
                                0.0,
                                1.0,
                            )
                        if (
                            neckline_preserve_for_post is not None
                            and neckline_preserve_for_post.shape == (H, W)
                        ):
                            chest_preclean_strand_mask = np.clip(
                                chest_preclean_strand_mask
                                - neckline_preserve_for_post * 0.55,
                                0.0,
                                1.0,
                            )
                        chest_preclean_u8 = (chest_preclean_strand_mask > 0.08).astype(
                            np.uint8
                        ) * 255
                        # cloth mask 교집합: 의상 바깥은 건드리지 않음
                        if cloth_mask_dilated.shape == (H, W):
                            chest_preclean_u8 = cv2.bitwise_and(
                                chest_preclean_u8,
                                (cloth_mask_dilated > 0.04).astype(np.uint8) * 255,
                            )
                        # long/medium은 가슴 중앙 corridor로 한정
                        if hair_length in ("long", "medium"):
                            x1f, y1f, x2f, y2f = [int(v) for v in face_bbox]
                            face_w_cp = max(int(x2f - x1f), 1)
                            face_h_cp = max(int(y2f - y1f), 1)
                            face_cx_cp = int(0.5 * (x1f + x2f))
                            corridor_cp = np.zeros((H, W), dtype=np.uint8)
                            cp_top = max(0, int(y2f + face_h_cp * 0.02))
                            cp_bottom = min(H, int(y2f + face_h_cp * 1.80))
                            cp_left = max(0, int(face_cx_cp - face_w_cp * 1.10))
                            cp_right = min(W, int(face_cx_cp + face_w_cp * 1.10))
                            if cp_top < cp_bottom and cp_left < cp_right:
                                corridor_cp[cp_top:cp_bottom, cp_left:cp_right] = 255
                            chest_preclean_u8 = cv2.bitwise_and(
                                chest_preclean_u8, corridor_cp
                            )
                            lama_min_px = 15
                        else:
                            lama_min_px = 20
                        chest_preclean_u8 = cv2.dilate(
                            chest_preclean_u8,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 13)),
                            iterations=1,
                        )
                        preclean_px = int((chest_preclean_u8 > 0).sum())
                        if preclean_px >= lama_min_px:
                            if debug_images_common is not None:
                                _store_rgb(
                                    "lama_chest_preclean_before", img_rgb_cleaned
                                )
                            img_rgb_cleaned = self._lama_inpaint(
                                img_rgb_cleaned, chest_preclean_u8
                            )
                            if debug_images_common is not None:
                                _store_rgb("lama_chest_preclean_after", img_rgb_cleaned)
                            if debug_data_common is not None:
                                ys_cp, xs_cp = np.where(chest_preclean_u8 > 0)
                                debug_data_common["chest_preclean_lama_bbox"] = (
                                    (
                                        int(np.min(xs_cp)),
                                        int(np.min(ys_cp)),
                                        int(np.max(xs_cp)),
                                        int(np.max(ys_cp)),
                                    )
                                    if len(xs_cp) > 0
                                    else (0, 0, 0, 0)
                                )
                                debug_data_common["chest_preclean_lama_pixels"] = (
                                    preclean_px
                                )
                                debug_data_common["chest_preclean_lama_hair_length"] = (
                                    hair_length
                                )
                            logger.info(
                                "[SDPipeline][v110] chest-center LaMa preclean applied"
                                " (hair_length=%s): pixels=%d",
                                hair_length,
                                preclean_px,
                            )
                        else:
                            logger.info(
                                "[SDPipeline][v110] chest-center LaMa preclean skipped"
                                " (hair_length=%s): px=%d < min=%d",
                                hair_length,
                                preclean_px,
                                lama_min_px,
                            )
                        _store_mask(
                            "pipeline_chest_preclean_lama_mask",
                            chest_preclean_strand_mask,
                        )
                    except Exception as e:
                        logger.warning(
                            "[SDPipeline][v110] chest-center LaMa preclean 실패(무시): %s",
                            e,
                        )
                # ── end v110 chest-center pre-clean ──────────────────────────────

                logger.info("[SDPipeline] cv2.inpaint 2-way 블렌딩 완료")
                if bg_mode == "sd":
                    try:
                        fill_seed = (
                            int(seeds[0]) if seeds else random.randint(0, 2**31 - 1)
                        )
                        face_crop_fill = self._crop_face(
                            Image.fromarray(img_rgb), face_bbox
                        )
                        img_rgb_cleaned = self._sd_refine_removed_region(
                            base_rgb=img_rgb_cleaned,
                            removal_mask=removal_mask,
                            face_bbox=face_bbox,
                            face_crop_pil=face_crop_fill,
                            protect_mask=protect_mask_for_sd,
                            cloth_mask=cloth_mask_dilated,
                            hair_length=hair_length,
                            seed=fill_seed,
                            subject_gender=subject_gender_mode,
                            white_tshirt_experiment=white_tshirt_experiment,
                        )
                        logger.info(
                            "[SDPipeline] bg_fill_mode=sd: 제거 영역 SD 보정 완료"
                        )
                    except Exception as e:
                        logger.warning(
                            f"[SDPipeline] bg_fill_mode=sd 실패, cv2 결과 사용: {e}"
                        )

                try:
                    img_rgb_cleaned = self._remove_residual_hair_below_cutoff(
                        img_rgb=img_rgb_cleaned,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y,
                        removal_mask=removal_mask,
                        shoulder_protect=shoulder_protect_for_post,
                        neckline_preserve=neckline_preserve_for_post,
                        lateral_preserve=lateral_neck_preserve_for_post,
                        hair_length=hair_length,
                        center_anchor_mask=center_chest_strand_removal_mask,
                    )
                except Exception as e:
                    logger.warning(f"[SDPipeline] residual hair 정리 실패(무시): {e}")
            else:
                img_rgb_cleaned = img_rgb

            if source_garment_prepass_enabled:
                try:
                    face_crop_fill = self._crop_face(
                        Image.fromarray(img_rgb), face_bbox
                    )
                    source_garment_seed = (
                        int(seeds[0]) if seeds else random.randint(0, 2**31 - 1)
                    ) + 1703
                    img_rgb_cleaned = self._sd_refine_removed_region(
                        base_rgb=img_rgb_cleaned,
                        removal_mask=source_garment_prepass_mask,
                        face_bbox=face_bbox,
                        face_crop_pil=face_crop_fill,
                        protect_mask=protect_mask_for_sd,
                        cloth_mask=cloth_mask_dilated,
                        hair_length=hair_length,
                        seed=source_garment_seed,
                        refine_mode="garment",
                        subject_gender=subject_gender_mode,
                        white_tshirt_experiment=white_tshirt_experiment,
                    )
                    source_garment_prepass_applied = True
                    source_garment_prepass_apply_mode = "sd"
                    logger.info(
                        "[SDPipeline] source garment prepass applied: pixels=%d seed=%d",
                        source_garment_prepass_px,
                        source_garment_seed,
                    )
                    if source_shoulder_contour_anchor_px >= 80:
                        img_rgb_cleaned = self._restore_cloth_overlap_from_source(
                            source_rgb=img_rgb,
                            current_rgb=img_rgb_cleaned,
                            restore_mask=source_shoulder_contour_anchor_mask,
                            final_hair_mask=None,
                            tone_reference_rgb=img_rgb,
                            tone_reference_mask=cloth_mask_dilated,
                        )
                        img_rgb_cleaned = self._cv2_refine_cloth_region(
                            img_rgb_cleaned,
                            source_shoulder_contour_anchor_mask,
                            reference_rgb=img_rgb,
                            reference_mask=cloth_mask_dilated,
                        )
                        img_rgb_cleaned = self._restore_reference_region(
                            img_rgb_cleaned,
                            img_rgb,
                            source_shoulder_contour_anchor_mask,
                            strength=0.988,
                        )
                        logger.info(
                            "[SDPipeline] source shoulder contour restore applied: pixels=%d",
                            source_shoulder_contour_anchor_px,
                        )
                    if (
                        hair_length == "short"
                        and short_torso_generation_freeze_active
                        and cloth_mask_dilated is not None
                        and cloth_mask_dilated.shape == (H, W)
                        and bool(
                            getattr(
                                self.config,
                                "short_generation_plain_cloth_stabilize",
                                True,
                            )
                        )
                    ):
                        prepass_stabilize_mask = np.maximum(
                            np.clip(
                                source_garment_prepass_mask.astype(np.float32), 0.0, 1.0
                            ),
                            np.clip(
                                short_torso_generation_freeze_mask.astype(np.float32)
                                * 0.96,
                                0.0,
                                1.0,
                            ),
                        ).astype(np.float32)
                        prepass_stabilize_mask = np.clip(
                            prepass_stabilize_mask.astype(np.float32)
                            * self._dilate_mask_with_px(
                                cloth_mask_dilated.astype(np.float32), 15
                            ),
                            0.0,
                            1.0,
                        ).astype(np.float32)
                        prepass_stabilize_px = self._count_active_mask_px(
                            prepass_stabilize_mask
                        )
                        if prepass_stabilize_px >= 180:
                            img_rgb_cleaned = self._cleanup_region_with_cloth_restore(
                                source_rgb=img_rgb,
                                current_rgb=img_rgb_cleaned,
                                cleanup_mask=prepass_stabilize_mask,
                                cloth_mask=cloth_mask_dilated,
                                final_hair_mask=None,
                                ignore_final_hair_for_cloth_restore=True,
                                cleanup_dark_tail=True,
                                prefer_plain_cloth_fill=bool(white_tshirt_experiment),
                                allow_plain_cloth_force=bool(white_tshirt_experiment),
                            )
                            source_garment_prepass_stabilized = True
                            _store_mask(
                                "pipeline_source_garment_prepass_stabilize_mask",
                                prepass_stabilize_mask,
                            )
                            _store_rgb(
                                "source_garment_prepass_stabilized_rgb", img_rgb_cleaned
                            )
                            logger.info(
                                "[SDPipeline] source garment prepass stabilized with cloth restore: pixels=%d",
                                prepass_stabilize_px,
                            )
                except Exception as e:
                    source_garment_prepass_error = f"{type(e).__name__}: {e}"
                    logger.warning(
                        f"[SDPipeline] source garment prepass 실패(무시): {e}"
                    )
                    try:
                        img_rgb_cleaned = self._cv2_refine_cloth_region(
                            img_rgb_cleaned,
                            source_garment_prepass_mask,
                            reference_rgb=img_rgb,
                            reference_mask=cloth_mask_dilated,
                        )
                        source_garment_prepass_applied = True
                        source_garment_prepass_apply_mode = "cv2_fallback"
                        logger.info(
                            "[SDPipeline] source garment prepass fallback applied: pixels=%d mode=%s",
                            source_garment_prepass_px,
                            source_garment_prepass_apply_mode,
                        )
                    except Exception as fallback_e:
                        fallback_message = f"{type(fallback_e).__name__}: {fallback_e}"
                        source_garment_prepass_error = f"{source_garment_prepass_error} | fallback={fallback_message}"
                        logger.warning(
                            "[SDPipeline] source garment prepass fallback 실패(무시): %s",
                            fallback_e,
                        )
                if debug_data_common is not None:
                    debug_data_common.setdefault("source_cloth_preclean", {})
                    debug_data_common["source_cloth_preclean"][
                        "garment_prepass_attempted"
                    ] = True
                    debug_data_common["source_cloth_preclean"][
                        "garment_prepass_applied"
                    ] = bool(source_garment_prepass_applied)
                    debug_data_common["source_cloth_preclean"][
                        "garment_prepass_apply_mode"
                    ] = source_garment_prepass_apply_mode
                    debug_data_common["source_cloth_preclean"][
                        "garment_prepass_stabilized"
                    ] = bool(source_garment_prepass_stabilized)
                    if source_garment_prepass_error:
                        debug_data_common["source_cloth_preclean"][
                            "garment_prepass_error"
                        ] = source_garment_prepass_error
                    debug_data_common["source_cloth_preclean"][
                        "shoulder_contour_restore_applied"
                    ] = bool(
                        source_shoulder_contour_anchor_px >= 80
                        and source_garment_prepass_applied
                    )

            if hair_length == "short" and not disable_short_postprocess_experiment:
                if (
                    not skip_source_cloth_preclean
                    and not source_garment_prepass_enabled
                ):
                    try:
                        preclean_cloth_hair_cleanup_mask = (
                            self._build_preclean_cloth_hair_cleanup_mask(
                                img_rgb=img_rgb_cleaned,
                                removal_mask=removal_mask_for_post,
                                cloth_mask=cloth_mask_dilated,
                                face_bbox=face_bbox,
                                cutoff_y=cutoff_y,
                                hair_length=hair_length,
                            )
                        )
                        preclean_cloth_hair_cleanup_u8 = (
                            np.clip(
                                preclean_cloth_hair_cleanup_mask.astype(np.float32),
                                0.0,
                                1.0,
                            )
                            > 0.08
                        ).astype(np.uint8) * 255
                        if int((preclean_cloth_hair_cleanup_u8 > 0).sum()) >= 120:
                            img_rgb_cleaned = self._lama_inpaint(
                                img_rgb_cleaned,
                                preclean_cloth_hair_cleanup_u8,
                            )
                            img_rgb_cleaned = self._cv2_cleanup_dark_tail_blob(
                                img_rgb_cleaned,
                                preclean_cloth_hair_cleanup_u8,
                            )
                            if (
                                cloth_mask_dilated is not None
                                and cloth_mask_dilated.shape == (H, W)
                            ):
                                img_rgb_cleaned = self._blend_neighbor_cloth_tone(
                                    img_rgb_cleaned,
                                    preclean_cloth_hair_cleanup_mask,
                                    cloth_mask=cloth_mask_dilated,
                                )
                            if hair_length != "short":
                                img_rgb_cleaned = (
                                    self._restore_cloth_overlap_from_source(
                                        source_rgb=img_rgb,
                                        current_rgb=img_rgb_cleaned,
                                        restore_mask=preclean_cloth_hair_cleanup_mask,
                                        final_hair_mask=None,
                                    )
                                )
                            img_rgb_cleaned = self._cv2_refine_cloth_region(
                                img_rgb_cleaned,
                                preclean_cloth_hair_cleanup_mask,
                            )
                        _store_mask(
                            "pipeline_preclean_cloth_hair_cleanup_mask",
                            preclean_cloth_hair_cleanup_mask,
                        )
                    except Exception as e:
                        logger.warning(
                            f"[SDPipeline] preclean cloth hair cleanup failed (ignored): {e}"
                        )

                    try:
                        preclean_side_candidate_mask = np.clip(
                            removal_mask_for_post.astype(np.float32),
                            0.0,
                            1.0,
                        )
                        if (
                            artifact_cleanup_mask_for_post is not None
                            and artifact_cleanup_mask_for_post.shape == (H, W)
                        ):
                            preclean_side_candidate_mask = np.maximum(
                                preclean_side_candidate_mask,
                                np.clip(
                                    artifact_cleanup_mask_for_post.astype(np.float32),
                                    0.0,
                                    1.0,
                                ),
                            ).astype(np.float32)
                        if (
                            shoulder_cloth_release_for_post is not None
                            and shoulder_cloth_release_for_post.shape == (H, W)
                        ):
                            preclean_side_candidate_mask = np.maximum(
                                preclean_side_candidate_mask,
                                np.clip(
                                    shoulder_cloth_release_for_post.astype(np.float32),
                                    0.0,
                                    1.0,
                                )
                                * 0.88,
                            ).astype(np.float32)
                        if (
                            below_bob_cloth_restore_for_post is not None
                            and below_bob_cloth_restore_for_post.shape == (H, W)
                        ):
                            preclean_side_candidate_mask = np.maximum(
                                preclean_side_candidate_mask,
                                np.clip(
                                    below_bob_cloth_restore_for_post.astype(np.float32),
                                    0.0,
                                    1.0,
                                ),
                            ).astype(np.float32)
                        preclean_side_restore_mask = (
                            self._build_side_column_cloth_restore_mask(
                                img_rgb=img_rgb_cleaned,
                                cloth_mask=cloth_mask_dilated,
                                candidate_mask=preclean_side_candidate_mask,
                                face_bbox=face_bbox,
                                cutoff_y=cutoff_y,
                                hair_length=hair_length,
                                final_hair_mask=None,
                            )
                        )
                        direct_preclean_side_restore_mask = (
                            self._build_direct_short_column_restore_mask(
                                removal_mask=removal_mask_for_post,
                                cloth_mask=cloth_mask_dilated,
                                face_bbox=face_bbox,
                                cutoff_y=cutoff_y,
                                hair_length=hair_length,
                            )
                        )
                        if float(direct_preclean_side_restore_mask.sum()) > 0.0:
                            preclean_side_restore_mask = np.maximum(
                                preclean_side_restore_mask,
                                direct_preclean_side_restore_mask,
                            ).astype(np.float32)
                        preclean_side_restore_u8 = (
                            np.clip(
                                preclean_side_restore_mask.astype(np.float32), 0.0, 1.0
                            )
                            > 0.08
                        ).astype(np.uint8) * 255
                        preclean_side_cleanup_mask = (
                            self._build_preclean_side_column_cleanup_mask(
                                img_rgb=img_rgb_cleaned,
                                cloth_mask=cloth_mask_dilated,
                                base_mask=preclean_side_restore_mask,
                                face_bbox=face_bbox,
                                cutoff_y=cutoff_y,
                                hair_length=hair_length,
                            )
                        )
                        preclean_side_cleanup_u8 = (
                            np.clip(
                                preclean_side_cleanup_mask.astype(np.float32), 0.0, 1.0
                            )
                            > 0.08
                        ).astype(np.uint8) * 255
                        if int((preclean_side_cleanup_u8 > 0).sum()) >= 48:
                            img_rgb_cleaned = self._cleanup_region_with_cloth_restore(
                                source_rgb=img_rgb,
                                current_rgb=img_rgb_cleaned,
                                cleanup_mask=preclean_side_cleanup_mask,
                                cloth_mask=cloth_mask_dilated,
                                ignore_final_hair_for_cloth_restore=True,
                                cleanup_dark_tail=True,
                            )
                        if int((preclean_side_restore_u8 > 0).sum()) >= 140:
                            if hair_length == "short":
                                img_rgb_cleaned = (
                                    self._cleanup_region_with_cloth_restore(
                                        source_rgb=img_rgb,
                                        current_rgb=img_rgb_cleaned,
                                        cleanup_mask=preclean_side_restore_mask,
                                        cloth_mask=cloth_mask_dilated,
                                        ignore_final_hair_for_cloth_restore=True,
                                        cleanup_dark_tail=True,
                                    )
                                )
                            else:
                                img_rgb_cleaned = (
                                    self._restore_cloth_overlap_from_source(
                                        source_rgb=img_rgb,
                                        current_rgb=img_rgb_cleaned,
                                        restore_mask=preclean_side_restore_mask,
                                        final_hair_mask=None,
                                    )
                                )
                                img_rgb_cleaned = self._cv2_refine_cloth_region(
                                    img_rgb_cleaned,
                                    preclean_side_restore_mask,
                                )
                        _store_mask(
                            "pipeline_preclean_side_column_cleanup_mask",
                            preclean_side_cleanup_mask,
                        )
                        _store_mask(
                            "pipeline_preclean_side_column_cloth_restore_mask",
                            preclean_side_restore_mask,
                        )
                        _store_mask(
                            "pipeline_preclean_direct_short_column_restore_mask",
                            direct_preclean_side_restore_mask,
                        )
                    except Exception as e:
                        logger.warning(
                            f"[SDPipeline] preclean side column cloth restore failed (ignored): {e}"
                        )
                    try:
                        preclean_dark_lane_mask = self._build_dark_lane_cleanup_mask(
                            img_rgb=img_rgb_cleaned,
                            cloth_mask=cloth_mask_dilated,
                            removal_mask=removal_mask_for_post,
                            face_bbox=face_bbox,
                            cutoff_y=cutoff_y,
                            hair_length=hair_length,
                        )
                        preclean_dark_lane_u8 = (
                            np.clip(
                                preclean_dark_lane_mask.astype(np.float32), 0.0, 1.0
                            )
                            > 0.08
                        ).astype(np.uint8) * 255
                        if int((preclean_dark_lane_u8 > 0).sum()) >= 40:
                            img_rgb_cleaned = self._lama_inpaint(
                                img_rgb_cleaned, preclean_dark_lane_u8
                            )
                            img_rgb_cleaned = self._cv2_refine_cloth_region(
                                img_rgb_cleaned,
                                preclean_dark_lane_mask,
                            )
                        _store_mask(
                            "pipeline_preclean_dark_lane_cleanup_mask",
                            preclean_dark_lane_mask,
                        )
                    except Exception as e:
                        logger.warning(
                            f"[SDPipeline] preclean dark lane cleanup failed (ignored): {e}"
                        )
                regen_tail_mask = self._build_short_regen_tail_mask(
                    img_rgb=img_rgb_cleaned,
                    removal_mask=removal_mask,
                    face_bbox=face_bbox,
                    cutoff_y=cutoff_y,
                    hair_length=hair_length,
                )
                face_w = max(int(face_bbox[2] - face_bbox[0]), 1)
                face_h = max(int(face_bbox[3] - face_bbox[1]), 1)
                if (
                    shoulder_protect_for_post is not None
                    and shoulder_protect_for_post.shape == (H, W)
                ):
                    regen_tail_mask = np.clip(
                        regen_tail_mask - shoulder_protect_for_post * 0.32,
                        0.0,
                        1.0,
                    )
                if (
                    neckline_preserve_for_post is not None
                    and neckline_preserve_for_post.shape == (H, W)
                ):
                    regen_tail_mask = np.clip(
                        regen_tail_mask - neckline_preserve_for_post * 0.38,
                        0.0,
                        1.0,
                    )
                regen_tail_mask = np.clip(
                    regen_tail_mask - cloth_mask_dilated * 0.48, 0.0, 1.0
                )
                regen_tail_u8 = cv2.dilate(
                    (regen_tail_mask > 0.08).astype(np.uint8) * 255,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 9)),
                    iterations=1,
                )
                if hair_length == "short":
                    regen_cap_y = min(H, int(cutoff_y + face_h * 0.88))
                    regen_center_x = int(0.5 * (face_bbox[0] + face_bbox[2]))
                    regen_x1 = max(0, int(face_bbox[0] - face_w * 1.16))
                    regen_x2 = min(W, int(face_bbox[2] + face_w * 1.16))
                    regen_corridor_u8 = np.zeros((H, W), dtype=np.uint8)
                    regen_top = max(0, int(cutoff_y - face_h * 0.02))
                    if regen_top < regen_cap_y and regen_x1 < regen_x2:
                        regen_corridor_u8[regen_top:regen_cap_y, regen_x1:regen_x2] = (
                            255
                        )
                        regen_tail_u8 = cv2.bitwise_and(
                            regen_tail_u8, regen_corridor_u8
                        )
                    if regen_cap_y < H:
                        regen_tail_u8[regen_cap_y:, :] = 0
                cloth_block_u8 = cv2.dilate(
                    (cloth_mask_dilated > 0.10).astype(np.uint8) * 255,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                    iterations=1,
                )
                regen_tail_u8 = cv2.bitwise_and(
                    regen_tail_u8, cv2.bitwise_not(cloth_block_u8)
                )
                if (
                    shoulder_protect_for_post is not None
                    and shoulder_protect_for_post.shape == (H, W)
                ):
                    shoulder_block_u8 = cv2.dilate(
                        (shoulder_protect_for_post > 0.04).astype(np.uint8) * 255,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 19)),
                        iterations=1,
                    )
                    regen_tail_u8 = cv2.bitwise_and(
                        regen_tail_u8, cv2.bitwise_not(shoulder_block_u8)
                    )
                if (
                    neckline_preserve_for_post is not None
                    and neckline_preserve_for_post.shape == (H, W)
                ):
                    neckline_block_u8 = cv2.dilate(
                        (neckline_preserve_for_post > 0.04).astype(np.uint8) * 255,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 15)),
                        iterations=1,
                    )
                    regen_tail_u8 = cv2.bitwise_and(
                        regen_tail_u8, cv2.bitwise_not(neckline_block_u8)
                    )
                if hair_length == "short" and int((regen_tail_u8 > 0).sum()) > 0:
                    filtered_regen_u8 = np.zeros((H, W), dtype=np.uint8)
                    num_labels, labels, stats, centroids = (
                        cv2.connectedComponentsWithStats(regen_tail_u8, 8)
                    )
                    center_half = max(18, int(face_w * 0.28))
                    side_offset = max(14, int(face_w * 0.14))
                    side_max_width = max(56, int(face_w * 0.88))
                    center_max_width = max(28, int(face_w * 0.42))
                    min_component_height = max(14, int(face_h * 0.10))
                    side_max_area = max(520, int(face_w * face_h * 0.22))
                    center_max_area = max(220, int(face_w * face_h * 0.09))
                    for idx in range(1, num_labels):
                        x = int(stats[idx, cv2.CC_STAT_LEFT])
                        y = int(stats[idx, cv2.CC_STAT_TOP])
                        w = int(stats[idx, cv2.CC_STAT_WIDTH])
                        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
                        area = int(stats[idx, cv2.CC_STAT_AREA])
                        bottom = y + h
                        if (
                            area < 18
                            or h < min_component_height
                            or bottom > regen_cap_y
                        ):
                            continue
                        comp_cx = float(centroids[idx][0])
                        if abs(comp_cx - regen_center_x) <= center_half:
                            if w > center_max_width or area > center_max_area:
                                continue
                        elif abs(comp_cx - regen_center_x) >= side_offset:
                            if w > side_max_width or area > side_max_area:
                                continue
                        else:
                            continue
                        filtered_regen_u8[labels == idx] = 255
                    regen_tail_u8 = filtered_regen_u8
                if (
                    hair_length == "short"
                    and below_bob_generation_block_for_post is not None
                    and below_bob_generation_block_for_post.shape == (H, W)
                    and int((regen_tail_u8 > 0).sum()) > 0
                ):
                    below_bob_generation_block_u8 = (
                        np.clip(
                            below_bob_generation_block_for_post.astype(np.float32),
                            0.0,
                            1.0,
                        )
                        > 0.08
                    ).astype(np.uint8) * 255
                    regen_tail_u8 = cv2.bitwise_and(
                        regen_tail_u8,
                        cv2.bitwise_not(below_bob_generation_block_u8),
                    )
                if (
                    hair_length == "short"
                    and below_bob_cloth_restore_for_post is not None
                    and below_bob_cloth_restore_for_post.shape == (H, W)
                    and int((regen_tail_u8 > 0).sum()) > 0
                ):
                    below_bob_block_u8 = (
                        np.clip(
                            below_bob_cloth_restore_for_post.astype(np.float32),
                            0.0,
                            1.0,
                        )
                        > 0.08
                    ).astype(np.uint8) * 255
                    regen_tail_u8 = cv2.bitwise_and(
                        regen_tail_u8,
                        cv2.bitwise_not(below_bob_block_u8),
                    )
                regen_tail_mask = regen_tail_u8.astype(np.float32) / 255.0
                if int((regen_tail_u8 > 0).sum()) > 60:
                    gen_mask = np.maximum(gen_mask, regen_tail_mask)
                if (
                    hair_length == "short"
                    and below_bob_generation_block_for_post is not None
                    and below_bob_generation_block_for_post.shape == (H, W)
                ):
                    gen_mask = np.clip(
                        gen_mask - below_bob_generation_block_for_post * 1.45,
                        0.0,
                        1.0,
                    )
                if (
                    hair_length == "short"
                    and below_bob_cloth_restore_for_post is not None
                    and below_bob_cloth_restore_for_post.shape == (H, W)
                ):
                    gen_mask = np.clip(
                        gen_mask - below_bob_cloth_restore_for_post * 1.35,
                        0.0,
                        1.0,
                    )
                _store_mask("pipeline_short_regen_tail_mask", regen_tail_mask)

            if hair_length == "short" and cutoff_y_for_post is not None:
                try:
                    short_generation_conditioning_cleanup_mask = self._build_short_generation_conditioning_cleanup_mask(
                        source_garment_prepass_mask=source_garment_prepass_mask,
                        upper_body_repaint_seed_mask=short_upper_body_repaint_seed_mask,
                        removal_mask=removal_mask_for_post,
                        source_torso_hair_mask=source_torso_hair_mask,
                        cloth_mask=cloth_mask_dilated,
                        protect_mask=protect_mask_for_sd,
                        face_bbox=face_bbox,
                        cutoff_y=int(cutoff_y_for_post),
                        hair_length=hair_length,
                    )
                    short_generation_conditioning_cleanup_px = (
                        self._count_active_mask_px(
                            short_generation_conditioning_cleanup_mask
                        )
                    )
                    if self._should_apply_cleanup_mask(
                        "short_generation_conditioning_cleanup",
                        short_generation_conditioning_cleanup_px,
                        hair_length,
                    ):
                        if (
                            cloth_mask_dilated is not None
                            and cloth_mask_dilated.shape == (H, W)
                        ):
                            img_rgb_cleaned = self._cleanup_region_with_cloth_restore(
                                source_rgb=img_rgb,
                                current_rgb=img_rgb_cleaned,
                                cleanup_mask=short_generation_conditioning_cleanup_mask,
                                cloth_mask=cloth_mask_dilated,
                                final_hair_mask=None,
                                ignore_final_hair_for_cloth_restore=True,
                                cleanup_dark_tail=True,
                                prefer_plain_cloth_fill=bool(white_tshirt_experiment),
                                allow_plain_cloth_force=bool(white_tshirt_experiment),
                            )
                        else:
                            short_generation_conditioning_cleanup_u8 = self._mask_to_u8(
                                short_generation_conditioning_cleanup_mask,
                                threshold=0.08,
                            )
                            img_rgb_cleaned = self._lama_inpaint(
                                img_rgb_cleaned,
                                short_generation_conditioning_cleanup_u8,
                            )
                            img_rgb_cleaned = self._cv2_cleanup_dark_tail_blob(
                                img_rgb_cleaned,
                                short_generation_conditioning_cleanup_u8,
                            )
                            img_rgb_cleaned = self._blend_neighbor_cloth_tone(
                                img_rgb_cleaned,
                                short_generation_conditioning_cleanup_mask,
                                cloth_mask=cloth_mask_dilated,
                                reference_rgb=img_rgb,
                            )
                            img_rgb_cleaned = self._cv2_refine_cloth_region(
                                img_rgb_cleaned,
                                short_generation_conditioning_cleanup_mask,
                                reference_rgb=img_rgb,
                                reference_mask=cloth_mask_dilated,
                            )
                        short_generation_conditioning_cleanup_applied = True
                        logger.info(
                            "[SDPipeline] short generation conditioning cleanup applied: pixels=%d",
                            short_generation_conditioning_cleanup_px,
                        )
                    if (
                        white_tshirt_experiment
                        and short_torso_generation_freeze_active
                        and cloth_mask_dilated is not None
                        and cloth_mask_dilated.shape == (H, W)
                        and bool(
                            getattr(
                                self.config,
                                "short_generation_white_tshirt_conditioning_fill",
                                True,
                            )
                        )
                    ):
                        short_generation_white_tshirt_conditioning_fill_mask = (
                            np.maximum(
                                np.clip(
                                    short_generation_conditioning_cleanup_mask.astype(
                                        np.float32
                                    ),
                                    0.0,
                                    1.0,
                                ),
                                np.clip(
                                    short_torso_generation_freeze_mask.astype(
                                        np.float32
                                    )
                                    * 0.96,
                                    0.0,
                                    1.0,
                                ),
                            ).astype(np.float32)
                        )
                        if source_garment_prepass_mask.shape == (H, W):
                            short_generation_white_tshirt_conditioning_fill_mask = np.maximum(
                                short_generation_white_tshirt_conditioning_fill_mask,
                                np.clip(
                                    source_garment_prepass_mask.astype(np.float32)
                                    * 0.92,
                                    0.0,
                                    1.0,
                                ),
                            ).astype(
                                np.float32
                            )
                        short_generation_white_tshirt_conditioning_fill_mask = np.clip(
                            short_generation_white_tshirt_conditioning_fill_mask
                            * self._dilate_mask_with_px(
                                cloth_mask_dilated.astype(np.float32), 13
                            ),
                            0.0,
                            1.0,
                        ).astype(np.float32)
                        if protect_mask_for_sd.shape == (H, W):
                            short_generation_white_tshirt_conditioning_fill_mask = (
                                np.clip(
                                    short_generation_white_tshirt_conditioning_fill_mask
                                    - protect_mask_for_sd.astype(np.float32) * 0.84,
                                    0.0,
                                    1.0,
                                ).astype(np.float32)
                            )
                        short_generation_white_tshirt_conditioning_fill_px = (
                            self._count_active_mask_px(
                                short_generation_white_tshirt_conditioning_fill_mask
                            )
                        )
                        if short_generation_white_tshirt_conditioning_fill_px >= 180:
                            plain_fill_rgb = img_rgb_cleaned.copy()
                            fill_bool = (
                                self._mask_to_u8(
                                    short_generation_white_tshirt_conditioning_fill_mask,
                                    threshold=0.08,
                                )
                                > 0
                            )
                            plain_fill_rgb[fill_bool] = np.array(
                                [242, 243, 239], dtype=np.uint8
                            )
                            img_rgb_cleaned = self._restore_reference_region(
                                img_rgb_cleaned,
                                plain_fill_rgb,
                                short_generation_white_tshirt_conditioning_fill_mask,
                                strength=float(
                                    getattr(
                                        self.config,
                                        "short_generation_white_tshirt_fill_strength",
                                        0.992,
                                    )
                                ),
                            )
                            short_generation_white_tshirt_conditioning_fill_applied = (
                                True
                            )
                            logger.info(
                                "[SDPipeline] short white-tshirt conditioning fill applied: pixels=%d",
                                short_generation_white_tshirt_conditioning_fill_px,
                            )
                    if (
                        (
                            explicit_no_bangs_requested
                            or short_no_bangs_target
                            and bool(
                                getattr(
                                    self.config,
                                    "short_no_bangs_forehead_lama_preclean",
                                    True,
                                )
                            )
                        )
                        and no_bangs_forehead_lama_preclean_seed_mask.shape == (H, W)
                    ):
                        no_bangs_forehead_lama_preclean_u8 = cv2.dilate(
                            self._mask_to_u8(
                                no_bangs_forehead_lama_preclean_seed_mask,
                                threshold=0.08,
                            ),
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11)),
                            iterations=1,
                        )
                        no_bangs_forehead_lama_preclean_mask = (
                            no_bangs_forehead_lama_preclean_u8.astype(np.float32)
                            / 255.0
                        )
                        no_bangs_forehead_lama_preclean_px = int(
                            (no_bangs_forehead_lama_preclean_u8 > 0).sum()
                        )
                        if no_bangs_forehead_lama_preclean_px >= 24:
                            img_rgb_cleaned = self._lama_inpaint(
                                img_rgb_cleaned,
                                no_bangs_forehead_lama_preclean_u8,
                            )
                            no_bangs_forehead_lama_preclean_applied = True
                            logger.info(
                                "[SDPipeline] short no-bangs forehead LaMa preclean applied: pixels=%d",
                                no_bangs_forehead_lama_preclean_px,
                            )
                except Exception as e:
                    logger.warning(
                        "[SDPipeline] short generation conditioning cleanup failed (ignored): %s",
                        e,
                    )
                if debug_data_common is not None:
                    debug_data_common.setdefault("source_cloth_preclean", {})
                    debug_data_common["source_cloth_preclean"][
                        "generation_conditioning_cleanup_px"
                    ] = short_generation_conditioning_cleanup_px
                    debug_data_common["source_cloth_preclean"][
                        "generation_conditioning_cleanup_applied"
                    ] = bool(short_generation_conditioning_cleanup_applied)
                    debug_data_common["source_cloth_preclean"][
                        "white_tshirt_conditioning_fill_px"
                    ] = int(short_generation_white_tshirt_conditioning_fill_px)
                    debug_data_common["source_cloth_preclean"][
                        "white_tshirt_conditioning_fill_applied"
                    ] = bool(short_generation_white_tshirt_conditioning_fill_applied)
                    debug_data_common["source_cloth_preclean"][
                        "no_bangs_forehead_lama_preclean_px"
                    ] = int(no_bangs_forehead_lama_preclean_px)
                    debug_data_common["source_cloth_preclean"][
                        "no_bangs_forehead_lama_preclean_applied"
                    ] = bool(no_bangs_forehead_lama_preclean_applied)
                    debug_data_common["source_cloth_preclean"][
                        "short_generation_tail_release_px"
                    ] = int(short_generation_tail_release_px)
                _store_mask(
                    "pipeline_short_generation_conditioning_cleanup_mask",
                    short_generation_conditioning_cleanup_mask,
                )
                _store_mask(
                    "pipeline_short_generation_white_tshirt_conditioning_fill_mask",
                    short_generation_white_tshirt_conditioning_fill_mask,
                )
                _store_mask(
                    "pipeline_no_bangs_forehead_lama_preclean_mask",
                    no_bangs_forehead_lama_preclean_mask,
                )
                _store_rgb("cv2_background_conditioning_cleaned_rgb", img_rgb_cleaned)

            _store_rgb("cv2_background_cleaned_rgb", img_rgb_cleaned)
            _store_mask(
                "pipeline_no_bangs_forehead_lama_preclean_mask",
                no_bangs_forehead_lama_preclean_mask,
            )
            _store_mask("pipeline_short_generation_mask", gen_mask)

            composite_hair_mask = gen_mask.astype(np.float32)
            hair_mask_for_sd = composite_hair_mask.copy()
            if (
                use_upper_clothes_overwrite
                and effective_upper_clothes_overwrite_mask.shape == (H, W)
                and not short_torso_generation_freeze_active
            ):
                hair_mask_for_sd = np.maximum(
                    hair_mask_for_sd,
                    effective_upper_clothes_overwrite_mask.astype(np.float32),
                ).astype(np.float32)
            hair_mask_for_sd_before_core = hair_mask_for_sd.copy()
            if (
                use_upper_clothes_overwrite
                and effective_upper_clothes_overwrite_core_mask.shape == (H, W)
                and not short_torso_generation_freeze_active
            ):
                hair_mask_for_sd = np.maximum(
                    hair_mask_for_sd,
                    effective_upper_clothes_overwrite_core_mask.astype(np.float32),
                ).astype(np.float32)
            img_rgb_for_sd = img_rgb_cleaned
        else:
            # long 헤어는 기존 단일 패스 유지
            hair_mask_for_sd = hair_mask
            composite_hair_mask = hair_mask_for_sd.astype(np.float32)
            hair_mask_for_sd_before_core = hair_mask_for_sd.copy()
            img_rgb_for_sd = img_rgb
            img_rgb_cleaned = img_rgb
            if (
                explicit_no_bangs_requested
                and no_bangs_forehead_lama_preclean_seed_mask.shape == (H, W)
            ):
                dilate_kernel = (9, 13) if hair_length == "medium" else (11, 15)
                no_bangs_forehead_lama_preclean_u8 = cv2.dilate(
                    self._mask_to_u8(
                        no_bangs_forehead_lama_preclean_seed_mask,
                        threshold=0.08,
                    ),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dilate_kernel),
                    iterations=1,
                )
                no_bangs_forehead_lama_preclean_mask = (
                    no_bangs_forehead_lama_preclean_u8.astype(np.float32) / 255.0
                )
                no_bangs_forehead_lama_preclean_px = int(
                    (no_bangs_forehead_lama_preclean_u8 > 0).sum()
                )
                if no_bangs_forehead_lama_preclean_px >= 24:
                    img_rgb_cleaned = self._lama_inpaint(
                        img_rgb_cleaned,
                        no_bangs_forehead_lama_preclean_u8,
                    )
                    img_rgb_for_sd = img_rgb_cleaned
                    no_bangs_forehead_lama_preclean_applied = True
                    logger.info(
                        "[SDPipeline] %s no-bangs forehead LaMa preclean applied: pixels=%d",
                        hair_length,
                        no_bangs_forehead_lama_preclean_px,
                    )
            if float(bangs_restore_for_sd.sum()) > 0.0:
                long_soft_bangs_mask = self._build_soft_bangs_generation_mask(
                    bangs_restore_for_sd,
                    face_bbox=face_bbox,
                    hair_length=hair_length,
                )
                if float(long_soft_bangs_mask.sum()) > 0.0:
                    long_soft_bangs_u8 = cv2.dilate(
                        (
                            np.clip(long_soft_bangs_mask.astype(np.float32), 0.0, 1.0)
                            > 0.04
                        ).astype(np.uint8)
                        * 255,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 13)),
                        iterations=1,
                    )
                    long_soft_bangs_mask = cv2.GaussianBlur(
                        long_soft_bangs_u8.astype(np.float32) / 255.0,
                        (0, 0),
                        sigmaX=2.6,
                        sigmaY=3.0,
                    ).astype(np.float32)
                    long_soft_bangs_mask = np.clip(
                        long_soft_bangs_mask * 1.10, 0.0, 1.0
                    )
                    hair_mask_for_sd = np.maximum(
                        hair_mask_for_sd.astype(np.float32),
                        long_soft_bangs_mask,
                    ).astype(np.float32)
                    composite_bangs_release_mask = np.maximum(
                        composite_bangs_release_mask,
                        np.clip(long_soft_bangs_mask * 1.28, 0.0, 1.0),
                    ).astype(np.float32)
                    _store_mask(
                        "pipeline_bangs_generation_soft_mask", long_soft_bangs_mask
                    )
                    _store_mask(
                        "pipeline_bangs_composite_release_mask",
                        composite_bangs_release_mask,
                    )

        # ── no-bangs composite release mask ────────────────────────────────────
        # face protect가 이마를 100% 차단하면, LaMA preclean이 성공해도 SD가 생성한
        # 깨끗한 이마(앞머리 없음)가 composite에 반영되지 않고 composite_base
        # (img_rgb_cleaned)만 표시된다. LaMA preclean 실패 시 원본 앞머리가 그대로 남음.
        # → 앞머리 없애기 요청 시 이마 영역의 face protect를 부분적으로 해제해서
        #   SD가 생성한 앞머리 없는 이마가 최종 composite에 반영되게 한다.
        #   (앞머리 내리기 release mask 와 동일한 원리, 방향만 반대)
        if (
            (short_no_bangs_target or explicit_no_bangs_requested)
            and face_bbox is not None
            and float(no_bangs_forehead_lama_preclean_seed_mask.sum()) > 0.0
        ):
            _nb_rx1, _nb_ry1, _nb_rx2, _nb_ry2 = face_bbox
            _nb_face_h = max(int(_nb_ry2 - _nb_ry1), 1)
            # 이마 release 깊이: 앞머리 내리기 cap과 동일 비율 사용 (눈썹 위까지)
            if subject_gender_mode == "male":
                _nb_ratio = (
                    0.22
                    if hair_length == "short"
                    else 0.20 if hair_length == "medium" else 0.18
                )
            else:
                _nb_ratio = (
                    0.27
                    if hair_length == "short"
                    else 0.23 if hair_length == "medium" else 0.20
                )
            _nb_y_limit = int(_nb_ry1 + _nb_face_h * _nb_ratio)
            _nb_seed_u8 = cv2.dilate(
                self._mask_to_u8(
                    no_bangs_forehead_lama_preclean_seed_mask, threshold=0.08
                ),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 9)),
                iterations=1,
            )
            _nb_soft = cv2.GaussianBlur(
                _nb_seed_u8.astype(np.float32) / 255.0,
                (0, 0),
                sigmaX=2.0,
                sigmaY=2.4,
            ).astype(np.float32)
            # 눈썹 위 이마 영역으로 제한 (eyes/nose에 SD artifact 유입 방지)
            _nb_cap = np.zeros((H, W), dtype=np.float32)
            _nb_fade_h = max(18, int(_nb_face_h * 0.14))
            _nb_fade_top = max(0, _nb_y_limit - _nb_fade_h)
            _nb_fade_bot = min(H, _nb_y_limit + _nb_fade_h)
            _nb_cap[:_nb_fade_top, :] = 1.0
            if _nb_fade_bot > _nb_fade_top:
                _nb_cap[_nb_fade_top:_nb_fade_bot, :] = np.linspace(
                    1.0,
                    0.0,
                    _nb_fade_bot - _nb_fade_top,
                    dtype=np.float32,
                )[:, np.newaxis]
            _nb_soft = np.clip(_nb_soft * _nb_cap * 0.95, 0.0, 1.0)
            if float(_nb_soft.sum()) > 10.0:
                composite_bangs_release_mask = np.maximum(
                    composite_bangs_release_mask,
                    _nb_soft,
                ).astype(np.float32)
                _store_mask("pipeline_no_bangs_composite_release_mask", _nb_soft)
                logger.debug(
                    "[SDPipeline] no-bangs composite release mask built: px=%.0f y_limit=%d (%s/%s)",
                    float(_nb_soft.sum()),
                    _nb_y_limit,
                    subject_gender_mode,
                    hair_length,
                )

        _store_mask(
            "pipeline_sd_inpaint_mask_before_core", hair_mask_for_sd_before_core
        )
        _store_mask("sd_inpaint_mask", hair_mask_for_sd)
        _store_rgb("sd_input_rgb", img_rgb_for_sd)
        if debug_data_common is not None:
            debug_data_common.setdefault("source_cloth_preclean", {})
            debug_data_common["source_cloth_preclean"][
                "no_bangs_forehead_lama_preclean_px"
            ] = int(no_bangs_forehead_lama_preclean_px)
            debug_data_common["source_cloth_preclean"][
                "no_bangs_forehead_lama_preclean_applied"
            ] = bool(no_bangs_forehead_lama_preclean_applied)
        if debug_images_common is not None:
            _store_rgb(
                "pipeline_source_with_overwrite_core_overlay",
                _overlay_mask_rgb(
                    img_rgb,
                    overwrite_core_restore_mask_for_debug,
                    (255, 96, 64),
                    alpha=0.52,
                ),
            )
            _store_rgb(
                "pipeline_source_with_sd_inpaint_mask_before_core_overlay",
                _overlay_mask_rgb(
                    img_rgb, hair_mask_for_sd_before_core, (255, 0, 0), alpha=0.38
                ),
            )
            _store_rgb(
                "pipeline_source_with_sd_inpaint_mask_overlay",
                _overlay_mask_rgb(img_rgb, hair_mask_for_sd, (255, 0, 0), alpha=0.38),
            )
        if debug_data_common is not None:
            torso_rect = diagnostic_rois.get("torso_front")
            _mask_stats(
                "guard_release", cloth_generation_guard_release_mask, torso_rect
            )
            _mask_stats("cloth_guard", cloth_generation_guard, torso_rect)
            _mask_stats("overwrite_mask", upper_clothes_overwrite_mask, torso_rect)
            _mask_stats(
                "overwrite_effective",
                effective_upper_clothes_overwrite_mask,
                torso_rect,
            )
            _mask_stats(
                "overwrite_core_restore",
                overwrite_core_restore_mask_for_debug,
                torso_rect,
            )
            _mask_stats(
                "overwrite_core_seed", short_upper_body_repaint_seed_mask, torso_rect
            )
            _mask_stats(
                "overwrite_core",
                effective_upper_clothes_overwrite_core_mask,
                torso_rect,
            )
            _mask_stats(
                "gen_mask_before_upper_overwrite",
                gen_mask_before_upper_overwrite,
                torso_rect,
            )
            _mask_stats(
                "gen_mask_after_upper_overwrite",
                gen_mask_after_upper_overwrite,
                torso_rect,
            )
            _mask_stats(
                "gen_mask_after_initial_core", gen_mask_after_initial_core, torso_rect
            )
            _mask_stats(
                "gen_mask_before_final_core", gen_mask_before_final_core, torso_rect
            )
            _mask_stats(
                "gen_mask_after_final_core", gen_mask_after_final_core, torso_rect
            )
            _mask_stats(
                "sd_inpaint_mask_before_core", hair_mask_for_sd_before_core, torso_rect
            )
            _mask_stats("garment_prepass", source_garment_prepass_mask, torso_rect)
            _mask_stats("short_repaint_mask", short_upper_body_repaint_mask, torso_rect)
            _mask_stats("final_inpaint_mask", hair_mask_for_sd, torso_rect)
            for name, mask in (
                locals().get("overwrite_core_seed_debug_masks") or {}
            ).items():
                _mask_stats(f"overwrite_core_seed_{name}", mask, torso_rect)
            diag = debug_data_common.setdefault("diagnostics", {})
            if isinstance(locals().get("overwrite_core_seed_debug"), dict):
                diag["overwrite_core_seed_debug"] = dict(
                    locals().get("overwrite_core_seed_debug")
                )
            mask_stats = diag.setdefault("mask_stats", {})
            guard_release_sum = float(
                mask_stats.get("guard_release", {}).get("sum", 0.0)
            )
            cloth_guard_sum = float(mask_stats.get("cloth_guard", {}).get("sum", 0.0))
            overwrite_effective_sum = float(
                mask_stats.get("overwrite_effective", {}).get("sum", 0.0)
            )
            overwrite_core_sum = float(
                mask_stats.get("overwrite_core", {}).get("sum", 0.0)
            )
            final_inpaint_sum = float(
                mask_stats.get("final_inpaint_mask", {}).get("sum", 0.0)
            )
            torso_roi_area = float(
                mask_stats.get("final_inpaint_mask", {}).get("torso_roi_area", 0.0)
            )
            diag["mask_ratios"] = {
                "cloth_guard_to_guard_release": (
                    cloth_guard_sum / guard_release_sum
                    if guard_release_sum > 1e-6
                    else 0.0
                ),
                "overwrite_core_to_effective": (
                    overwrite_core_sum / overwrite_effective_sum
                    if overwrite_effective_sum > 1e-6
                    else 0.0
                ),
                "final_inpaint_sum_to_torso_roi_area": (
                    final_inpaint_sum / torso_roi_area if torso_roi_area > 1e-6 else 0.0
                ),
            }
            overwrite_core_seed_px = int(
                (
                    np.clip(
                        short_upper_body_repaint_seed_mask.astype(np.float32), 0.0, 1.0
                    )
                    > 0.08
                ).sum()
            )
            overwrite_core_px = int(
                (
                    np.clip(
                        upper_clothes_overwrite_core_mask.astype(np.float32), 0.0, 1.0
                    )
                    > 0.08
                ).sum()
            )
            protect_overlap_px = int(
                (
                    (
                        np.clip(
                            short_upper_body_repaint_seed_mask.astype(np.float32),
                            0.0,
                            1.0,
                        )
                        > 0.08
                    )
                    & (np.clip(protect_mask_for_sd.astype(np.float32), 0.0, 1.0) > 0.08)
                ).sum()
            )
            core_reason = "active"
            if overwrite_core_px <= 0:
                if overwrite_core_seed_px <= 0:
                    core_reason = "seed_mask_empty"
                elif protect_overlap_px >= max(1, int(overwrite_core_seed_px * 0.95)):
                    core_reason = "seed_removed_by_face_protect"
                else:
                    core_reason = "core_mask_not_promoted"
            diag["overwrite_core_debug"] = {
                "seed_mask_px": overwrite_core_seed_px,
                "core_mask_px": overwrite_core_px,
                "protect_overlap_px": protect_overlap_px,
                "short_repaint_px": int(short_upper_body_repaint_px),
                "reason": core_reason,
                "seed_builder_reason": (
                    locals().get("overwrite_core_seed_debug") or {}
                ).get("reason"),
                "seed_trim_removed_px": (
                    locals().get("overwrite_core_seed_debug") or {}
                ).get("trim_removed_px"),
            }
            logger.info(
                "[SDPipeline][diag][mask-ratio] cloth_guard/release=%.4f overwrite_core/effective=%.4f final_inpaint/torso=%.4f",
                diag["mask_ratios"]["cloth_guard_to_guard_release"],
                diag["mask_ratios"]["overwrite_core_to_effective"],
                diag["mask_ratios"]["final_inpaint_sum_to_torso_roi_area"],
            )
        if debug_images_common is not None:
            debug_images_common["pipeline_source_with_final_inpaint_mask_overlay"] = (
                cv2.cvtColor(
                    _overlay_mask_rgb(
                        img_rgb_for_sd, hair_mask_for_sd, (255, 64, 64), alpha=0.48
                    ),
                    cv2.COLOR_RGB2BGR,
                )
            )
            debug_images_common["pipeline_source_with_cloth_guard_overlay"] = (
                cv2.cvtColor(
                    _overlay_mask_rgb(
                        img_rgb_for_sd,
                        cloth_generation_guard,
                        (64, 120, 255),
                        alpha=0.46,
                    ),
                    cv2.COLOR_RGB2BGR,
                )
            )
            debug_images_common["pipeline_source_with_guard_release_overlay"] = (
                cv2.cvtColor(
                    _overlay_mask_rgb(
                        img_rgb_for_sd,
                        cloth_generation_guard_release_mask,
                        (64, 220, 96),
                        alpha=0.40,
                    ),
                    cv2.COLOR_RGB2BGR,
                )
            )
            _store_roi_crops("source", img_rgb_for_sd)

        # ── Step 4: SD 입력 준비 ─────────────────────────────────────────────
        img_pil = Image.fromarray(img_rgb_cleaned)
        # short/medium에서는 기존 long-hair 윤곽도 억제해 ControlNet이
        # 원본 긴머리 edge를 새 단발 형상으로 따라가지 않게 한다.
        canny_suppress = None
        if hair_length in ("short", "medium") or use_upper_clothes_overwrite:
            canny_suppress = hair_mask_for_removal.astype(np.float32)
            if (
                use_upper_clothes_overwrite
                and self.config.controlnet_use_masked_edges
                and effective_upper_clothes_overwrite_mask.shape == canny_suppress.shape
            ):
                canny_suppress = np.maximum(
                    canny_suppress,
                    effective_upper_clothes_overwrite_mask.astype(np.float32),
                ).astype(np.float32)
            if (
                use_upper_clothes_overwrite
                and self.config.controlnet_use_masked_edges
                and effective_upper_clothes_overwrite_core_mask.shape
                == canny_suppress.shape
            ):
                canny_suppress = np.maximum(
                    canny_suppress,
                    effective_upper_clothes_overwrite_core_mask.astype(np.float32),
                ).astype(np.float32)
            if hair_length == "short":
                canny_suppress = np.maximum(
                    canny_suppress,
                    self._dilate_mask_with_px(
                        hair_mask_for_removal.astype(np.float32), 33
                    ),
                )
                canny_suppress = np.maximum(
                    canny_suppress,
                    self._dilate_mask_with_px(hair_mask_for_sd.astype(np.float32), 25),
                )
                # front=down/bangs 요청 시: requested_front_coverage_mask를 직접
                # canny_suppress에 병합해 이마 머리선(hairline)의 원본 edge를
                # ControlNet이 따라가지 못하게 한다. 눈썹 아래로는 suppress하지 않아
                # 눈/코 edge가 보존되도록 y-cap을 적용한다.
                if (
                    bangs_requested
                    and face_bbox is not None
                    and requested_front_coverage_mask.shape == canny_suppress.shape
                    and float(requested_front_coverage_mask.sum()) > 0.0
                ):
                    _cs_rx1, _cs_ry1, _cs_rx2, _cs_ry2 = face_bbox
                    _cs_face_h = max(int(_cs_ry2 - _cs_ry1), 1)
                    _cs_top_ratio = 0.28 if hair_length == "short" else 0.25
                    _cs_bot_ratio = 0.36 if hair_length == "short" else 0.33
                    _cs_y_top = int(_cs_ry1 + _cs_face_h * _cs_top_ratio)
                    _cs_y_bot = int(_cs_ry1 + _cs_face_h * _cs_bot_ratio)
                    _cs_cap = np.ones_like(requested_front_coverage_mask)
                    if _cs_y_bot > _cs_y_top:
                        _cs_fade_h = _cs_y_bot - _cs_y_top
                        _cs_cap[_cs_y_top:_cs_y_bot, :] = np.linspace(
                            1.0, 0.0, _cs_fade_h, dtype=np.float32
                        )[:, np.newaxis]
                    _cs_cap[_cs_y_bot:, :] = 0.0
                    canny_suppress = np.maximum(
                        canny_suppress,
                        self._dilate_mask_with_px(
                            (
                                requested_front_coverage_mask.astype(np.float32)
                                * _cs_cap
                            ),
                            21,
                        ),
                    )
                if (
                    isinstance(source_garment_prepass_mask, np.ndarray)
                    and source_garment_prepass_mask.shape == canny_suppress.shape
                ):
                    canny_suppress = np.maximum(
                        canny_suppress,
                        self._dilate_mask_with_px(
                            source_garment_prepass_mask.astype(np.float32), 17
                        ),
                    )
                if (
                    lower_tail_support_mask is not None
                    and lower_tail_support_mask.shape == canny_suppress.shape
                ):
                    canny_suppress = np.maximum(
                        canny_suppress,
                        self._dilate_mask_with_px(
                            lower_tail_support_mask.astype(np.float32), 21
                        ),
                    )
                if (
                    short_torso_generation_freeze_active
                    and short_torso_generation_freeze_mask.shape == canny_suppress.shape
                ):
                    canny_suppress = np.maximum(
                        canny_suppress,
                        self._dilate_mask_with_px(
                            short_torso_generation_freeze_mask.astype(np.float32), 19
                        ),
                    )
                if (
                    short_generation_conditioning_cleanup_mask.shape
                    == canny_suppress.shape
                ):
                    canny_suppress = np.maximum(
                        canny_suppress,
                        self._dilate_mask_with_px(
                            short_generation_conditioning_cleanup_mask.astype(
                                np.float32
                            ),
                            17,
                        ),
                    )
                if (
                    short_generation_white_tshirt_conditioning_fill_applied
                    and short_generation_white_tshirt_conditioning_fill_mask.shape
                    == canny_suppress.shape
                ):
                    canny_suppress = np.maximum(
                        canny_suppress,
                        self._dilate_mask_with_px(
                            short_generation_white_tshirt_conditioning_fill_mask.astype(
                                np.float32
                            ),
                            21,
                        ),
                    )
        sd_input_debug: Optional[Dict[str, np.ndarray]] = (
            {} if return_intermediates else None
        )
        img_512, mask_512, canny_512, scale, pad = self._prepare_sd_inputs(
            img_rgb_for_sd,
            hair_mask_for_sd,
            canny_suppress_mask=canny_suppress,
            debug_outputs=sd_input_debug,
            target_size=generation_canvas_size,
        )
        if debug_images_common is not None:
            debug_images_common[f"sd_input_{generation_canvas_size}"] = cv2.cvtColor(
                np.array(img_512), cv2.COLOR_RGB2BGR
            )
            debug_images_common[f"sd_inpaint_mask_{generation_canvas_size}"] = cv2.cvtColor(
                np.array(mask_512).astype(np.uint8), cv2.COLOR_GRAY2BGR
            )
            debug_images_common[f"controlnet_canny_{generation_canvas_size}"] = cv2.cvtColor(
                np.array(canny_512), cv2.COLOR_RGB2BGR
            )
            if generation_canvas_size == SD_SIZE:
                debug_images_common["sd_input_512"] = debug_images_common[
                    f"sd_input_{generation_canvas_size}"
                ]
                debug_images_common["sd_inpaint_mask_512"] = debug_images_common[
                    f"sd_inpaint_mask_{generation_canvas_size}"
                ]
                debug_images_common["controlnet_canny_512"] = debug_images_common[
                    f"controlnet_canny_{generation_canvas_size}"
                ]
            if isinstance(sd_input_debug, dict):
                if isinstance(sd_input_debug.get("source_canny_raw"), np.ndarray):
                    debug_images_common["controlnet_canny_source_raw"] = cv2.cvtColor(
                        sd_input_debug["source_canny_raw"],
                        cv2.COLOR_GRAY2BGR,
                    )
                if isinstance(
                    sd_input_debug.get("source_canny_suppressed"), np.ndarray
                ):
                    debug_images_common["controlnet_canny_source_suppressed"] = (
                        cv2.cvtColor(
                            sd_input_debug["source_canny_suppressed"],
                            cv2.COLOR_GRAY2BGR,
                        )
                    )
                    debug_images_common["pipeline_source_with_canny_overlay"] = (
                        cv2.cvtColor(
                            _overlay_edges_rgb(
                                img_rgb_for_sd,
                                sd_input_debug["source_canny_suppressed"],
                                (255, 230, 0),
                                alpha=0.82,
                            ),
                            cv2.COLOR_RGB2BGR,
                        )
                    )
                if isinstance(
                    sd_input_debug.get("source_canny_suppress_mask"), np.ndarray
                ):
                    debug_images_common["controlnet_canny_suppress_mask_source"] = (
                        cv2.cvtColor(
                            sd_input_debug["source_canny_suppress_mask"],
                            cv2.COLOR_GRAY2BGR,
                        )
                    )
                if isinstance(sd_input_debug.get("control_canny_raw_512"), np.ndarray):
                    debug_images_common["controlnet_canny_raw_512"] = cv2.cvtColor(
                        sd_input_debug["control_canny_raw_512"],
                        cv2.COLOR_GRAY2BGR,
                    )
                if isinstance(
                    sd_input_debug.get("control_canny_suppress_mask_512"), np.ndarray
                ):
                    debug_images_common["controlnet_canny_suppress_mask_512"] = (
                        cv2.cvtColor(
                            sd_input_debug["control_canny_suppress_mask_512"],
                            cv2.COLOR_GRAY2BGR,
                        )
                    )
        if isinstance(sd_input_debug, dict):
            _edge_stats(
                "source_canny_raw",
                sd_input_debug.get("source_canny_raw"),
                hair_mask_for_removal,
                diagnostic_rois,
            )
            _edge_stats(
                "source_canny_suppressed",
                sd_input_debug.get("source_canny_suppressed"),
                hair_mask_for_removal,
                diagnostic_rois,
            )
            if debug_data_common is not None:
                conditioning = debug_data_common.setdefault(
                    "diagnostics", {}
                ).setdefault("conditioning_ratios", {})
                raw_stats = (
                    debug_data_common["diagnostics"]
                    .get("conditioning_stats", {})
                    .get("source_canny_raw", {})
                )
                suppressed_stats = (
                    debug_data_common["diagnostics"]
                    .get("conditioning_stats", {})
                    .get("source_canny_suppressed", {})
                )
                conditioning["hair_edge_retention_ratio"] = float(
                    suppressed_stats.get("hair_roi_edge_sum", 0.0)
                ) / max(float(raw_stats.get("hair_roi_edge_sum", 0.0)), 1e-6)
                conditioning["chest_edge_retention_ratio"] = float(
                    suppressed_stats.get("chest_center_edge_sum", 0.0)
                ) / max(float(raw_stats.get("chest_center_edge_sum", 0.0)), 1e-6)
                conditioning["left_side_edge_retention_ratio"] = float(
                    suppressed_stats.get("left_side_edge_sum", 0.0)
                ) / max(float(raw_stats.get("left_side_edge_sum", 0.0)), 1e-6)
                conditioning["right_side_edge_retention_ratio"] = float(
                    suppressed_stats.get("right_side_edge_sum", 0.0)
                ) / max(float(raw_stats.get("right_side_edge_sum", 0.0)), 1e-6)
                logger.info(
                    "[SDPipeline][diag][edge-ratio] hair=%.4f chest=%.4f left=%.4f right=%.4f",
                    conditioning["hair_edge_retention_ratio"],
                    conditioning["chest_edge_retention_ratio"],
                    conditioning["left_side_edge_retention_ratio"],
                    conditioning["right_side_edge_retention_ratio"],
                )

        # ── Step 5: 얼굴 crop (IP-Adapter) ───────────────────────────────────
        # bangs_requested=True 인 경우, img_pil(LaMA 처리 후 앞머리 없음)을 레퍼런스로
        # 쓰면 IP-Adapter가 "앞머리 없는 얼굴"을 참조해 SD가 앞머리를 올리는 방향으로
        # 생성하는 역전 현상이 발생한다. 앞머리 유지/생성 요청 시에는 원본 이미지로 crop.
        _face_crop_src = Image.fromarray(img_rgb) if bangs_requested else img_pil
        face_crop_pil = self._crop_face(_face_crop_src, face_bbox)
        if debug_images_common is not None:
            debug_images_common["ip_adapter_face_crop"] = cv2.cvtColor(
                np.array(face_crop_pil),
                cv2.COLOR_RGB2BGR,
            )
            debug_images_common["ip_adapter_face_crop_source"] = (
                "original_rgb" if bangs_requested else "cleaned_rgb"
            )

        source_garment_prompt_hints: Dict[str, Any] = {}
        source_garment_prompt_support_mask = np.zeros((H, W), dtype=np.float32)
        source_garment_prompt_hints_enabled = bool(
            subject_profile.use_source_garment_prompt_hints
        )
        if (
            hair_length in ("short", "medium", "long")
            and source_garment_prompt_hints_enabled
        ):
            try:
                (
                    source_garment_prompt_hints,
                    source_garment_prompt_support_mask,
                ) = self._extract_source_garment_prompt_hints(
                    img_rgb,
                    cloth_mask=cloth_mask,
                    torso_candidate_mask=subject_torso_candidate_mask,
                    source_cloth_overlap_mask=source_cloth_overlap_mask,
                    hair_mask_for_removal=hair_mask_for_removal,
                    protect_mask=protect_mask_for_sd,
                    face_bbox=face_bbox,
                )
            except Exception as e:
                logger.warning(
                    f"[SDPipeline] source garment prompt hint extraction failed (ignored): {e}"
                )
                source_garment_prompt_hints = {}
                source_garment_prompt_support_mask = np.zeros((H, W), dtype=np.float32)
        elif hair_length in ("short", "medium", "long"):
            logger.info(
                "[SDPipeline] source garment prompt hint extraction skipped for subject branch=%s",
                subject_profile.key,
            )
        _store_mask(
            "pipeline_source_garment_prompt_support_mask",
            source_garment_prompt_support_mask,
        )
        if debug_data_common is not None and source_garment_prompt_hints:
            debug_data_common["source_garment_prompt_hints"] = (
                source_garment_prompt_hints
            )
        if debug_data_common is not None:
            debug_data_common["source_garment_prompt_hints_enabled"] = (
                source_garment_prompt_hints_enabled
            )
        if source_garment_prompt_hints:
            logger.info(
                "[SDPipeline] source garment prompt hints: color=%s pattern=%s material=%s neckline=%s support_px=%s",
                source_garment_prompt_hints.get("color_name"),
                source_garment_prompt_hints.get("pattern_type"),
                source_garment_prompt_hints.get("material_hint"),
                source_garment_prompt_hints.get("neckline_hint"),
                source_garment_prompt_hints.get("support_pixels"),
            )

        # ── Step 6: 프롬프트 ─────────────────────────────────────────────────
        prompt, neg_prompt, guidance, prompt_meta = self._build_prompt(
            effective_hairstyle_text,
            normalized_color_text,
            hair_length,
            subject_gender=subject_gender_mode,
            sd_prompt_data=sd_prompt_data,
            prompt_context=normalized_prompt_context,
            source_garment_hints=source_garment_prompt_hints,
            white_tshirt_experiment=white_tshirt_experiment,
        )
        request_resolution = {
            "resolved_gender_branch": prompt_meta.get(
                "resolved_gender_branch", subject_gender_mode
            ),
            "resolved_canonical_preferences": prompt_meta.get(
                "canonical_preferences", {}
            ),
            "generation_backend": generation_spec.key,
            "generation_model_id": generation_spec.model_id,
            "generation_canvas_size": int(generation_canvas_size),
            "final_internal_prompt": prompt,
            "negative_prompt": neg_prompt,
            "blocked_vocabulary": prompt_meta.get("blocked_vocabulary", []),
            "fallback_mode": bool(prompt_meta.get("fallback_mode")),
            "structured_payload_used": bool(prompt_meta.get("structured_payload_used")),
            "style_source": prompt_meta.get("style_source"),
            "normalized_style": prompt_meta.get("normalized_style"),
        }
        generation_ip_scale, generation_control_scale = (
            self._resolve_generation_conditioning(
                hair_length,
                subject_gender=subject_gender_mode,
            )
        )
        logger.info(f"[SDPipeline] 프롬프트: {prompt}")
        logger.info(f"[SDPipeline] 네거티브: {neg_prompt}")
        logger.info(f"[SDPipeline] guidance_scale: {guidance}")
        logger.info("[SDPipeline] request_resolution=%s", request_resolution)
        logger.info(
            "[SDPipeline] generation conditioning: ip_adapter_scale=%.4f controlnet_scale=%.4f internal_candidates=%d",
            generation_ip_scale,
            generation_control_scale,
            len(seeds),
        )
        if debug_data_common is not None:
            debug_data_common["generation_prompt"] = {
                "positive": prompt,
                "negative": neg_prompt,
                "guidance_scale": float(guidance),
            }
            debug_data_common["request_resolution"] = request_resolution
            debug_data_common["generation_conditioning"] = {
                "backend": generation_spec.key,
                "canvas_size": int(generation_canvas_size),
                "steps": int(generation_steps),
                "ip_adapter_scale": float(generation_ip_scale),
                "controlnet_scale": float(generation_control_scale),
                "internal_candidate_count": int(len(seeds)),
            }

        # ── Step 7: SD Inpainting ─────────────────────────────────────────────
        gen_images = self._generate(
            img_512,
            mask_512,
            canny_512,
            face_crop_pil,
            prompt,
            neg_prompt,
            guidance,
            seeds,
            hair_length=hair_length,
            subject_gender=subject_gender_mode,
        )
        logger.info(
            "[SDPipeline] generation batch returned: images=%d requested_top_k=%d internal_candidates=%d",
            len(gen_images),
            requested_top_k,
            len(seeds),
        )

        # ── Step 8: Composite → 원본 해상도 ───────────────────────────────────
        # 전략 2는 원본 위에 short 생성물을 합성한 뒤, cutoff 아래 잔여 긴머리만 정리한다.
        composite_base_rgb = img_rgb_cleaned
        composite_base_bgr = cv2.cvtColor(composite_base_rgb, cv2.COLOR_RGB2BGR)
        if debug_images_common is not None:
            debug_images_common["pipeline_composite_base_rgb"] = (
                composite_base_bgr.copy()
            )
        male_medium_source_profile: Optional[Dict[str, float]] = None
        if hair_length == "medium" and subject_gender_mode == "male":
            male_medium_source_profile = self._estimate_hair_shape_profile(
                hair_mask_base,
                face_bbox,
                hair_length="medium",
            )

        candidates: List[Dict[str, Any]] = []
        for gen_idx, (gen_pil, seed) in enumerate(zip(gen_images, seeds)):
            logger.info(
                "[SDPipeline] candidate postprocess start: idx=%d/%d seed=%d",
                gen_idx + 1,
                len(gen_images),
                int(seed),
            )
            gen_preview_bgr = cv2.cvtColor(np.array(gen_pil), cv2.COLOR_RGB2BGR)
            generated_resized_rgb = _project_generated_to_original(
                gen_pil, scale, pad, (W, H)
            )
            composite_before_core_mask = gen_mask_after_upper_overwrite.astype(
                np.float32
            )
            composite_after_core_mask = gen_mask_after_final_core.astype(np.float32)
            composite_before_core_bgr = self._composite(
                composite_base_bgr,
                composite_base_rgb,
                gen_pil,
                composite_before_core_mask,
                scale,
                pad,
                (W, H),
                garment_mask=None,
                protect_mask=protect_mask_for_sd,
                protect_release_mask=(
                    composite_bangs_release_mask
                    if float(composite_bangs_release_mask.sum()) > 0.0
                    else None
                ),
                hair_length=hair_length,
                subject_gender=subject_gender_mode,
                fringe_requested=bangs_requested,
            )
            composite_after_core_mask_bgr = self._composite(
                composite_base_bgr,
                composite_base_rgb,
                gen_pil,
                composite_after_core_mask,
                scale,
                pad,
                (W, H),
                garment_mask=None,
                protect_mask=protect_mask_for_sd,
                protect_release_mask=(
                    composite_bangs_release_mask
                    if float(composite_bangs_release_mask.sum()) > 0.0
                    else None
                ),
                hair_length=hair_length,
                subject_gender=subject_gender_mode,
                fringe_requested=bangs_requested,
            )
            composite_mask = composite_hair_mask.astype(np.float32)
            if hair_length == "short":
                # 숏컷 변환 시, 원본 긴머리가 있던 곳을 생성물로 확실히 덮어씌워야 함
                if (
                    hair_mask_for_removal is not None
                    and hair_mask_for_removal.shape == (H, W)
                ):
                    composite_mask = np.maximum(
                        composite_mask,
                        np.clip(
                            hair_mask_for_removal.astype(np.float32) * 1.05, 0.0, 1.0
                        ),
                    )
                if (
                    removal_mask_for_post is not None
                    and removal_mask_for_post.shape == (H, W)
                ):
                    composite_mask = np.maximum(
                        composite_mask,
                        np.clip(
                            removal_mask_for_post.astype(np.float32) * 1.05, 0.0, 1.0
                        ),
                    )
                if (
                    short_torso_generation_freeze_active
                    and short_torso_generation_freeze_mask.shape == (H, W)
                ):
                    composite_mask = np.clip(
                        composite_mask.astype(np.float32)
                        - short_torso_generation_freeze_mask.astype(np.float32) * 1.20,
                        0.0,
                        1.0,
                    ).astype(np.float32)
                if (
                    short_generation_tail_release_mask.shape == (H, W)
                    and short_generation_tail_release_px >= 80
                ):
                    composite_mask = np.maximum(
                        composite_mask.astype(np.float32),
                        np.clip(
                            short_generation_tail_release_mask.astype(np.float32)
                            * 1.05,
                            0.0,
                            1.0,
                        ),
                    ).astype(np.float32)

            garment_composite_mask = None
            if (
                not short_torso_generation_freeze_active
                and use_upper_clothes_overwrite
                and effective_upper_clothes_overwrite_core_mask.shape
                == hair_mask_for_sd.shape
            ):
                if float(effective_upper_clothes_overwrite_core_mask.sum()) > 0.0:
                    garment_composite_mask = (
                        effective_upper_clothes_overwrite_core_mask.astype(np.float32)
                    )
            if (
                garment_composite_mask is None
                and not short_torso_generation_freeze_active
                and use_upper_clothes_overwrite
                and effective_upper_clothes_overwrite_mask.shape
                == hair_mask_for_sd.shape
                and source_garment_prepass_mask.shape == hair_mask_for_sd.shape
            ):
                garment_composite_mask = np.minimum(
                    effective_upper_clothes_overwrite_mask.astype(np.float32),
                    np.clip(
                        source_garment_prepass_mask.astype(np.float32) * 1.08, 0.0, 1.0
                    ),
                ).astype(np.float32)
            if garment_composite_mask is not None:
                composite_mask = np.maximum(
                    composite_mask.astype(np.float32),
                    garment_composite_mask.astype(np.float32),
                ).astype(np.float32)
            # ── composite_mask에서 face protect 영역 선제 제거 ─────────────────
            # gen_mask(SD inpaint mask)에 requested_front_coverage_mask
            # (y1+0.56·face_h = 코끝 레벨)가 포함되어 composite_mask로 이어지면
            # hair_alpha가 눈/코 레벨까지 높게 계산됨.
            # _composite() 내부 face protect(alpha=0 강제)만으로는 soft-edge 경계에서
            # alpha가 새어나와 SD 생성물(dark artifact)이 눈/코에 찍히는 문제 발생.
            # → protect 영역을 미리 composite_mask에서 빼면 hair_alpha≈0 확보.
            #   이마 앞머리(이마 release mask)는 _composite() 내부에서 alpha를 복원함.
            # front=down 요청 + 원본 앞머리 존재 케이스에서 원본 bangs release를 그대로 쓰면
            # 이마 중앙에 얇은 원본 앞머리 띠가 남으면서 "두상이 두 개"처럼 보이는
            # band artifact가 생긴다. 생성된 fringe가 원본 앞머리를 완전히 덮도록
            # composite_bangs_release_mask만 사용하고, 원본 앞머리 preserve 분기는 끈다.
            _original_bangs_detected = (
                bangs_requested
                and _original_bang_px_in_protect > 100.0
            )
            _preserve_original_bangs = False
            if _original_bangs_detected:
                logger.info(
                    "[SDPipeline] front=down + original bangs detected (px=%.0f): "
                    "preferring generated fringe coverage over original-bangs preserve mask",
                    _original_bang_px_in_protect,
                )
            composite_mask_for_blend = composite_mask.astype(np.float32).copy()
            # [v185+] composite_mask 확장은 두상 왜곡을 유발하므로 제거.
            # protect_release_mask 만 사용: 앞머리 영역의 protect 해제 → 상단 생성 hair의
            # Gaussian blur alpha가 자연스럽게 이마까지 흘러내려 원본 앞머리를 부드럽게 덮음.
            if (
                bangs_requested
                and not _preserve_original_bangs
                and protect_mask_for_sd.shape == (H, W)
                and float(composite_bangs_release_mask.sum()) > 0.0
            ):
                _protect_subtract_weight = np.clip(
                    protect_mask_for_sd.astype(np.float32) * 1.4, 0.0, 1.0
                )
                composite_mask_for_blend = np.clip(
                    composite_mask_for_blend - _protect_subtract_weight, 0.0, 1.0
                )
                logger.debug(
                    "[SDPipeline] bangs composite_mask protect-subtracted: "
                    "px_before=%.0f px_after=%.0f",
                    float(composite_mask.sum()),
                    float(composite_mask_for_blend.sum()),
                )
            # protect_release_mask:
            # _preserve_original_bangs: 앞머리 마스크를 release로 써서
            # protect 경계가 부드럽게 해제되어 상단 생성 hair alpha가 이마쪽으로 서서히 번짐
            _bangs_release_for_composite = None
            if _preserve_original_bangs and float(_saved_original_bangs_mask.sum()) > 0.0:
                _bangs_release_for_composite = cv2.GaussianBlur(
                    np.clip(_saved_original_bangs_mask, 0.0, 1.0),
                    (0, 0),
                    sigmaX=5.0,
                    sigmaY=5.0,
                ).astype(np.float32)
            elif float(composite_bangs_release_mask.sum()) > 0.0:
                _release_sigma = 7.0 if bangs_requested else 9.0
                _bangs_release_for_composite = cv2.GaussianBlur(
                    np.clip(composite_bangs_release_mask, 0.0, 1.0),
                    (0, 0),
                    sigmaX=_release_sigma,
                    sigmaY=_release_sigma,
                ).astype(np.float32)

            composited_bgr = self._composite(
                composite_base_bgr,
                composite_base_rgb,
                gen_pil,
                composite_mask_for_blend,
                scale,
                pad,
                (W, H),
                garment_mask=garment_composite_mask,
                protect_mask=protect_mask_for_sd,  # 얼굴 영역 alpha 침범 방지
                protect_release_mask=_bangs_release_for_composite,
                hair_length=hair_length,
                subject_gender=subject_gender_mode,
                fringe_requested=bangs_requested,
            )
            if (
                not _preserve_original_bangs
                and float(composite_bangs_release_mask.sum()) > 60.0
            ):
                try:
                    # hair mask 영역을 제외한 hairline 전환 구간만 색보정
                    # → 두상 변형 없이 경계 색상 아티팩트만 제거
                    _hairline_only_mask = np.clip(
                        composite_bangs_release_mask
                        - np.clip(composite_mask.astype(np.float32) * 1.5, 0.0, 1.0),
                        0.0,
                        1.0,
                    ).astype(np.float32)
                    if float(_hairline_only_mask.sum()) > 40.0:
                        composited_bgr = cv2.cvtColor(
                            self._cv2_refine_cloth_region(
                                cv2.cvtColor(composited_bgr, cv2.COLOR_BGR2RGB),
                                _hairline_only_mask,
                                reference_rgb=img_rgb,
                                reference_mask=_hairline_only_mask,
                            ),
                            cv2.COLOR_RGB2BGR,
                        )
                except Exception:
                    pass
            if (
                hair_length == "short"
                and subject_gender_mode == "female"
                and not bangs_requested
                and (short_no_bangs_target or explicit_no_bangs_requested)
                and isinstance(generated_resized_rgb, np.ndarray)
                and generated_resized_rgb.shape[:2] == (H, W)
            ):
                try:
                    comp_rgb_for_halo = cv2.cvtColor(composited_bgr, cv2.COLOR_BGR2RGB)
                    halo_mask = self._build_no_bangs_hairline_halo_mask(
                        comp_rgb_for_halo,
                        generated_resized_rgb,
                        composite_mask.astype(np.float32),
                        no_bangs_forehead_lama_preclean_mask,
                        composite_bangs_release_mask,
                        protect_mask_for_sd,
                        face_bbox,
                        hair_length=hair_length,
                        subject_gender=subject_gender_mode,
                    )
                    if float(halo_mask.sum()) > 8.0:
                        halo_alpha = halo_mask[..., np.newaxis]
                        repaired_rgb = (
                            generated_resized_rgb.astype(np.float32) * halo_alpha
                            + comp_rgb_for_halo.astype(np.float32) * (1.0 - halo_alpha)
                        )
                        composited_bgr = cv2.cvtColor(
                            np.clip(repaired_rgb, 0, 255).astype(np.uint8),
                            cv2.COLOR_RGB2BGR,
                        )
                        if debug_images_common is not None and gen_idx == 0:
                            debug_images_common[
                                "pipeline_no_bangs_hairline_halo_repair_mask"
                            ] = cv2.cvtColor(
                                (np.clip(halo_mask, 0.0, 1.0) * 255).astype(np.uint8),
                                cv2.COLOR_GRAY2BGR,
                            )
                        if debug_data_common is not None and gen_idx == 0:
                            debug_data_common.setdefault("diagnostics", {})[
                                "no_bangs_hairline_halo_repair_px"
                            ] = int((halo_mask > 0.04).sum())
                except Exception as e:
                    logger.warning(
                        "[SDPipeline] no-bangs hairline halo repair failed (ignored): %s",
                        e,
                    )
            composite_pre_cleanup_bgr = composited_bgr.copy()
            candidate_cleanup_trace: List[Dict[str, Any]] = []
            candidate_prev_rgb = cv2.cvtColor(
                composite_pre_cleanup_bgr, cv2.COLOR_BGR2RGB
            )
            post_final_cutoff_cleanup_bgr: Optional[np.ndarray] = None
            post_remove_residual_below_cutoff_bgr: Optional[np.ndarray] = None
            final_cutoff_debug_masks: Dict[str, np.ndarray] = {}
            residual_cleanup_debug_masks: Dict[str, np.ndarray] = {}

            def _record_candidate_cleanup_stage(
                stage_name: str, current_bgr: np.ndarray
            ) -> None:
                nonlocal candidate_prev_rgb
                current_rgb = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2RGB)
                entry: Dict[str, Any] = {
                    "stage_name": stage_name,
                    "prev_mean_abs_diff": _mean_abs_diff(
                        candidate_prev_rgb, current_rgb
                    ),
                    "source_similarity_ratio": _source_similarity_ratio(
                        composite_base_rgb, current_rgb
                    ),
                }
                for roi_name in (
                    "chest_center",
                    "left_side",
                    "right_side",
                    "neckline",
                    "torso_front",
                ):
                    rect = diagnostic_rois.get(roi_name)
                    entry[f"{roi_name}_prev_mean_abs_diff"] = _mean_abs_diff(
                        candidate_prev_rgb, current_rgb, rect
                    )
                    entry[f"{roi_name}_source_similarity_ratio"] = (
                        _source_similarity_ratio(
                            composite_base_rgb,
                            current_rgb,
                            rect,
                        )
                    )
                candidate_cleanup_trace.append(entry)
                candidate_prev_rgb = current_rgb

            if (
                hair_length in ("short", "medium")
                and cutoff_y_for_post is not None
                and removal_mask_for_post is not None
            ):
                try:
                    # [v183 fix] 원본 앞머리 보존 모드: LaMA Cleanup이 앞머리를 잔머리로
                    # 오인하여 회색 박스 스머지를 만드는 것을 방지하기 위해,
                    # removal_mask_for_post 에서 _saved_original_bangs_mask 구역을 뺀다.
                    _cleanup_removal_mask = removal_mask_for_post
                    if _preserve_original_bangs and float(_saved_original_bangs_mask.sum()) > 0.0:
                        _bangs_exclusion = cv2.dilate(
                            (_saved_original_bangs_mask > 0.08).astype(np.uint8) * 255,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                            iterations=1,
                        ).astype(np.float32) / 255.0
                        _cleanup_removal_mask = np.clip(
                            removal_mask_for_post.astype(np.float32) - _bangs_exclusion,
                            0.0,
                            1.0,
                        ).astype(np.float32)
                        logger.info(
                            "[SDPipeline] preserve_original_bangs: bangs excluded from "
                            "removal_mask_for_post (px_before=%.0f px_after=%.0f)",
                            float(removal_mask_for_post.sum()),
                            float(_cleanup_removal_mask.sum()),
                        )
                    post_rgb = cv2.cvtColor(composited_bgr, cv2.COLOR_BGR2RGB)
                    post_rgb = self._final_cutoff_cleanup(
                        post_rgb,
                        face_bbox=face_bbox,
                        removal_mask=_cleanup_removal_mask,
                        cutoff_y=cutoff_y_for_post,
                        shoulder_protect=shoulder_protect_for_post,
                        neckline_preserve=neckline_preserve_for_post,
                        lateral_preserve=lateral_neck_preserve_for_post,
                        hair_length=hair_length,
                        center_anchor_mask=center_chest_strand_removal_mask,
                        debug_outputs=(
                            final_cutoff_debug_masks
                            if debug_images_common is not None and rank == 0
                            else None
                        ),
                    )
                    post_final_cutoff_cleanup_bgr = cv2.cvtColor(
                        post_rgb, cv2.COLOR_RGB2BGR
                    )
                    _record_candidate_cleanup_stage(
                        "final_cutoff_cleanup", post_final_cutoff_cleanup_bgr
                    )
                    post_rgb = self._remove_residual_hair_below_cutoff(
                        post_rgb,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        removal_mask=_cleanup_removal_mask,
                        shoulder_protect=shoulder_protect_for_post,
                        neckline_preserve=neckline_preserve_for_post,
                        lateral_preserve=lateral_neck_preserve_for_post,
                        hair_length=hair_length,
                        center_anchor_mask=center_chest_strand_removal_mask,
                        debug_outputs=(
                            residual_cleanup_debug_masks
                            if debug_images_common is not None and rank == 0
                            else None
                        ),
                    )
                    composited_bgr = cv2.cvtColor(post_rgb, cv2.COLOR_RGB2BGR)
                    post_remove_residual_below_cutoff_bgr = composited_bgr.copy()
                    _record_candidate_cleanup_stage(
                        "remove_residual_hair_below_cutoff",
                        post_remove_residual_below_cutoff_bgr,
                    )
                except Exception as e:
                    logger.warning(
                        f"[SDPipeline] short/medium 잔여물 cleanup 실패(무시): {e}"
                    )

            if not has_color_request:
                try:
                    post_rgb = cv2.cvtColor(composited_bgr, cv2.COLOR_BGR2RGB)
                    post_rgb = self._preserve_original_hair_tone(
                        source_rgb=img_rgb,
                        target_rgb=post_rgb,
                        face_bbox=face_bbox,
                    )
                    composited_bgr = cv2.cvtColor(post_rgb, cv2.COLOR_RGB2BGR)
                    _record_candidate_cleanup_stage(
                        "preserve_original_hair_tone", composited_bgr
                    )
                except Exception as e:
                    logger.warning(f"[SDPipeline] 원본 컬러 유지 보정 실패(무시): {e}")

            color_distance: Optional[float] = None
            color_score = 0.0
            if has_color_request and target_hair_lab is not None:
                try:
                    post_rgb = cv2.cvtColor(composited_bgr, cv2.COLOR_BGR2RGB)
                    color_distance = self._estimate_hair_color_distance(
                        img_rgb=post_rgb,
                        face_bbox=face_bbox,
                        target_lab=target_hair_lab,
                    )
                    if color_distance is not None:
                        color_score = float(
                            np.clip(1.0 - (color_distance / 80.0), 0.0, 1.0)
                        )
                except Exception as e:
                    logger.warning(f"[SDPipeline] 색상 거리 계산 실패(무시): {e}")

            tail_penalty: Optional[float] = None
            short_silhouette_penalty: Optional[float] = None
            if (
                hair_length == "short"
                and cutoff_y_for_post is not None
                and removal_mask_for_post is not None
            ):
                try:
                    short_silhouette_penalty = self._estimate_short_silhouette_penalty(
                        img_rgb=generated_resized_rgb,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        removal_mask=removal_mask_for_post,
                    )
                except Exception as e:
                    logger.warning(
                        f"[SDPipeline] short silhouette penalty 계산 실패(무시): {e}"
                    )
                try:
                    post_rgb = cv2.cvtColor(composited_bgr, cv2.COLOR_BGR2RGB)
                    tail_penalty = self._estimate_short_tail_penalty(
                        img_rgb=post_rgb,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        removal_mask=removal_mask_for_post,
                    )
                except Exception as e:
                    logger.warning(
                        f"[SDPipeline] short tail penalty 계산 실패(무시): {e}"
                    )

            accessory_penalty: Optional[float] = None
            accessory_details: Optional[Dict[str, Any]] = None
            accessory_excluded = False
            try:
                post_rgb = cv2.cvtColor(composited_bgr, cv2.COLOR_BGR2RGB)
                accessory_details = self._estimate_accessory_penalty_details(
                    img_rgb=post_rgb,
                    face_bbox=face_bbox,
                    source_accessory_profile=source_accessory_profile,
                )
                if isinstance(accessory_details, dict):
                    accessory_penalty = float(
                        accessory_details.get("total_penalty", 0.0)
                    )
                    accessory_excluded = bool(accessory_details.get("exclude"))
            except Exception as e:
                logger.warning(f"[SDPipeline] accessory penalty 계산 실패(무시): {e}")

            male_medium_fit_penalty: Optional[float] = None
            if hair_length == "medium" and subject_gender_mode == "male":
                try:
                    post_rgb = cv2.cvtColor(composited_bgr, cv2.COLOR_BGR2RGB)
                    male_medium_fit_penalty = self._estimate_male_medium_fit_penalty(
                        img_rgb=post_rgb,
                        face_bbox=face_bbox,
                        source_profile=male_medium_source_profile,
                    )
                except Exception as e:
                    logger.warning(
                        f"[SDPipeline] male medium fit penalty 怨꾩궛 ?ㅽ뙣(臾댁떆): {e}"
                    )

            candidates.append(
                {
                    "seed": seed,
                    "image_bgr": composited_bgr,
                    "preview_bgr": gen_preview_bgr,
                    "generated_resized_rgb": generated_resized_rgb,
                    "composite_before_core_bgr": composite_before_core_bgr,
                    "composite_after_core_mask_bgr": composite_after_core_mask_bgr,
                    "composite_pre_cleanup_bgr": composite_pre_cleanup_bgr,
                    "post_final_cutoff_cleanup_bgr": post_final_cutoff_cleanup_bgr,
                    "post_remove_residual_below_cutoff_bgr": post_remove_residual_below_cutoff_bgr,
                    "final_cutoff_debug_masks": final_cutoff_debug_masks,
                    "residual_cleanup_debug_masks": residual_cleanup_debug_masks,
                    "candidate_cleanup_trace": candidate_cleanup_trace,
                    "composite_before_core_mask": composite_before_core_mask.copy(),
                    "composite_after_core_mask": composite_after_core_mask.copy(),
                    "composite_mask": composite_mask.astype(np.float32),
                    "garment_composite_mask": (
                        garment_composite_mask.astype(np.float32)
                        if isinstance(garment_composite_mask, np.ndarray)
                        else None
                    ),
                    "color_distance": color_distance,
                    "color_score": color_score,
                    "short_silhouette_penalty": short_silhouette_penalty,
                    "tail_penalty": tail_penalty,
                    "accessory_penalty": accessory_penalty,
                    "accessory_details": accessory_details,
                    "accessory_excluded": accessory_excluded,
                    "male_medium_fit_penalty": male_medium_fit_penalty,
                    "gen_idx": gen_idx,
                }
            )
            logger.info(
                "[SDPipeline] candidate postprocess done: idx=%d/%d seed=%d short_silhouette_penalty=%s tail_penalty=%s accessory_penalty=%s accessory_excluded=%s",
                gen_idx + 1,
                len(gen_images),
                int(seed),
                (
                    "none"
                    if short_silhouette_penalty is None
                    else f"{float(short_silhouette_penalty):.4f}"
                ),
                "none" if tail_penalty is None else f"{float(tail_penalty):.4f}",
                (
                    "none"
                    if accessory_penalty is None
                    else f"{float(accessory_penalty):.4f}"
                ),
                accessory_excluded,
            )

        if has_color_request and target_hair_lab is not None and len(candidates) > 1:
            sortable_count = sum(c["color_distance"] is not None for c in candidates)
            if sortable_count >= 2:
                if subject_profile.key == "male":
                    sort_key = lambda c: (
                        bool(c.get("accessory_excluded", False)),
                        c["male_medium_fit_penalty"] is None,
                        (
                            c["male_medium_fit_penalty"]
                            if c["male_medium_fit_penalty"] is not None
                            else 1e9
                        ),
                        c["accessory_penalty"] is None,
                        (
                            c["accessory_penalty"]
                            if c["accessory_penalty"] is not None
                            else 1e9
                        ),
                        c["color_distance"] is None,
                        c["color_distance"] if c["color_distance"] is not None else 1e9,
                        c["gen_idx"],
                    )
                else:
                    sort_key = lambda c: (
                        bool(c.get("accessory_excluded", False)),
                        c["accessory_penalty"] is None,
                        (
                            c["accessory_penalty"]
                            if c["accessory_penalty"] is not None
                            else 1e9
                        ),
                        c["male_medium_fit_penalty"] is None,
                        (
                            c["male_medium_fit_penalty"]
                            if c["male_medium_fit_penalty"] is not None
                            else 1e9
                        ),
                        c["color_distance"] is None,
                        c["color_distance"] if c["color_distance"] is not None else 1e9,
                        c["gen_idx"],
                    )
                candidates.sort(key=sort_key)
                logger.info("[SDPipeline] 컬러 유사도 기준으로 결과 재정렬 완료")
            else:
                logger.info("[SDPipeline] 컬러 유사도 재정렬 스킵 (유효 샘플 부족)")
        elif len(candidates) > 1:
            accessory_sortable = sum(
                c["accessory_penalty"] is not None for c in candidates
            )
            fit_sortable = sum(
                c["male_medium_fit_penalty"] is not None for c in candidates
            )
            if accessory_sortable >= 2 or fit_sortable >= 2:
                if subject_profile.key == "male":
                    sort_key = lambda c: (
                        bool(c.get("accessory_excluded", False)),
                        c["male_medium_fit_penalty"] is None,
                        (
                            c["male_medium_fit_penalty"]
                            if c["male_medium_fit_penalty"] is not None
                            else 1e9
                        ),
                        c["accessory_penalty"] is None,
                        (
                            c["accessory_penalty"]
                            if c["accessory_penalty"] is not None
                            else 1e9
                        ),
                        c["gen_idx"],
                    )
                else:
                    sort_key = lambda c: (
                        bool(c.get("accessory_excluded", False)),
                        c["accessory_penalty"] is None,
                        (
                            c["accessory_penalty"]
                            if c["accessory_penalty"] is not None
                            else 1e9
                        ),
                        c["male_medium_fit_penalty"] is None,
                        (
                            c["male_medium_fit_penalty"]
                            if c["male_medium_fit_penalty"] is not None
                            else 1e9
                        ),
                        c["gen_idx"],
                    )
                candidates.sort(key=sort_key)
                if fit_sortable >= 2:
                    logger.info("[SDPipeline] male medium fit ranking applied")
                else:
                    logger.info("[SDPipeline] accessory penalty ranking applied")

        if hair_length == "short" and len(candidates) > 1:
            silhouette_sortable = sum(
                c["short_silhouette_penalty"] is not None for c in candidates
            )
            tail_sortable = sum(c["tail_penalty"] is not None for c in candidates)
            accessory_sortable = sum(
                c["accessory_penalty"] is not None for c in candidates
            )
            if (
                silhouette_sortable >= 2
                or tail_sortable >= 2
                or accessory_sortable >= 2
            ):
                candidates.sort(
                    key=lambda c: (
                        bool(c.get("accessory_excluded", False)),
                        c["short_silhouette_penalty"] is None,
                        (
                            c["short_silhouette_penalty"]
                            if c["short_silhouette_penalty"] is not None
                            else 1e9
                        ),
                        c["tail_penalty"] is None,
                        c["tail_penalty"] if c["tail_penalty"] is not None else 1e9,
                        c["accessory_penalty"] is None,
                        (
                            c["accessory_penalty"]
                            if c["accessory_penalty"] is not None
                            else 1e9
                        ),
                        c["color_distance"] is None,
                        c["color_distance"] if c["color_distance"] is not None else 1e9,
                        c["gen_idx"],
                    )
                )
                logger.info(
                    "[SDPipeline] short candidate ranking applied (silhouette_sortable=%d tail_sortable=%d accessory_sortable=%d)",
                    silhouette_sortable,
                    tail_sortable,
                    accessory_sortable,
                )

        if debug_data_common is not None:
            diag = debug_data_common.setdefault("diagnostics", {})
            diag["candidate_ranking"] = [
                {
                    "final_order": int(idx),
                    "seed": int(c["seed"]),
                    "gen_idx": int(c["gen_idx"]),
                    "accessory_excluded": bool(c.get("accessory_excluded", False)),
                    "short_silhouette_penalty": (
                        None
                        if c["short_silhouette_penalty"] is None
                        else float(c["short_silhouette_penalty"])
                    ),
                    "tail_penalty": (
                        None if c["tail_penalty"] is None else float(c["tail_penalty"])
                    ),
                    "accessory_penalty": (
                        None
                        if c["accessory_penalty"] is None
                        else float(c["accessory_penalty"])
                    ),
                    "accessory_details": (
                        None
                        if not isinstance(c.get("accessory_details"), dict)
                        else {
                            "exclude": bool(
                                c["accessory_details"].get("exclude", False)
                            ),
                            "total_penalty": float(
                                c["accessory_details"].get("total_penalty", 0.0)
                            ),
                            "headwear_penalty": float(
                                c["accessory_details"].get("headwear_penalty", 0.0)
                            ),
                            "headwear_surface_penalty": float(
                                c["accessory_details"].get(
                                    "headwear_surface_penalty", 0.0
                                )
                            ),
                            "headwear_brim_penalty": float(
                                c["accessory_details"].get("headwear_brim_penalty", 0.0)
                            ),
                            "glasses_penalty": float(
                                c["accessory_details"].get("glasses_penalty", 0.0)
                            ),
                            "jewelry_penalty": float(
                                c["accessory_details"].get("jewelry_penalty", 0.0)
                            ),
                            "candidate_profile": {
                                key: float(value)
                                for key, value in (
                                    c["accessory_details"].get("candidate_profile", {})
                                    or {}
                                ).items()
                            },
                        }
                    ),
                    "color_distance": (
                        None
                        if c["color_distance"] is None
                        else float(c["color_distance"])
                    ),
                }
                for idx, c in enumerate(candidates)
            ]
            eligible_candidates = [
                c for c in candidates if not bool(c.get("accessory_excluded", False))
            ]
            diag["candidate_selection"] = {
                "requested_top_k": int(requested_top_k),
                "total_candidates": int(len(candidates)),
                "eligible_candidates": int(len(eligible_candidates)),
                "excluded_candidates": int(len(candidates) - len(eligible_candidates)),
            }

        selected_candidates = [
            c for c in candidates if not bool(c.get("accessory_excluded", False))
        ]
        if selected_candidates:
            selected_candidates = selected_candidates[:requested_top_k]
        elif candidates:
            selected_candidates = candidates[:requested_top_k]
            logger.warning(
                "[SDPipeline] all candidates were accessory-filtered; falling back to the least-penalized candidate"
            )
            if debug_data_common is not None:
                diag = debug_data_common.setdefault("diagnostics", {})
                diag.setdefault("candidate_selection", {})
                diag["candidate_selection"][
                    "fallback_reason"
                ] = "all_candidates_excluded"
        else:
            selected_candidates = []

        results: List[SDInpaintResult] = []
        for rank, cand in enumerate(selected_candidates):
            if debug_images_common is not None and rank == 0:
                debug_images_common[
                    f"sd_generated_rank0_{generation_canvas_size}"
                ] = cand["preview_bgr"]
                if generation_canvas_size == SD_SIZE:
                    debug_images_common["sd_generated_rank0_512"] = cand["preview_bgr"]
                if isinstance(cand.get("generated_resized_rgb"), np.ndarray):
                    debug_images_common["pipeline_generated_rank0_resized_rgb"] = (
                        cv2.cvtColor(
                            cand["generated_resized_rgb"],
                            cv2.COLOR_RGB2BGR,
                        )
                    )
                    _store_rgb(
                        "pipeline_generated_rank0_resized_with_overwrite_core_overlay",
                        _overlay_mask_rgb(
                            cand["generated_resized_rgb"],
                            overwrite_core_restore_mask_for_debug,
                            (255, 96, 64),
                            alpha=0.52,
                        ),
                    )
                    _store_roi_crops("generated_resized", cand["generated_resized_rgb"])
                if isinstance(cand.get("composite_before_core_bgr"), np.ndarray):
                    debug_images_common["pipeline_composite_rank0_before_core"] = cand[
                        "composite_before_core_bgr"
                    ].copy()
                    _store_roi_crops(
                        "composite_before_core",
                        cv2.cvtColor(
                            cand["composite_before_core_bgr"], cv2.COLOR_BGR2RGB
                        ),
                    )
                if isinstance(cand.get("composite_after_core_mask_bgr"), np.ndarray):
                    debug_images_common["pipeline_composite_rank0_after_core_mask"] = (
                        cand["composite_after_core_mask_bgr"].copy()
                    )
                    _store_roi_crops(
                        "composite_after_core_mask",
                        cv2.cvtColor(
                            cand["composite_after_core_mask_bgr"], cv2.COLOR_BGR2RGB
                        ),
                    )
                if isinstance(cand.get("composite_pre_cleanup_bgr"), np.ndarray):
                    debug_images_common["pipeline_composite_rank0_pre_cleanup"] = cand[
                        "composite_pre_cleanup_bgr"
                    ].copy()
                    _store_roi_crops(
                        "composite_pre_cleanup",
                        cv2.cvtColor(
                            cand["composite_pre_cleanup_bgr"], cv2.COLOR_BGR2RGB
                        ),
                    )
                if isinstance(cand.get("post_final_cutoff_cleanup_bgr"), np.ndarray):
                    debug_images_common["pipeline_post_final_cutoff_cleanup_rank0"] = (
                        cand["post_final_cutoff_cleanup_bgr"].copy()
                    )
                    _store_roi_crops(
                        "post_final_cutoff_cleanup",
                        cv2.cvtColor(
                            cand["post_final_cutoff_cleanup_bgr"], cv2.COLOR_BGR2RGB
                        ),
                    )
                if isinstance(
                    cand.get("post_remove_residual_below_cutoff_bgr"), np.ndarray
                ):
                    debug_images_common[
                        "pipeline_post_remove_residual_below_cutoff_rank0"
                    ] = cand["post_remove_residual_below_cutoff_bgr"].copy()
                    _store_roi_crops(
                        "post_remove_residual_below_cutoff",
                        cv2.cvtColor(
                            cand["post_remove_residual_below_cutoff_bgr"],
                            cv2.COLOR_BGR2RGB,
                        ),
                    )
                for name, mask in (cand.get("final_cutoff_debug_masks") or {}).items():
                    _store_mask(f"pipeline_{name}", mask)
                for name, mask in (
                    cand.get("residual_cleanup_debug_masks") or {}
                ).items():
                    _store_mask(f"pipeline_{name}", mask)
                if isinstance(cand.get("composite_mask"), np.ndarray):
                    debug_images_common["pipeline_composite_mask_rank0"] = cv2.cvtColor(
                        (
                            (
                                np.clip(
                                    cand["composite_mask"].astype(np.float32), 0.0, 1.0
                                )
                                > 0.08
                            ).astype(np.uint8)
                            * 255
                        ),
                        cv2.COLOR_GRAY2BGR,
                    )
                if isinstance(cand.get("garment_composite_mask"), np.ndarray):
                    debug_images_common["pipeline_garment_composite_mask_rank0"] = (
                        cv2.cvtColor(
                            (
                                (
                                    np.clip(
                                        cand["garment_composite_mask"].astype(
                                            np.float32
                                        ),
                                        0.0,
                                        1.0,
                                    )
                                    > 0.08
                                ).astype(np.uint8)
                                * 255
                            ),
                            cv2.COLOR_GRAY2BGR,
                        )
                    )
                if isinstance(cand.get("composite_before_core_mask"), np.ndarray):
                    debug_images_common["pipeline_composite_before_core_mask_rank0"] = (
                        cv2.cvtColor(
                            (
                                (
                                    np.clip(
                                        cand["composite_before_core_mask"].astype(
                                            np.float32
                                        ),
                                        0.0,
                                        1.0,
                                    )
                                    > 0.08
                                ).astype(np.uint8)
                                * 255
                            ),
                            cv2.COLOR_GRAY2BGR,
                        )
                    )
                if isinstance(cand.get("composite_after_core_mask"), np.ndarray):
                    debug_images_common["pipeline_composite_after_core_mask_rank0"] = (
                        cv2.cvtColor(
                            (
                                (
                                    np.clip(
                                        cand["composite_after_core_mask"].astype(
                                            np.float32
                                        ),
                                        0.0,
                                        1.0,
                                    )
                                    > 0.08
                                ).astype(np.uint8)
                                * 255
                            ),
                            cv2.COLOR_GRAY2BGR,
                        )
                    )
            final_bgr = cand["image_bgr"]
            rank0_cleanup_stage_trace: Optional[List[Dict[str, Any]]] = None
            rank0_prev_stage_rgb: Optional[np.ndarray] = None
            rank0_composite_pre_cleanup_rgb = (
                cv2.cvtColor(cand["composite_pre_cleanup_bgr"], cv2.COLOR_BGR2RGB)
                if isinstance(cand.get("composite_pre_cleanup_bgr"), np.ndarray)
                else None
            )
            if debug_data_common is not None and rank == 0:
                diag = debug_data_common.setdefault("diagnostics", {})
                diag["candidate_cleanup_trace"] = (
                    cand.get("candidate_cleanup_trace") or []
                )
                torso_rect = diagnostic_rois.get("torso_front")
                for name, mask in (cand.get("final_cutoff_debug_masks") or {}).items():
                    _mask_stats(name, mask, torso_rect, bucket="cleanup_mask_stats")
                for name, mask in (
                    cand.get("residual_cleanup_debug_masks") or {}
                ).items():
                    _mask_stats(name, mask, torso_rect, bucket="cleanup_mask_stats")
                rank0_cleanup_stage_trace = []
                diag["cleanup_stage_trace"] = rank0_cleanup_stage_trace
                rank0_prev_stage_rgb = (
                    rank0_composite_pre_cleanup_rgb.copy()
                    if isinstance(rank0_composite_pre_cleanup_rgb, np.ndarray)
                    else cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                )

                def _record_rank0_cleanup_stage(
                    stage_name: str,
                    image_bgr: np.ndarray,
                    *,
                    trigger_mask: Optional[np.ndarray] = None,
                    mask_label: Optional[str] = None,
                    force: bool = False,
                    extra: Optional[Dict[str, Any]] = None,
                ) -> None:
                    nonlocal rank0_prev_stage_rgb
                    if rank0_cleanup_stage_trace is None:
                        return
                    current_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    prev_rgb = rank0_prev_stage_rgb
                    prev_mean_abs_diff = _mean_abs_diff(prev_rgb, current_rgb)
                    if (
                        not force
                        and prev_mean_abs_diff <= 0.01
                        and trigger_mask is None
                    ):
                        rank0_prev_stage_rgb = current_rgb
                        return
                    stage_idx = len(rank0_cleanup_stage_trace)
                    stage_key = f"cleanup_stage_{stage_idx:02d}_{stage_name}"
                    entry: Dict[str, Any] = {
                        "stage_name": stage_name,
                        "stage_key": stage_key,
                        "prev_mean_abs_diff": prev_mean_abs_diff,
                        "source_similarity_ratio": _source_similarity_ratio(
                            composite_base_rgb, current_rgb
                        ),
                    }
                    prev_source_similarity = _source_similarity_ratio(
                        composite_base_rgb, prev_rgb
                    )
                    entry["source_similarity_delta"] = (
                        entry["source_similarity_ratio"] - prev_source_similarity
                    )
                    for roi_name in (
                        "chest_center",
                        "left_side",
                        "right_side",
                        "neckline",
                        "torso_front",
                    ):
                        rect = diagnostic_rois.get(roi_name)
                        entry[f"{roi_name}_prev_mean_abs_diff"] = _mean_abs_diff(
                            prev_rgb, current_rgb, rect
                        )
                        current_sim = _source_similarity_ratio(
                            composite_base_rgb, current_rgb, rect
                        )
                        prev_sim = _source_similarity_ratio(
                            composite_base_rgb, prev_rgb, rect
                        )
                        entry[f"{roi_name}_source_similarity_ratio"] = current_sim
                        entry[f"{roi_name}_source_similarity_delta"] = (
                            current_sim - prev_sim
                        )
                    if extra:
                        entry.update(extra)
                    if trigger_mask is not None:
                        cleanup_mask_name = mask_label or stage_name
                        _mask_stats(
                            cleanup_mask_name,
                            trigger_mask,
                            diagnostic_rois.get("torso_front"),
                            bucket="cleanup_mask_stats",
                        )
                        entry["mask_label"] = cleanup_mask_name
                    if debug_images_common is not None:
                        debug_images_common[stage_key] = image_bgr.copy()
                        _store_roi_crops(stage_key, current_rgb)
                    rank0_cleanup_stage_trace.append(entry)
                    logger.info(
                        "[SDPipeline][diag][cleanup-stage] %s prev=%.2f chest=%.2f left=%.2f right=%.2f src_delta=%.4f",
                        stage_name,
                        entry["prev_mean_abs_diff"],
                        entry.get("chest_center_prev_mean_abs_diff", 0.0),
                        entry.get("left_side_prev_mean_abs_diff", 0.0),
                        entry.get("right_side_prev_mean_abs_diff", 0.0),
                        entry.get("source_similarity_delta", 0.0),
                    )
                    rank0_prev_stage_rgb = current_rgb

                _record_rank0_cleanup_stage(
                    "candidate_postprocess_base",
                    final_bgr,
                    force=True,
                    extra={
                        "reference_stage": "composite_pre_cleanup",
                    },
                )
            if (
                self.config.enable_post_cloth_refine
                and hair_length in ("short", "medium")
                and cloth_mask_dilated is not None
                and removal_mask_for_post is not None
                and cutoff_y_for_post is not None
            ):
                try:
                    post_cloth_refine_applied = False
                    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                    final_hair_mask, _, _ = self._segface_hair_mask(
                        final_rgb, face_bbox
                    )
                    cloth_refine_mask = self._build_post_cloth_refine_mask(
                        removal_mask=removal_mask_for_post,
                        cloth_mask=cloth_mask_dilated,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        hair_length=hair_length,
                        protect_mask=protect_mask_for_sd,
                        final_hair_mask=final_hair_mask,
                        artifact_cleanup_mask=artifact_cleanup_mask_for_post,
                    )
                    cloth_refine_u8 = (cloth_refine_mask > 0.08).astype(np.uint8) * 255
                    if int((cloth_refine_u8 > 0).sum()) >= 100:
                        final_rgb = self._sd_refine_removed_region(
                            base_rgb=final_rgb,
                            removal_mask=cloth_refine_mask,
                            face_bbox=face_bbox,
                            face_crop_pil=face_crop_pil,
                            protect_mask=protect_mask_for_sd,
                            cloth_mask=cloth_mask_dilated,
                            hair_length=hair_length,
                            seed=int(cand["seed"]) + 1701,
                            refine_mode="cloth",
                            subject_gender=subject_gender_mode,
                            white_tshirt_experiment=white_tshirt_experiment,
                        )
                        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                        post_cloth_refine_applied = True
                        if debug_images_common is not None and rank == 0:
                            debug_images_common["pipeline_post_cloth_refine_mask"] = (
                                cv2.cvtColor(
                                    cloth_refine_u8,
                                    cv2.COLOR_GRAY2BGR,
                                )
                            )
                    if int((cloth_refine_u8 > 0).sum()) >= 80:
                        final_rgb = self._cv2_refine_cloth_region(
                            final_rgb,
                            cloth_refine_mask,
                            reference_rgb=img_rgb,
                            reference_mask=cloth_mask_dilated,
                        )
                        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                        post_cloth_refine_applied = True
                    if (
                        debug_data_common is not None
                        and rank == 0
                        and post_cloth_refine_applied
                    ):
                        _record_rank0_cleanup_stage(
                            "post_cloth_refine",
                            final_bgr,
                            trigger_mask=cloth_refine_mask,
                            mask_label="post_cloth_refine",
                        )
                except Exception as e:
                    logger.warning(f"[SDPipeline] post cloth refine 실패(무시): {e}")
            if (
                hair_length in ("short", "medium")
                and cloth_mask_dilated is not None
                and artifact_cleanup_mask_for_post is not None
                and cutoff_y_for_post is not None
            ):
                try:
                    final_artifact_cleanup_applied = False
                    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                    final_hair_mask, _, _ = self._segface_hair_mask(
                        final_rgb, face_bbox
                    )
                    x1, y1, x2, y2 = face_bbox
                    face_w = max(int(x2 - x1), 1)
                    face_h = max(int(y2 - y1), 1)
                    final_dark_tail_u8 = np.zeros((H, W), dtype=np.uint8)

                    cleanup_u8 = (
                        np.clip(
                            artifact_cleanup_mask_for_post.astype(np.float32), 0.0, 1.0
                        )
                        > 0.18
                    ).astype(np.uint8) * 255
                    cleanup_u8 = cv2.bitwise_and(
                        cleanup_u8,
                        (
                            np.clip(cloth_mask_dilated.astype(np.float32), 0.0, 1.0)
                            > 0.10
                        ).astype(np.uint8)
                        * 255,
                    )
                    cleanup_corridor_u8 = np.zeros((H, W), dtype=np.uint8)
                    cleanup_top = max(0, int(cutoff_y_for_post - face_h * 0.02))
                    cleanup_bottom = min(
                        H,
                        int(
                            cutoff_y_for_post
                            + face_h * (0.78 if hair_length == "short" else 0.92)
                        ),
                    )
                    cleanup_left = max(
                        0, int(x1 - face_w * (0.84 if hair_length == "short" else 1.12))
                    )
                    cleanup_right = min(
                        W, int(x2 + face_w * (0.84 if hair_length == "short" else 1.12))
                    )
                    if cleanup_top < cleanup_bottom and cleanup_left < cleanup_right:
                        cleanup_corridor_u8[
                            cleanup_top:cleanup_bottom, cleanup_left:cleanup_right
                        ] = 255
                        cleanup_u8 = cv2.bitwise_and(cleanup_u8, cleanup_corridor_u8)
                    cleanup_anchor_u8 = cleanup_u8.copy()
                    if hair_length == "short":
                        cleanup_u8 = self._filter_short_center_cleanup_mask(
                            cleanup_u8,
                            face_bbox,
                            cutoff_y_for_post,
                            anchor_u8=cleanup_anchor_u8,
                            top_scale=0.00,
                            bottom_scale=0.78,
                            half_w_scale=0.25,
                            shrink_half_w_scale=0.18,
                            max_area_scale=0.10,
                            max_width_scale=0.34,
                            min_height_scale=0.10,
                            center_allow_scale=0.18,
                            max_total_scale=0.05,
                        )
                    support_cleanup_u8 = np.zeros((H, W), dtype=np.uint8)
                    support_cleanup_mask = lower_tail_support_for_post
                    if support_cleanup_mask is None or support_cleanup_mask.shape != (
                        H,
                        W,
                    ):
                        support_cleanup_mask = lower_tail_support_mask
                    support_corridor_u8 = np.zeros((H, W), dtype=np.uint8)
                    support_top = max(0, int(cutoff_y_for_post - face_h * 0.03))
                    support_bottom = min(
                        H,
                        int(
                            cutoff_y_for_post
                            + face_h * (0.92 if hair_length == "short" else 1.18)
                        ),
                    )
                    support_left = max(
                        0, int(x1 - face_w * (0.62 if hair_length == "short" else 0.92))
                    )
                    support_right = min(
                        W, int(x2 + face_w * (0.62 if hair_length == "short" else 0.92))
                    )
                    if support_top < support_bottom and support_left < support_right:
                        support_corridor_u8[
                            support_top:support_bottom, support_left:support_right
                        ] = 255
                    center_anchor_u8 = np.zeros((H, W), dtype=np.uint8)
                    if float(center_chest_strand_mask.sum()) > 0.0:
                        center_anchor_u8 = cv2.dilate(
                            (
                                np.clip(
                                    center_chest_strand_mask.astype(np.float32),
                                    0.0,
                                    1.0,
                                )
                                > 0.05
                            ).astype(np.uint8)
                            * 255,
                            cv2.getStructuringElement(
                                cv2.MORPH_ELLIPSE,
                                (7, 13) if hair_length == "short" else (11, 25),
                            ),
                            iterations=1,
                        )
                        center_anchor_u8 = cv2.bitwise_and(
                            center_anchor_u8, support_corridor_u8
                        )
                        center_anchor_u8 = cv2.bitwise_and(
                            center_anchor_u8,
                            (
                                np.clip(cloth_mask_dilated.astype(np.float32), 0.0, 1.0)
                                > 0.04
                            ).astype(np.uint8)
                            * 255,
                        )
                        cleanup_anchor_u8 = cv2.bitwise_or(
                            cleanup_anchor_u8, center_anchor_u8
                        )
                    if (
                        support_cleanup_mask is not None
                        and support_cleanup_mask.shape == (H, W)
                    ):
                        support_cleanup_u8 = (
                            np.clip(support_cleanup_mask.astype(np.float32), 0.0, 1.0)
                            > 0.08
                        ).astype(np.uint8) * 255
                        support_cleanup_u8 = cv2.bitwise_and(
                            support_cleanup_u8, support_corridor_u8
                        )
                        support_cleanup_u8 = cv2.bitwise_and(
                            support_cleanup_u8,
                            (
                                np.clip(cloth_mask_dilated.astype(np.float32), 0.0, 1.0)
                                > 0.04
                            ).astype(np.uint8)
                            * 255,
                        )
                        support_cleanup_u8 = cv2.dilate(
                            support_cleanup_u8,
                            cv2.getStructuringElement(
                                cv2.MORPH_ELLIPSE,
                                (7, 13) if hair_length == "short" else (9, 17),
                            ),
                            iterations=1,
                        )
                        if hair_length == "short":
                            support_cleanup_u8 = self._filter_short_center_cleanup_mask(
                                support_cleanup_u8,
                                face_bbox,
                                cutoff_y_for_post,
                                anchor_u8=cleanup_anchor_u8,
                                top_scale=0.00,
                                bottom_scale=0.86,
                                half_w_scale=0.24,
                                shrink_half_w_scale=0.18,
                                max_area_scale=0.10,
                                max_width_scale=0.34,
                                min_height_scale=0.10,
                                center_allow_scale=0.18,
                                max_total_scale=0.05,
                            )
                        cleanup_u8 = cv2.bitwise_or(cleanup_u8, support_cleanup_u8)
                    if float(center_chest_strand_mask.sum()) > 0.0:
                        center_cleanup_u8 = center_anchor_u8.copy()
                        if hair_length == "short":
                            center_cleanup_u8 = self._filter_short_center_cleanup_mask(
                                center_cleanup_u8,
                                face_bbox,
                                cutoff_y_for_post,
                                anchor_u8=center_anchor_u8,
                                top_scale=0.00,
                                bottom_scale=0.86,
                                half_w_scale=0.22,
                                shrink_half_w_scale=0.17,
                                max_area_scale=0.08,
                                max_width_scale=0.30,
                                min_height_scale=0.12,
                                center_allow_scale=0.16,
                                max_total_scale=0.04,
                            )
                        cleanup_u8 = cv2.bitwise_or(cleanup_u8, center_cleanup_u8)
                    final_dark_tail = self._build_dark_tail_residual_mask(
                        img_rgb=final_rgb,
                        removal_mask=removal_mask_for_post,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        hair_length=hair_length,
                    )
                    final_dark_tail_u8 = (final_dark_tail > 0.08).astype(np.uint8) * 255
                    if int((final_dark_tail_u8 > 0).sum()) > 0:
                        final_dark_tail_u8 = cv2.bitwise_and(
                            final_dark_tail_u8, cleanup_corridor_u8
                        )
                        final_dark_tail_u8 = cv2.bitwise_and(
                            final_dark_tail_u8,
                            (
                                np.clip(cloth_mask_dilated.astype(np.float32), 0.0, 1.0)
                                > 0.08
                            ).astype(np.uint8)
                            * 255,
                        )
                        if hair_length == "short":
                            final_dark_tail_u8 = self._filter_short_center_cleanup_mask(
                                final_dark_tail_u8,
                                face_bbox,
                                cutoff_y_for_post,
                                anchor_u8=cleanup_anchor_u8,
                                top_scale=0.00,
                                bottom_scale=0.82,
                                half_w_scale=0.25,
                                shrink_half_w_scale=0.18,
                                max_area_scale=0.10,
                                max_width_scale=0.34,
                                min_height_scale=0.10,
                                center_allow_scale=0.18,
                                max_total_scale=0.05,
                            )
                        cleanup_u8 = cv2.bitwise_or(cleanup_u8, final_dark_tail_u8)
                    final_hair_u8 = cv2.dilate(
                        (
                            np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.24
                        ).astype(np.uint8)
                        * 255,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 13)),
                        iterations=1,
                    )
                    cleanup_u8 = cv2.bitwise_and(
                        cleanup_u8, cv2.bitwise_not(final_hair_u8)
                    )
                    if hair_length == "short":
                        cleanup_u8 = self._filter_short_center_cleanup_mask(
                            cleanup_u8,
                            face_bbox,
                            cutoff_y_for_post,
                            anchor_u8=cleanup_anchor_u8,
                            top_scale=0.00,
                            bottom_scale=0.84,
                            half_w_scale=0.26,
                            shrink_half_w_scale=0.18,
                            max_area_scale=0.11,
                            max_width_scale=0.36,
                            min_height_scale=0.10,
                            center_allow_scale=0.18,
                            max_total_scale=0.05,
                        )
                    cleanup_u8 = cv2.dilate(
                        cleanup_u8,
                        cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE,
                            (7, 13) if hair_length == "short" else (11, 19),
                        ),
                        iterations=1,
                    )
                    if hair_length == "short":
                        cleanup_u8 = self._filter_short_center_cleanup_mask(
                            cleanup_u8,
                            face_bbox,
                            cutoff_y_for_post,
                            anchor_u8=cleanup_anchor_u8,
                            top_scale=0.00,
                            bottom_scale=0.84,
                            half_w_scale=0.27,
                            shrink_half_w_scale=0.19,
                            max_area_scale=0.12,
                            max_width_scale=0.38,
                            min_height_scale=0.10,
                            center_allow_scale=0.19,
                            max_total_scale=0.06,
                        )
                    micro_cleanup_u8 = (
                        self._build_micro_cloth_artifact_mask(
                            img_rgb=final_rgb,
                            cloth_mask=cloth_mask_dilated,
                            face_bbox=face_bbox,
                            cutoff_y=cutoff_y_for_post,
                            hair_length=hair_length,
                            final_hair_mask=final_hair_mask,
                        )
                        > 0.08
                    ).astype(np.uint8) * 255
                    if int((micro_cleanup_u8 > 0).sum()) > 0:
                        cleanup_u8 = cv2.bitwise_or(cleanup_u8, micro_cleanup_u8)
                        if debug_images_common is not None and rank == 0:
                            debug_images_common[
                                "pipeline_micro_cloth_artifact_mask"
                            ] = cv2.cvtColor(
                                micro_cleanup_u8,
                                cv2.COLOR_GRAY2BGR,
                            )
                    if hair_length == "short":
                        cleanup_u8 = self._filter_short_center_cleanup_mask(
                            cleanup_u8,
                            face_bbox,
                            cutoff_y_for_post,
                            anchor_u8=cleanup_anchor_u8,
                            top_scale=0.02,
                            bottom_scale=0.82,
                            half_w_scale=0.24,
                            shrink_half_w_scale=0.17,
                            max_area_scale=0.10,
                            max_width_scale=0.34,
                            min_height_scale=0.10,
                            center_allow_scale=0.18,
                            max_total_scale=0.045,
                        )
                    if int((cleanup_u8 > 0).sum()) >= 80:
                        final_rgb = self._lama_inpaint(final_rgb, cleanup_u8)
                        if int((final_dark_tail_u8 > 0).sum()) >= 40:
                            final_rgb = self._cv2_cleanup_dark_tail_blob(
                                final_rgb, final_dark_tail_u8
                            )
                        if hair_length != "short":
                            final_rgb = self._restore_cloth_overlap_from_source(
                                source_rgb=img_rgb,
                                current_rgb=final_rgb,
                                restore_mask=cleanup_u8.astype(np.float32) / 255.0,
                                final_hair_mask=final_hair_mask,
                            )
                        final_rgb = self._cv2_refine_cloth_region(
                            final_rgb,
                            cleanup_u8.astype(np.float32) / 255.0,
                            reference_rgb=img_rgb,
                            reference_mask=cloth_mask_dilated,
                        )
                        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                        final_artifact_cleanup_applied = True
                        if debug_images_common is not None and rank == 0:
                            debug_images_common[
                                "pipeline_final_artifact_cloth_cleanup_mask"
                            ] = cv2.cvtColor(
                                cleanup_u8,
                                cv2.COLOR_GRAY2BGR,
                            )
                    if (
                        debug_data_common is not None
                        and rank == 0
                        and final_artifact_cleanup_applied
                    ):
                        _record_rank0_cleanup_stage(
                            "final_artifact_cloth_cleanup",
                            final_bgr,
                            trigger_mask=cleanup_u8.astype(np.float32) / 255.0,
                            mask_label="final_artifact_cloth_cleanup",
                            extra={
                                "micro_cleanup_px": int((micro_cleanup_u8 > 0).sum()),
                                "dark_tail_px": int((final_dark_tail_u8 > 0).sum()),
                            },
                        )
                except Exception as e:
                    logger.warning(
                        f"[SDPipeline] final artifact cloth cleanup ?ㅽ뙣(臾댁떆): {e}"
                    )
            if (
                self.config.enable_post_cloth_refine
                and hair_length in ("short", "medium")
                and cloth_mask_dilated is not None
                and removal_mask_for_post is not None
                and cutoff_y_for_post is not None
            ):
                try:
                    if (
                        hair_length == "short"
                        and short_torso_generation_freeze_active
                        and bool(
                            getattr(
                                self.config,
                                "short_generation_freeze_skip_shoulder_refine",
                                True,
                            )
                        )
                    ):
                        shoulder_cloth_refine_skipped_for_freeze = True
                        logger.info(
                            "[SDPipeline] shoulder cloth refine skipped for short freeze mode"
                        )
                    else:
                        shoulder_refine_applied = False
                        final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                        final_hair_mask, _, _ = self._segface_hair_mask(
                            final_rgb, face_bbox
                        )
                        _, shoulder_y1, _, shoulder_y2 = face_bbox
                        shoulder_face_h = max(int(shoulder_y2 - shoulder_y1), 1)
                        shoulder_cloth_refine_mask = (
                            self._build_shoulder_cloth_restore_mask(
                                removal_mask=removal_mask_for_post,
                                cloth_mask=cloth_mask_dilated,
                                face_bbox=face_bbox,
                                cutoff_y=cutoff_y_for_post,
                                hair_length=hair_length,
                                protect_mask=protect_mask_for_sd,
                                final_hair_mask=final_hair_mask,
                            )
                        )
                        if (
                            hair_length == "short"
                            and shoulder_cloth_release_for_post is not None
                            and shoulder_cloth_release_for_post.shape == (H, W)
                        ):
                            shoulder_source_restore_mask = np.clip(
                                shoulder_cloth_release_for_post.astype(np.float32),
                                0.0,
                                1.0,
                            )
                            shoulder_source_restore_mask[
                                : max(
                                    0, int(cutoff_y_for_post - shoulder_face_h * 0.02)
                                ),
                                :,
                            ] = 0.0
                            shoulder_source_restore_mask = np.clip(
                                shoulder_source_restore_mask
                                * np.clip(
                                    cloth_mask_dilated.astype(np.float32), 0.0, 1.0
                                ),
                                0.0,
                                1.0,
                            )
                            if float(shoulder_cloth_refine_mask.sum()) > 0.0:
                                shoulder_cloth_refine_mask = np.maximum(
                                    shoulder_cloth_refine_mask,
                                    shoulder_source_restore_mask * 0.92,
                                ).astype(np.float32)
                            else:
                                shoulder_cloth_refine_mask = (
                                    shoulder_source_restore_mask.astype(np.float32)
                                )
                        shoulder_cloth_refine_u8 = (
                            np.clip(
                                shoulder_cloth_refine_mask.astype(np.float32), 0.0, 1.0
                            )
                            > 0.08
                        ).astype(np.uint8) * 255
                        shoulder_refine_px = int((shoulder_cloth_refine_u8 > 0).sum())
                        if shoulder_refine_px >= 180:
                            final_rgb = self._sd_refine_removed_region(
                                base_rgb=final_rgb,
                                removal_mask=shoulder_cloth_refine_mask,
                                face_bbox=face_bbox,
                                face_crop_pil=face_crop_pil,
                                protect_mask=protect_mask_for_sd,
                                cloth_mask=cloth_mask_dilated,
                                hair_length=hair_length,
                                seed=int(cand["seed"]) + 1739,
                                refine_mode="cloth",
                                subject_gender=subject_gender_mode,
                                white_tshirt_experiment=white_tshirt_experiment,
                            )
                        if 100 <= shoulder_refine_px < 8000:
                            final_rgb = self._cv2_refine_cloth_region(
                                final_rgb,
                                shoulder_cloth_refine_mask,
                                reference_rgb=img_rgb,
                                reference_mask=cloth_mask_dilated,
                            )
                        if shoulder_refine_px >= 120:
                            final_rgb = self._restore_cloth_overlap_from_source(
                                source_rgb=img_rgb,
                                current_rgb=final_rgb,
                                restore_mask=shoulder_cloth_refine_mask,
                                final_hair_mask=final_hair_mask,
                            )
                        if shoulder_refine_px >= 100:
                            final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                            shoulder_refine_applied = True
                        if debug_images_common is not None and rank == 0:
                            debug_images_common[
                                "pipeline_shoulder_cloth_refine_mask"
                            ] = cv2.cvtColor(
                                shoulder_cloth_refine_u8,
                                cv2.COLOR_GRAY2BGR,
                            )
                        if (
                            debug_data_common is not None
                            and rank == 0
                            and shoulder_refine_applied
                        ):
                            _record_rank0_cleanup_stage(
                                "shoulder_cloth_refine",
                                final_bgr,
                                trigger_mask=shoulder_cloth_refine_mask,
                                mask_label="shoulder_cloth_refine",
                                extra={"shoulder_refine_px": shoulder_refine_px},
                            )
                except Exception as e:
                    logger.warning(
                        f"[SDPipeline] shoulder cloth refine failed (ignored): {e}"
                    )
                if debug_data_common is not None and rank == 0:
                    debug_data_common.setdefault("short_postprocess", {})
                    debug_data_common["short_postprocess"][
                        "shoulder_cloth_refine_skipped_for_freeze"
                    ] = bool(shoulder_cloth_refine_skipped_for_freeze)
            if (
                hair_length in ("short", "medium")
                and cloth_mask_dilated is not None
                and cutoff_y_for_post is not None
            ):
                try:
                    side_column_restore_applied = False
                    side_column_restore_skipped_for_freeze = bool(
                        hair_length == "short"
                        and short_torso_generation_freeze_active
                        and getattr(
                            self.config,
                            "short_generation_freeze_skip_side_column_restore",
                            False,
                        )
                    )
                    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                    final_hair_mask, _, _ = self._segface_hair_mask(
                        final_rgb, face_bbox
                    )
                    side_column_debug_info: Dict[str, Any] = {}
                    side_column_debug_masks: Dict[str, np.ndarray] = {}
                    side_column_candidate_mask = np.zeros((H, W), dtype=np.float32)
                    if (
                        removal_mask_for_post is not None
                        and removal_mask_for_post.shape == (H, W)
                    ):
                        side_column_candidate_mask = np.maximum(
                            side_column_candidate_mask,
                            np.clip(removal_mask_for_post.astype(np.float32), 0.0, 1.0)
                            * 0.34,
                        ).astype(np.float32)
                    if (
                        artifact_cleanup_mask_for_post is not None
                        and artifact_cleanup_mask_for_post.shape == (H, W)
                    ):
                        side_column_candidate_mask = np.maximum(
                            side_column_candidate_mask,
                            np.clip(
                                artifact_cleanup_mask_for_post.astype(np.float32),
                                0.0,
                                1.0,
                            ),
                        ).astype(np.float32)
                    if (
                        shoulder_cloth_release_for_post is not None
                        and shoulder_cloth_release_for_post.shape == (H, W)
                    ):
                        side_column_candidate_mask = np.maximum(
                            side_column_candidate_mask,
                            np.clip(
                                shoulder_cloth_release_for_post.astype(np.float32),
                                0.0,
                                1.0,
                            )
                            * 0.96,
                        ).astype(np.float32)
                    if (
                        below_bob_cloth_restore_for_post is not None
                        and below_bob_cloth_restore_for_post.shape == (H, W)
                    ):
                        side_column_candidate_mask = np.maximum(
                            side_column_candidate_mask,
                            np.clip(
                                below_bob_cloth_restore_for_post.astype(np.float32),
                                0.0,
                                1.0,
                            ),
                        ).astype(np.float32)
                    side_column_restore_mask = (
                        self._build_side_column_cloth_restore_mask(
                            img_rgb=final_rgb,
                            cloth_mask=cloth_mask_dilated,
                            candidate_mask=side_column_candidate_mask,
                            face_bbox=face_bbox,
                            cutoff_y=cutoff_y_for_post,
                            hair_length=hair_length,
                            final_hair_mask=final_hair_mask,
                            debug_info=side_column_debug_info,
                            debug_masks=side_column_debug_masks,
                        )
                    )
                    direct_side_column_restore_mask = (
                        self._build_direct_short_column_restore_mask(
                            removal_mask=removal_mask_for_post,
                            cloth_mask=cloth_mask_dilated,
                            face_bbox=face_bbox,
                            cutoff_y=cutoff_y_for_post,
                            hair_length=hair_length,
                        )
                    )
                    if float(direct_side_column_restore_mask.sum()) > 0.0:
                        side_column_restore_mask = np.maximum(
                            side_column_restore_mask,
                            direct_side_column_restore_mask,
                        ).astype(np.float32)
                    side_column_restore_u8 = (
                        np.clip(side_column_restore_mask.astype(np.float32), 0.0, 1.0)
                        > 0.08
                    ).astype(np.uint8) * 255
                    side_column_px = int((side_column_restore_u8 > 0).sum())
                    if (
                        side_column_px >= 160
                        and not side_column_restore_skipped_for_freeze
                    ):
                        if hair_length != "short":
                            final_rgb = self._sd_refine_removed_region(
                                base_rgb=final_rgb,
                                removal_mask=side_column_restore_mask,
                                face_bbox=face_bbox,
                                face_crop_pil=face_crop_pil,
                                protect_mask=protect_mask_for_sd,
                                cloth_mask=cloth_mask_dilated,
                                hair_length=hair_length,
                                seed=int(cand["seed"]) + 1787,
                                refine_mode="cloth",
                                subject_gender=subject_gender_mode,
                                white_tshirt_experiment=white_tshirt_experiment,
                            )
                        if hair_length == "short":
                            side_column_cleanup_trace: List[Tuple[str, np.ndarray]] = []
                            side_column_cleanup_debug: Dict[str, Any] = {}
                            side_column_cleanup_masks: Dict[str, np.ndarray] = {}
                            final_rgb = self._cleanup_region_with_cloth_restore(
                                source_rgb=img_rgb,
                                current_rgb=final_rgb,
                                cleanup_mask=side_column_restore_mask,
                                cloth_mask=cloth_mask_dilated,
                                final_hair_mask=final_hair_mask,
                                ignore_final_hair_for_cloth_restore=False,
                                cleanup_dark_tail=True,
                                prefer_plain_cloth_fill=bool(
                                    getattr(
                                        self.config,
                                        "short_side_column_restore_plain_fill",
                                        False,
                                    )
                                ),
                                allow_plain_cloth_force=bool(
                                    getattr(
                                        self.config,
                                        "short_side_column_allow_plain_cloth_force",
                                        False,
                                    )
                                ),
                                debug_trace=side_column_cleanup_trace,
                                debug_info=side_column_cleanup_debug,
                                debug_masks=side_column_cleanup_masks,
                            )
                        else:
                            final_rgb = self._restore_cloth_overlap_from_source(
                                source_rgb=img_rgb,
                                current_rgb=final_rgb,
                                restore_mask=side_column_restore_mask,
                                final_hair_mask=final_hair_mask,
                            )
                            final_rgb = self._cv2_refine_cloth_region(
                                final_rgb, side_column_restore_mask
                            )
                        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                        side_column_restore_applied = True
                    if debug_images_common is not None and rank == 0:
                        debug_images_common[
                            "pipeline_side_column_cloth_restore_mask"
                        ] = cv2.cvtColor(
                            side_column_restore_u8,
                            cv2.COLOR_GRAY2BGR,
                        )
                        debug_images_common[
                            "pipeline_direct_short_column_restore_mask"
                        ] = cv2.cvtColor(
                            (
                                (
                                    np.clip(
                                        direct_side_column_restore_mask.astype(
                                            np.float32
                                        ),
                                        0.0,
                                        1.0,
                                    )
                                    > 0.08
                                ).astype(np.uint8)
                                * 255
                            ),
                            cv2.COLOR_GRAY2BGR,
                        )
                        for name, mask in side_column_debug_masks.items():
                            _store_mask(
                                f"pipeline_side_column_cloth_restore_{name}_mask", mask
                            )
                        if hair_length == "short":
                            for name, mask in side_column_cleanup_masks.items():
                                _store_mask(
                                    f"pipeline_side_column_cloth_restore_cleanup_{name}_mask",
                                    mask,
                                )
                    if (
                        debug_data_common is not None
                        and rank == 0
                        and side_column_restore_applied
                    ):
                        diag = debug_data_common.setdefault("diagnostics", {})
                        diag["side_column_cloth_restore_debug"] = side_column_debug_info
                        for name, mask in side_column_debug_masks.items():
                            _mask_stats(
                                f"side_column_cloth_restore_{name}",
                                mask,
                                diagnostic_rois.get("torso_front"),
                                bucket="cleanup_mask_stats",
                            )
                        if hair_length == "short":
                            diag["side_column_cloth_restore_cleanup_debug"] = (
                                side_column_cleanup_debug
                            )
                            for name, mask in side_column_cleanup_masks.items():
                                _mask_stats(
                                    f"side_column_cloth_restore_cleanup_{name}",
                                    mask,
                                    diagnostic_rois.get("torso_front"),
                                    bucket="cleanup_mask_stats",
                                )
                            for (
                                substage_name,
                                substage_rgb,
                            ) in side_column_cleanup_trace:
                                _record_rank0_cleanup_stage(
                                    f"side_column_cloth_restore_{substage_name}",
                                    cv2.cvtColor(substage_rgb, cv2.COLOR_RGB2BGR),
                                )
                        _record_rank0_cleanup_stage(
                            "side_column_cloth_restore",
                            final_bgr,
                            trigger_mask=side_column_restore_mask,
                            mask_label="side_column_cloth_restore",
                            extra={"side_column_px": side_column_px},
                        )
                except Exception as e:
                    logger.warning(
                        f"[SDPipeline] side column cloth restore failed (ignored): {e}"
                    )
                if debug_data_common is not None and rank == 0:
                    debug_data_common.setdefault("short_postprocess", {})
                    debug_data_common["short_postprocess"][
                        "side_column_cloth_restore_skipped_for_freeze"
                    ] = bool(side_column_restore_skipped_for_freeze)
            if (
                hair_length == "short"
                and cloth_mask_dilated is not None
                and removal_mask_for_post is not None
                and cutoff_y_for_post is not None
            ):
                try:
                    short_side_lane_refine_applied = False
                    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                    final_hair_mask, _, _ = self._segface_hair_mask(
                        final_rgb, face_bbox
                    )
                    short_side_lane_refine_mask = (
                        self._build_short_final_side_lane_refine_mask(
                            img_rgb=final_rgb,
                            final_hair_mask=final_hair_mask,
                            cloth_mask=cloth_mask_dilated,
                            removal_mask=removal_mask_for_post,
                            face_bbox=face_bbox,
                            cutoff_y=cutoff_y_for_post,
                            hair_length=hair_length,
                        )
                    )
                    short_side_lane_refine_u8 = self._mask_to_u8(
                        short_side_lane_refine_mask, threshold=0.08
                    )
                    short_side_lane_px = self._count_active_mask_px(
                        short_side_lane_refine_mask, threshold=0.08
                    )
                    if self._should_apply_cleanup_mask(
                        "short_final_side_lane_refine",
                        short_side_lane_px,
                        hair_length,
                    ):
                        final_rgb = self._cleanup_region_with_cloth_restore(
                            source_rgb=img_rgb,
                            current_rgb=final_rgb,
                            cleanup_mask=short_side_lane_refine_mask,
                            cloth_mask=cloth_mask_dilated,
                            final_hair_mask=final_hair_mask,
                            ignore_final_hair_for_cloth_restore=True,
                            cleanup_dark_tail=True,
                            prefer_plain_cloth_fill=True,
                        )
                        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                        short_side_lane_refine_applied = True
                    if debug_images_common is not None and rank == 0:
                        debug_images_common[
                            "pipeline_short_final_side_lane_refine_mask"
                        ] = cv2.cvtColor(
                            short_side_lane_refine_u8,
                            cv2.COLOR_GRAY2BGR,
                        )
                    if (
                        debug_data_common is not None
                        and rank == 0
                        and short_side_lane_refine_applied
                    ):
                        _record_rank0_cleanup_stage(
                            "short_final_side_lane_refine",
                            final_bgr,
                            trigger_mask=short_side_lane_refine_mask,
                            mask_label="short_final_side_lane_refine",
                            extra={"short_side_lane_px": short_side_lane_px},
                        )
                except Exception as e:
                    logger.warning(
                        f"[SDPipeline] short side lane refine failed (ignored): {e}"
                    )
            use_short_torso_garment_repaint = (
                hair_length == "short" and not use_upper_clothes_overwrite
            )
            if (
                hair_length in ("short", "medium")
                and (hair_length != "short" or not disable_short_postprocess_experiment)
                and cloth_mask_dilated is not None
                and removal_mask_for_post is not None
                and cutoff_y_for_post is not None
                and not use_short_torso_garment_repaint
            ):
                if hair_length == "short":
                    try:
                        short_lower_tail_cleanup_applied = False
                        final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                        final_hair_mask, _, _ = self._segface_hair_mask(
                            final_rgb, face_bbox
                        )
                        short_lower_tail_mask = (
                            self._build_short_lower_tail_cleanup_mask(
                                img_rgb=final_rgb,
                                cloth_mask=cloth_mask_dilated,
                                removal_mask=removal_mask_for_post,
                                final_hair_mask=final_hair_mask,
                                face_bbox=face_bbox,
                                cutoff_y=cutoff_y_for_post,
                                hair_length=hair_length,
                            )
                        )
                        short_lower_tail_u8 = self._mask_to_u8(
                            short_lower_tail_mask, threshold=0.08
                        )
                        short_lower_tail_px = self._count_active_mask_px(
                            short_lower_tail_mask, threshold=0.08
                        )
                        if self._should_apply_cleanup_mask(
                            "short_lower_tail_cleanup",
                            short_lower_tail_px,
                            hair_length,
                        ):
                            final_rgb = self._cleanup_region_with_cloth_restore(
                                source_rgb=img_rgb,
                                current_rgb=final_rgb,
                                cleanup_mask=short_lower_tail_mask,
                                cloth_mask=cloth_mask_dilated,
                                ignore_final_hair_for_cloth_restore=True,
                                cleanup_dark_tail=True,
                            )
                            final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                            short_lower_tail_cleanup_applied = True
                        if debug_images_common is not None and rank == 0:
                            debug_images_common[
                                "pipeline_short_lower_tail_cleanup_mask"
                            ] = cv2.cvtColor(
                                short_lower_tail_u8,
                                cv2.COLOR_GRAY2BGR,
                            )
                        if (
                            debug_data_common is not None
                            and rank == 0
                            and short_lower_tail_cleanup_applied
                        ):
                            _record_rank0_cleanup_stage(
                                "short_lower_tail_cleanup",
                                final_bgr,
                                trigger_mask=short_lower_tail_mask,
                                mask_label="short_lower_tail_cleanup",
                                extra={"short_lower_tail_px": short_lower_tail_px},
                            )
                    except Exception as e:
                        logger.warning(
                            f"[SDPipeline] short lower tail cleanup failed (ignored): {e}"
                        )
                try:
                    dark_lane_cleanup_applied = False
                    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                    dark_lane_mask = self._build_dark_lane_cleanup_mask(
                        img_rgb=final_rgb,
                        cloth_mask=cloth_mask_dilated,
                        removal_mask=removal_mask_for_post,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        hair_length=hair_length,
                    )
                    dark_lane_u8 = self._mask_to_u8(dark_lane_mask, threshold=0.08)
                    dark_lane_px = self._count_active_mask_px(
                        dark_lane_mask, threshold=0.08
                    )
                    if self._should_apply_cleanup_mask(
                        "dark_lane_cleanup",
                        dark_lane_px,
                        hair_length,
                    ):
                        final_rgb = self._lama_inpaint(final_rgb, dark_lane_u8)
                        final_rgb = self._cv2_refine_cloth_region(
                            final_rgb,
                            dark_lane_mask,
                            reference_rgb=img_rgb,
                            reference_mask=cloth_mask_dilated,
                        )
                        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                        dark_lane_cleanup_applied = True
                    if debug_images_common is not None and rank == 0:
                        debug_images_common["pipeline_dark_lane_cleanup_mask"] = (
                            cv2.cvtColor(
                                dark_lane_u8,
                                cv2.COLOR_GRAY2BGR,
                            )
                        )
                    if (
                        debug_data_common is not None
                        and rank == 0
                        and dark_lane_cleanup_applied
                    ):
                        _record_rank0_cleanup_stage(
                            "dark_lane_cleanup",
                            final_bgr,
                            trigger_mask=dark_lane_mask,
                            mask_label="dark_lane_cleanup",
                            extra={"dark_lane_px": dark_lane_px},
                        )
                except Exception as e:
                    logger.warning(
                        f"[SDPipeline] dark lane cleanup failed (ignored): {e}"
                    )
            if (
                hair_length in ("short", "medium")
                and cloth_mask_dilated is not None
                and removal_mask_for_post is not None
                and cutoff_y_for_post is not None
                and not use_short_torso_garment_repaint
            ):
                try:
                    final_hair_lane_cleanup_applied = False
                    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                    final_hair_mask, _, _ = self._segface_hair_mask(
                        final_rgb, face_bbox
                    )
                    final_hair_lane_debug_info: Dict[str, Any] = {}
                    final_hair_lane_debug_masks: Dict[str, np.ndarray] = {}
                    final_hair_lane_mask = self._build_final_hair_lane_cleanup_mask(
                        final_hair_mask=final_hair_mask,
                        cloth_mask=cloth_mask_dilated,
                        removal_mask=removal_mask_for_post,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        hair_length=hair_length,
                        debug_info=final_hair_lane_debug_info,
                        debug_masks=final_hair_lane_debug_masks,
                    )
                    final_hair_lane_u8 = self._mask_to_u8(
                        final_hair_lane_mask, threshold=0.08
                    )
                    final_hair_lane_px = self._count_active_mask_px(
                        final_hair_lane_mask, threshold=0.08
                    )
                    if self._should_apply_cleanup_mask(
                        "final_hair_lane_cleanup",
                        final_hair_lane_px,
                        hair_length,
                    ):
                        final_rgb = self._lama_inpaint(final_rgb, final_hair_lane_u8)
                        if debug_data_common is not None and rank == 0:
                            _record_rank0_cleanup_stage(
                                "final_hair_lane_cleanup_lama",
                                cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR),
                            )
                        final_rgb = self._cv2_cleanup_dark_tail_blob(
                            final_rgb, final_hair_lane_u8
                        )
                        if debug_data_common is not None and rank == 0:
                            _record_rank0_cleanup_stage(
                                "final_hair_lane_cleanup_dark_tail",
                                cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR),
                            )
                        final_rgb = self._cv2_refine_cloth_region(
                            final_rgb,
                            final_hair_lane_mask,
                            reference_rgb=img_rgb,
                            reference_mask=cloth_mask_dilated,
                        )
                        if debug_data_common is not None and rank == 0:
                            _record_rank0_cleanup_stage(
                                "final_hair_lane_cleanup_cloth_refine",
                                cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR),
                            )
                        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                        final_hair_lane_cleanup_applied = True
                    if debug_images_common is not None and rank == 0:
                        debug_images_common["pipeline_final_hair_lane_cleanup_mask"] = (
                            cv2.cvtColor(
                                final_hair_lane_u8,
                                cv2.COLOR_GRAY2BGR,
                            )
                        )
                        for name, mask in final_hair_lane_debug_masks.items():
                            _store_mask(
                                f"pipeline_final_hair_lane_cleanup_{name}_mask", mask
                            )
                    if (
                        debug_data_common is not None
                        and rank == 0
                        and final_hair_lane_cleanup_applied
                    ):
                        diag = debug_data_common.setdefault("diagnostics", {})
                        diag["final_hair_lane_cleanup_debug"] = (
                            final_hair_lane_debug_info
                        )
                        for name, mask in final_hair_lane_debug_masks.items():
                            _mask_stats(
                                f"final_hair_lane_cleanup_{name}",
                                mask,
                                diagnostic_rois.get("torso_front"),
                                bucket="cleanup_mask_stats",
                            )
                        _record_rank0_cleanup_stage(
                            "final_hair_lane_cleanup",
                            final_bgr,
                            trigger_mask=final_hair_lane_mask,
                            mask_label="final_hair_lane_cleanup",
                            extra={"final_hair_lane_px": final_hair_lane_px},
                        )
                except Exception as e:
                    logger.warning(
                        f"[SDPipeline] final hair lane cleanup failed (ignored): {e}"
                    )
            if (
                hair_length == "short"
                and not disable_short_postprocess_experiment
                and cloth_mask_dilated is not None
                and removal_mask_for_post is not None
                and cutoff_y_for_post is not None
            ):
                try:
                    short_bob_tail_cleanup_applied = False
                    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                    final_hair_mask, _, _ = self._segface_hair_mask(
                        final_rgb, face_bbox
                    )
                    short_bob_tail_mask = self._build_short_bob_tail_suppress_mask(
                        img_rgb=final_rgb,
                        final_hair_mask=final_hair_mask,
                        cloth_mask=cloth_mask_dilated,
                        removal_mask=removal_mask_for_post,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        hair_length=hair_length,
                    )
                    short_bob_tail_u8 = self._mask_to_u8(
                        short_bob_tail_mask, threshold=0.08
                    )
                    short_bob_tail_px = self._count_active_mask_px(
                        short_bob_tail_mask, threshold=0.08
                    )
                    if self._should_apply_cleanup_mask(
                        "short_bob_tail_suppress",
                        short_bob_tail_px,
                        hair_length,
                    ):
                        final_rgb = self._cleanup_region_with_cloth_restore(
                            source_rgb=img_rgb,
                            current_rgb=final_rgb,
                            cleanup_mask=short_bob_tail_mask,
                            cloth_mask=cloth_mask_dilated,
                            ignore_final_hair_for_cloth_restore=True,
                            cleanup_dark_tail=True,
                        )
                        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                        short_bob_tail_cleanup_applied = True
                    if debug_images_common is not None and rank == 0:
                        debug_images_common["pipeline_short_bob_tail_suppress_mask"] = (
                            cv2.cvtColor(
                                short_bob_tail_u8,
                                cv2.COLOR_GRAY2BGR,
                            )
                        )
                    if (
                        debug_data_common is not None
                        and rank == 0
                        and short_bob_tail_cleanup_applied
                    ):
                        _record_rank0_cleanup_stage(
                            "short_bob_tail_suppress",
                            final_bgr,
                            trigger_mask=short_bob_tail_mask,
                            mask_label="short_bob_tail_suppress",
                            extra={"short_bob_tail_px": short_bob_tail_px},
                        )
                except Exception as e:
                    logger.warning(
                        f"[SDPipeline] short bob tail suppress failed (ignored): {e}"
                    )
            if (
                hair_length == "short"
                and not disable_short_postprocess_experiment
                and removal_mask_for_post is not None
                and cutoff_y_for_post is not None
                and not source_garment_prepass_applied
            ):
                try:
                    short_lower_garment_cleanup_applied = False
                    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                    final_hair_mask, _, _ = self._segface_hair_mask(
                        final_rgb, face_bbox
                    )
                    short_lower_garment_cleanup_mask = (
                        self._build_short_lower_garment_cleanup_mask(
                            current_rgb=final_rgb,
                            source_rgb=img_rgb,
                            removal_mask=removal_mask_for_post,
                            cloth_mask=cloth_mask_dilated,
                            face_bbox=face_bbox,
                            cutoff_y=cutoff_y_for_post,
                            hair_length=hair_length,
                            protect_mask=protect_mask_for_sd,
                            final_hair_mask=final_hair_mask,
                        )
                    )
                    short_lower_garment_cleanup_u8 = self._mask_to_u8(
                        short_lower_garment_cleanup_mask,
                        threshold=0.08,
                    )
                    short_lower_garment_cleanup_px = self._count_active_mask_px(
                        short_lower_garment_cleanup_mask,
                        threshold=0.08,
                    )
                    if self._should_apply_cleanup_mask(
                        "short_lower_garment_cleanup",
                        short_lower_garment_cleanup_px,
                        hair_length,
                    ):
                        final_rgb = self._cleanup_region_with_cloth_restore(
                            source_rgb=img_rgb,
                            current_rgb=final_rgb,
                            cleanup_mask=short_lower_garment_cleanup_mask,
                            cloth_mask=cloth_mask_dilated,
                            final_hair_mask=final_hair_mask,
                            ignore_final_hair_for_cloth_restore=False,
                            cleanup_dark_tail=True,
                            prefer_plain_cloth_fill=True,
                        )
                        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                        short_lower_garment_cleanup_applied = True
                    if debug_images_common is not None and rank == 0:
                        debug_images_common[
                            "pipeline_short_lower_garment_cleanup_mask"
                        ] = cv2.cvtColor(
                            short_lower_garment_cleanup_u8,
                            cv2.COLOR_GRAY2BGR,
                        )
                    if (
                        debug_data_common is not None
                        and rank == 0
                        and short_lower_garment_cleanup_applied
                    ):
                        _record_rank0_cleanup_stage(
                            "short_lower_garment_cleanup",
                            final_bgr,
                            trigger_mask=short_lower_garment_cleanup_mask,
                            mask_label="short_lower_garment_cleanup",
                            extra={
                                "short_lower_garment_px": short_lower_garment_cleanup_px
                            },
                        )
                except Exception as e:
                    logger.warning(
                        f"[SDPipeline] short lower garment cleanup failed (ignored): {e}"
                    )
                try:
                    short_lower_cloth_hard_override_applied = False
                    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                    final_hair_mask, _, _ = self._segface_hair_mask(
                        final_rgb, face_bbox
                    )
                    short_lower_cloth_hard_override_mask = (
                        self._build_short_lower_cloth_hard_override_mask(
                            current_rgb=final_rgb,
                            source_rgb=img_rgb,
                            removal_mask=removal_mask_for_post,
                            cloth_mask=cloth_mask_dilated,
                            face_bbox=face_bbox,
                            cutoff_y=cutoff_y_for_post,
                            hair_length=hair_length,
                            final_hair_mask=final_hair_mask,
                        )
                    )
                    short_lower_cloth_hard_override_u8 = self._mask_to_u8(
                        short_lower_cloth_hard_override_mask,
                        threshold=0.08,
                    )
                    short_lower_cloth_hard_override_px = self._count_active_mask_px(
                        short_lower_cloth_hard_override_mask,
                        threshold=0.08,
                    )
                    if self._should_apply_cleanup_mask(
                        "short_lower_cloth_hard_override",
                        short_lower_cloth_hard_override_px,
                        hair_length,
                    ):
                        final_rgb = self._cleanup_region_with_cloth_restore(
                            source_rgb=img_rgb,
                            current_rgb=final_rgb,
                            cleanup_mask=short_lower_cloth_hard_override_mask,
                            cloth_mask=cloth_mask_dilated,
                            final_hair_mask=final_hair_mask,
                            ignore_final_hair_for_cloth_restore=False,
                            cleanup_dark_tail=True,
                            prefer_plain_cloth_fill=True,
                        )
                        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                        short_lower_cloth_hard_override_applied = True
                    if debug_images_common is not None and rank == 0:
                        debug_images_common[
                            "pipeline_short_lower_cloth_hard_override_mask"
                        ] = cv2.cvtColor(
                            short_lower_cloth_hard_override_u8,
                            cv2.COLOR_GRAY2BGR,
                        )
                    if (
                        debug_data_common is not None
                        and rank == 0
                        and short_lower_cloth_hard_override_applied
                    ):
                        _record_rank0_cleanup_stage(
                            "short_lower_cloth_hard_override",
                            final_bgr,
                            trigger_mask=short_lower_cloth_hard_override_mask,
                            mask_label="short_lower_cloth_hard_override",
                            extra={
                                "short_lower_cloth_hard_override_px": short_lower_cloth_hard_override_px
                            },
                        )
                except Exception as e:
                    logger.warning(
                        f"[SDPipeline] short lower cloth hard override failed (ignored): {e}"
                    )
            if (
                hair_length in ("short", "medium")
                and (hair_length != "short" or not disable_short_postprocess_experiment)
                and cloth_mask_dilated is not None
                and removal_mask_for_post is not None
                and cutoff_y_for_post is not None
                and not use_short_torso_garment_repaint
            ):
                try:
                    residual_strand_cleanup_applied = False
                    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                    final_hair_mask, _, _ = self._segface_hair_mask(
                        final_rgb, face_bbox
                    )
                    residual_strand_mask = self._build_residual_strand_cleanup_mask(
                        img_rgb=final_rgb,
                        cloth_mask=cloth_mask_dilated,
                        removal_mask=removal_mask_for_post,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        hair_length=hair_length,
                        final_hair_mask=final_hair_mask,
                    )
                    residual_strand_u8 = self._mask_to_u8(
                        residual_strand_mask, threshold=0.08
                    )
                    residual_strand_px = self._count_active_mask_px(
                        residual_strand_mask, threshold=0.08
                    )
                    if self._should_apply_cleanup_mask(
                        "residual_strand_cleanup",
                        residual_strand_px,
                        hair_length,
                    ):
                        final_rgb = self._lama_inpaint(final_rgb, residual_strand_u8)
                        final_rgb = self._cv2_cleanup_dark_tail_blob(
                            final_rgb, residual_strand_u8
                        )
                        final_rgb = self._cv2_refine_cloth_region(
                            final_rgb,
                            residual_strand_mask,
                            reference_rgb=img_rgb,
                            reference_mask=cloth_mask_dilated,
                        )
                        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                        residual_strand_cleanup_applied = True
                    if debug_images_common is not None and rank == 0:
                        debug_images_common["pipeline_residual_strand_cleanup_mask"] = (
                            cv2.cvtColor(
                                residual_strand_u8,
                                cv2.COLOR_GRAY2BGR,
                            )
                        )
                    if (
                        debug_data_common is not None
                        and rank == 0
                        and residual_strand_cleanup_applied
                    ):
                        _record_rank0_cleanup_stage(
                            "residual_strand_cleanup",
                            final_bgr,
                            trigger_mask=residual_strand_mask,
                            mask_label="residual_strand_cleanup",
                            extra={"residual_strand_px": residual_strand_px},
                        )
                except Exception as e:
                    logger.warning(
                        f"[SDPipeline] residual strand cleanup failed (ignored): {e}"
                    )
            if (
                hair_length in ("short", "medium")
                and (hair_length != "short" or not disable_short_postprocess_experiment)
                and cloth_mask_dilated is not None
                and removal_mask_for_post is not None
                and cutoff_y_for_post is not None
                and not use_short_torso_garment_repaint
            ):
                try:
                    final_source_cloth_rescue_applied = False
                    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                    final_hair_mask, _, _ = self._segface_hair_mask(
                        final_rgb, face_bbox
                    )
                    final_source_cloth_rescue_mask = (
                        self._build_final_source_cloth_rescue_mask(
                            current_rgb=final_rgb,
                            source_rgb=img_rgb,
                            removal_mask=removal_mask_for_post,
                            cloth_mask=cloth_mask_dilated,
                            face_bbox=face_bbox,
                            cutoff_y=cutoff_y_for_post,
                            hair_length=hair_length,
                            protect_mask=protect_mask_for_sd,
                            final_hair_mask=final_hair_mask,
                        )
                    )
                    final_source_cloth_rescue_u8 = self._mask_to_u8(
                        final_source_cloth_rescue_mask,
                        threshold=0.08,
                    )
                    final_source_cloth_rescue_px = self._count_active_mask_px(
                        final_source_cloth_rescue_mask,
                        threshold=0.08,
                    )
                    if self._should_apply_cleanup_mask(
                        "final_source_cloth_rescue",
                        final_source_cloth_rescue_px,
                        hair_length,
                    ):
                        final_rgb = self._cleanup_region_with_cloth_restore(
                            source_rgb=img_rgb,
                            current_rgb=final_rgb,
                            cleanup_mask=final_source_cloth_rescue_mask,
                            cloth_mask=cloth_mask_dilated,
                            final_hair_mask=final_hair_mask,
                            ignore_final_hair_for_cloth_restore=(
                                hair_length == "short"
                            ),
                            cleanup_dark_tail=(hair_length == "short"),
                        )
                        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                        final_source_cloth_rescue_applied = True
                    if debug_images_common is not None and rank == 0:
                        debug_images_common[
                            "pipeline_final_source_cloth_rescue_mask"
                        ] = cv2.cvtColor(
                            final_source_cloth_rescue_u8,
                            cv2.COLOR_GRAY2BGR,
                        )
                    if (
                        debug_data_common is not None
                        and rank == 0
                        and final_source_cloth_rescue_applied
                    ):
                        _record_rank0_cleanup_stage(
                            "final_source_cloth_rescue",
                            final_bgr,
                            trigger_mask=final_source_cloth_rescue_mask,
                            mask_label="final_source_cloth_rescue",
                            extra={
                                "final_source_cloth_rescue_px": final_source_cloth_rescue_px
                            },
                        )
                except Exception as e:
                    logger.warning(
                        f"[SDPipeline] final source cloth rescue failed (ignored): {e}"
                    )
            if (
                hair_length == "short"
                and cloth_mask_dilated is not None
                and cutoff_y_for_post is not None
                and not use_short_torso_garment_repaint
            ):
                try:
                    short_subject_cloth_cleanup_applied = False
                    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                    final_hair_mask, _, _ = self._segface_hair_mask(
                        final_rgb, face_bbox
                    )
                    short_below_bob_torso_mask = self._build_short_below_bob_torso_mask(
                        cloth_mask=cloth_mask_dilated,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        hair_length=hair_length,
                        final_hair_mask=final_hair_mask,
                        torso_candidate_mask=subject_torso_candidate_mask,
                        completed_torso_fill_mask=completed_torso_fill_mask,
                        shoulder_anchor_mask=upper_clothes_overwrite_anchor_mask,
                    )
                    short_subject_cloth_cleanup_mask = (
                        self._build_short_subject_cloth_cleanup_mask(
                            current_rgb=final_rgb,
                            source_rgb=img_rgb,
                            cloth_mask=cloth_mask_dilated,
                            torso_mask=short_below_bob_torso_mask,
                            face_mask=face_region_mask,
                            face_bbox=face_bbox,
                            cutoff_y=cutoff_y_for_post,
                            hair_length=hair_length,
                            final_hair_mask=final_hair_mask,
                        )
                    )
                    short_subject_cloth_cleanup_u8 = self._mask_to_u8(
                        short_subject_cloth_cleanup_mask,
                        threshold=0.08,
                    )
                    short_subject_cloth_cleanup_px = self._count_active_mask_px(
                        short_subject_cloth_cleanup_mask,
                        threshold=0.08,
                    )
                    if self._should_apply_cleanup_mask(
                        "short_subject_cloth_cleanup",
                        short_subject_cloth_cleanup_px,
                        hair_length,
                    ):
                        final_rgb = self._cleanup_region_with_cloth_restore(
                            source_rgb=img_rgb,
                            current_rgb=final_rgb,
                            cleanup_mask=short_subject_cloth_cleanup_mask,
                            cloth_mask=cloth_mask_dilated,
                            final_hair_mask=final_hair_mask,
                            ignore_final_hair_for_cloth_restore=True,
                            cleanup_dark_tail=True,
                        )
                        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                        short_subject_cloth_cleanup_applied = True
                    if debug_images_common is not None and rank == 0:
                        debug_images_common["pipeline_short_below_bob_torso_mask"] = (
                            cv2.cvtColor(
                                (
                                    (
                                        np.clip(
                                            short_below_bob_torso_mask.astype(
                                                np.float32
                                            ),
                                            0.0,
                                            1.0,
                                        )
                                        > 0.08
                                    ).astype(np.uint8)
                                    * 255
                                ),
                                cv2.COLOR_GRAY2BGR,
                            )
                        )
                        debug_images_common[
                            "pipeline_short_subject_cloth_cleanup_mask"
                        ] = cv2.cvtColor(
                            short_subject_cloth_cleanup_u8,
                            cv2.COLOR_GRAY2BGR,
                        )
                    if (
                        debug_data_common is not None
                        and rank == 0
                        and short_subject_cloth_cleanup_applied
                    ):
                        _record_rank0_cleanup_stage(
                            "short_subject_cloth_cleanup",
                            final_bgr,
                            trigger_mask=short_subject_cloth_cleanup_mask,
                            mask_label="short_subject_cloth_cleanup",
                            extra={
                                "short_subject_cloth_cleanup_px": short_subject_cloth_cleanup_px
                            },
                        )
                except Exception as e:
                    logger.warning(
                        f"[SDPipeline] short subject cloth cleanup failed (ignored): {e}"
                    )
            if (
                hair_length == "short"
                and cloth_mask_dilated is not None
                and cutoff_y_for_post is not None
                and not source_garment_prepass_applied
            ):
                try:
                    garment_repaint_applied = False
                    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                    final_hair_mask, _, _ = self._segface_hair_mask(
                        final_rgb, face_bbox
                    )
                    short_below_bob_torso_mask = self._build_short_below_bob_torso_mask(
                        cloth_mask=cloth_mask_dilated,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        hair_length=hair_length,
                        final_hair_mask=final_hair_mask,
                        torso_candidate_mask=subject_torso_candidate_mask,
                        completed_torso_fill_mask=completed_torso_fill_mask,
                        shoulder_anchor_mask=upper_clothes_overwrite_anchor_mask,
                    )
                    garment_repaint_mask = self._build_short_torso_garment_repaint_mask(
                        cloth_mask=cloth_mask_dilated,
                        torso_mask=short_below_bob_torso_mask,
                        torso_anchor_mask=subject_torso_anchor_mask,
                        torso_candidate_mask=subject_torso_candidate_mask,
                        shoulder_bridge_mask=subject_shoulder_bridge_mask,
                        sam2_hair_mask=hair_mask_for_removal,
                        face_mask=face_region_mask,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        hair_length=hair_length,
                        final_hair_mask=final_hair_mask,
                        protect_mask=protect_mask_for_sd,
                    )
                    garment_repaint_u8 = self._mask_to_u8(
                        garment_repaint_mask, threshold=0.08
                    )
                    garment_repaint_px = self._count_active_mask_px(
                        garment_repaint_mask, threshold=0.08
                    )
                    generated_resized_rgb = cand.get("generated_resized_rgb")
                    if (
                        isinstance(generated_resized_rgb, np.ndarray)
                        and debug_images_common is not None
                    ):
                        _store_rgb("generated_resized_rgb_debug", generated_resized_rgb)
                    if debug_images_common is not None:
                        _store_rgb("composite_pre_cleanup_debug", final_rgb)
                    if debug_images_common is not None and rank == 0:
                        debug_images_common["pipeline_short_below_bob_torso_mask"] = (
                            cv2.cvtColor(
                                (
                                    (
                                        np.clip(
                                            short_below_bob_torso_mask.astype(
                                                np.float32
                                            ),
                                            0.0,
                                            1.0,
                                        )
                                        > 0.08
                                    ).astype(np.uint8)
                                    * 255
                                ),
                                cv2.COLOR_GRAY2BGR,
                            )
                        )
                        debug_images_common[
                            "pipeline_controlnet_garment_repaint_mask"
                        ] = cv2.cvtColor(
                            garment_repaint_u8,
                            cv2.COLOR_GRAY2BGR,
                        )
                    if self._should_apply_cleanup_mask(
                        "controlnet_garment_repaint",
                        garment_repaint_px,
                        hair_length,
                    ):
                        final_rgb = self._sd_refine_removed_region(
                            base_rgb=final_rgb,
                            removal_mask=garment_repaint_mask,
                            face_bbox=face_bbox,
                            face_crop_pil=face_crop_pil,
                            protect_mask=protect_mask_for_sd,
                            cloth_mask=cloth_mask_dilated,
                            hair_length=hair_length,
                            seed=int(cand["seed"]) + 2411,
                            refine_mode="garment",
                            subject_gender=subject_gender_mode,
                            white_tshirt_experiment=white_tshirt_experiment,
                        )
                        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                        garment_repaint_applied = True
                    if (
                        debug_data_common is not None
                        and rank == 0
                        and garment_repaint_applied
                    ):
                        _record_rank0_cleanup_stage(
                            "controlnet_garment_repaint",
                            final_bgr,
                            trigger_mask=garment_repaint_mask,
                            mask_label="controlnet_garment_repaint",
                            extra={"garment_repaint_px": garment_repaint_px},
                        )
                except Exception as e:
                    logger.warning(
                        f"[SDPipeline] controlnet garment repaint failed (ignored): {e}"
                    )
            if debug_images_common is not None and rank == 0:
                debug_images_common["pipeline_cleanup_rank0_pre_eye_restore"] = (
                    final_bgr.copy()
                )
                _store_roi_crops(
                    "cleanup_post", cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                )
            if debug_data_common is not None and rank == 0:
                _record_rank0_cleanup_stage(
                    "cleanup_pre_eye_restore",
                    final_bgr,
                    force=True,
                )
            try:
                eye_restore_applied = False
                final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                final_hair_mask, _, _ = self._segface_hair_mask(final_rgb, face_bbox)
                eye_restore_mask = self._build_eye_region_restore_mask(
                    landmark_debug_data=landmark_debug_data,
                    image_shape=(H, W),
                    face_bbox=face_bbox,
                    hair_length=hair_length,
                    final_hair_mask=final_hair_mask,
                    subject_gender=subject_gender_mode,
                    fringe_requested=bangs_requested,
                )
                if float(eye_restore_mask.sum()) <= 0.0 and hair_length == "long":
                    eye_restore_mask = self._build_face_eye_band_restore_mask(
                        face_mask=landmark_face_mask,
                        image_shape=(H, W),
                        face_bbox=face_bbox,
                    )
                if float(eye_restore_mask.sum()) > 0.0:
                    eye_restore_strength = 0.96 if hair_length == "long" else 0.92
                    if (
                        subject_gender_mode == "male"
                        and bangs_requested
                        and hair_length in ("short", "medium")
                    ):
                        eye_restore_strength = 0.84
                    final_rgb = self._restore_reference_region(
                        final_rgb,
                        img_rgb,
                        eye_restore_mask,
                        strength=eye_restore_strength,
                    )
                    final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                    eye_restore_applied = True
                    if debug_images_common is not None and rank == 0:
                        debug_images_common["pipeline_eye_region_restore_mask"] = (
                            cv2.cvtColor(
                                (
                                    (
                                        np.clip(
                                            eye_restore_mask.astype(np.float32),
                                            0.0,
                                            1.0,
                                        )
                                        > 0.05
                                    ).astype(np.uint8)
                                    * 255
                                ),
                                cv2.COLOR_GRAY2BGR,
                            )
                        )
                if (
                    debug_data_common is not None
                    and rank == 0
                    and float(eye_restore_mask.sum()) > 0.0
                ):
                    _mask_stats(
                        "eye_region_restore",
                        eye_restore_mask,
                        diagnostic_rois.get("torso_front"),
                        bucket="cleanup_mask_stats",
                    )
                if debug_data_common is not None and rank == 0 and eye_restore_applied:
                    _record_rank0_cleanup_stage(
                        "eye_region_restore",
                        final_bgr,
                        trigger_mask=eye_restore_mask,
                        mask_label="eye_region_restore",
                    )
            except Exception as e:
                logger.warning(f"[SDPipeline] eye region restore failed (ignored): {e}")
            if debug_images_common is not None and rank == 0:
                debug_images_common["pipeline_final_result_rank0"] = final_bgr.copy()
                _store_roi_crops(
                    "final_result", cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                )
            if debug_data_common is not None and rank == 0:
                final_rgb_for_diag = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                generated_resized_rgb = cand.get("generated_resized_rgb")
                composite_before_core_bgr = cand.get("composite_before_core_bgr")
                composite_before_core_rgb = (
                    cv2.cvtColor(composite_before_core_bgr, cv2.COLOR_BGR2RGB)
                    if isinstance(composite_before_core_bgr, np.ndarray)
                    else None
                )
                composite_after_core_mask_bgr = cand.get(
                    "composite_after_core_mask_bgr"
                )
                composite_after_core_mask_rgb = (
                    cv2.cvtColor(composite_after_core_mask_bgr, cv2.COLOR_BGR2RGB)
                    if isinstance(composite_after_core_mask_bgr, np.ndarray)
                    else None
                )
                composite_pre_cleanup_bgr = cand.get("composite_pre_cleanup_bgr")
                composite_pre_cleanup_rgb = (
                    cv2.cvtColor(composite_pre_cleanup_bgr, cv2.COLOR_BGR2RGB)
                    if isinstance(composite_pre_cleanup_bgr, np.ndarray)
                    else None
                )
                composite_before_core_mask = cand.get("composite_before_core_mask")
                composite_after_core_mask = cand.get("composite_after_core_mask")
                composite_mask_for_diag = cand.get("composite_mask")
                boundary_band = _build_boundary_band(composite_mask_for_diag, (H, W))
                if debug_images_common is not None:
                    debug_images_common["pipeline_composite_boundary_band"] = (
                        cv2.cvtColor(
                            boundary_band,
                            cv2.COLOR_GRAY2BGR,
                        )
                    )
                    for name, diff_rgb in (
                        (
                            "pipeline_diff_generated_to_composite_before_core",
                            _build_abs_diff_heatmap_rgb(
                                generated_resized_rgb, composite_before_core_rgb
                            ),
                        ),
                        (
                            "pipeline_diff_composite_before_core_to_after_core_mask",
                            _build_abs_diff_heatmap_rgb(
                                composite_before_core_rgb, composite_after_core_mask_rgb
                            ),
                        ),
                        (
                            "pipeline_diff_composite_after_core_mask_to_pre_cleanup",
                            _build_abs_diff_heatmap_rgb(
                                composite_after_core_mask_rgb, composite_pre_cleanup_rgb
                            ),
                        ),
                        (
                            "pipeline_diff_generated_to_pre_cleanup",
                            _build_abs_diff_heatmap_rgb(
                                generated_resized_rgb, composite_pre_cleanup_rgb
                            ),
                        ),
                        (
                            "pipeline_diff_generated_to_final",
                            _build_abs_diff_heatmap_rgb(
                                generated_resized_rgb, final_rgb_for_diag
                            ),
                        ),
                    ):
                        if diff_rgb is None:
                            continue
                        _store_rgb(name, diff_rgb)
                        _store_roi_crops(name, diff_rgb)
                    _store_rgb(
                        "pipeline_source_with_composite_before_core_mask_overlay",
                        _overlay_mask_rgb(
                            img_rgb,
                            composite_before_core_mask,
                            (255, 196, 0),
                            alpha=0.42,
                        ),
                    )
                    _store_rgb(
                        "pipeline_source_with_composite_after_core_mask_overlay",
                        _overlay_mask_rgb(
                            img_rgb,
                            composite_after_core_mask,
                            (255, 96, 64),
                            alpha=0.46,
                        ),
                    )
                    _store_rgb(
                        "pipeline_source_with_garment_composite_mask_overlay",
                        _overlay_mask_rgb(
                            img_rgb,
                            cand.get("garment_composite_mask"),
                            (0, 208, 255),
                            alpha=0.52,
                        ),
                    )
                _mask_stats(
                    "composite_boundary_band",
                    boundary_band.astype(np.float32) / 255.0,
                    diagnostic_rois.get("torso_front"),
                    bucket="cleanup_mask_stats",
                )
                _mask_stats(
                    "composite_before_core_mask",
                    composite_before_core_mask,
                    diagnostic_rois.get("torso_front"),
                    bucket="cleanup_mask_stats",
                )
                _mask_stats(
                    "composite_after_core_mask",
                    composite_after_core_mask,
                    diagnostic_rois.get("torso_front"),
                    bucket="cleanup_mask_stats",
                )
                _mask_stats(
                    "garment_composite_mask",
                    cand.get("garment_composite_mask"),
                    diagnostic_rois.get("torso_front"),
                    bucket="cleanup_mask_stats",
                )
                diag = debug_data_common.setdefault("diagnostics", {})
                stage_diffs = diag.setdefault("stage_diffs", {})
                stage_diffs[
                    "generated_resized_vs_composite_before_core_mean_abs_diff"
                ] = _mean_abs_diff(
                    generated_resized_rgb,
                    composite_before_core_rgb,
                )
                stage_diffs[
                    "generated_resized_vs_composite_after_core_mask_mean_abs_diff"
                ] = _mean_abs_diff(
                    generated_resized_rgb,
                    composite_after_core_mask_rgb,
                )
                stage_diffs[
                    "generated_resized_vs_composite_pre_cleanup_mean_abs_diff"
                ] = _mean_abs_diff(
                    generated_resized_rgb,
                    composite_pre_cleanup_rgb,
                )
                stage_diffs[
                    "composite_before_core_vs_composite_after_core_mask_mean_abs_diff"
                ] = _mean_abs_diff(
                    composite_before_core_rgb,
                    composite_after_core_mask_rgb,
                )
                stage_diffs[
                    "composite_after_core_mask_vs_composite_pre_cleanup_mean_abs_diff"
                ] = _mean_abs_diff(
                    composite_after_core_mask_rgb,
                    composite_pre_cleanup_rgb,
                )
                stage_diffs["generated_resized_vs_final_mean_abs_diff"] = (
                    _mean_abs_diff(
                        generated_resized_rgb,
                        final_rgb_for_diag,
                    )
                )
                stage_diffs["composite_pre_cleanup_vs_final_mean_abs_diff"] = (
                    _mean_abs_diff(
                        composite_pre_cleanup_rgb,
                        final_rgb_for_diag,
                    )
                )
                for roi_name in ("chest_center", "left_side", "right_side", "neckline"):
                    rect = diagnostic_rois.get(roi_name)
                    stage_diffs[
                        f"{roi_name}_generated_resized_vs_composite_before_core_mean_abs_diff"
                    ] = _mean_abs_diff(
                        generated_resized_rgb,
                        composite_before_core_rgb,
                        rect,
                    )
                    stage_diffs[
                        f"{roi_name}_generated_resized_vs_composite_after_core_mask_mean_abs_diff"
                    ] = _mean_abs_diff(
                        generated_resized_rgb,
                        composite_after_core_mask_rgb,
                        rect,
                    )
                    stage_diffs[
                        f"{roi_name}_generated_resized_vs_composite_pre_cleanup_mean_abs_diff"
                    ] = _mean_abs_diff(
                        generated_resized_rgb,
                        composite_pre_cleanup_rgb,
                        rect,
                    )
                    stage_diffs[
                        f"{roi_name}_composite_before_core_vs_composite_after_core_mask_mean_abs_diff"
                    ] = _mean_abs_diff(
                        composite_before_core_rgb,
                        composite_after_core_mask_rgb,
                        rect,
                    )
                    stage_diffs[
                        f"{roi_name}_composite_after_core_mask_vs_composite_pre_cleanup_mean_abs_diff"
                    ] = _mean_abs_diff(
                        composite_after_core_mask_rgb,
                        composite_pre_cleanup_rgb,
                        rect,
                    )
                    stage_diffs[
                        f"{roi_name}_generated_resized_vs_final_mean_abs_diff"
                    ] = _mean_abs_diff(
                        generated_resized_rgb,
                        final_rgb_for_diag,
                        rect,
                    )
                    stage_diffs[
                        f"{roi_name}_composite_pre_cleanup_vs_final_mean_abs_diff"
                    ] = _mean_abs_diff(
                        composite_pre_cleanup_rgb,
                        final_rgb_for_diag,
                        rect,
                    )
                    stage_diffs[f"{roi_name}_source_similarity_ratio_in_final"] = (
                        _source_similarity_ratio(
                            composite_base_rgb,
                            final_rgb_for_diag,
                            rect,
                        )
                    )
                boundary_mask = boundary_band > 0
                if bool(boundary_mask.any()) and isinstance(
                    generated_resized_rgb, np.ndarray
                ):
                    gen_final_diff = np.abs(
                        generated_resized_rgb.astype(np.float32)
                        - final_rgb_for_diag.astype(np.float32)
                    ).mean(axis=2)
                    src_final_diff = np.abs(
                        composite_base_rgb.astype(np.float32)
                        - final_rgb_for_diag.astype(np.float32)
                    ).mean(axis=2)
                    stage_diffs[
                        "composite_boundary_generated_vs_final_mean_abs_diff"
                    ] = float(gen_final_diff[boundary_mask].mean())
                    stage_diffs["composite_boundary_source_vs_final_mean_abs_diff"] = (
                        float(src_final_diff[boundary_mask].mean())
                    )
                    stage_diffs["composite_boundary_source_similarity_ratio"] = float(
                        (src_final_diff[boundary_mask] <= 12.0).mean()
                    )
                if rank0_cleanup_stage_trace:
                    cleanup_summary: Dict[str, Any] = {}
                    for roi_name, threshold in (
                        ("chest_center", 8.0),
                        ("left_side", 6.0),
                        ("right_side", 6.0),
                    ):
                        key = f"{roi_name}_prev_mean_abs_diff"
                        strongest_stage = max(
                            rank0_cleanup_stage_trace,
                            key=lambda entry: float(entry.get(key, 0.0)),
                        )
                        cleanup_summary[f"{roi_name}_largest_delta_stage"] = (
                            strongest_stage.get("stage_name")
                        )
                        cleanup_summary[f"{roi_name}_largest_delta_value"] = float(
                            strongest_stage.get(key, 0.0)
                        )
                        first_material = next(
                            (
                                entry
                                for entry in rank0_cleanup_stage_trace
                                if float(entry.get(key, 0.0)) >= threshold
                            ),
                            None,
                        )
                        cleanup_summary[f"{roi_name}_first_material_stage"] = (
                            first_material.get("stage_name")
                            if first_material is not None
                            else None
                        )
                    strongest_source_reintro = max(
                        rank0_cleanup_stage_trace,
                        key=lambda entry: float(
                            entry.get("source_similarity_delta", 0.0)
                        ),
                    )
                    cleanup_summary["largest_source_reintro_stage"] = (
                        strongest_source_reintro.get("stage_name")
                    )
                    cleanup_summary["largest_source_reintro_delta"] = float(
                        strongest_source_reintro.get("source_similarity_delta", 0.0)
                    )
                    diag["cleanup_stage_summary"] = cleanup_summary
                candidate_cleanup_trace = cand.get("candidate_cleanup_trace") or []
                if candidate_cleanup_trace:
                    candidate_summary: Dict[str, Any] = {}
                    for roi_name, threshold in (
                        ("chest_center", 8.0),
                        ("left_side", 6.0),
                        ("right_side", 6.0),
                    ):
                        key = f"{roi_name}_prev_mean_abs_diff"
                        strongest_stage = max(
                            candidate_cleanup_trace,
                            key=lambda entry: float(entry.get(key, 0.0)),
                        )
                        candidate_summary[f"{roi_name}_largest_delta_stage"] = (
                            strongest_stage.get("stage_name")
                        )
                        candidate_summary[f"{roi_name}_largest_delta_value"] = float(
                            strongest_stage.get(key, 0.0)
                        )
                        first_material = next(
                            (
                                entry
                                for entry in candidate_cleanup_trace
                                if float(entry.get(key, 0.0)) >= threshold
                            ),
                            None,
                        )
                        candidate_summary[f"{roi_name}_first_material_stage"] = (
                            first_material.get("stage_name")
                            if first_material is not None
                            else None
                        )
                    diag["candidate_cleanup_summary"] = candidate_summary
                logger.info(
                    "[SDPipeline][diag][stage] gen->precore=%.2f gen->core=%.2f core->pre=%.2f gen->final=%.2f chest(core)=%.2f chest(final)=%.2f",
                    stage_diffs.get(
                        "generated_resized_vs_composite_before_core_mean_abs_diff", 0.0
                    ),
                    stage_diffs.get(
                        "generated_resized_vs_composite_after_core_mask_mean_abs_diff",
                        0.0,
                    ),
                    stage_diffs.get(
                        "composite_after_core_mask_vs_composite_pre_cleanup_mean_abs_diff",
                        0.0,
                    ),
                    stage_diffs.get("generated_resized_vs_final_mean_abs_diff", 0.0),
                    stage_diffs.get(
                        "chest_center_generated_resized_vs_composite_after_core_mask_mean_abs_diff",
                        0.0,
                    ),
                    stage_diffs.get(
                        "chest_center_generated_resized_vs_final_mean_abs_diff", 0.0
                    ),
                )
            final_rgb_for_output = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
            final_hair_mask_for_output, _, _ = self._segface_hair_mask(
                final_rgb_for_output, face_bbox
            )
            output_crop_box = build_length_aware_output_crop_box(
                self.config,
                final_bgr.shape[:2],
                face_bbox,
                hair_length,
                final_hair_mask=final_hair_mask_for_output,
            )
            result_bgr = final_bgr
            result_mask = hair_mask_for_sd
            result_face_bbox = face_bbox
            if output_crop_box is not None:
                crop_x1, crop_y1, crop_x2, crop_y2 = output_crop_box
                if (
                    crop_x1 < 0
                    or crop_y1 < 0
                    or crop_x2 > final_bgr.shape[1]
                    or crop_y2 > final_bgr.shape[0]
                ):
                    cropped_bgr = crop_with_padding(final_bgr, output_crop_box)
                else:
                    cropped_bgr = final_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
                if cropped_bgr.size > 0:
                    result_bgr = cropped_bgr.copy()
                    if (
                        isinstance(hair_mask_for_sd, np.ndarray)
                        and hair_mask_for_sd.shape[:2] == final_bgr.shape[:2]
                    ):
                        result_mask = crop_with_padding(
                            hair_mask_for_sd,
                            output_crop_box,
                            border_mode=cv2.BORDER_CONSTANT,
                            constant_value=0,
                            blur_padding=False,
                        ).copy()
                    if face_bbox is not None:
                        fx1, fy1, fx2, fy2 = [int(v) for v in face_bbox]
                        result_face_bbox = (
                            max(0, fx1 - crop_x1),
                            max(0, fy1 - crop_y1),
                            min(result_bgr.shape[1], max(0, fx2 - crop_x1)),
                            min(result_bgr.shape[0], max(0, fy2 - crop_y1)),
                        )
            if debug_data_common is not None and rank == 0:
                debug_data_common["output_crop"] = {
                    "applied": bool(output_crop_box is not None),
                    "hair_length": str(hair_length),
                    "original_shape": [
                        int(final_bgr.shape[0]),
                        int(final_bgr.shape[1]),
                    ],
                    "output_shape": [
                        int(result_bgr.shape[0]),
                        int(result_bgr.shape[1]),
                    ],
                    "crop_box": (
                        list(output_crop_box) if output_crop_box is not None else None
                    ),
                    "hair_bbox": (
                        self._mask_bbox(
                            final_hair_mask_for_output,
                            threshold=float(
                                getattr(
                                    self.config, "output_crop_hair_mask_threshold", 0.18
                                )
                            ),
                        )
                        if isinstance(final_hair_mask_for_output, np.ndarray)
                        else None
                    ),
                }
            results.append(
                SDInpaintResult(
                    image=result_bgr,
                    image_pil=Image.fromarray(
                        cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
                    ),
                    seed=cand["seed"],
                    rank=rank,
                    mask_used=mask_source,
                    mask_refine_mode=mask_refine_mode_used,
                    clip_score=float(cand["color_score"]),
                    mask=result_mask,
                    face_bbox=result_face_bbox,
                    debug_images=(
                        debug_images_common
                        if (debug_images_common is not None and rank == 0)
                        else None
                    ),
                    debug_data=(
                        debug_data_common
                        if (debug_data_common is not None and rank == 0)
                        else None
                    ),
                    style_meta=request_resolution if rank == 0 else None,
                    output_crop_box=output_crop_box,
                )
            )

        return results


from pipeline_sd_components import (
    bind_loading_methods_to_pipeline,
    bind_prompt_methods_to_pipeline,
    bind_segmentation_methods_to_pipeline,
    bind_mask_builder_methods_to_pipeline,
    bind_cloth_preserve_methods_to_pipeline,
    bind_refinement_methods_to_pipeline,
    bind_scoring_methods_to_pipeline,
)

bind_loading_methods_to_pipeline(MirrAISDPipeline)
bind_prompt_methods_to_pipeline(MirrAISDPipeline)
bind_segmentation_methods_to_pipeline(MirrAISDPipeline)
bind_mask_builder_methods_to_pipeline(MirrAISDPipeline)
bind_cloth_preserve_methods_to_pipeline(MirrAISDPipeline)
bind_refinement_methods_to_pipeline(MirrAISDPipeline)
bind_scoring_methods_to_pipeline(MirrAISDPipeline)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--hairstyle", default="wolf cut, layered")
    parser.add_argument("--color", default="auburn")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", default="./sd_output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    pipe = MirrAISDPipeline()
    pipe.load()

    results = pipe.run(img, args.hairstyle, args.color, args.top_k)

    os.makedirs(args.output, exist_ok=True)
    for r in results:
        path = os.path.join(args.output, f"rank{r.rank}_seed{r.seed}_{r.mask_used}.jpg")
        cv2.imwrite(path, r.image)
        print(f"저장: {path}")
