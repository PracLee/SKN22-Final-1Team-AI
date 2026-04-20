"""
MirrAI SD Inpainting — SD / CV2 / LaMa 정제 로직
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
postprocess.py에서 분리된 정제/복원/합성 함수 모음.
"""

from __future__ import annotations

import gc
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .config import (
    _COMMON_STYLE_BLOCK_NEGATIVE,
    _WHITE_TSHIRT_NEGATIVE_HINTS,
    _WHITE_TSHIRT_POSITIVE_HINTS,
)
from .generation_backends import (
    get_generation_backend_spec,
    resolve_generation_canvas_size,
    resolve_generation_guidance_scale,
    resolve_generation_steps,
)

logger = logging.getLogger(__name__)


def _active_generation_backend_spec(self):
    spec = getattr(self, "_generation_backend_spec", None)
    if spec is None:
        spec = get_generation_backend_spec(self.config)
        self._generation_backend_spec = spec
    return spec


def _make_backend_generator(self, spec, seed: int) -> torch.Generator:
    if getattr(spec, "generator_device", "pipeline") == "cpu":
        return torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.Generator(device=self.device).manual_seed(int(seed))


def _run_inpaint_backend(
    self,
    *,
    image: Image.Image,
    mask_image: Image.Image,
    control_image: Image.Image,
    face_crop_pil: Image.Image,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float,
    controlnet_conditioning_scale: float,
    seeds: List[int],
    strength: float,
) -> List[Image.Image]:
    spec = _active_generation_backend_spec(self)
    if self._sd_pipe is None:
        raise RuntimeError("Generation pipeline is not loaded.")

    steps = resolve_generation_steps(self.config)
    guidance = resolve_generation_guidance_scale(self.config, guidance_scale)
    height = int(image.height)
    width = int(image.width)

    def build_kwargs(seed_or_generators, n: int) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "image": image,
            "mask_image": mask_image,
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "generator": seed_or_generators,
        }
        if spec.supports_negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        if spec.supports_controlnet:
            kwargs["control_image"] = control_image
            kwargs["controlnet_conditioning_scale"] = float(controlnet_conditioning_scale)
        if spec.supports_ip_adapter:
            kwargs["ip_adapter_image"] = [face_crop_pil]
        if spec.supports_strength:
            kwargs["strength"] = float(strength)
        if n > 1:
            kwargs["num_images_per_prompt"] = int(n)
        kwargs["guidance_scale"] = float(guidance)
        return kwargs

    if spec.batchable:
        if len(seeds) == 1:
            generator = _make_backend_generator(self, spec, int(seeds[0]))
            out = self._sd_pipe(**build_kwargs(generator, 1))
            return list(out.images)
        generators = [_make_backend_generator(self, spec, seed) for seed in seeds]
        out = self._sd_pipe(**build_kwargs(generators, len(generators)))
        return list(out.images)

    images: List[Image.Image] = []
    for seed in seeds:
        generator = _make_backend_generator(self, spec, seed)
        out = self._sd_pipe(**build_kwargs(generator, 1))
        images.extend(list(out.images))
    return images


def _resolve_generation_conditioning(
    self,
    hair_length: str,
    subject_gender: Optional[str] = None,
) -> Tuple[float, float]:
    subject_profile = self._resolve_subject_pipeline_profile(subject_gender)
    if hair_length == "short":
        ip_scale = (
            float(subject_profile.short_ip_adapter_scale)
            if subject_profile.short_ip_adapter_scale is not None
            else float(self.config.short_generation_ip_adapter_scale)
        )
        control_cap = (
            float(subject_profile.short_controlnet_scale_cap)
            if subject_profile.short_controlnet_scale_cap is not None
            else float(self.config.short_generation_controlnet_scale_cap)
        )
        control_scale = min(
            float(self.config.controlnet_conditioning_scale),
            control_cap,
        )
    elif hair_length == "medium":
        ip_scale = (
            float(subject_profile.medium_ip_adapter_scale)
            if subject_profile.medium_ip_adapter_scale is not None
            else 0.18
        )
        control_cap = (
            float(subject_profile.medium_controlnet_scale_cap)
            if subject_profile.medium_controlnet_scale_cap is not None
            else 0.20
        )
        control_scale = min(float(self.config.controlnet_conditioning_scale), control_cap)
    else:
        ip_scale = (
            float(subject_profile.long_ip_adapter_scale)
            if subject_profile.long_ip_adapter_scale is not None
            else float(self.config.ip_adapter_scale)
        )
        control_scale = (
            min(
                float(self.config.controlnet_conditioning_scale),
                float(subject_profile.long_controlnet_scale),
            )
            if subject_profile.long_controlnet_scale is not None
            else float(self.config.controlnet_conditioning_scale)
        )
    return float(ip_scale), float(control_scale)


def _generate(
    self,
    img_512: Image.Image,
    mask_512: Image.Image,
    canny_512: Image.Image,
    face_crop_pil: Image.Image,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float,
    seeds: List[int],
    hair_length: str = "long",
    subject_gender: Optional[str] = None,
) -> List[Image.Image]:
    """
    모든 seed를 단일 배치 forward pass로 생성 (순차 대비 ~절반 시간).

    diffusers는 generator를 리스트로 받으면 num_images_per_prompt 개의
    이미지를 각자 다른 seed로 한 번의 파이프라인 실행에 처리함.
    """
    spec = _active_generation_backend_spec(self)
    # 숏컷/중단발 변환 시 IP-Adapter / ControlNet 비중을 낮춰
    # 원본 긴머리 실루엣 고착을 줄인다.
    subject_profile = self._resolve_subject_pipeline_profile(subject_gender)
    ip_scale, control_scale = self._resolve_generation_conditioning(
        hair_length,
        subject_gender=subject_gender,
    )
    if spec.supports_ip_adapter and hasattr(self._sd_pipe, "set_ip_adapter_scale"):
        self._sd_pipe.set_ip_adapter_scale(ip_scale)
    logger.info(
        "[SDPipeline] generation backend=%s ip_adapter_scale=%.4f controlnet_scale=%.4f "
        "(hair_length=%s, subject_branch=%s)",
        spec.key,
        ip_scale if spec.supports_ip_adapter else 0.0,
        control_scale if spec.supports_controlnet else 0.0,
        hair_length,
        subject_profile.key,
    )

    n = len(seeds)
    logger.info(f"[SDPipeline] 배치 생성 시작 (n={n}, seeds={seeds})")
    diffusion_started = time.time()
    try:
        free_gb = None
        total_gb = None
        if torch.cuda.is_available():
            free_mem, total_mem = torch.cuda.mem_get_info()
            free_gb = free_mem / (1024 ** 3)
            total_gb = total_mem / (1024 ** 3)
        logger.info(
            "[SDPipeline] diffusion forward dispatch: steps=%d size=%dx%d free_gpu_gb=%s total_gpu_gb=%s",
            int(resolve_generation_steps(self.config)),
            int(img_512.width),
            int(img_512.height),
            "n/a" if free_gb is None else f"{free_gb:.2f}",
            "n/a" if total_gb is None else f"{total_gb:.2f}",
        )
    except Exception as e:
        logger.warning(f"[SDPipeline] diffusion dispatch stats failed (ignored): {e}")

    with torch.inference_mode():
        images = _run_inpaint_backend(
            self,
            image=img_512,
            mask_image=mask_512,
            control_image=canny_512,
            face_crop_pil=face_crop_pil,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=control_scale,
            seeds=seeds,
            strength=float(getattr(spec, "default_strength", 1.0)),
        )

    logger.info(
        "[SDPipeline] 배치 생성 완료 → %d장 (%.2fs)",
        len(images),
        time.time() - diffusion_started,
    )
    return images

def _cv2_refine_cloth_region(
    base_rgb: np.ndarray,
    cloth_refine_mask: np.ndarray,
    reference_rgb: Optional[np.ndarray] = None,
    reference_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Clean residual shirt/blouse blur with a small deterministic inpaint pass.
    The mask is expected to already exclude hair and face-protect regions.
    """
    H, W = base_rgb.shape[:2]
    if cloth_refine_mask.shape != (H, W):
        return base_rgb

    ref_rgb = base_rgb
    if reference_rgb is not None and reference_rgb.shape[:2] == (H, W):
        ref_rgb = reference_rgb

    mask_u8 = (np.clip(cloth_refine_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    if int((mask_u8 > 0).sum()) < 60:
        return base_rgb

    mask_u8 = cv2.dilate(
        mask_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    ring_u8 = cv2.dilate(
        mask_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
        iterations=1,
    )
    ring_u8 = cv2.subtract(
        ring_u8,
        cv2.dilate(
            mask_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        ),
    )
    telea = cv2.inpaint(base_rgb, mask_u8, 5, cv2.INPAINT_TELEA)
    ns = cv2.inpaint(base_rgb, mask_u8, 4, cv2.INPAINT_NS)
    refined = cv2.addWeighted(telea, 0.66, ns, 0.34, 0.0)
    mask_bool = mask_u8 > 0
    ring_bool = ring_u8 > 0
    if reference_rgb is not None and reference_rgb.shape[:2] == (H, W):
        ref_telea = cv2.inpaint(ref_rgb, mask_u8, 5, cv2.INPAINT_TELEA)
        ref_ns = cv2.inpaint(ref_rgb, mask_u8, 4, cv2.INPAINT_NS)
        ref_refined = cv2.addWeighted(ref_telea, 0.62, ref_ns, 0.38, 0.0)
        refined = cv2.addWeighted(refined, 0.42, ref_refined, 0.58, 0.0)
    if int(ring_bool.sum()) >= 80 and int(mask_bool.sum()) >= 60:
        refined_lab = cv2.cvtColor(refined, cv2.COLOR_RGB2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        ring_ref_bool = ring_bool.copy()
        if reference_mask is not None and reference_mask.shape == (H, W):
            ref_mask_u8 = cv2.dilate(
                (np.clip(reference_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                iterations=1,
            )
            masked_ring = np.logical_and(ring_ref_bool, ref_mask_u8 > 0)
            if int(masked_ring.sum()) >= 40:
                ring_ref_bool = masked_ring
        ring_vals = ref_lab[ring_ref_bool]
        mask_vals = refined_lab[mask_bool]
        ring_mean = ring_vals.mean(axis=0)
        mask_mean = mask_vals.mean(axis=0)
        ring_std = ring_vals.std(axis=0)
        mask_std = np.maximum(mask_vals.std(axis=0), 1.0)
        tone_matched = mask_vals.copy()
        tone_matched[:, 0] = np.clip(
            (tone_matched[:, 0] - mask_mean[0]) * np.clip(ring_std[0] / mask_std[0], 0.82, 1.18)
            + mask_mean[0]
            + np.clip(ring_mean[0] - mask_mean[0], -16.0, 16.0) * 0.72,
            0.0,
            255.0,
        )
        tone_matched[:, 1] = np.clip(
            tone_matched[:, 1] + np.clip(ring_mean[1] - mask_mean[1], -5.0, 5.0) * 0.55,
            0.0,
            255.0,
        )
        tone_matched[:, 2] = np.clip(
            tone_matched[:, 2] + np.clip(ring_mean[2] - mask_mean[2], -5.0, 5.0) * 0.55,
            0.0,
            255.0,
        )
        refined_lab[mask_bool] = tone_matched
        refined = cv2.cvtColor(refined_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

        detail_reference_rgb = ref_rgb
        lowpass = cv2.GaussianBlur(detail_reference_rgb, (0, 0), sigmaX=3.2, sigmaY=3.2)
        detail_src = np.clip(
            detail_reference_rgb.astype(np.float32) - lowpass.astype(np.float32) + 128.0,
            0.0,
            255.0,
        ).astype(np.uint8)
        detail_telea = cv2.inpaint(detail_src, mask_u8, 3, cv2.INPAINT_TELEA)
        detail_ns = cv2.inpaint(detail_src, mask_u8, 3, cv2.INPAINT_NS)
        detail_fill = cv2.addWeighted(detail_telea, 0.70, detail_ns, 0.30, 0.0)
        detail_signed = detail_fill.astype(np.float32) - 128.0

        gray = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        ring_detail = float(np.mean(np.abs(lap[ring_bool]))) if int(ring_bool.sum()) > 0 else 0.0
        mask_detail = float(np.mean(np.abs(detail_signed[mask_bool]))) if int(mask_bool.sum()) > 0 else 0.0
        texture_gain = float(np.clip(ring_detail / max(mask_detail, 1.0), 0.65, 1.35))
        textured = np.clip(
            refined.astype(np.float32) + detail_signed * (0.46 * texture_gain),
            0.0,
            255.0,
        )
        refined = textured.astype(np.uint8)

    alpha = cv2.GaussianBlur(
        (mask_u8 > 0).astype(np.float32),
        (0, 0),
        sigmaX=2.4,
        sigmaY=2.4,
    )[..., np.newaxis]
    alpha = np.clip(alpha * 0.92, 0.0, 1.0)
    out = refined.astype(np.float32) * alpha + base_rgb.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def _sd_refine_removed_region(
    self,
    base_rgb: np.ndarray,          # H×W×3 RGB (cv2 inpaint 1차 결과)
    removal_mask: np.ndarray,      # H×W float32 (긴머리 제거 영역)
    face_bbox: Tuple[int, int, int, int],
    face_crop_pil: Image.Image,    # IP-Adapter conditioning face
    protect_mask: Optional[np.ndarray],  # H×W float32 (얼굴 보호)
    cloth_mask: Optional[np.ndarray],    # H×W float32 (의상 영역)
    hair_length: str,
    seed: int,
    refine_mode: str = "generic",
    subject_gender: Optional[str] = None,
    white_tshirt_experiment: bool = False,
) -> np.ndarray:
    """
    긴머리 제거 후 남는 어색한 영역(목/어깨/배경)을 SD로 한 번 더 정리.
    """
    H, W = base_rgb.shape[:2]
    removal_mask = self._resize_mask_to_shape(removal_mask, (H, W))
    protect_mask = self._resize_mask_to_shape(protect_mask, (H, W))
    cloth_mask = self._resize_mask_to_shape(cloth_mask, (H, W))
    if removal_mask.shape != (H, W):
        raise ValueError(f"removal_mask shape mismatch: {removal_mask.shape} vs {(H, W)}")

    # removal 영역 중심으로만 SD를 적용하기 위해 그대로 letterbox 변환
    fill_mask = (removal_mask > 0.5).astype(np.float32)
    img_512, mask_512, canny_512, scale, pad = self._prepare_sd_inputs(
        base_rgb,
        fill_mask,
        mask_edge_suppression=0.45,
        target_size=resolve_generation_canvas_size(self.config),
    )
    white_tshirt_positive = ", ".join(_WHITE_TSHIRT_POSITIVE_HINTS[:2])
    white_tshirt_negative = ", ".join(_WHITE_TSHIRT_NEGATIVE_HINTS)
    normalized_gender = self._normalize_subject_gender(subject_gender)

    if refine_mode == "garment":
        if white_tshirt_experiment:
            fill_prompt = (
                "professional portrait photo, restore a plain white t-shirt, "
                "simple white crew-neck t-shirt, clean connected shoulder cloth, "
                "preserve clean white cotton fabric continuity, natural tee folds and seams, "
                "no hair strands on clothes, photorealistic details"
            )
        else:
            fill_prompt = (
                "professional portrait photo, restore the same original outfit, "
                "clean connected shoulder cloth, continuous cardigan blouse shirt or jacket fabric, "
                "preserve neckline collar seams and buttons, realistic garment folds and texture, "
                "no hair strands on clothes, photorealistic details"
            )
        fill_guidance = 6.2 if hair_length == "short" else 6.6
        fill_negative = (
            f"{white_tshirt_negative}, " if white_tshirt_experiment else ""
        ) + (
            "hair strands on clothes, loose dangling hair, long hair, black blob, disconnected clothing, "
            "broken neckline, missing collar, missing buttons, warped garment, melted fabric, exposed shoulder skin, "
            "deformed neck, artifacts, blurry, cartoon, painting, "
            f"{_COMMON_STYLE_BLOCK_NEGATIVE}"
        )
    elif refine_mode == "cloth":
        if white_tshirt_experiment:
            fill_prompt = (
                "professional portrait photo, preserve a plain white t-shirt shape, "
                f"{white_tshirt_positive}, realistic white cotton fabric texture continuity, "
                "coherent tee folds and seams, clean neck and shoulders, "
                "no hair strands in masked region, photorealistic details"
            )
        else:
            fill_prompt = (
                "professional portrait photo, preserve the original shirt or blouse shape, "
                "realistic clothing fabric texture continuity, coherent folds and seams, "
                "clean neck and shoulders, no hair strands in masked region, photorealistic details"
            )
        fill_guidance = 6.8 if hair_length == "short" else 7.0
        fill_negative = (
            f"{white_tshirt_negative}, " if white_tshirt_experiment else ""
        ) + (
            "hair strands, loose dangling hair, long hair, blur, blurry cloth, smudged cloth, "
            "melted fabric, duplicate collar, broken neckline, extra folds, extra buttons, "
            "warped shirt, warped blouse, deformed neck, artifacts, cartoon, painting, "
            f"{_COMMON_STYLE_BLOCK_NEGATIVE}"
        )
    elif refine_mode == "short_tail" and hair_length == "short":
        garment_phrase = "same plain white t-shirt preserved" if white_tshirt_experiment else "same shirt or blouse preserved"
        cloth_phrase = "realistic white cotton tee texture continuity" if white_tshirt_experiment else "realistic clothing fabric texture continuity"
        if normalized_gender == "male":
            fill_prompt = (
                "professional portrait photo, clean male short haircut with a soft two-block balance, "
                "clean side line above the ears, visible neck and shoulders, "
                f"{garment_phrase}, {cloth_phrase}, "
                "non-bob masculine short silhouette, no hair below jawline, no dangling strands in masked region, "
                "photorealistic details"
            )
        else:
            fill_prompt = (
                "professional portrait photo, neat compact short jaw-length bob haircut, "
                "clean side silhouette above the shoulders, visible neck and shoulders, "
                f"{garment_phrase}, {cloth_phrase}, "
                "clean neckline, no hair below jawline, no shoulder-length side hair, "
                "no dangling strands in masked region, photorealistic details"
            )
        fill_guidance = 8.2
        if normalized_gender == "male":
            fill_negative = (
                f"{white_tshirt_negative}, " if white_tshirt_experiment else ""
            ) + (
                "bob, lob, mini bob, c-curl bob, feminine bob silhouette, feminine face-framing layers, "
                "long hair, shoulder-length hair, medium hair, hair below jawline, "
                "hair touching shoulders, dangling side tails, loose strands, extra hair mass, "
                "warped shirt, warped blouse, melted fabric, deformed neck, artifacts, blurry, "
                "smudged texture, cartoon, painting, "
                f"{_COMMON_STYLE_BLOCK_NEGATIVE}"
            )
        else:
            fill_negative = (
                f"{white_tshirt_negative}, " if white_tshirt_experiment else ""
            ) + (
                "long hair, shoulder-length hair, medium hair, lob haircut, hair below jawline, "
                "hair touching shoulders, dangling side tails, loose strands, extra hair mass, "
                "warped shirt, warped blouse, melted fabric, deformed neck, artifacts, blurry, "
                "smudged texture, cartoon, painting, "
                f"{_COMMON_STYLE_BLOCK_NEGATIVE}"
            )
    elif hair_length == "short":
        garment_phrase = "same plain white t-shirt preserved" if white_tshirt_experiment else "same shirt or blouse preserved"
        cloth_phrase = "realistic white cotton tee texture continuity" if white_tshirt_experiment else "realistic clothing fabric texture continuity"
        fill_prompt = (
            "professional portrait photo, clean natural neck and shoulders, "
            f"{garment_phrase}, {cloth_phrase}, "
            "coherent neckline, collar and sleeve folds, coherent background, "
            "short-hair silhouette maintained, no long hair below jawline, "
            "no loose dangling strands in masked region, photorealistic details"
        )
        fill_guidance = 7.1
        fill_negative = (
            f"{white_tshirt_negative}, " if white_tshirt_experiment else ""
        ) + (
            "long hair, hair below chin, hair below shoulders, loose hair strands, "
            "wavy hair, straight long hair, wig, ponytail, braid, bangs, side locks, "
            "deformed neck, artifacts, blurry, smudged texture, melted details, cartoon, painting, "
            f"{_COMMON_STYLE_BLOCK_NEGATIVE}"
        )
    else:
        fill_prompt = (
            "professional portrait photo, clean neck and shoulders, "
            "natural skin and clothing texture continuity, coherent background, "
            "no loose long hair strands in masked region, photorealistic details"
        )
        fill_guidance = 7.6
        fill_negative = (
            "long hair, hair below chin, hair below shoulders, loose hair strands, "
            "wavy hair, straight long hair, wig, ponytail, braid, bangs, side locks, "
            "deformed neck, artifacts, blurry, smudged texture, melted details, cartoon, painting, "
            f"{_COMMON_STYLE_BLOCK_NEGATIVE}"
        )

    # 배경 복원은 identity 영향이 과하면 긴머리가 다시 생길 수 있어 scale을 낮춘다.
    spec = _active_generation_backend_spec(self)
    if spec.supports_ip_adapter and hasattr(self._sd_pipe, "set_ip_adapter_scale"):
        self._sd_pipe.set_ip_adapter_scale(0.0)
    if refine_mode == "garment":
        fill_control = float(np.clip(max(self.config.controlnet_conditioning_scale, 0.36), 0.32, 0.52))
        fill_steps = max(24, self.config.num_inference_steps - 4)
        fill_strength = 0.68
    elif refine_mode == "cloth":
        fill_control = float(np.clip(max(self.config.controlnet_conditioning_scale, 0.16), 0.10, 0.24))
        fill_steps = max(20, self.config.num_inference_steps - 8)
        fill_strength = 0.84
    elif refine_mode == "short_tail" and hair_length == "short":
        fill_control = float(np.clip(max(self.config.controlnet_conditioning_scale, 0.14), 0.10, 0.20))
        fill_steps = max(22, self.config.num_inference_steps - 6)
        fill_strength = 0.90
    else:
        fill_control = float(np.clip(max(self.config.controlnet_conditioning_scale, 0.18), 0.12, 0.30))
        fill_steps = max(24, self.config.num_inference_steps - 4)
        fill_strength = 0.88

    previous_backend_steps = getattr(self.config, "generation_backend_steps", None)
    self.config.generation_backend_steps = fill_steps
    try:
        with torch.inference_mode():
            images = _run_inpaint_backend(
                self,
                image=img_512,
                mask_image=mask_512,
                control_image=canny_512,
                face_crop_pil=face_crop_pil,
                prompt=fill_prompt,
                negative_prompt=fill_negative,
                guidance_scale=fill_guidance,
                controlnet_conditioning_scale=fill_control,
                seeds=[int(seed)],
                strength=fill_strength,
            )
    finally:
        self.config.generation_backend_steps = previous_backend_steps

    gen_np = np.array(images[0])  # square RGB

    # letterbox 역변환
    pad_l, pad_t = pad
    new_w = int(W * scale)
    new_h = int(H * scale)
    gen_cropped = gen_np[pad_t:pad_t + new_h, pad_l:pad_l + new_w]
    gen_orig = cv2.resize(gen_cropped, (W, H), interpolation=cv2.INTER_LANCZOS4)

    alpha = cv2.GaussianBlur(fill_mask, (0, 0), sigmaX=7.0, sigmaY=7.0)
    alpha = np.clip(alpha, 0.0, 1.0)

    # 중앙 편향을 완화해 side 잔존 영역도 자연스럽게 복원한다.
    x1, y1, x2, y2 = face_bbox
    cx = 0.5 * (x1 + x2)
    face_w = max(float(x2 - x1), 1.0)
    sigma_x = max(face_w * 1.45, 44.0)
    xs = np.arange(W, dtype=np.float32)
    center_weight = np.exp(-0.5 * ((xs - cx) / sigma_x) ** 2)
    alpha = alpha * (0.65 + 0.35 * center_weight[np.newaxis, :])

    # 의상 영역은 과도한 hallucination을 줄이기 위해 SD 블렌딩 가중치를 낮춘다.
    if cloth_mask is not None and cloth_mask.shape == (H, W):
        cloth_w = np.clip(cloth_mask.astype(np.float32), 0.0, 1.0)
        alpha = alpha * (1.0 - 0.18 * cloth_w)

    # 얼굴은 기존 픽셀 고정
    if refine_mode == "garment":
        alpha = cv2.GaussianBlur(fill_mask, (0, 0), sigmaX=5.2, sigmaY=5.2)
        alpha = np.clip(alpha * 0.76, 0.0, 0.86)
        if cloth_mask is not None and cloth_mask.shape == (H, W):
            cloth_w = np.clip(cloth_mask.astype(np.float32), 0.0, 1.0)
            alpha = np.clip(alpha * (0.82 + 0.12 * cloth_w), 0.0, 0.92)
    elif refine_mode == "cloth":
        alpha = cv2.GaussianBlur(fill_mask, (0, 0), sigmaX=6.0, sigmaY=6.0)
        alpha = np.clip(alpha, 0.0, 1.0)
        if cloth_mask is not None and cloth_mask.shape == (H, W):
            cloth_w = np.clip(cloth_mask.astype(np.float32), 0.0, 1.0)
            alpha = np.clip(alpha * (0.94 + 0.18 * cloth_w), 0.0, 1.0)

    if protect_mask is not None:
        protect_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        protect = cv2.dilate(protect_mask.astype(np.float32), protect_k)
        alpha = alpha * (1.0 - np.clip(protect, 0.0, 1.0))

    alpha = alpha[..., np.newaxis]
    refined = (
        gen_orig.astype(np.float32) * alpha
        + base_rgb.astype(np.float32) * (1.0 - alpha)
    )
    return np.clip(refined, 0, 255).astype(np.uint8)

def _filter_short_center_cleanup_mask(
    self,
    mask_u8: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    *,
    anchor_u8: Optional[np.ndarray] = None,
    top_scale: float = 0.02,
    bottom_scale: float = 0.82,
    half_w_scale: float = 0.24,
    shrink_half_w_scale: float = 0.18,
    max_area_scale: float = 0.10,
    max_width_scale: float = 0.34,
    min_height_scale: float = 0.10,
    center_allow_scale: float = 0.18,
    max_total_scale: float = 0.05,
) -> np.ndarray:
    H, W = mask_u8.shape[:2]
    filtered_u8 = (mask_u8 > 0).astype(np.uint8) * 255
    if int((filtered_u8 > 0).sum()) == 0:
        return np.zeros((H, W), dtype=np.uint8)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = int(0.5 * (x1 + x2))

    gate_u8 = np.zeros((H, W), dtype=np.uint8)
    gate_half_w = max(14, int(face_w * half_w_scale))
    gate_left = max(0, cx - gate_half_w)
    gate_right = min(W, cx + gate_half_w)
    gate_top = max(0, int(cutoff_y + face_h * top_scale))
    gate_bottom = min(H, int(cutoff_y + face_h * bottom_scale))
    if gate_top >= gate_bottom or gate_left >= gate_right:
        return np.zeros((H, W), dtype=np.uint8)
    gate_u8[gate_top:gate_bottom, gate_left:gate_right] = 255
    filtered_u8 = cv2.bitwise_and(filtered_u8, gate_u8)
    if int((filtered_u8 > 0).sum()) == 0:
        return np.zeros((H, W), dtype=np.uint8)

    anchor_local_u8 = np.zeros((H, W), dtype=np.uint8)
    if anchor_u8 is not None and anchor_u8.shape == (H, W):
        anchor_local_u8 = cv2.bitwise_and((anchor_u8 > 0).astype(np.uint8) * 255, gate_u8)

    max_component_area = max(96, int(face_w * face_h * max_area_scale))
    max_component_width = max(24, int(face_w * max_width_scale))
    min_component_height = max(12, int(face_h * min_height_scale))
    center_allow = max(14, int(face_w * center_allow_scale))
    keep_u8 = np.zeros((H, W), dtype=np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filtered_u8, 8)
    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        comp_u8 = (labels == idx).astype(np.uint8) * 255
        comp_cx = x + (w * 0.5)
        if area < 14 or area > max_component_area:
            continue
        if w > max_component_width or h < min_component_height:
            continue
        if (y + h) > gate_bottom:
            continue
        if abs(comp_cx - cx) > center_allow:
            continue
        if w > max(20, int(face_w * 0.28)) and h < max(20, int(face_h * 0.22)):
            continue
        if int((anchor_local_u8 > 0).sum()) > 0:
            anchor_overlap = int((cv2.bitwise_and(comp_u8, anchor_local_u8) > 0).sum())
            if anchor_overlap < max(4, int(area * 0.04)):
                continue
        keep_u8 = cv2.bitwise_or(keep_u8, comp_u8)

    if int((keep_u8 > 0).sum()) == 0:
        return np.zeros((H, W), dtype=np.uint8)

    max_total_px = max(72, int(face_w * face_h * max_total_scale))
    current_px = int((keep_u8 > 0).sum())
    if current_px > max_total_px:
        shrink_u8 = np.zeros((H, W), dtype=np.uint8)
        shrink_half_w = max(12, int(face_w * shrink_half_w_scale))
        shrink_left = max(0, cx - shrink_half_w)
        shrink_right = min(W, cx + shrink_half_w)
        shrink_top = max(gate_top, int(cutoff_y + face_h * max(top_scale, 0.04)))
        shrink_bottom = min(H, int(cutoff_y + face_h * max(bottom_scale - 0.06, 0.10)))
        if shrink_top < shrink_bottom and shrink_left < shrink_right:
            shrink_u8[shrink_top:shrink_bottom, shrink_left:shrink_right] = 255
            keep_u8 = cv2.bitwise_and(keep_u8, shrink_u8)

    return keep_u8


def _filter_short_torso_box_mask(
    self,
    img_rgb: Optional[np.ndarray],
    removal_mask: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    support_mask: Optional[np.ndarray],
    center_support_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
) -> np.ndarray:
    H, W = removal_mask.shape[:2]
    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = float(0.5 * (x1 + x2))

    removal_u8 = (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    original_px = int((removal_u8 > 0).sum())
    if original_px < 40:
        return np.clip(removal_mask, 0.0, 1.0).astype(np.float32)

    if cloth_mask is not None and cloth_mask.shape == (H, W):
        cloth_u8 = (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.16).astype(np.uint8)
    else:
        cloth_u8 = np.zeros((H, W), dtype=np.uint8)

    support_hint_u8 = np.zeros((H, W), dtype=np.uint8)
    if support_mask is not None and support_mask.shape == (H, W):
        support_hint_u8 = cv2.bitwise_or(
            support_hint_u8,
            cv2.dilate(
                (np.clip(support_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 17)),
                iterations=1,
            ),
        )
    if center_support_mask is not None and center_support_mask.shape == (H, W):
        support_hint_u8 = cv2.bitwise_or(
            support_hint_u8,
            cv2.dilate(
                (np.clip(center_support_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 19)),
                iterations=1,
            ),
        )

    dark_evidence_u8 = np.zeros((H, W), dtype=np.uint8)
    bright_cloth_evidence_u8 = np.zeros((H, W), dtype=np.uint8)
    if img_rgb is not None and img_rgb.shape[:2] == (H, W):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=5.0, sigmaY=5.0)
        sat = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
        blackhat = cv2.morphologyEx(
            gray.astype(np.uint8),
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 13)),
        )
        dark_evidence_u8 = (
            (
                ((gray < 168.0) & ((blur - gray) > 2.4))
                | (blackhat > 8)
            ).astype(np.uint8)
            * 255
        )
        dark_zone_u8 = np.zeros((H, W), dtype=np.uint8)
        dark_top = max(0, int(cutoff_y - face_h * 0.04))
        dark_bottom = min(H, int(cutoff_y + face_h * 0.96))
        dark_left = max(0, int(x1 - face_w * 1.12))
        dark_right = min(W, int(x2 + face_w * 1.12))
        if dark_top < dark_bottom and dark_left < dark_right:
            dark_zone_u8[dark_top:dark_bottom, dark_left:dark_right] = 255
            dark_evidence_u8 = cv2.bitwise_and(dark_evidence_u8, dark_zone_u8)
        if int((cloth_u8 > 0).sum()) > 0:
            dark_evidence_u8 = cv2.bitwise_and(dark_evidence_u8, cloth_u8.astype(np.uint8) * 255)
        bright_cloth_evidence_u8 = (
            (
                (gray > 178.0)
                & (blur > 182.0)
                & (sat < 60.0)
            ).astype(np.uint8)
            * 255
        )
        bright_zone_u8 = np.zeros((H, W), dtype=np.uint8)
        bright_top = max(0, int(cutoff_y + face_h * 0.02))
        bright_bottom = min(H, int(cutoff_y + face_h * 1.04))
        bright_left = max(0, int(x1 - face_w * 1.16))
        bright_right = min(W, int(x2 + face_w * 1.16))
        if bright_top < bright_bottom and bright_left < bright_right:
            bright_zone_u8[bright_top:bright_bottom, bright_left:bright_right] = 255
            bright_cloth_evidence_u8 = cv2.bitwise_and(bright_cloth_evidence_u8, bright_zone_u8)
        if int((cloth_u8 > 0).sum()) > 0:
            bright_cloth_evidence_u8 = cv2.bitwise_and(
                bright_cloth_evidence_u8,
                cloth_u8.astype(np.uint8) * 255,
            )

    keep_u8 = np.zeros((H, W), dtype=np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(removal_u8, 8)
    boxy_width = max(26, int(face_w * 0.30))
    boxy_height = max(24, int(face_h * 0.30))
    boxy_area = max(120, int(face_w * face_h * 0.075))
    center_half = max(16, int(face_w * 0.24))
    low_top = int(cutoff_y + face_h * 0.16)
    low_bottom = int(cutoff_y + face_h * 0.82)
    center_strand_width = max(16, int(face_w * 0.18))
    center_strand_height = max(30, int(face_h * 0.30))

    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < 16:
            continue

        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        bottom = y + h
        comp_mask = labels == idx
        cloth_overlap = int(cloth_u8[comp_mask].sum())
        overlap_ratio = float(cloth_overlap) / float(area) if area > 0 else 0.0
        comp_cx = float(centroids[idx][0])
        comp_u8 = comp_mask.astype(np.uint8) * 255
        support_overlap = int((cv2.bitwise_and(comp_u8, support_hint_u8) > 0).sum())
        dark_overlap = int((cv2.bitwise_and(comp_u8, dark_evidence_u8) > 0).sum())
        dark_ratio = float(dark_overlap) / float(area) if area > 0 else 0.0
        bright_overlap = int((cv2.bitwise_and(comp_u8, bright_cloth_evidence_u8) > 0).sum())
        bright_ratio = float(bright_overlap) / float(area) if area > 0 else 0.0

        is_boxy = w >= boxy_width and h >= boxy_height and area >= boxy_area
        is_center_box = abs(comp_cx - cx) <= center_half and w >= max(22, int(face_w * 0.28))
        is_center_strand = (
            abs(comp_cx - cx) <= center_half
            and w <= center_strand_width
            and h >= center_strand_height
            and bottom >= int(cutoff_y + face_h * 0.30)
        )
        is_low = y >= low_top or bottom >= low_bottom
        is_side_component = abs(comp_cx - cx) >= max(22, int(face_w * 0.28))
        is_side_blob = (
            is_side_component
            and overlap_ratio >= 0.30
            and w >= max(20, int(face_w * 0.22))
            and area >= max(72, int(face_w * face_h * 0.022))
        )
        is_bright_side_blob = (
            is_side_component
            and overlap_ratio >= 0.26
            and bright_ratio >= 0.18
            and w >= max(24, int(face_w * 0.28))
            and area >= max(96, int(face_w * face_h * 0.024))
            and bottom >= int(cutoff_y + face_h * 0.14)
        )
        if is_center_strand:
            keep_u8[comp_mask] = 255
            continue
        if is_bright_side_blob:
            if support_overlap < max(16, int(area * 0.08)):
                if dark_ratio < 0.16 or bright_ratio > (dark_ratio * 1.8 + 0.06):
                    continue
        if is_side_blob and (dark_ratio < 0.09 or dark_overlap < max(8, int(area * 0.04))):
            if support_overlap < max(14, int(area * 0.08)):
                continue
            if h < max(28, int(face_h * 0.42)):
                continue
        if overlap_ratio > 0.46 and (is_boxy or is_center_box or is_low):
            if support_overlap >= max(18, int(area * 0.08)):
                supported_u8 = cv2.bitwise_and(comp_u8, support_hint_u8)
                supported_u8 = cv2.dilate(
                    supported_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 19)),
                    iterations=1,
                )
                supported_u8 = cv2.bitwise_and(supported_u8, comp_u8)
                keep_u8 = cv2.bitwise_or(keep_u8, supported_u8)
            continue
        if is_low and is_boxy and support_overlap < max(10, int(area * 0.05)):
            continue
        if is_side_component and overlap_ratio > 0.34 and dark_ratio < 0.07 and area >= max(80, int(face_w * face_h * 0.028)):
            continue
        if (
            is_side_component
            and overlap_ratio > 0.30
            and bright_ratio >= 0.22
            and dark_ratio < 0.11
            and support_overlap < max(12, int(area * 0.06))
            and area >= max(110, int(face_w * face_h * 0.030))
        ):
            continue

        keep_u8[comp_mask] = 255

    if int((support_hint_u8 > 0).sum()) > 0:
        deep_torso_u8 = np.zeros((H, W), dtype=np.uint8)
        deep_x1 = max(0, int(x1 - face_w * 1.05))
        deep_x2 = min(W, int(x2 + face_w * 1.05))
        deep_y1 = max(0, int(cutoff_y + face_h * 0.22))
        if deep_x1 < deep_x2 and deep_y1 < H:
            deep_torso_u8[deep_y1:, deep_x1:deep_x2] = 255
            if int((cloth_u8 > 0).sum()) > 0:
                deep_torso_u8 = cv2.bitwise_and(deep_torso_u8, cloth_u8.astype(np.uint8) * 255)
            precise_keep_u8 = cv2.dilate(
                support_hint_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 13)),
                iterations=1,
            )
            precise_keep_u8 = cv2.bitwise_and(precise_keep_u8, deep_torso_u8)
            keep_u8 = cv2.bitwise_and(keep_u8, cv2.bitwise_not(deep_torso_u8))
            keep_u8 = cv2.bitwise_or(
                keep_u8,
                cv2.bitwise_and(removal_u8, precise_keep_u8),
            )

    kept_px = int((keep_u8 > 0).sum())
    if kept_px <= 0:
        return np.clip(removal_mask, 0.0, 1.0).astype(np.float32)

    filtered = removal_mask.astype(np.float32) * (keep_u8.astype(np.float32) / 255.0)
    filtered = cv2.GaussianBlur(filtered, (0, 0), sigmaX=1.1, sigmaY=1.1)
    return np.clip(filtered, 0.0, 1.0).astype(np.float32)


def _restrict_short_removal_to_tail_lanes(
    self,
    removal_mask: np.ndarray,
    support_mask: Optional[np.ndarray],
    center_support_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
) -> np.ndarray:
    if hair_length != "short":
        return np.clip(removal_mask, 0.0, 1.0).astype(np.float32)

    H, W = removal_mask.shape[:2]
    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)

    removal_u8 = (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    original_px = int((removal_u8 > 0).sum())
    if original_px < 60:
        return np.clip(removal_mask, 0.0, 1.0).astype(np.float32)

    side_seed_u8 = np.zeros((H, W), dtype=np.uint8)
    if support_mask is not None and support_mask.shape == (H, W):
        side_seed_u8 = (np.clip(support_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255

    center_seed_u8 = np.zeros((H, W), dtype=np.uint8)
    if center_support_mask is not None and center_support_mask.shape == (H, W):
        center_seed_u8 = (np.clip(center_support_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255

    if int((side_seed_u8 > 0).sum()) < 24:
        removal_side_seed_u8 = np.zeros((H, W), dtype=np.uint8)
        seed_top = max(0, int(cutoff_y + face_h * 0.08))
        seed_bottom = min(H, int(cutoff_y + face_h * 1.52))
        left_outer = max(0, int(x1 - face_w * 0.30))
        left_inner = min(W, int(x1 + face_w * 0.02))
        right_inner = max(0, int(x2 - face_w * 0.02))
        right_outer = min(W, int(x2 + face_w * 0.30))
        if seed_top < seed_bottom:
            if left_outer < left_inner:
                removal_side_seed_u8[seed_top:seed_bottom, left_outer:left_inner] = 255
            if right_inner < right_outer:
                removal_side_seed_u8[seed_top:seed_bottom, right_inner:right_outer] = 255
        removal_side_seed_u8 = cv2.bitwise_and(removal_side_seed_u8, removal_u8)
        if int((removal_side_seed_u8 > 0).sum()) >= 24:
            removal_side_seed_u8 = cv2.morphologyEx(
                removal_side_seed_u8,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 17)),
            )
            side_seed_u8 = cv2.bitwise_or(side_seed_u8, removal_side_seed_u8)

    if int((side_seed_u8 > 0).sum()) < 24 and int((center_seed_u8 > 0).sum()) < 12:
        return np.clip(removal_mask, 0.0, 1.0).astype(np.float32)

    side_lane_u8 = cv2.dilate(
        side_seed_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 57)),
        iterations=1,
    )
    center_lane_u8 = cv2.dilate(
        center_seed_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 53)),
        iterations=1,
    )
    lane_u8 = cv2.bitwise_or(side_lane_u8, center_lane_u8)

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(cutoff_y - face_h * 0.04))
    bottom = min(H, int(cutoff_y + face_h * 1.62))
    left = max(0, int(x1 - face_w * 1.24))
    right = min(W, int(x2 + face_w * 1.24))
    if top >= bottom or left >= right:
        return np.clip(removal_mask, 0.0, 1.0).astype(np.float32)
    corridor_u8[top:bottom, left:right] = 255
    lane_u8 = cv2.bitwise_and(lane_u8, corridor_u8)
    if int((lane_u8 > 0).sum()) < 120:
        return np.clip(removal_mask, 0.0, 1.0).astype(np.float32)

    filtered_u8 = cv2.bitwise_and(removal_u8, lane_u8)
    filtered_u8 = cv2.morphologyEx(
        filtered_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 21)),
    )
    filtered_px = int((filtered_u8 > 0).sum())
    if filtered_px < max(220, int(original_px * 0.34)):
        return np.clip(removal_mask, 0.0, 1.0).astype(np.float32)

    filtered = removal_mask.astype(np.float32) * (filtered_u8.astype(np.float32) / 255.0)
    filtered = cv2.GaussianBlur(filtered, (0, 0), sigmaX=1.0, sigmaY=1.2)
    return np.clip(filtered, 0.0, 1.0).astype(np.float32)


def _cv2_cleanup_dark_tail_blob(
    img_rgb: np.ndarray,
    dark_tail_u8: np.ndarray,
) -> np.ndarray:
    """Run a small focused cv2 inpaint pass over deep dark residual tail blobs."""
    if dark_tail_u8.shape[:2] != img_rgb.shape[:2]:
        return img_rgb
    if int((dark_tail_u8 > 0).sum()) < 40:
        return img_rgb

    mask_u8 = cv2.dilate(
        dark_tail_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 25)),
        iterations=1,
    )
    telea = cv2.inpaint(img_rgb, mask_u8, 7, cv2.INPAINT_TELEA)
    ns = cv2.inpaint(img_rgb, mask_u8, 6, cv2.INPAINT_NS)
    fill = cv2.addWeighted(telea, 0.74, ns, 0.26, 0.0)
    alpha = cv2.GaussianBlur(
        (mask_u8 > 0).astype(np.float32),
        (0, 0),
        sigmaX=3.6,
        sigmaY=3.6,
    )[..., np.newaxis]
    out = fill.astype(np.float32) * alpha + img_rgb.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def _build_no_bangs_hairline_halo_mask(
    composited_rgb: np.ndarray,
    generated_rgb: np.ndarray,
    composite_mask: np.ndarray,
    no_bangs_seed_mask: Optional[np.ndarray],
    release_mask: Optional[np.ndarray],
    protect_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    hair_length: str = "short",
    subject_gender: str = "unknown",
) -> np.ndarray:
    """
    Build a thin repair mask for the bright fringe left between protected face
    pixels and generated short hair in no-bangs composites.
    """
    H, W = composited_rgb.shape[:2]
    if generated_rgb.shape[:2] != (H, W) or composite_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    seed = np.zeros((H, W), dtype=np.float32)
    if no_bangs_seed_mask is not None and no_bangs_seed_mask.shape == (H, W):
        seed = np.maximum(
            seed,
            np.clip(no_bangs_seed_mask.astype(np.float32), 0.0, 1.0),
        )
    if release_mask is not None and release_mask.shape == (H, W):
        seed = np.maximum(seed, np.clip(release_mask.astype(np.float32), 0.0, 1.0))
    if float(seed.sum()) <= 8.0:
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    face_w = max(x2 - x1, 1)
    face_h = max(y2 - y1, 1)
    forehead_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(y1 - face_h * 0.18))
    bottom_ratio = (
        0.38
        if hair_length == "short" and subject_gender == "female"
        else 0.34 if hair_length == "short" else 0.30
    )
    bottom = min(H, int(y1 + face_h * bottom_ratio))
    left = max(0, int(x1 - face_w * 0.34))
    right = min(W, int(x2 + face_w * 0.34))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)
    forehead_u8[top:bottom, left:right] = 255

    hair_core_u8 = (
        np.clip(composite_mask.astype(np.float32), 0.0, 1.0) > 0.08
    ).astype(np.uint8) * 255
    if int((hair_core_u8 > 0).sum()) < 60:
        return np.zeros((H, W), dtype=np.float32)

    outer = cv2.dilate(
        hair_core_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        iterations=1,
    )
    inner = cv2.erode(
        hair_core_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    hair_boundary_u8 = cv2.subtract(outer, inner)

    seed_u8 = cv2.dilate(
        (seed > 0.03).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )
    candidate_u8 = cv2.bitwise_and(hair_boundary_u8, seed_u8)
    candidate_u8 = cv2.bitwise_and(candidate_u8, forehead_u8)
    if protect_mask is not None and protect_mask.shape == (H, W):
        protect_u8 = cv2.dilate(
            (np.clip(protect_mask.astype(np.float32), 0.0, 1.0) > 0.03).astype(
                np.uint8
            )
            * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        candidate_u8 = cv2.bitwise_and(candidate_u8, protect_u8)

    if int((candidate_u8 > 0).sum()) < 12:
        return np.zeros((H, W), dtype=np.float32)

    comp_gray = cv2.cvtColor(composited_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gen_gray = cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    comp_sat = cv2.cvtColor(composited_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(
        np.float32
    )
    bright_halo = (
        ((comp_gray > 212.0) & (comp_sat < 86.0))
        | ((comp_gray > 188.0) & ((comp_gray - gen_gray) > 14.0) & (comp_sat < 112.0))
    )
    candidate_u8 = cv2.bitwise_and(
        candidate_u8,
        bright_halo.astype(np.uint8) * 255,
    )
    if int((candidate_u8 > 0).sum()) < 8:
        return np.zeros((H, W), dtype=np.float32)

    candidate_u8 = cv2.morphologyEx(
        candidate_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    candidate_u8 = cv2.dilate(
        candidate_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    mask = cv2.GaussianBlur(
        candidate_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=1.15,
        sigmaY=1.25,
    )
    return np.clip(mask * 0.84, 0.0, 0.84).astype(np.float32)


def _remove_residual_hair_below_cutoff(
    self,
    img_rgb: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    removal_mask: Optional[np.ndarray] = None,
    shoulder_protect: Optional[np.ndarray] = None,
    neckline_preserve: Optional[np.ndarray] = None,
    lateral_preserve: Optional[np.ndarray] = None,
    hair_length: str = "short",
    center_anchor_mask: Optional[np.ndarray] = None,
    debug_outputs: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """
    short/medium 변환 후 cutoff 아래에 남은 머리카락을 재검출해 정리.
    """
    H, W = img_rgb.shape[:2]
    cutoff_y = int(np.clip(cutoff_y, 0, H - 1))
    _, y1, _, y2 = face_bbox
    face_h = max(int(y2 - y1), 1)
    soft_zone = max(10, int(face_h * 0.22))
    soft_end = min(H - 1, cutoff_y + soft_zone)

    hair_now, _, _ = self._segface_hair_mask(img_rgb, face_bbox)
    residual = hair_now.copy()
    residual[:cutoff_y, :] = 0.0

    residual_u8 = (residual > 0.5).astype(np.uint8) * 255
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    residual_u8 = cv2.morphologyEx(residual_u8, cv2.MORPH_OPEN, open_k)

    if soft_end > cutoff_y:
        ramp = np.ones((H,), dtype=np.float32)
        ramp[:cutoff_y] = 0.0
        ramp[cutoff_y:soft_end + 1] = np.linspace(
            0.0, 1.0, soft_end - cutoff_y + 1, dtype=np.float32
        )
        residual_soft = (residual_u8.astype(np.float32) / 255.0) * ramp[:, np.newaxis]
        residual_u8 = (residual_soft > 0.50).astype(np.uint8) * 255

    dark_tail_u8 = np.zeros((H, W), dtype=np.uint8)
    front_cleanup_u8 = np.zeros((H, W), dtype=np.uint8)
    residual_near_u8 = np.zeros((H, W), dtype=np.uint8)
    if removal_mask is not None and removal_mask.shape == (H, W):
        tail_hint = self._build_side_tail_cleanup_mask(
            removal_mask=removal_mask,
            face_bbox=face_bbox,
            cutoff_y=cutoff_y,
            hair_length=hair_length,
        )
        tail_hint_u8 = (tail_hint > 0.0).astype(np.uint8) * 255
        if int((tail_hint_u8 > 0).sum()) > 0:
            deep_start = min(H, int(cutoff_y + face_h * (0.18 if hair_length == "short" else 0.24)))
            deep_zone = np.zeros((H, W), dtype=np.uint8)
            if deep_start < H:
                deep_zone[deep_start:, :] = 255
            if int((residual_u8 > 0).sum()) > 0:
                residual_near_u8 = cv2.dilate(
                    residual_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
                    iterations=1,
                )
                tail_hint_u8 = cv2.bitwise_and(
                    tail_hint_u8,
                    cv2.bitwise_or(residual_near_u8, deep_zone),
                )
            else:
                tail_hint_u8 = cv2.bitwise_and(tail_hint_u8, deep_zone)
            residual_u8 = cv2.bitwise_or(residual_u8, tail_hint_u8)

        if hair_length == "short":
            tail_core = self._build_short_tail_core_mask(
                removal_mask=removal_mask,
                face_bbox=face_bbox,
                cutoff_y=cutoff_y,
                hair_length=hair_length,
            )
            tail_core_u8 = (tail_core > 0.0).astype(np.uint8) * 255
            if int((tail_core_u8 > 0).sum()) > 0:
                tail_core_gate_u8 = np.zeros((H, W), dtype=np.uint8)
                shallow_limit = min(H, int(cutoff_y + face_h * 0.24))
                if shallow_limit > cutoff_y:
                    center_band_u8 = np.zeros((H, W), dtype=np.uint8)
                    face_w_local = max(int(face_bbox[2] - face_bbox[0]), 1)
                    center_half = max(16, int(face_w_local * 0.22))
                    center_band_u8[
                        cutoff_y:shallow_limit,
                        max(0, int(0.5 * (face_bbox[0] + face_bbox[2])) - center_half):min(W, int(0.5 * (face_bbox[0] + face_bbox[2])) + center_half),
                    ] = 255
                    shallow_gate_u8 = cv2.bitwise_and(residual_near_u8, center_band_u8)
                    tail_core_gate_u8[cutoff_y:shallow_limit, :] = shallow_gate_u8[cutoff_y:shallow_limit, :]
                deep_core_start = min(H, int(cutoff_y + face_h * 0.40))
                if deep_core_start < H:
                    tail_core_gate_u8[deep_core_start:, :] = 255
                tail_core_u8 = cv2.bitwise_and(tail_core_u8, tail_core_gate_u8)
            if int((tail_core_u8 > 0).sum()) > 0:
                residual_u8 = cv2.bitwise_or(residual_u8, tail_core_u8)

            front_cleanup = self._build_front_strand_cleanup_mask(
                removal_mask=removal_mask,
                face_bbox=face_bbox,
                cutoff_y=cutoff_y,
                hair_length=hair_length,
                anchor_mask=center_anchor_mask,
            )
            front_cleanup_u8 = (front_cleanup > 0.0).astype(np.uint8) * 255
            if int((front_cleanup_u8 > 0).sum()) > 0:
                residual_u8 = cv2.bitwise_or(residual_u8, front_cleanup_u8)

        dark_tail = self._build_dark_tail_residual_mask(
            img_rgb=img_rgb,
            removal_mask=removal_mask,
            face_bbox=face_bbox,
            cutoff_y=cutoff_y,
            hair_length=hair_length,
        )
        dark_tail_u8 = (dark_tail > 0.0).astype(np.uint8) * 255
        if int((dark_tail_u8 > 0).sum()) > 0:
            residual_u8 = cv2.bitwise_or(residual_u8, dark_tail_u8)

    if int((residual_u8 > 0).sum()) < 60:
        return img_rgb

    dilate_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 13) if hair_length == "short" else (9, 9),
    )
    residual_u8 = cv2.dilate(residual_u8, dilate_k, iterations=1)

    if shoulder_protect is not None and shoulder_protect.shape == (H, W):
        protect_threshold = 0.62 if hair_length == "short" else 0.34
        protect_u8 = (shoulder_protect > protect_threshold).astype(np.uint8) * 255
        if hair_length == "short" and int((protect_u8 > 0).sum()) > 0:
            deep_release_y = min(H, int(cutoff_y + face_h * 0.28))
            if deep_release_y < H:
                protect_u8[deep_release_y:, :] = 0
            protect_u8 = cv2.erode(
                protect_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )
        if int((protect_u8 > 0).sum()) > 0:
            residual_u8 = cv2.bitwise_and(residual_u8, cv2.bitwise_not(protect_u8))

    if neckline_preserve is not None and neckline_preserve.shape == (H, W):
        preserve_u8 = (neckline_preserve > (0.34 if hair_length == "short" else 0.26)).astype(np.uint8) * 255
        if int((preserve_u8 > 0).sum()) > 0:
            if hair_length == "short":
                deep_release_y = min(H, int(cutoff_y + face_h * 0.22))
                if deep_release_y < H:
                    preserve_u8[deep_release_y:, :] = 0
            preserve_u8 = cv2.dilate(
                preserve_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                iterations=1,
            )
            residual_u8 = cv2.bitwise_and(residual_u8, cv2.bitwise_not(preserve_u8))

    if lateral_preserve is not None and lateral_preserve.shape == (H, W):
        lateral_u8 = (lateral_preserve > 0.18).astype(np.uint8) * 255
        if int((lateral_u8 > 0).sum()) > 0:
            lateral_u8 = cv2.dilate(
                lateral_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 9)),
                iterations=1,
            )
            residual_u8 = cv2.bitwise_and(residual_u8, cv2.bitwise_not(lateral_u8))

    if debug_outputs is not None:
        debug_outputs["residual_below_cutoff_mask"] = residual_u8.astype(np.float32) / 255.0
        debug_outputs["residual_below_cutoff_near_mask"] = residual_near_u8.astype(np.float32) / 255.0
        debug_outputs["residual_below_cutoff_dark_tail_mask"] = dark_tail_u8.astype(np.float32) / 255.0
        debug_outputs["residual_below_cutoff_front_cleanup_mask"] = front_cleanup_u8.astype(np.float32) / 255.0

    if int((residual_u8 > 0).sum()) < 60:
        return img_rgb

    cleaned = self._lama_inpaint(img_rgb, residual_u8)
    if int((dark_tail_u8 > 0).sum()) > 0:
        cleaned = self._cv2_cleanup_dark_tail_blob(cleaned, dark_tail_u8)
    if int((front_cleanup_u8 > 0).sum()) > 0:
        cleaned = self._cv2_cleanup_dark_tail_blob(cleaned, front_cleanup_u8)
    return cleaned

def _final_cutoff_cleanup(
    self,
    img_rgb: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    removal_mask: np.ndarray,
    cutoff_y: int,
    shoulder_protect: Optional[np.ndarray] = None,
    neckline_preserve: Optional[np.ndarray] = None,
    lateral_preserve: Optional[np.ndarray] = None,
    hair_length: str = "short",
    center_anchor_mask: Optional[np.ndarray] = None,
    debug_outputs: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """
    최종 결과에서 cutoff 아래 long-hair 제거 마스크 영역을 한 번 더 정리.
    """
    H, W = img_rgb.shape[:2]
    if removal_mask.shape != (H, W):
        return img_rgb

    x1, y1, x2, y2 = face_bbox
    face_h = max(int(y2 - y1), 1)
    face_w = max(int(x2 - x1), 1)
    cx = int(0.5 * (x1 + x2))
    soft_zone = max(12, int(face_h * 0.25))

    force = removal_mask.copy().astype(np.float32)
    cutoff_y = int(np.clip(cutoff_y, 0, H - 1))
    force[:cutoff_y, :] = 0.0
    soft_end = min(H - 1, cutoff_y + soft_zone)
    if soft_end > cutoff_y:
        ramp = np.ones((H,), dtype=np.float32)
        ramp[:cutoff_y] = 0.0
        ramp[cutoff_y:soft_end + 1] = np.linspace(
            0.0, 1.0, soft_end - cutoff_y + 1, dtype=np.float32
        )
        force = force * ramp[:, np.newaxis]

    # 얼굴 주변 corridor 안에서만 cleanup을 허용해 의상/배경 훼손을 줄인다.
    corridor_ratio = 1.55 if hair_length == "short" else 1.35
    x_min = max(0, int(x1 - face_w * corridor_ratio))
    x_max = min(W, int(x2 + face_w * corridor_ratio))
    corridor = np.zeros((H, W), dtype=np.uint8)
    if x_min < x_max:
        corridor[:, x_min:x_max] = 255

    force_thresh = 0.56 if hair_length == "short" else 0.54
    force_u8 = ((force > force_thresh).astype(np.uint8) * 255)
    force_u8 = cv2.bitwise_and(force_u8, corridor)
    dark_tail_u8 = np.zeros((H, W), dtype=np.uint8)
    front_cleanup_u8 = np.zeros((H, W), dtype=np.uint8)

    # 실제 남아있는 hair 픽셀과 교집합을 우선 적용해 의상/배경 훼손 방지
    hair_now, _, _ = self._segface_hair_mask(img_rgb, face_bbox)
    hair_now[:cutoff_y, :] = 0.0
    hair_now_u8 = (hair_now > 0.5).astype(np.uint8) * 255
    if int((hair_now_u8 > 0).sum()) > 0:
        hair_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (11, 11) if hair_length == "short" else (9, 9),
        )
        hair_now_u8 = cv2.dilate(hair_now_u8, hair_k, iterations=1)

    hair_inter_u8 = cv2.bitwise_and(force_u8, hair_now_u8)
    if hair_length == "short":
        # SegFace miss를 보완하기 위해 side-zone에 한해 high-confidence force를 추가 반영
        center_half = max(18, int(face_w * 0.42))
        side_zone = corridor.copy()
        side_zone[:, max(0, cx - center_half):min(W, cx + center_half)] = 0
        fallback_u8 = ((force > 0.78).astype(np.uint8) * 255)
        fallback_u8 = cv2.bitwise_and(fallback_u8, side_zone)
        force_u8 = cv2.bitwise_or(hair_inter_u8, fallback_u8)
        hair_near_u8 = np.zeros((H, W), dtype=np.uint8)
        if int((hair_now_u8 > 0).sum()) > 0:
            hair_near_u8 = cv2.dilate(
                hair_now_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)),
                iterations=1,
            )

        tail_hint = self._build_side_tail_cleanup_mask(
            removal_mask=removal_mask,
            face_bbox=face_bbox,
            cutoff_y=cutoff_y,
            hair_length=hair_length,
        )
        tail_hint_u8 = (tail_hint > 0.0).astype(np.uint8) * 255
        if int((tail_hint_u8 > 0).sum()) > 0:
            deep_start = min(H, int(cutoff_y + face_h * 0.18))
            deep_zone = np.zeros((H, W), dtype=np.uint8)
            if deep_start < H:
                deep_zone[deep_start:, :] = 255
            forced_tail_u8 = cv2.bitwise_and(
                tail_hint_u8,
                cv2.bitwise_or(hair_near_u8, deep_zone),
            )
            force_u8 = cv2.bitwise_or(force_u8, forced_tail_u8)

        tail_core = self._build_short_tail_core_mask(
            removal_mask=removal_mask,
            face_bbox=face_bbox,
            cutoff_y=cutoff_y,
            hair_length=hair_length,
        )
        tail_core_u8 = (tail_core > 0.0).astype(np.uint8) * 255
        if int((tail_core_u8 > 0).sum()) > 0:
            tail_core_gate_u8 = np.zeros((H, W), dtype=np.uint8)
            shallow_limit = min(H, int(cutoff_y + face_h * 0.24))
            if shallow_limit > cutoff_y:
                center_band_u8 = np.zeros((H, W), dtype=np.uint8)
                center_half = max(16, int(face_w * 0.22))
                center_band_u8[
                    cutoff_y:shallow_limit,
                    max(0, cx - center_half):min(W, cx + center_half),
                ] = 255
                shallow_gate_u8 = cv2.bitwise_and(hair_near_u8, center_band_u8)
                tail_core_gate_u8[cutoff_y:shallow_limit, :] = shallow_gate_u8[cutoff_y:shallow_limit, :]
            deep_core_start = min(H, int(cutoff_y + face_h * 0.40))
            if deep_core_start < H:
                tail_core_gate_u8[deep_core_start:, :] = 255
            tail_core_u8 = cv2.bitwise_and(tail_core_u8, tail_core_gate_u8)
        if int((tail_core_u8 > 0).sum()) > 0:
            force_u8 = cv2.bitwise_or(force_u8, tail_core_u8)

        front_cleanup = self._build_front_strand_cleanup_mask(
            removal_mask=removal_mask,
            face_bbox=face_bbox,
            cutoff_y=cutoff_y,
            hair_length=hair_length,
            anchor_mask=center_anchor_mask,
        )
        front_cleanup_u8 = (front_cleanup > 0.0).astype(np.uint8) * 255
        if int((front_cleanup_u8 > 0).sum()) > 0:
            force_u8 = cv2.bitwise_or(force_u8, front_cleanup_u8)
    else:
        # medium도 SegFace miss 보완용 fallback force 일부 허용
        fallback_u8 = ((force > 0.74).astype(np.uint8) * 255)
        fallback_u8 = cv2.bitwise_and(fallback_u8, corridor)
        force_u8 = cv2.bitwise_or(hair_inter_u8, fallback_u8)

    dark_tail = self._build_dark_tail_residual_mask(
        img_rgb=img_rgb,
        removal_mask=removal_mask,
        face_bbox=face_bbox,
        cutoff_y=cutoff_y,
        hair_length=hair_length,
    )
    dark_tail_u8 = (dark_tail > 0.0).astype(np.uint8) * 255
    if int((dark_tail_u8 > 0).sum()) > 0:
        force_u8 = cv2.bitwise_or(force_u8, cv2.bitwise_and(dark_tail_u8, corridor))

    if shoulder_protect is not None and shoulder_protect.shape == (H, W):
        protect_threshold = 0.62 if hair_length == "short" else 0.34
        protect_u8 = (shoulder_protect > protect_threshold).astype(np.uint8) * 255
        if hair_length == "short" and int((protect_u8 > 0).sum()) > 0:
            deep_release_y = min(H, int(cutoff_y + face_h * 0.28))
            if deep_release_y < H:
                protect_u8[deep_release_y:, :] = 0
            protect_u8 = cv2.erode(
                protect_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )
        if int((protect_u8 > 0).sum()) > 0:
            force_u8 = cv2.bitwise_and(force_u8, cv2.bitwise_not(protect_u8))

    if neckline_preserve is not None and neckline_preserve.shape == (H, W):
        preserve_u8 = (neckline_preserve > (0.34 if hair_length == "short" else 0.26)).astype(np.uint8) * 255
        if int((preserve_u8 > 0).sum()) > 0:
            if hair_length == "short":
                deep_release_y = min(H, int(cutoff_y + face_h * 0.22))
                if deep_release_y < H:
                    preserve_u8[deep_release_y:, :] = 0
            preserve_u8 = cv2.dilate(
                preserve_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                iterations=1,
            )
            force_u8 = cv2.bitwise_and(force_u8, cv2.bitwise_not(preserve_u8))

    if lateral_preserve is not None and lateral_preserve.shape == (H, W):
        lateral_u8 = (lateral_preserve > 0.18).astype(np.uint8) * 255
        if int((lateral_u8 > 0).sum()) > 0:
            lateral_u8 = cv2.dilate(
                lateral_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 9)),
                iterations=1,
            )
            force_u8 = cv2.bitwise_and(force_u8, cv2.bitwise_not(lateral_u8))

    if debug_outputs is not None:
        debug_outputs["final_cutoff_force_mask"] = force_u8.astype(np.float32) / 255.0
        debug_outputs["final_cutoff_corridor_mask"] = corridor.astype(np.float32) / 255.0
        debug_outputs["final_cutoff_hair_intersection_mask"] = hair_inter_u8.astype(np.float32) / 255.0
        debug_outputs["final_cutoff_dark_tail_mask"] = dark_tail_u8.astype(np.float32) / 255.0
        debug_outputs["final_cutoff_front_cleanup_mask"] = front_cleanup_u8.astype(np.float32) / 255.0

    min_cleanup_px = 28 if hair_length == "short" else 40
    if int((force_u8 > 0).sum()) < min_cleanup_px:
        return img_rgb

    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 13) if hair_length == "short" else (7, 7),
    )
    force_u8 = cv2.dilate(force_u8, k, iterations=1)

    # LaMa로 잔존 hair 제거
    cleaned = self._lama_inpaint(img_rgb, force_u8)
    if int((dark_tail_u8 > 0).sum()) > 0:
        cleaned = self._cv2_cleanup_dark_tail_blob(cleaned, dark_tail_u8)
    if int((front_cleanup_u8 > 0).sum()) > 0:
        cleaned = self._cv2_cleanup_dark_tail_blob(cleaned, front_cleanup_u8)
    return cleaned

def _composite(
    self,
    orig_bgr: np.ndarray,
    orig_rgb: np.ndarray,
    gen_pil: Image.Image,            # 512×512 RGB
    hair_mask: np.ndarray,           # H×W float32 (original resolution)
    scale: float,
    pad: Tuple[int, int],            # (pad_left, pad_top)
    original_size: Tuple[int, int],  # (W, H)
    garment_mask: Optional[np.ndarray] = None,
    protect_mask: Optional[np.ndarray] = None,  # H×W float32: 이 영역은 alpha=0 강제 (얼굴 보호)
    protect_release_mask: Optional[np.ndarray] = None,
    hair_length: str = "long",
    subject_gender: str = "unknown",
    fringe_requested: bool = False,
) -> np.ndarray:
    """
    SD 생성 이미지를 원본에 합성.
    - hair / garment regenerate 영역: SD 생성 결과
    - 그 외 (+ protect_mask): 원본 (얼굴/배경 유지)
    """
    W, H = original_size
    hair_mask = self._resize_mask_to_shape(hair_mask, (H, W))
    garment_mask = self._resize_mask_to_shape(garment_mask, (H, W))
    protect_mask = self._resize_mask_to_shape(protect_mask, (H, W))
    protect_release_mask = self._resize_mask_to_shape(protect_release_mask, (H, W))
    pad_l, pad_t = pad
    new_w = int(W * scale)
    new_h = int(H * scale)

    # letterbox 제거 → 원본 비율로 crop
    gen_np = np.array(gen_pil)   # 512×512×3 RGB
    gen_cropped = gen_np[pad_t:pad_t + new_h, pad_l:pad_l + new_w]

    # 원본 해상도로 upscale
    gen_orig = cv2.resize(gen_cropped, (W, H), interpolation=cv2.INTER_LANCZOS4)

    # alpha 블렌딩: short/medium는 경계를 더 또렷하게 유지
    preserve_fringe_detail = bool(
        subject_gender == "male"
        and fringe_requested
        and hair_length in ("short", "medium")
    )
    sigma = 6.0
    if hair_length == "short":
        sigma = 4.2
    elif hair_length == "medium":
        sigma = 4.1 if preserve_fringe_detail else 4.8
    hair_alpha = cv2.GaussianBlur(hair_mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if hair_length == "short":
        hair_alpha = np.clip((hair_alpha - 0.10) / 0.90, 0.0, 1.0)
    elif hair_length == "medium":
        if preserve_fringe_detail:
            hair_alpha = np.clip((hair_alpha - 0.04) / 0.96, 0.0, 1.0)
        else:
            hair_alpha = np.clip((hair_alpha - 0.07) / 0.93, 0.0, 1.0)
    if preserve_fringe_detail and protect_release_mask is not None:
        hair_alpha = np.maximum(
            hair_alpha,
            np.clip(protect_release_mask.astype(np.float32), 0.0, 1.0) * 0.72,
        )
    hair_alpha = np.clip(hair_alpha, 0.0, 1.0)

    garment_alpha = np.zeros((H, W), dtype=np.float32)
    if garment_mask is not None:
        garment_alpha = cv2.GaussianBlur(
            np.clip(garment_mask.astype(np.float32), 0.0, 1.0),
            (0, 0),
            sigmaX=3.2,
            sigmaY=3.8,
        )
        garment_alpha = np.clip((garment_alpha - 0.04) / 0.96, 0.0, 1.0)

    alpha = np.maximum(hair_alpha, garment_alpha)

    # 얼굴/귀/눈 등 보호 영역: alpha를 0으로 강제
    # → Gaussian blur가 얼굴 경계로 번지더라도 원본 픽셀 100% 유지
    if protect_mask is not None:
        # protect_mask도 살짝 dilate해서 경계까지 확실히 보호
        protect_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3) if preserve_fringe_detail else (5, 5),
        )
        protect_dilated = cv2.dilate(protect_mask.astype(np.float32), protect_k)
        if protect_release_mask is not None:
            protect_dilated = np.clip(
                protect_dilated
                - np.clip(
                    protect_release_mask.astype(np.float32)
                    * (1.55 if preserve_fringe_detail else 1.35),
                    0.0,
                    1.0,
                ),
                0.0,
                1.0,
            )
        alpha = alpha * (1.0 - np.clip(protect_dilated, 0.0, 1.0))
        if preserve_fringe_detail and protect_release_mask is not None:
            alpha = np.maximum(
                alpha,
                np.clip(protect_release_mask.astype(np.float32), 0.0, 1.0) * 0.75,
            )

    alpha = alpha[..., np.newaxis]   # H×W×1

    orig_f = orig_rgb.astype(np.float32)
    gen_f  = gen_orig.astype(np.float32)
    blend  = gen_f * alpha + orig_f * (1.0 - alpha)
    blend  = np.clip(blend, 0, 255).astype(np.uint8)

    return cv2.cvtColor(blend, cv2.COLOR_RGB2BGR)

def unload(self) -> None:
    """VRAM 해제"""
    import gc
    self._sd_pipe = None
    self._generation_backend_spec = None
    self._sam2_factory = None
    if self._mp_face:
        self._mp_face.close()
    if self._mp_face_mesh:
        self._mp_face_mesh.close()
    self._mp_face = None
    self._mp_face_mesh = None
    self._loaded = False
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("[SDPipeline] 모델 언로드 완료")


def bind_refinement_methods_to_pipeline(cls) -> None:
    """정제/복원/합성 메서드를 MirrAISDPipeline에 바인딩."""
    cls._resolve_generation_conditioning = _resolve_generation_conditioning
    cls._generate = _generate
    cls._cv2_refine_cloth_region = staticmethod(_cv2_refine_cloth_region)
    cls._sd_refine_removed_region = _sd_refine_removed_region
    cls._filter_short_center_cleanup_mask = _filter_short_center_cleanup_mask
    cls._filter_short_torso_box_mask = _filter_short_torso_box_mask
    cls._restrict_short_removal_to_tail_lanes = _restrict_short_removal_to_tail_lanes
    cls._cv2_cleanup_dark_tail_blob = staticmethod(_cv2_cleanup_dark_tail_blob)
    cls._build_no_bangs_hairline_halo_mask = staticmethod(
        _build_no_bangs_hairline_halo_mask
    )
    cls._remove_residual_hair_below_cutoff = _remove_residual_hair_below_cutoff
    cls._final_cutoff_cleanup = _final_cutoff_cleanup
    cls._composite = _composite
    cls.unload = unload
