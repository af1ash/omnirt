"""Standalone FeatherTalk inference engine migrated from OpenTalking.

Weights and person-specific frame/landmark assets are configured by the runtime.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from types import SimpleNamespace

@dataclass
class AudioChunk:
    data: np.ndarray
    sample_rate: int
    duration_ms: float

@dataclass
class VideoFrameData:
    data: np.ndarray
    width: int
    height: int
    timestamp_ms: float

def numpy_bgr_to_videoframe(image, timestamp_ms):
    return VideoFrameData(image, image.shape[1], image.shape[0], timestamp_ms)

AvatarManifest = Any

FPS = 25
log = logging.getLogger(__name__)


@dataclass
class FeatherTalkFeatures:
    vector: np.ndarray  # [frames, 40, 1024], including padded audio context
    frame_count: int


@dataclass
class FeatherTalkState:
    manifest: AvatarManifest
    frame_paths: list[Path]
    landmark_dir: Path
    model: Any
    hubert: Any
    extra: dict[str, Any] = field(default_factory=dict)


def _frame_index(index: int, count: int) -> int:
    if count == 1:
        return 0
    position = index % (2 * count - 2)
    return min(position, 2 * count - 2 - position)


class FeatherTalkEngine:
    model_type = "feathertalk"

    def __init__(self) -> None:
        self._device = os.getenv("OMNIRT_FEATHERTALK_DEVICE", "auto")
        self._hubert: Any = None
        self._model: Any = None

    def load_model(self, device: str = "cuda") -> None:
        # Generator weights can be person-specific and are loaded with the avatar.
        log.info("FeatherTalk initializing device: %s", device)
        import torch

        self._device = os.getenv("OMNIRT_FEATHERTALK_DEVICE", device)
        if self._device == "auto":
            from .data_utils.feather_hubert.feather_hubert import get_best_device

            self._device = str(get_best_device())
        torch.empty(0, device=self._device)  # Fail early for unavailable devices.
        log.info("FeatherTalk device ready: %s", self._device)

    def load_reference(self, dataset: Path, checkpoint: Path, hubert_checkpoint: Path, width: int, height: int) -> FeatherTalkState:
        import torch
        from .model import Model, load_checkpoint_state
        from .data_utils.feather_hubert.feather_hubert import load_feather_hubert

        paths = sorted((dataset / "full_body_img").glob("*.jpg"), key=lambda path: int(path.stem))
        if not paths:
            raise ValueError(f"No FeatherTalk frames in {dataset / 'full_body_img'}")
        model = Model().to(self._device)
        model.load_state_dict(load_checkpoint_state(str(checkpoint), self._device))
        model.eval()
        hubert = load_feather_hubert(hubert_checkpoint, device=torch.device(self._device))
        self._model, self._hubert = model, hubert
        manifest = SimpleNamespace(width=width, height=height, fps=FPS, sample_rate=16000)
        return FeatherTalkState(manifest, paths, dataset / "landmarks", model, hubert)

    def warmup(self, avatar_state: FeatherTalkState | None = None) -> None:
        import torch

        model = avatar_state.model if avatar_state is not None else self._model
        if model is None:
            return
        with torch.inference_mode():
            model(torch.zeros(1, 6, 144, 144, device=self._device),
                        torch.zeros(1, 40, 1024, device=self._device))

    @staticmethod
    def _pcm(chunk: AudioChunk) -> np.ndarray:
        data = np.asarray(chunk.data)
        if data.ndim not in (1, 2) or chunk.sample_rate <= 0:
            raise ValueError("FeatherTalk requires mono/stereo PCM with a positive sample rate")
        pcm = data.astype(np.float32)
        if np.issubdtype(data.dtype, np.signedinteger):
            pcm /= float(-np.iinfo(data.dtype).min)
        elif np.issubdtype(data.dtype, np.unsignedinteger):
            midpoint = float(np.iinfo(data.dtype).max + 1) / 2
            pcm = (pcm - midpoint) / midpoint
        if pcm.ndim == 2:
            pcm = pcm.mean(axis=1)
        if chunk.sample_rate != 16000 and pcm.size:
            from scipy.signal import resample_poly
            from math import gcd

            divisor = gcd(chunk.sample_rate, 16000)
            pcm = resample_poly(pcm, 16000 // divisor, chunk.sample_rate // divisor)
        return pcm.astype(np.float32)

    def _features(self, pcm: np.ndarray, hubert: Any, start: int, count: int) -> FeatherTalkFeatures:
        import torch

        from .data_utils.feather_hubert.feather_hubert import (
            get_feather_hubert_from_16k_speech, make_even_first_dim,
        )
        from .face_utils import gather_audio_window, reshape_audio_feat

        if count == 0:
            return FeatherTalkFeatures(np.empty((0, 40, 1024), np.float32), 0)
        # Match inference.py: normalize real audio, encode, then drop an odd token.
        # Pad feature frames only; synthetic waveform samples distort normalization.
        hidden = get_feather_hubert_from_16k_speech(pcm, hubert, device=torch.device(self._device))
        features = make_even_first_dim(hidden).numpy().reshape(-1, 2, 1024)
        missing = start + count - len(features)
        if missing > 0:
            features = np.pad(features, ((0, missing), (0, 0), (0, 0)))
        vector = np.stack([
            reshape_audio_feat(gather_audio_window(features, i)).numpy()
            for i in range(start, start + count)
        ])
        return FeatherTalkFeatures(vector, count)

    def extract_features(self, audio_chunk: AudioChunk) -> FeatherTalkFeatures:
        if self._hubert is None:
            raise RuntimeError("Load a FeatherTalk avatar before extracting features")
        pcm = self._pcm(audio_chunk)
        return self._features(pcm, self._hubert, 0, (len(pcm) + 639) // 640)

    def extract_features_for_stream(
        self, audio_chunk: AudioChunk, avatar_state: FeatherTalkState,
    ) -> FeatherTalkFeatures:
        """Extract only the supplied chunk, without history or lookahead.

        Keep the existing [frames, 40, 1024] feature layout and boundary zero
        padding. This variant does not read or update avatar_state.extra.
        A 640-sample chunk yields one HuBERT token, discarded by the existing
        even-token pairing rule, so its returned feature window is all zeros.
        """
        pcm = self._pcm(audio_chunk)
        count = (len(pcm) + 639) // 640
        return self._features(pcm, avatar_state.hubert, 0, count)

    def extract_features_for_stream1(self, audio_chunk: AudioChunk, avatar_state: FeatherTalkState) -> FeatherTalkFeatures:
        pcm = self._pcm(audio_chunk)
        extra = avatar_state.extra
        previous = np.asarray(extra.get("audio_context_pcm", np.empty(0)), dtype=np.float32)
        total_before = int(extra.get("audio_total_samples", 0))
        total = total_before + len(pcm)
        # Round cumulative time so repeated short chunks do not accumulate drift.
        start = (total_before + 639) // 640
        count = (total + 639) // 640 - start
        history = np.concatenate([previous, pcm])
        lookahead = np.asarray(extra.pop("feathertalk_lookahead_pcm", np.empty(0)), dtype=np.float32)
        joined = np.concatenate([history, lookahead])
        context_start = total_before - len(previous)
        features = self._features(joined, avatar_state.hubert, start - context_start // 640, count)
        # Preserve a frame-aligned 2s left context. Future audio is not committed
        # to history until it arrives as the next driving chunk.
        keep_start = max(0, (len(history) // 640 - 50) * 640)
        extra["audio_context_pcm"] = history[keep_start:].copy()
        extra["audio_total_samples"] = total
        return features

    @staticmethod
    def _read_frame(avatar_state: FeatherTalkState, index: int) -> np.ndarray:
        import cv2

        path = avatar_state.frame_paths[index]
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read FeatherTalk frame: {path}")
        if image.shape[:2] != (avatar_state.manifest.height, avatar_state.manifest.width):
            raise ValueError(f"Frame dimensions do not match avatar manifest: {path}")
        return image

    def infer(self, features: FeatherTalkFeatures, avatar_state: FeatherTalkState) -> list[np.ndarray]:
        """Read and prepare only the current resource frame, then paste its prediction.

        Return composed BGR images so compose_frame never has to read the same
        image/landmarks again or retain all prepared avatar assets in memory.
        """
        import cv2
        import torch

        from .face_utils import (
            FACE_BORDER, FACE_INNER_SIZE, compute_face_bbox, crop_face,
            extract_inner, hwc_to_chw_tensor, mask_mouth, read_landmarks,
        )

        start = int(avatar_state.extra.get("frame_index_start", 0))
        predictions = []
        with torch.inference_mode():
            for i, audio in enumerate(features.vector):
                index = _frame_index(start + i, len(avatar_state.frame_paths))
                path = avatar_state.frame_paths[index]
                image = self._read_frame(avatar_state, index)
                landmark_path = avatar_state.landmark_dir / f"{path.stem}.lms"
                box = compute_face_bbox(read_landmarks(str(landmark_path)))
                x1, y1, x2, y2 = box
                if not (0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]):
                    raise ValueError(f"Invalid FeatherTalk face bounding box {box}: {path}")
                crop = crop_face(image, box)
                inner = extract_inner(crop)
                model_input = torch.cat([
                    hwc_to_chw_tensor(inner), hwc_to_chw_tensor(mask_mouth(inner.copy())),
                ], dim=0).unsqueeze(0).to(self._device)
                prediction = avatar_state.model(
                    model_input,
                    torch.from_numpy(audio).unsqueeze(0).to(self._device),
                )[0]
                prediction = (prediction.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                crop[FACE_BORDER:FACE_BORDER + FACE_INNER_SIZE,
                     FACE_BORDER:FACE_BORDER + FACE_INNER_SIZE] = prediction
                image[y1:y2, x1:x2] = cv2.resize(crop, (x2 - x1, y2 - y1))
                predictions.append(image)
        return predictions

    def compose_frame(self, avatar_state: FeatherTalkState, frame_idx: int, prediction: Any) -> VideoFrameData:
        if prediction is None:
            index = _frame_index(frame_idx, len(avatar_state.frame_paths))
            prediction = self._read_frame(avatar_state, index)
        return numpy_bgr_to_videoframe(prediction, frame_idx * 1000.0 / FPS)

    def idle_frame(self, avatar_state: FeatherTalkState, frame_idx: int) -> VideoFrameData:
        index = _frame_index(frame_idx, len(avatar_state.frame_paths))
        return numpy_bgr_to_videoframe(self._read_frame(avatar_state, index), frame_idx * 1000.0 / FPS)
