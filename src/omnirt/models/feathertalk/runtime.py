"""Resident FeatherTalk runtime; no OpenTalking package dependency."""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import numpy as np

from omnirt.server.realtime_avatar import encode_jpeg_sequence


class FeatherTalkRealtimeRuntime:
    def __init__(self, *, device=None, engine_factory=None):
        self.device = device or os.getenv('OMNIRT_FEATHERTALK_DEVICE', 'auto')
        self._engine_factory = engine_factory
        self._references = OrderedDict()
        self._states = {}
        self._lock = threading.RLock()

    def _paths(self, session):
        paths = []
        for key in ('dataset', 'checkpoint', 'hubert_checkpoint'):
            raw = session.runtime_config.get(f'feathertalk_{key}') or os.getenv(f'OMNIRT_FEATHERTALK_{key.upper()}')
            if not raw:
                raise ValueError(f'Missing feathertalk_{key} or OMNIRT_FEATHERTALK_{key.upper()}')
            path = Path(raw).expanduser().resolve()
            if not (path.is_dir() if key == 'dataset' else path.is_file()):
                raise FileNotFoundError(f'FeatherTalk {key} not found: {path}')
            paths.append(path)
        return paths

    def _state(self, session):
        if session.session_id in self._states:
            return self._states[session.session_id]
        dataset, checkpoint, hubert = self._paths(session)
        key = (str(dataset), str(checkpoint), checkpoint.stat().st_mtime_ns,
               str(hubert), hubert.stat().st_mtime_ns, self.device)
        cached = self._references.get(key)
        if cached is None:
            from .engine import FeatherTalkEngine
            engine = (self._engine_factory or FeatherTalkEngine)()
            engine.load_model(self.device)
            # Source coordinates must be preserved; never resize before cropping.
            import cv2
            paths = sorted((dataset / 'full_body_img').glob('*.jpg'), key=lambda p: int(p.stem))
            if not paths:
                raise ValueError(f'No FeatherTalk frames in {dataset}')
            image = cv2.imread(str(paths[0]))
            if image is None:
                raise ValueError(f'Unreadable FeatherTalk image: {paths[0]}')
            state = engine.load_reference(dataset, checkpoint, hubert, image.shape[1], image.shape[0])
            engine.warmup(state)
            cached = (engine, state)
            self._references[key] = cached
            while len(self._references) > 1:
                self._references.popitem(last=False)
        self._references.move_to_end(key)
        engine, reference = cached
        state = replace(reference, extra={})
        session.video.width = state.manifest.width
        session.video.height = state.manifest.height
        session.video.frame_count = len(state.frame_paths)
        self._states[session.session_id] = (engine, state)
        return engine, state

    def preload_reference(self, session):
        with self._lock:
            _, state = self._state(session)
            return {'type': 'preload_result', 'frames': len(state.frame_paths)}

    def set_audio_context(self, session, pcm, reset=False):
        with self._lock:
            _, state = self._state(session)
            if reset:
                frame = int(state.extra.get('frame_index_start', 0))
                state.extra.clear()
                state.extra['frame_index_start'] = frame
            state.extra['feathertalk_lookahead_pcm'] = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768

    def render_chunk(self, session, pcm_s16le):
        from .engine import AudioChunk
        import cv2
        with self._lock:
            engine, state = self._state(session)
            pcm = np.frombuffer(pcm_s16le, dtype='<i2').copy()
            features = engine.extract_features_for_stream(AudioChunk(pcm, 16000, len(pcm) / 16), state)
            frames = engine.infer(features, state)
            state.extra['frame_index_start'] = int(state.extra.get('frame_index_start', 0)) + len(frames)
            encoded = []
            for frame in frames:
                ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    raise RuntimeError('FeatherTalk JPEG encoding failed')
                encoded.append(jpeg.tobytes())
            return encode_jpeg_sequence(encoded)

    def render_idle(self, session, frame_index):
        import cv2
        with self._lock:
            engine, state = self._state(session)
            frame = engine.idle_frame(state, frame_index).data
            ok, jpeg = cv2.imencode('.jpg', frame)
            if not ok:
                raise RuntimeError('FeatherTalk idle JPEG encoding failed')
            return encode_jpeg_sequence([jpeg.tobytes()])

    def close_session(self, session_id):
        with self._lock:
            self._states.pop(session_id, None)
