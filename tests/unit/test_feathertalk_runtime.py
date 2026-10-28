import base64
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from omnirt.server.avatar_app import create_avatar_app
from omnirt.server.realtime_avatar import MAGIC_AUDIO, decode_jpeg_sequence
from omnirt.models.feathertalk.runtime import FeatherTalkRealtimeRuntime


@dataclass
class State:
    manifest: object
    frame_paths: list
    extra: dict = field(default_factory=dict)


class Engine:
    loaded = 0
    def load_model(self, device):
        pass
    def load_reference(self, dataset, checkpoint, hubert, width, height):
        Engine.loaded += 1
        return State(SimpleNamespace(width=width, height=height), [dataset / 'full_body_img/0.jpg'])
    def warmup(self, state):
        pass
    def extract_features_for_stream(self, chunk, state):
        state.extra['samples'] = state.extra.get('samples', 0) + len(chunk.data)
        return chunk
    def infer(self, features, state):
        return [np.full((16, 16, 3), state.extra['samples'] // 640, np.uint8)]
    def idle_frame(self, state, index):
        return SimpleNamespace(data=np.full((16, 16, 3), index, np.uint8))


@pytest.fixture
def service(tmp_path, monkeypatch):
    import cv2
    dataset = tmp_path / 'data'
    (dataset / 'full_body_img').mkdir(parents=True)
    cv2.imwrite(str(dataset / 'full_body_img/0.jpg'), np.zeros((16, 16, 3), np.uint8))
    for name in ('model.pth', 'hubert.pth'):
        (tmp_path / name).touch()
    monkeypatch.setenv('OMNIRT_FEATHERTALK_RUNTIME', '1')
    monkeypatch.setenv('OMNIRT_FEATHERTALK_DATASET', str(dataset))
    monkeypatch.setenv('OMNIRT_FEATHERTALK_CHECKPOINT', str(tmp_path / 'model.pth'))
    monkeypatch.setenv('OMNIRT_FEATHERTALK_HUBERT_CHECKPOINT', str(tmp_path / 'hubert.pth'))
    monkeypatch.setenv('OMNIRT_ALLOWED_FRAME_ROOTS', str(tmp_path))
    app = create_avatar_app()
    runtime = FeatherTalkRealtimeRuntime(device='cpu', engine_factory=Engine)
    app.state.realtime_avatar_service.runtime.feathertalk = runtime
    return app, runtime


def init(ws, **kwargs):
    ws.send_json({'type': 'init', 'ref_image': base64.b64encode(b'image').decode(), **kwargs})
    return ws.receive_json()


def test_ws_discovery_render_context_idle_close(service):
    app, runtime = service
    with TestClient(app) as client:
        assert 'feathertalk' in client.get('/v1/audio2video/models').json()['models']
        with client.websocket_connect('/v1/audio2video/feathertalk') as ws:
            response = init(ws)
            assert response['type'] == 'init_ok'
            assert response['chunk_samples'] == 640
            assert response['slice_len'] == 1
            assert (response['width'], response['height']) == (16, 16)
            for i in range(3):
                ws.send_json({'type': 'audio_context', 'reset': i == 0, 'pcm_s16le': base64.b64encode(b'\0\0' * 6400).decode()})
                assert ws.receive_json()['type'] == 'audio_context_ok'
                ws.send_bytes(MAGIC_AUDIO + b'\0\0' * 640)
                assert len(decode_jpeg_sequence(ws.receive_bytes())) == 1
            ws.send_json({'type': 'idle', 'frame_index': 3})
            assert len(decode_jpeg_sequence(ws.receive_bytes())) == 1
            ws.send_json({'type': 'close'})
            assert ws.receive_json()['type'] == 'close_ok'
        assert runtime._states == {}


def test_sessions_share_weights_but_not_audio_state(service):
    app, runtime = service
    with TestClient(app) as client:
        with client.websocket_connect('/v1/avatar/feathertalk') as first, client.websocket_connect('/v1/audio2video/feathertalk') as second:
            assert init(first)['type'] == 'init_ok'
            assert init(second)['type'] == 'init_ok'
            entries = list(runtime._states.values())
            assert entries[0][0] is entries[1][0]
            assert entries[0][1].extra is not entries[1][1].extra
            first.send_bytes(MAGIC_AUDIO + b'\0\0' * 640)
            first.receive_bytes()
            assert entries[0][1].extra['samples'] == 640
            assert 'samples' not in entries[1][1].extra


def test_rejects_request_paths_outside_allowed_roots(service):
    app, _ = service
    with TestClient(app) as client, client.websocket_connect('/v1/audio2video/feathertalk') as ws:
        response = init(ws, feathertalk_dataset='/')
        assert response['type'] == 'error'
        assert response['code'] == 'bad_feathertalk_path'


def test_disabled_runtime_does_not_render_fake_frames(monkeypatch):
    monkeypatch.delenv('OMNIRT_FEATHERTALK_RUNTIME', raising=False)
    app = create_avatar_app()
    with TestClient(app) as client:
        assert 'feathertalk' not in client.get('/v1/audio2video/models').json()['models']
        with client.websocket_connect('/v1/audio2video/feathertalk') as ws:
            assert init(ws)['type'] == 'error'


@pytest.mark.parametrize('payload', [b'AUDI' + b'\0\0' * 639, b'XXXX' + b'\0\0' * 640])
def test_rejects_malformed_audio(service, payload):
    app, _ = service
    with TestClient(app) as client, client.websocket_connect('/v1/audio2video/feathertalk') as ws:
        assert init(ws)['type'] == 'init_ok'
        ws.send_bytes(payload)
        assert ws.receive_json()['type'] == 'error'


def test_context_limit_and_disconnect_cleanup(service):
    app, runtime = service
    with TestClient(app) as client:
        with client.websocket_connect('/v1/audio2video/feathertalk') as ws:
            assert init(ws)['type'] == 'init_ok'
            ws.send_json({'type': 'audio_context', 'pcm_s16le': base64.b64encode(b'\0\0' * 6401).decode()})
            assert ws.receive_json()['type'] == 'error'
    assert runtime._states == {}
