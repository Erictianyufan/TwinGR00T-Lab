"""Small numeric-only client for GR00T's local ZeroMQ policy server."""

from __future__ import annotations

from typing import Any

import numpy as np


def _encode_numeric(value: Any) -> Any:
    import msgpack_numpy

    if isinstance(value, np.ndarray) and value.dtype.kind == "O":
        raise TypeError("Refusing to encode object-dtype arrays")
    return msgpack_numpy.encode(value)


def _decode_numeric(value: Any) -> Any:
    import msgpack_numpy

    if isinstance(value, dict):
        nd_marker = value.get(b"nd", value.get("nd"))
        kind = value.get(b"kind", value.get("kind"))
        if nd_marker and kind in (b"O", "O"):
            raise ValueError("Refusing to decode an object-dtype array payload")
    return msgpack_numpy.decode(value)


class PolicyTransportClient:
    """Use GR00T's msgpack-numpy wire protocol without importing its model stack."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout_ms: int = 120_000):
        try:
            import msgpack
            import msgpack_numpy
            import zmq
        except ImportError as exc:  # pragma: no cover - packaging guard
            raise RuntimeError("Install the bridge extra: pip install -e '.[bridge]'") from exc
        self._msgpack_numpy = msgpack_numpy
        self._msgpack = msgpack
        self._zmq = zmq
        self._context = zmq.Context()
        self._host = host
        self._port = port
        self._timeout_ms = timeout_ms
        self._closed = False
        self._connect()

    def _connect(self) -> None:
        self._socket = self._context.socket(self._zmq.REQ)
        self._socket.setsockopt(self._zmq.RCVTIMEO, self._timeout_ms)
        self._socket.setsockopt(self._zmq.SNDTIMEO, self._timeout_ms)
        self._socket.connect(f"tcp://{self._host}:{self._port}")

    def _reset_socket(self) -> None:
        self._socket.close(linger=0)
        self._connect()

    def call(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        try:
            self._socket.send(self._msgpack.packb(request, default=_encode_numeric, use_bin_type=True))
            message = self._socket.recv()
        except self._zmq.error.Again:
            self._reset_socket()
            raise TimeoutError(f"GR00T policy endpoint timed out: {endpoint}") from None
        if message == b"ERROR":
            raise RuntimeError("GR00T policy server returned an unstructured error")
        response = self._msgpack.unpackb(message, object_hook=_decode_numeric, raw=False)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T policy server error: {response['error']}")
        return response

    def ping(self) -> bool:
        try:
            response = self.call("ping")
        except TimeoutError:
            return False
        return isinstance(response, dict) and response.get("status") == "ok"

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.call("reset", {"options": options})

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self.call("get_action", {"observation": observation, "options": options})
        if not isinstance(response, list | tuple) or len(response) != 2:
            raise RuntimeError(f"Unexpected get_action response: {type(response).__name__}")
        return response[0], response[1]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._socket.close(linger=0)
        self._context.term()

    def __enter__(self) -> PolicyTransportClient:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
