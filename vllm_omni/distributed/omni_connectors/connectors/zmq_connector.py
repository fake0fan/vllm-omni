# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ZMQ-based connector for high-performance inter-stage communication.

This connector uses ZeroMQ for efficient data transfer between stages,
providing better performance than queue-based communication for large payloads.
"""

import os
import tempfile
import threading
import time
from typing import Any

import zmq

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)


class ZMQConnector(OmniConnectorBase):
    """
    Connector that uses ZeroMQ for high-performance inter-stage communication.

    This connector creates PUSH/PULL socket pairs for each stage-to-stage edge,
    providing efficient point-to-point data transfer with automatic load balancing.

    Features:
        - Uses IPC (Unix socket) transport for best local performance
        - Falls back to TCP for cross-machine communication
        - Supports both blocking and non-blocking operations
        - Thread-safe with proper socket management

    Configuration options:
        - transport: "ipc" (default) or "tcp"
        - base_port: Starting port for TCP transport (default: 15555)
        - socket_dir: Directory for IPC sockets (default: system temp dir)
        - send_timeout: Send timeout in ms (default: 5000)
        - recv_timeout: Receive timeout in ms (default: 5000)
        - hwm: High water mark for socket buffers (default: 1000)
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config

        # ZMQ configuration
        self.transport = config.get("transport", "ipc")
        self.base_port = int(config.get("base_port", 15555))
        self.socket_dir = config.get("socket_dir", tempfile.gettempdir())
        self.send_timeout = int(config.get("send_timeout", 5000))
        self.recv_timeout = int(config.get("recv_timeout", 5000))
        self.hwm = int(config.get("hwm", 1000))

        # ZMQ context (shared across all sockets)
        self._context = zmq.Context.instance()

        # Socket management
        # Key: (from_stage, to_stage), Value: socket
        self._push_sockets: dict[tuple[str, str], zmq.Socket] = {}
        self._pull_sockets: dict[tuple[str, str], zmq.Socket] = {}
        self._socket_lock = threading.Lock()

        # Data store for async retrieval
        # Key: (from_stage, to_stage, request_id), Value: (data, size, timestamp)
        self._data_store: dict[tuple[str, str, str], tuple[Any, int, float]] = {}
        self._store_lock = threading.Lock()

        # Metrics
        self._metrics = {
            "puts": 0,
            "gets": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "send_time_ms": 0.0,
            "recv_time_ms": 0.0,
        }

        logger.info(
            f"ZMQConnector initialized: transport={self.transport}, socket_dir={self.socket_dir}, hwm={self.hwm}"
        )

    def _get_endpoint(self, from_stage: str, to_stage: str) -> str:
        """Generate endpoint address for a stage-to-stage edge."""
        if self.transport == "ipc":
            socket_name = f"omni_zmq_{from_stage}_to_{to_stage}.ipc"
            return f"ipc://{os.path.join(self.socket_dir, socket_name)}"
        else:
            # TCP transport - use port based on stage IDs
            port = self.base_port + int(from_stage) * 100 + int(to_stage)
            return f"tcp://127.0.0.1:{port}"

    def _get_push_socket(self, from_stage: str, to_stage: str) -> zmq.Socket:
        """Get or create a PUSH socket for the given edge."""
        key = (from_stage, to_stage)

        with self._socket_lock:
            if key not in self._push_sockets:
                socket = self._context.socket(zmq.PUSH)
                socket.setsockopt(zmq.SNDTIMEO, self.send_timeout)
                socket.setsockopt(zmq.SNDHWM, self.hwm)
                socket.setsockopt(zmq.LINGER, 0)

                endpoint = self._get_endpoint(from_stage, to_stage)
                socket.bind(endpoint)
                self._push_sockets[key] = socket
                logger.debug(f"Created PUSH socket: {endpoint}")

            return self._push_sockets[key]

    def _get_pull_socket(self, from_stage: str, to_stage: str) -> zmq.Socket:
        """Get or create a PULL socket for the given edge."""
        key = (from_stage, to_stage)

        with self._socket_lock:
            if key not in self._pull_sockets:
                socket = self._context.socket(zmq.PULL)
                socket.setsockopt(zmq.RCVTIMEO, self.recv_timeout)
                socket.setsockopt(zmq.RCVHWM, self.hwm)
                socket.setsockopt(zmq.LINGER, 0)

                endpoint = self._get_endpoint(from_stage, to_stage)
                socket.connect(endpoint)
                self._pull_sockets[key] = socket
                logger.debug(f"Created PULL socket: {endpoint}")

            return self._pull_sockets[key]

    def put(
        self, from_stage: str, to_stage: str, request_id: str, data: Any
    ) -> tuple[bool, int, dict[str, Any] | None]:
        """Send data via ZMQ PUSH socket.

        The data is serialized and sent through the ZMQ socket. A lightweight
        metadata dict is returned containing the request_id for tracking.

        Args:
            from_stage: Source stage identifier
            to_stage: Destination stage identifier
            request_id: Unique request identifier
            data: Python object to send

        Returns:
            tuple: (success, serialized_size, metadata)
        """
        try:
            t_start = time.time()

            # Serialize the data
            payload = self.serialize_obj(data)
            size = len(payload)

            # Create message with request_id header
            message = {
                "request_id": request_id,
                "from_stage": from_stage,
                "to_stage": to_stage,
                "payload": payload,
                "timestamp": time.time(),
            }

            # Get or create PUSH socket
            socket = self._get_push_socket(from_stage, to_stage)

            # Send serialized message
            socket.send_pyobj(message, flags=zmq.NOBLOCK)

            t_end = time.time()
            send_ms = (t_end - t_start) * 1000.0

            # Update metrics
            self._metrics["puts"] += 1
            self._metrics["bytes_sent"] += size
            self._metrics["send_time_ms"] += send_ms

            # Return metadata for tracking (lightweight - actual data sent via ZMQ)
            metadata = {
                "zmq": True,
                "request_id": request_id,
                "from_stage": from_stage,
                "to_stage": to_stage,
                "size": size,
                "send_time_ms": send_ms,
            }

            logger.debug(f"ZMQ PUT: {from_stage}->{to_stage} req={request_id} size={size} time={send_ms:.2f}ms")

            return True, size, metadata

        except zmq.Again:
            logger.warning(f"ZMQ PUT timeout: {from_stage}->{to_stage} req={request_id}")
            return False, 0, None
        except Exception as e:
            logger.error(f"ZMQ PUT failed: {from_stage}->{to_stage} req={request_id}: {e}")
            return False, 0, None

    def get(
        self, from_stage: str, to_stage: str, request_id: str, metadata: dict[str, Any] | None = None
    ) -> tuple[Any, int] | None:
        """Receive data via ZMQ PULL socket.

        This method receives data from the ZMQ socket. If the received message
        doesn't match the expected request_id, it stores the message for later
        retrieval.

        Args:
            from_stage: Source stage identifier
            to_stage: Destination stage identifier
            request_id: Expected request identifier
            metadata: Optional metadata from put operation (used for size tracking)

        Returns:
            Tuple of (Python object, serialized byte size) if found, None otherwise
        """
        # First check the data store for previously received messages
        store_key = (from_stage, to_stage, request_id)
        with self._store_lock:
            if store_key in self._data_store:
                data, size, _ = self._data_store.pop(store_key)
                self._metrics["gets"] += 1
                return data, size

        try:
            t_start = time.time()

            # Get or create PULL socket
            socket = self._get_pull_socket(from_stage, to_stage)

            # Receive with timeout - may need multiple attempts
            # if messages arrive out of order
            max_attempts = 100  # Prevent infinite loop
            for _ in range(max_attempts):
                try:
                    message = socket.recv_pyobj(flags=zmq.NOBLOCK)
                except zmq.Again:
                    # No message available, wait a bit
                    time.sleep(0.001)
                    continue

                msg_request_id = message.get("request_id")
                payload = message.get("payload")
                size = len(payload) if payload else 0

                if msg_request_id == request_id:
                    # Found our message - deserialize and return
                    data = self.deserialize_obj(payload)

                    t_end = time.time()
                    recv_ms = (t_end - t_start) * 1000.0

                    self._metrics["gets"] += 1
                    self._metrics["bytes_received"] += size
                    self._metrics["recv_time_ms"] += recv_ms

                    logger.debug(f"ZMQ GET: {from_stage}->{to_stage} req={request_id} size={size} time={recv_ms:.2f}ms")

                    return data, size
                else:
                    # Message for different request - store for later
                    store_key_other = (
                        message.get("from_stage", from_stage),
                        message.get("to_stage", to_stage),
                        msg_request_id,
                    )
                    data = self.deserialize_obj(payload)
                    with self._store_lock:
                        self._data_store[store_key_other] = (data, size, time.time())
                    logger.debug(
                        f"ZMQ GET: stored out-of-order message for req={msg_request_id} while waiting for {request_id}"
                    )

            # Timeout - no matching message found
            logger.warning(f"ZMQ GET timeout: {from_stage}->{to_stage} req={request_id}")
            return None

        except Exception as e:
            logger.error(f"ZMQ GET failed: {from_stage}->{to_stage} req={request_id}: {e}")
            return None

    def cleanup(self, request_id: str) -> None:
        """Clean up resources for a request.

        Removes any stored messages for the given request_id from the data store.
        """
        with self._store_lock:
            keys_to_remove = [key for key in self._data_store.keys() if key[2] == request_id]
            for key in keys_to_remove:
                del self._data_store[key]

        if keys_to_remove:
            logger.debug(f"ZMQ cleanup: removed {len(keys_to_remove)} stored messages for req={request_id}")

    def health(self) -> dict[str, Any]:
        """Return health status and metrics."""
        with self._store_lock:
            pending_messages = len(self._data_store)

        return {
            "status": "healthy",
            "transport": self.transport,
            "push_sockets": len(self._push_sockets),
            "pull_sockets": len(self._pull_sockets),
            "pending_messages": pending_messages,
            **self._metrics,
        }

    def close(self) -> None:
        """Close all sockets and clean up resources."""
        with self._socket_lock:
            for socket in self._push_sockets.values():
                socket.close()
            for socket in self._pull_sockets.values():
                socket.close()
            self._push_sockets.clear()
            self._pull_sockets.clear()

        # Clean up IPC socket files
        if self.transport == "ipc":
            try:
                import glob

                pattern = os.path.join(self.socket_dir, "omni_zmq_*.ipc")
                for f in glob.glob(pattern):
                    try:
                        os.unlink(f)
                    except OSError:
                        pass
            except Exception as e:
                logger.debug(f"Error cleaning up IPC socket files: {e}")

        logger.info("ZMQConnector closed")

    def __del__(self):
        """Destructor - ensure sockets are closed."""
        try:
            self.close()
        except Exception:
            pass
