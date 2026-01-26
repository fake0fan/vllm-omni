# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
StageCore: Core engine for a single stage in vLLM-Omni multi-stage pipeline.

This module extends vLLM's EngineCore/EngineCoreProc to support multi-stage
pipelines, reusing as much of vLLM's infrastructure as possible.

Key reused components from vLLM:
- EngineCore: Core scheduling and execution logic
- EngineCoreProc: ZMQ-based process wrapper
- EngineCoreRequest/EngineCoreOutput: Message types
- MsgpackEncoder/Decoder: Serialization
"""

import queue
import signal
import threading
import time
from contextlib import ExitStack
from typing import Any

import zmq
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    EngineCoreRequestType,
)
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.executor import Executor
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

logger = init_logger(__name__)

# Constants (reused from vLLM)
POLLING_TIMEOUT_S = 2.5


class StageCoreProc(EngineCoreProc):
    """
    ZMQ-wrapper for running StageCore in a background process.

    Extends vLLM's EngineCoreProc with:
    - Stage-specific identification
    - Integration with multi-stage pipeline
    - Custom input/output handling for stage data

    Reused from vLLM's EngineCoreProc:
    - ZMQ socket management
    - Input/output threading
    - Core busy loop structure
    - Msgpack serialization
    """

    def __init__(
        self,
        stage_id: int,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        input_address: str,
        output_address: str,
    ):
        """
        Initialize StageCoreProc.

        Unlike the parent class which does handshake, this class
        connects to pre-specified addresses.
        """
        self.stage_id = stage_id
        self.input_address = input_address
        self.output_address = output_address

        # Queues for thread communication (same as parent)
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
        self.output_queue = queue.Queue[tuple[int, EngineCoreOutputs] | bytes]()

        # Initialize the base EngineCore
        # Note: We don't call super().__init__ because EngineCoreProc's
        # __init__ does handshake which we don't need
        self.engine_index = stage_id
        self.engines_running = False

        # Create EngineCore directly
        executor_fail_callback = lambda: self.input_queue.put_nowait((EngineCoreRequestType.EXECUTOR_FAILED, b""))
        EngineCore.__init__(
            self,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
            executor_fail_callback=executor_fail_callback,
            include_finished_set=False,
        )

        self.aborts_queue = queue.Queue[list[str]]()
        self.client_count = 1

    def run_busy_loop(self):
        """
        Core busy loop - reuses vLLM's pattern.

        This is similar to EngineCoreProc.run_busy_loop() but simplified
        for single-stage operation.
        """
        while True:
            # 1) Poll the input queue until there is work to do
            self._process_input_queue()
            # 2) Step the engine core and return the outputs
            self._process_engine_step()

    def _process_input_queue(self):
        """Process input queue - same as parent."""
        waited = False
        while not self.engines_running and not self.scheduler.has_requests() and not self.batch_queue:
            if self.input_queue.empty():
                with self.aborts_queue.mutex:
                    self.aborts_queue.queue.clear()
                if logger.isEnabledFor(10):  # DEBUG level
                    logger.debug(f"[Stage-{self.stage_id}] Waiting for work")
                    waited = True
            req = self.input_queue.get()
            self._handle_client_request(*req)

        if waited:
            logger.debug(f"[Stage-{self.stage_id}] Loop active")

        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)

    def _process_engine_step(self) -> bool:
        """Step the engine - same as parent."""
        outputs, model_executed = self.step_fn()
        for output in outputs.items() if outputs else ():
            self.output_queue.put_nowait(output)
        self.post_step(model_executed)

        if not model_executed and self.scheduler.has_unfinished_requests():
            time.sleep(0.001)

        return model_executed

    def process_input_sockets(
        self,
        identity: bytes,
        ready_event: threading.Event,
    ):
        """
        Input socket IO thread.

        Simplified from parent - connects to known address.
        """
        add_request_decoder = MsgpackDecoder(EngineCoreRequest)
        generic_decoder = MsgpackDecoder()

        with ExitStack() as stack, zmq.Context() as ctx:
            input_socket = stack.enter_context(
                make_zmq_socket(
                    ctx,
                    self.input_address,
                    zmq.DEALER,
                    identity=identity,
                    bind=False,
                )
            )

            # Send ready message
            input_socket.send(b"READY")
            ready_event.set()

            poller = zmq.Poller()
            poller.register(input_socket, zmq.POLLIN)

            while True:
                for socket, _ in poller.poll():
                    type_frame, *data_frames = socket.recv_multipart(copy=False)
                    request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                    if request_type == EngineCoreRequestType.ADD:
                        req = add_request_decoder.decode(data_frames)
                        try:
                            request = self.preprocess_add_request(req)
                        except Exception:
                            self._handle_request_preproc_error(req)
                            continue
                    else:
                        request = generic_decoder.decode(data_frames)
                        if request_type == EngineCoreRequestType.ABORT:
                            self.aborts_queue.put_nowait(request)

                    self.input_queue.put_nowait((request_type, request))

    def process_output_sockets(self):
        """
        Output socket IO thread.

        Simplified from parent - connects to known address.
        """
        encoder = MsgpackEncoder()

        with ExitStack() as stack, zmq.Context() as ctx:
            output_socket = stack.enter_context(make_zmq_socket(ctx, self.output_address, zmq.PUSH, linger=4000))

            while True:
                output = self.output_queue.get()
                if output == EngineCoreProc.ENGINE_CORE_DEAD:
                    output_socket.send(output)
                    break

                assert not isinstance(output, bytes)
                client_index, outputs = output
                outputs.engine_index = self.engine_index

                buffers = encoder.encode(outputs)
                output_socket.send_multipart(buffers, copy=False)

    def _handle_request_preproc_error(self, request: EngineCoreRequest) -> None:
        """Handle preprocessing errors - same as parent."""
        from vllm.v1.engine import FinishReason

        logger.exception(
            f"[Stage-{self.stage_id}] Error preprocessing request %s",
            request.request_id,
        )
        self.output_queue.put_nowait(
            (
                request.client_index,
                EngineCoreOutputs(
                    engine_index=self.engine_index,
                    finished_requests={request.request_id},
                    outputs=[
                        EngineCoreOutput(
                            request_id=request.request_id,
                            new_token_ids=[],
                            finish_reason=FinishReason.ERROR,
                        )
                    ],
                ),
            )
        )

    @staticmethod
    def run_stage_core(
        stage_id: int,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        input_address: str,
        output_address: str,
    ):
        """
        Entry point for running StageCore in a subprocess.

        Similar to EngineCoreProc.run_engine_core but for stages.
        """
        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        stage_core = None
        try:
            logger.info(f"[Stage-{stage_id}] Starting StageCore process")

            stage_core = StageCoreProc(
                stage_id=stage_id,
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=log_stats,
                input_address=input_address,
                output_address=output_address,
            )

            # Start IO threads
            identity = stage_id.to_bytes(length=2, byteorder="little")
            ready_event = threading.Event()

            input_thread = threading.Thread(
                target=stage_core.process_input_sockets,
                args=(identity, ready_event),
                daemon=True,
            )
            input_thread.start()

            output_thread = threading.Thread(
                target=stage_core.process_output_sockets,
                daemon=True,
            )
            output_thread.start()

            # Wait for ready
            if not ready_event.wait(timeout=300):
                raise TimeoutError(f"[Stage-{stage_id}] Input thread failed to become ready")

            logger.info(f"[Stage-{stage_id}] StageCore ready, entering busy loop")

            # Run the core loop
            stage_core.run_busy_loop()

        except SystemExit:
            logger.debug(f"[Stage-{stage_id}] Exiting")
        except Exception as e:
            logger.exception(f"[Stage-{stage_id}] Fatal error: {e}")
            if stage_core is not None:
                stage_core.output_queue.put_nowait(EngineCoreProc.ENGINE_CORE_DEAD)
        finally:
            if stage_core is not None:
                stage_core.shutdown()


def launch_stage_core(
    stage_id: int,
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    base_dir: str = "/tmp/vllm_omni",
) -> tuple[Any, str, str]:
    """
    Launch a StageCore process.

    Args:
        stage_id: Unique stage identifier
        vllm_config: vLLM configuration
        executor_class: Executor class
        log_stats: Whether to log statistics
        base_dir: Base directory for IPC sockets

    Returns:
        Tuple of (process, input_address, output_address)
    """
    import os

    os.makedirs(base_dir, exist_ok=True)

    input_address = f"ipc://{base_dir}/stage_{stage_id}_input.ipc"
    output_address = f"ipc://{base_dir}/stage_{stage_id}_output.ipc"

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    process = ctx.Process(
        target=StageCoreProc.run_stage_core,
        args=(
            stage_id,
            vllm_config,
            executor_class,
            log_stats,
            input_address,
            output_address,
        ),
        daemon=True,
    )
    process.start()

    return process, input_address, output_address
