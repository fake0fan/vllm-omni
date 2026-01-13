# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import multiprocessing as mp
import os
import time
import uuid
import weakref
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pprint import pformat
from typing import Any

from omegaconf import OmegaConf
from tqdm.auto import tqdm
from vllm.inputs import PromptType
from vllm.logger import init_logger

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.distributed.omni_connectors import (
    get_stage_connector_config,
    initialize_orchestrator_connectors,
)
from vllm_omni.distributed.omni_connectors.adapter import try_send_via_connector
from vllm_omni.distributed.ray_utils.utils import (
    create_placement_group,
    get_ray_queue_class,
    try_close_ray,
)
from vllm_omni.entrypoints.log_utils import OrchestratorMetrics
from vllm_omni.entrypoints.omni_stage import OmniStage
from vllm_omni.entrypoints.stage_utils import SHUTDOWN_TASK, OmniStageTaskType
from vllm_omni.entrypoints.stage_utils import maybe_load_from_ipc as _load
from vllm_omni.entrypoints.utils import (
    get_final_stage_id_for_e2e,
    load_stage_configs_from_model,
    load_stage_configs_from_yaml,
    resolve_model_config_path,
)
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


def _weak_close_cleanup(stage_list, stage_in_queues, ray_pg):
    """Weak reference cleanup function for OmniBase instances."""
    if stage_list:
        for q in stage_in_queues:
            try:
                q.put_nowait(SHUTDOWN_TASK)
            except Exception as e:
                logger.warning(f"Failed to send shutdown signal to stage input queue: {e}")
        for stage in stage_list:
            try:
                stage.stop_stage_worker()
            except Exception as e:
                logger.warning(f"Failed to stop stage worker: {e}")
    try_close_ray(ray_pg)


def _dummy_snapshot_download(model_id):
    return model_id


def omni_snapshot_download(model_id) -> str:
    # TODO: this is just a workaround for quickly use modelscope, we should support
    # modelscope in weight loading feature instead of using `snapshot_download`
    if os.environ.get("VLLM_USE_MODELSCOPE", False):
        from modelscope.hub.snapshot_download import snapshot_download

        return snapshot_download(model_id)
    else:
        return _dummy_snapshot_download(model_id)


class OmniBase:
    """Base class for serving Omni models.

    Args:
        model: Model name or path to load.

        Stage Management:
            stage_configs_path: Optional path to YAML file containing stage
                configurations. If None, configurations are loaded from the model.
            log_stats: Whether to enable statistics logging.
            stage_init_timeout: Per-stage init watchdog (seconds).
            init_timeout: Timeout in seconds for waiting for all stages to initialize.
            batch_timeout: Timeout in seconds for batching requests within a stage.

        Distributed/IPC:
            worker_backend: Backend for worker processes. Default is "multi_process".
            ray_address: Address of Ray cluster for Ray backend.
            shm_threshold_bytes: Threshold in bytes for using shared memory for IPC.

        LLM:
            tokenizer: Optional tokenizer name or path. If None, uses model's tokenizer.

        Diffusion:
            vae_use_slicing: Enable VAE slicing for memory optimization.
            vae_use_tiling: Enable VAE tiling for memory optimization.
            cache_backend: Cache backend type ("none", "cache_dit", "tea_cache").
            cache_config: Cache configuration dictionary.
            parallel_config: Diffusion parallel configuration.
            enforce_eager: Force eager execution mode.
            boundary_ratio: MoE boundary ratio for Wan2.2.
            flow_shift: Scheduler flow_shift for Wan2.2.

        **kwargs: Additional keyword arguments passed to stage engines.
    """

    def __init__(
        self,
        model: str,
        *,
        # === Stage Management ===
        stage_configs_path: str | None = None,
        log_stats: bool = False,
        stage_init_timeout: int = 20,
        init_timeout: int = 300,
        batch_timeout: int = 10,
        # === Distributed/IPC ===
        worker_backend: str = "multi_process",
        ray_address: str | None = None,
        shm_threshold_bytes: int = 65536,
        # === LLM ===
        tokenizer: str | None = None,
        # === Diffusion ===
        vae_use_slicing: bool = False,
        vae_use_tiling: bool = False,
        cache_backend: str | None = None,
        cache_config: dict[str, Any] | None = None,
        parallel_config: Any | None = None,
        enforce_eager: bool = False,
        boundary_ratio: float | None = None,
        flow_shift: float | None = None,
        # === Additional ===
        **kwargs: dict[str, Any]
    ) -> None:
        model = omni_snapshot_download(model)

        # Stage management attributes
        self.stage_list: list[OmniStage] = []
        self._stage_in_queues: list[mp.Queue] = []
        self._stage_out_queues: list[mp.Queue] = []
        self._stages_ready: set[int] = set()
        self._ray_pg = None
        self._queue_cls = None
        self._ctx = None

        # Initialize stages - each stage will create appropriate instance based on stage_type
        # Stage workers will automatically create OmniLLM or OmniDiffusion instances
        # based on stage_type in YAML config (handled in omni_stage.py)
        logger.info(f"Initializing stages for model: {model}")

        # Merge explicit diffusion params into kwargs for stage configuration
        diffusion_params = {
            "vae_use_slicing": vae_use_slicing,
            "vae_use_tiling": vae_use_tiling,
            "cache_backend": cache_backend,
            "cache_config": cache_config,
            "parallel_config": parallel_config,
            "enforce_eager": enforce_eager,
            "boundary_ratio": boundary_ratio,
            "flow_shift": flow_shift,
        }
        # Only include non-None values to allow kwargs to override defaults
        diffusion_kwargs = {k: v for k, v in diffusion_params.items() if v is not None}
        engine_kwargs = {**diffusion_kwargs, **kwargs}

        # Base engine args for LLM stages
        base_engine_args = {"tokenizer": tokenizer} if tokenizer is not None else None

        # Load stage configurations from YAML
        if stage_configs_path is None:
            self.config_path = resolve_model_config_path(model)
            self.stage_configs = load_stage_configs_from_model(model, base_engine_args=base_engine_args)
            if not self.stage_configs:
                default_stage_cfg = self._create_default_diffusion_stage_cfg(engine_kwargs)
                self.stage_configs = OmegaConf.create(default_stage_cfg)
        else:
            self.config_path = stage_configs_path
            self.stage_configs = load_stage_configs_from_yaml(stage_configs_path, base_engine_args=base_engine_args)

        # Initialize connectors
        self.omni_transfer_config, self.connectors = initialize_orchestrator_connectors(
            self.config_path, worker_backend=worker_backend, shm_threshold_bytes=shm_threshold_bytes
        )

        # Initialize stats paths
        self._enable_stats: bool = bool(log_stats)
        self.worker_backend = worker_backend
        self.ray_address = ray_address
        self.batch_timeout = batch_timeout

        # Build OmniStage instances in parallel, preserve original order
        def _build_stage(idx_cfg: tuple[int, Any]) -> tuple[int, OmniStage]:
            idx, cfg = idx_cfg
            return idx, OmniStage(cfg, stage_init_timeout=stage_init_timeout)

        with ThreadPoolExecutor(max_workers=min(len(self.stage_configs), max(1, os.cpu_count() or 1))) as executor:
            futures = [executor.submit(_build_stage, (idx, cfg)) for idx, cfg in enumerate(self.stage_configs)]
            results: list[tuple[int, OmniStage]] = []
            for fut in as_completed(futures):
                results.append(fut.result())
        results.sort(key=lambda x: x[0])
        self.stage_list = [st for _, st in results]
        self.default_sampling_params_list = [st.default_sampling_params for st in self.stage_list]
        self.output_modalities = [st.final_output_type for st in self.stage_list]
        logger.debug(f"[{self._name}] Loaded {len(self.stage_list)} stages")

        if self.worker_backend == "ray":
            self._queue_cls = get_ray_queue_class()
        else:
            self._ctx = mp.get_context("spawn")
            self._queue_cls = lambda: self._ctx.Queue(maxsize=0)

        self._shm_threshold_bytes = max(0, int(shm_threshold_bytes))
        self._start_stages(model)
        # Wait for all stages to report readiness before seeding
        self._wait_for_stages_ready(timeout=init_timeout)

    def _get_default_cache_config(self, cache_backend: str | None) -> dict[str, Any] | None:
        if cache_backend == "cache_dit":
            return {
                "Fn_compute_blocks": 1,
                "Bn_compute_blocks": 0,
                "max_warmup_steps": 4,
                "residual_diff_threshold": 0.24,
                "max_continuous_cached_steps": 3,
                "enable_taylorseer": False,
                "taylorseer_order": 1,
                "scm_steps_mask_policy": None,
                "scm_steps_policy": "dynamic",
            }
        if cache_backend == "tea_cache":
            return {
                "rel_l1_thresh": 0.2,
            }
        return None

    def _normalize_cache_config(self, cache_backend: str | None, cache_config: Any | None) -> Any | None:
        if isinstance(cache_config, str):
            try:
                cache_config = json.loads(cache_config)
            except json.JSONDecodeError:
                logger.warning("Invalid cache_config JSON, using defaults.")
                cache_config = None
        if cache_config is None and cache_backend not in (None, "", "none"):
            cache_config = self._get_default_cache_config(cache_backend)
        return cache_config

    def _create_default_diffusion_stage_cfg(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Create default diffusion stage configuration."""
        # We temporally create a default config for diffusion stage.
        # In the future, we should merge the default config with the user-provided config.
        # TODO: hack, convert dtype to string to avoid non-premitive omegaconf create error.
        if "dtype" in kwargs:
            kwargs["dtype"] = str(kwargs["dtype"])
        cache_backend = kwargs.get("cache_backend", "none")
        cache_config = self._normalize_cache_config(cache_backend, kwargs.get("cache_config", None))
        # TODO: hack, calculate devices based on parallel config.
        devices = "0"
        if "parallel_config" in kwargs:
            num_devices = kwargs["parallel_config"].world_size
            for i in range(1, num_devices):
                devices += f",{i}"
        default_stage_cfg = [
            {
                "stage_id": 0,
                "stage_type": "diffusion",
                "runtime": {
                    "process": True,
                    "devices": devices,
                    "max_batch_size": 1,
                },
                "engine_args": OmegaConf.create(
                    {
                        **kwargs,
                        "cache_backend": cache_backend,
                        "cache_config": cache_config,
                    }
                ),
                "final_output": True,
                "final_output_type": "image",
            }
        ]
        default_stage_cfg[0]["engine_args"]["model_stage"] = "diffusion"
        return default_stage_cfg

    def _start_stages(self, model: str) -> None:
        """Start all stage processes."""
        if self.worker_backend == "ray":
            # Initialize Ray Cluster
            self._ray_pg = create_placement_group(
                number_of_stages=len(self.stage_list), address=self.ray_address, strategy="PACK"
            )

        for stage_id, stage in enumerate[OmniStage](self.stage_list):
            in_q = self._queue_cls()
            out_q = self._queue_cls()
            self._stage_in_queues.append(in_q)
            self._stage_out_queues.append(out_q)
            stage.attach_queues(in_q, out_q)

            stage_connectors_config = get_stage_connector_config(
                self.omni_transfer_config,
                stage_id,
            )

            stage.init_stage_worker(
                model,
                is_async=self.is_async,
                shm_threshold_bytes=self._shm_threshold_bytes,
                ctx=self._ctx if self.worker_backend != "ray" else None,
                batch_timeout=self.batch_timeout,
                connectors_config=stage_connectors_config,
                worker_backend=self.worker_backend,
                ray_placement_group=self._ray_pg,
            )

            logger.debug(f"[{self._name}] Stage-{stage_id} process started")

    def _process_stage_ready(self, stage: OmniStage, stage_id: int, result: dict[str, Any]) -> None:
        self._stages_ready.add(stage_id)
        logger.info(f"[{self._name}] Stage-{stage_id} reported ready")

    def _wait_for_stages_ready(self, timeout: int = 120) -> None:
        """Wait for all stages to report readiness with optimized polling."""
        num_stages = len(self.stage_list)
        deadline = time.time() + max(0, int(timeout))

        logger.info(f"[{self._name}] Waiting for {num_stages} stages to initialize (timeout: {timeout}s)")

        while len(self._stages_ready) < num_stages and time.time() < deadline:
            progressed = False
            for stage_id, stage in enumerate(self.stage_list):
                if stage_id in self._stages_ready:
                    continue

                # Check if the stage has reported status
                if result := stage.try_collect():
                    progressed = True
                    if result.get("type") == "stage_ready":
                        self._process_stage_ready(stage, stage_id, result)

            if not progressed:
                time.sleep(0.05)

        # Handle Final State
        if len(self._stages_ready) == num_stages:
            logger.info(f"[{self._name}] All stages initialized successfully")
            return

        # Handle Timeout/Failure
        not_ready = sorted(set(range(num_stages)) - set(self._stages_ready))
        logger.warning(
            f"[{self._name}] Initialization timeout: {len(self._stages_ready)}/{num_stages} "
            f"stages ready. Missing stages: {not_ready}"
        )

        suggestions = [
            "Verify GPU/device assignment in config (runtime.devices) is correct.",
            "Check GPU/host memory availability; reduce model or batch size if needed.",
            "Check model weights path and network reachability (if loading remotely).",
            "Increase initialization wait time (stage_init_timeout or call-site timeout).",
        ]

        formatted_suggestions = "\n".join(f"  {i + 1}) {msg}" for i, msg in enumerate(suggestions))

        logger.error(f"[{self._name}] Stage initialization failed. Troubleshooting Steps:\n{formatted_suggestions}")

    def start_profile(self, stages: list[int] | None = None) -> None:
        """Start profiling for specified stages.

        Sends start_profile command to stage workers. Profiling must be enabled
        via VLLM_TORCH_PROFILER_DIR environment variable.

        Args:
            stages: List of stage IDs to start profiling. If None, starts
                profiling for all stages that have profiling enabled.

        Example:
            >>> # Profile all stages
            >>> omni.start_profile()
            >>> outputs = omni.generate(prompts, sampling_params)
            >>> omni.stop_profile()

            >>> # Profile only stage 0 and 2
            >>> omni.start_profile(stages=[0, 2])
        """
        if stages is None:
            stages = list(range(len(self.stage_list)))

        for stage_id in stages:
            if stage_id < len(self.stage_list):
                try:
                    self.stage_list[stage_id].submit({"type": OmniStageTaskType.PROFILER_START})
                    logger.info("[%s] Sent start_profile to stage-%s", self._name, stage_id)
                except Exception as e:
                    logger.warning(
                        "[%s] Failed to send start_profile to stage-%s: %s",
                        self._name,
                        stage_id,
                        e,
                    )

    def stop_profile(self, stages: list[int] | None = None) -> None:
        """Stop profiling for specified stages.

        Sends stop_profile command to stage workers to finalize and save traces.

        Args:
            stages: List of stage IDs to stop profiling. If None, stops
                profiling for all stages.

        Example:
            >>> omni.start_profile()
            >>> outputs = omni.generate(prompts, sampling_params)
            >>> omni.stop_profile()
            >>> # Traces saved to VLLM_TORCH_PROFILER_DIR/stage_X_<model_stage>/
        """
        if stages is None:
            stages = list(range(len(self.stage_list)))

        for stage_id in stages:
            if stage_id < len(self.stage_list):
                try:
                    self.stage_list[stage_id].submit({"type": OmniStageTaskType.PROFILER_STOP})
                    logger.info("[%s] Sent stop_profile to stage-%s", self._name, stage_id)
                except Exception as e:
                    logger.warning(
                        "[%s] Failed to send stop_profile to stage-%s: %s",
                        self._name,
                        stage_id,
                        e,
                    )

    def _update_metrics(
        self,
        metrics: OrchestratorMetrics,
        result: dict[str, Any],
        stage_id: int,
        request_id: str,
        pbar: Any,
    ) -> None:
        """Update metrics from stage result and refresh progress bar."""
        try:
            stage_metrics = result.get("metrics")
            if stage_metrics is None:
                return
            if not isinstance(stage_metrics, dict):
                stage_metrics = asdict(stage_metrics)
            metrics.on_stage_metrics(stage_id, request_id, stage_metrics)

            if pbar:
                elapsed = pbar.format_dict["elapsed"] or 1e-6
                total_out = sum(metrics.stage_total_tokens)
                speed = total_out / elapsed
                unit = "img" if self.output_modalities[stage_id] == "image" else "tok"
                avg_latency = metrics.e2e_total_ms / metrics.e2e_count if metrics.e2e_count > 0 else 0
                pbar.postfix = f"stage-{stage_id} {speed:.1f} {unit}/s, avg latency: {avg_latency:.1f}ms"
        except Exception as e:
            logger.exception(f"[{self._name}] Metrics error for {request_id} at stage-{stage_id}: {e}")

    def _forward_to_next_stage(
        self,
        request_id: str,
        current_stage_id: int,
        next_stage_id: int,
        sampling_params_list: list[Any],
        request_id_to_prompt: dict[str, Any],
        metrics: OrchestratorMetrics,
    ) -> None:
        """Forward request output to the next stage via connector."""
        next_stage = self.stage_list[next_stage_id]
        try:
            next_inputs = next_stage.process_engine_inputs(
                self.stage_list, [request_id_to_prompt[request_id]]
            )
        except Exception as e:
            logger.exception(f"[{self._name}] Failed to process inputs for stage-{next_stage_id}: {e}")
            raise

        connector_key = (str(current_stage_id), str(next_stage_id))
        connector = self.connectors.get(connector_key)
        if not connector:
            raise RuntimeError(
                f"No connector configured for edge {current_stage_id} -> {next_stage_id}"
            )

        success = try_send_via_connector(
            connector=connector,
            stage_id=current_stage_id,
            next_stage_id=next_stage_id,
            req_id=request_id,
            next_inputs=next_inputs,
            sampling_params=sampling_params_list[next_stage_id],
            original_prompt=request_id_to_prompt[request_id],
            next_stage_queue_submit_fn=next_stage.submit,
            metrics=metrics,
        )
        if not success:
            raise RuntimeError(f"Failed to send {request_id} to stage-{next_stage_id}")
        logger.debug(f"[{self._name}] Forwarded {request_id} to stage-{next_stage_id}")

    def close(self) -> None:
        """Close all stage processes and clean up resources."""
        if hasattr(self, "_weak_finalizer"):
            self._weak_finalizer()

    @property
    def _name(self) -> str:
        return "OmniBase"

    @property
    def is_async(self) -> bool:
        return False


class Omni(OmniBase):
    """Unified entrypoint for both LLM and Diffusion models for better usability.

    See OmniBase for full parameter documentation.

    Example:
        >>> # LLM model
        >>> omni = Omni(model="Qwen/Qwen2.5-Omni-7B")
        >>> outputs = omni.generate(prompts="Hello, world!")

        >>> # Diffusion model with cache acceleration
        >>> omni = Omni(
        ...     model="Qwen/Qwen-Image",
        ...     cache_backend="cache_dit",
        ...     vae_use_slicing=True,
        ... )
        >>> outputs = omni.generate(prompts="a cat sitting on a couch")
    """

    def __init__(
        self,
        model: str,
        *,
        # === Stage Management ===
        stage_configs_path: str | None = None,
        log_stats: bool = False,
        stage_init_timeout: int = 20,
        init_timeout: int = 300,
        batch_timeout: int = 10,
        # === Distributed/IPC ===
        worker_backend: str = "multi_process",
        ray_address: str | None = None,
        shm_threshold_bytes: int = 65536,
        # === LLM ===
        tokenizer: str | None = None,
        # === Diffusion ===
        vae_use_slicing: bool = False,
        vae_use_tiling: bool = False,
        cache_backend: str | None = None,
        cache_config: dict[str, Any] | None = None,
        parallel_config: Any | None = None,
        enforce_eager: bool = False,
        boundary_ratio: float | None = None,
        flow_shift: float | None = None,
        # === Additional ===
        **kwargs: dict[str, Any]
    ) -> None:
        super().__init__(
            model=model,
            stage_configs_path=stage_configs_path,
            log_stats=log_stats,
            stage_init_timeout=stage_init_timeout,
            init_timeout=init_timeout,
            batch_timeout=batch_timeout,
            worker_backend=worker_backend,
            ray_address=ray_address,
            shm_threshold_bytes=shm_threshold_bytes,
            tokenizer=tokenizer,
            vae_use_slicing=vae_use_slicing,
            vae_use_tiling=vae_use_tiling,
            cache_backend=cache_backend,
            cache_config=cache_config,
            parallel_config=parallel_config,
            enforce_eager=enforce_eager,
            boundary_ratio=boundary_ratio,
            flow_shift=flow_shift,
            **kwargs
        )

        # Register weak reference cleanup (called on garbage collection)
        self._weak_finalizer = weakref.finalize(
            self,
            _weak_close_cleanup,
            self.stage_list,
            self._stage_in_queues,
            self._ray_pg,
        )

    def generate(
        self,
        prompts: str | list[str] | None = None,
        sampling_params_list: list[Any] | None = None,
        *,
        py_generator: bool = False,
        **kwargs: dict[str, Any]
    ) -> Generator[OmniRequestOutput, None, None] | list[OmniRequestOutput]:
        """Generate outputs for the given prompts.

        Orchestrates the multi-stage pipeline based on YAML configuration.
        Each stage will use OmniLLM or OmniDiffusion based on stage_type.

        Args:
            prompts: Input prompts for generation. Can be a single string or list of strings.
            sampling_params_list: Optional list of per-stage parameters. If None, default
                parameters will be used for all stages.
            py_generator: Whether to return a Python generator for streaming results.
            **kwargs: Additional keyword arguments passed to stage engines. Also supports
                'prompt' as an alternative to 'prompts' for backward compatibility.

        Returns:
            List of OmniRequestOutput objects (or generator if py_generator=True), one for 
            each input prompt. Each output contains the stage_id, final_output_type, and
            the request_output from the final stage.

        Raises:
            ValueError: If prompts is None or sampling_params_list has incorrect length.
        """
        # Handle backward compatibility with 'prompt' keyword
        if prompts is None:
            prompts = kwargs.get("prompt")
            if prompts is None:
                raise ValueError("prompts is required for generation")

        if sampling_params_list is None:
            # For Omni LLM, the params are parsed via the yaml file. For the current version,
            # diffusion params can parsed via the command line.
            omni_params_kwargs = {
                k: v for k, v in kwargs.items() if k not in ["prompt", "request_id", "output_modalities"]
            }

            per_stage_params: list[Any] = []
            for stage_id, stage in enumerate(self.stage_list):
                stage_type = getattr(stage, "stage_type", "llm")
                if stage_type == "diffusion":
                    default_dict = self.default_sampling_params_list[stage_id]
                    # Merge user-provided kwargs
                    merged = {**default_dict, **omni_params_kwargs}
                    # Diffusion only needs to keep diff params, will be used via OmniDiffusionRequest
                    per_stage_params.append(merged)
                else:
                    # LLM directly constructs SamplingParams, don't use the merged params
                    per_stage_params.append(self.default_sampling_params_list[stage_id])

            sampling_params_list = per_stage_params
        try:
            if py_generator:
                return self._run_generation_with_generator(prompts, sampling_params_list)
            else:
                outputs = list(self._run_generation(prompts, sampling_params_list))
                self.close()
                return outputs
        except Exception as e:
            logger.exception("[Orchestrator] Failed to run generation: %s", e)
            # Always close on exception to ensure cleanup
            self.close()
            raise e

    def _run_generation_with_generator(
        self,
        prompts: PromptType | Sequence[PromptType] | OmniDiffusionRequest | Sequence[OmniDiffusionRequest],
        sampling_params_list: Any | Sequence[Any] | None,
    ) -> Generator[OmniRequestOutput, None, None]:
        """Run generation through all stages in the pipeline and return a generator."""
        gen = self._run_generation(prompts, sampling_params_list)
        try:
            yield from gen
        except Exception as e:
            logger.exception("[Orchestrator] Failed to run generation: %s", e)
            raise e
        finally:
            # Cleanup when generator is exhausted or closed
            self.close()

    def _run_generation(
        self,
        prompts: PromptType | Sequence[PromptType] | OmniDiffusionRequest | Sequence[OmniDiffusionRequest],
        sampling_params_list: Any | Sequence[Any] | None = None,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> Generator[OmniRequestOutput, None, None]:
        """Run generation through all stages in the pipeline."""
        logger.debug(f"[{self._name}] generate() called")
        assert sampling_params_list is not None, "sampling_params_list is required for pipelined generation"

        # Normalize sampling_params_list to a list
        if not isinstance(sampling_params_list, (list, tuple)):
            sampling_params_list = [sampling_params_list]

        assert len(sampling_params_list) == len(self.stage_list), \
            f"Expected {len(self.stage_list)} sampling params, got {len(sampling_params_list)}"

        # Normalize prompts to a list for per-request iteration
        request_prompts = [prompts] if not isinstance(prompts, (list, tuple)) else prompts

        # Orchestrator keeps stage objects for input derivation
        num_stages = len(self.stage_list)

        # Generate globally unique request IDs and map them to original prompts
        request_ids: list[str] = [f"{i}_{uuid.uuid4()}" for i in range(len(request_prompts))]
        request_id_to_prompt: dict[str, PromptType] = {rid: p for rid, p in zip(request_ids, request_prompts)}

        # Determine the final stage for E2E stats (highest stage_id with final_output=True; fallback to last stage)
        final_stage_id_to_prompt: dict[str, int] = {}
        for rid, prompt in request_id_to_prompt.items():
            if isinstance(prompt, dict):
                prompt_modalities = prompt.get("modalities", None)
            else:
                prompt_modalities = None
            final_stage_id_for_e2e = get_final_stage_id_for_e2e(
                prompt_modalities, self.output_modalities, self.stage_list
            )
            final_stage_id_to_prompt[rid] = final_stage_id_for_e2e

        # Metrics/aggregation helper (manages request timing internally)
        metrics = OrchestratorMetrics(num_stages, self._enable_stats)

        # Submit all requests to stage-0
        logger.debug(f"[{self._name}] Submitting {len(request_prompts)} requests to stage-0")
        stage0_params = sampling_params_list[0]
        for request_id, prompt in request_id_to_prompt.items():
            self.stage_list[0].submit({
                "request_id": request_id,
                "engine_inputs": prompt,
                "sampling_params": stage0_params,
            })
            metrics.on_request_submit(request_id)
            logger.debug(f"[{self._name}] Submitted request {request_id} to stage-0")

        pbar = None
        if use_tqdm:
            tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
            pbar = tqdm_func(
                total=len(request_prompts),
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} unit/s, output: {0:.2f} unit/s"),
            )
        # Pipeline scheduling loop: poll stages, forward results, collect finals
        completed = 0
        total = len(request_prompts)
        logger.debug(f"[{self._name}] Starting pipeline: {total} requests, {num_stages} stages")

        while completed < total:
            made_progress = False

            for stage_id, stage in enumerate(self.stage_list):
                result = stage.try_collect()
                if result is None:
                    continue

                made_progress = True
                request_id = result.get("request_id")

                # Handle errors
                if "error" in result:
                    logger.error(f"[{self._name}] Stage-{stage_id} error: {result['error']}")
                    continue

                # Handle stage initialization signal
                if result.get("type") == "stage_ready":
                    time.sleep(0.05)
                    continue

                # Process stage output
                engine_outputs = _load(result, obj_key="engine_outputs", shm_key="engine_outputs_shm")
                metrics.stage_last_ts[stage_id] = max(metrics.stage_last_ts[stage_id] or 0.0, time.time())
                stage.set_engine_outputs(engine_outputs)

                # Update metrics
                self._update_metrics(metrics, result, stage_id, request_id, pbar)

                # Yield final output if this is a final stage
                final_stage_for_request = final_stage_id_to_prompt[request_id]
                if getattr(stage, "final_output", False):
                    if stage_id == final_stage_for_request and str(request_id) not in metrics.e2e_done:
                        metrics.on_finalize_request(stage_id, request_id)
                    yield OmniRequestOutput(
                        stage_id=stage_id,
                        final_output_type=stage.final_output_type,
                        request_output=engine_outputs,
                    )

                # Forward to next stage or mark completed
                next_stage_id = stage_id + 1
                if next_stage_id <= final_stage_for_request:
                    self._forward_to_next_stage(
                        request_id, stage_id, next_stage_id,
                        sampling_params_list, request_id_to_prompt, metrics
                    )
                else:
                    completed += 1
                    if pbar:
                        pbar.unit = "img" if self.output_modalities[final_stage_for_request] == "image" else "req"
                        pbar.update(1)
                    logger.debug(f"[{self._name}] Completed {request_id} ({completed}/{total})")

            if not made_progress:
                time.sleep(0.005)

        logger.debug(f"[{self._name}] All {total} requests completed")

        if pbar:
            pbar.close()

        # Summarize and print stats
        try:
            summary = metrics.build_and_log_summary(final_stage_id_to_prompt)
            logger.info("[Summary] %s", pformat(summary, sort_dicts=False))
        except Exception as e:
            logger.exception(f"[{self._name}] Failed to build/log summary: {e}")

    @property
    def _name(self) -> str:
        return "Orchestrator"
