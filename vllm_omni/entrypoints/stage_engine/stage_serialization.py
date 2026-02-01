"""Msgpack serialization for vLLM-Omni stage communication.

This module provides custom msgpack encoders and decoders for serializing
vLLM-Omni types (OmniPromptType, OmniSamplingParams, OmniRequestOutput) with
support for zero-copy tensor transfer via shared memory.

Based on vLLM's serial_utils.py pattern.
"""

import io
from typing import Any, Sequence

import msgspec
import torch
from PIL import Image
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.v1.serial_utils import MsgpackDecoder as VllmMsgpackDecoder
from vllm.v1.serial_utils import MsgpackEncoder as VllmMsgpackEncoder

from vllm_omni.inputs.data import (
    OmniDiffusionSamplingParams,
    OmniEmbedsPrompt,
    OmniTextPrompt,
    OmniTokenInputs,
    OmniTokensPrompt,
)
from vllm_omni.lora.request import LoRARequest
from vllm_omni.outputs import OmniRequestOutput

# Custom type codes for vLLM-Omni types
CUSTOM_TYPE_OMNI_REQUEST_OUTPUT = 100
CUSTOM_TYPE_OMNI_DIFFUSION_SAMPLING_PARAMS = 101
CUSTOM_TYPE_OMNI_TEXT_PROMPT = 102
CUSTOM_TYPE_OMNI_TOKENS_PROMPT = 103
CUSTOM_TYPE_OMNI_TOKEN_INPUTS = 104
CUSTOM_TYPE_OMNI_EMBEDS_PROMPT = 105
CUSTOM_TYPE_PIL_IMAGE = 106
CUSTOM_TYPE_LORA_REQUEST = 107
CUSTOM_TYPE_SAMPLING_PARAMS = 108
CUSTOM_TYPE_REQUEST_OUTPUT = 109

bytestr = bytes | bytearray | memoryview


class StageMsgpackEncoder(VllmMsgpackEncoder):
    """Msgpack encoder for vLLM-Omni stage communication.

    Extends vLLM's MsgpackEncoder to handle vLLM-Omni specific types:
    - OmniPromptType variants (OmniTextPrompt, OmniTokensPrompt, etc.)
    - OmniSamplingParams (SamplingParams, OmniDiffusionSamplingParams)
    - OmniRequestOutput
    - PIL Images
    - LoRARequest

    Supports zero-copy tensor serialization via shared memory for large tensors.
    """

    def enc_hook(self, obj: Any) -> Any:
        """Custom encoding hook for vLLM-Omni types."""
        # Handle OmniRequestOutput
        if isinstance(obj, OmniRequestOutput):
            return self._encode_omni_request_output(obj)

        # Handle OmniDiffusionSamplingParams
        if isinstance(obj, OmniDiffusionSamplingParams):
            return self._encode_omni_diffusion_sampling_params(obj)

        # Handle SamplingParams (vLLM type)
        if isinstance(obj, SamplingParams):
            return self._encode_sampling_params(obj)

        # Handle RequestOutput (vLLM type)
        if isinstance(obj, RequestOutput):
            return self._encode_request_output(obj)

        # Handle OmniPromptType variants
        if isinstance(obj, dict):
            prompt_type = obj.get("type")
            if prompt_type == "text" and "prompt" in obj:
                # Could be OmniTextPrompt
                if any(k in obj for k in ["prompt_embeds", "negative_prompt_embeds", "additional_information"]):
                    return self._encode_omni_text_prompt(obj)
            elif prompt_type == "token":
                # Could be OmniTokensPrompt or OmniTokenInputs
                if "prompt_token_ids" in obj:
                    if any(k in obj for k in ["prompt_embeds", "negative_prompt_embeds", "additional_information"]):
                        if "multi_modal_data" in obj or "multi_modal_placeholders" in obj:
                            return self._encode_omni_token_inputs(obj)
                        else:
                            return self._encode_omni_tokens_prompt(obj)
            elif prompt_type == "embeds":
                # Could be OmniEmbedsPrompt
                if any(k in obj for k in ["prompt_embeds", "negative_prompt_embeds", "additional_information"]):
                    return self._encode_omni_embeds_prompt(obj)

        # Handle PIL Image
        if isinstance(obj, Image.Image):
            return self._encode_pil_image(obj)

        # Handle LoRARequest
        if isinstance(obj, LoRARequest):
            return self._encode_lora_request(obj)

        # Fall back to parent encoder
        return super().enc_hook(obj)

    def _encode_omni_request_output(self, obj: OmniRequestOutput) -> Any:
        """Encode OmniRequestOutput."""
        data = {
            "request_id": obj.request_id,
            "finished": obj.finished,
            "stage_id": obj.stage_id,
            "final_output_type": obj.final_output_type,
            "request_output": obj.request_output,
            "images": obj.images,
            "prompt": obj.prompt,
            "latents": obj.latents,
            "metrics": obj.metrics,
            "multimodal_output": obj.multimodal_output,
        }
        return msgspec.msgpack.Ext(CUSTOM_TYPE_OMNI_REQUEST_OUTPUT,
                                    self.encoder.encode(data))

    def _encode_omni_diffusion_sampling_params(self, obj: OmniDiffusionSamplingParams) -> Any:
        """Encode OmniDiffusionSamplingParams."""
        # Convert dataclass to dict, handling special fields
        data = {}
        for field_name in obj.__dataclass_fields__:
            value = getattr(obj, field_name)
            # Skip non-serializable fields
            if field_name in ["generator"]:
                continue
            data[field_name] = value

        return msgspec.msgpack.Ext(CUSTOM_TYPE_OMNI_DIFFUSION_SAMPLING_PARAMS,
                                    self.encoder.encode(data))

    def _encode_sampling_params(self, obj: SamplingParams) -> Any:
        """Encode vLLM SamplingParams."""
        # Use vLLM's built-in to_dict method if available
        if hasattr(obj, "to_dict"):
            data = obj.to_dict()
        else:
            # Fallback: manually extract fields
            data = {k: getattr(obj, k) for k in dir(obj)
                   if not k.startswith("_") and not callable(getattr(obj, k))}

        return msgspec.msgpack.Ext(CUSTOM_TYPE_SAMPLING_PARAMS,
                                    self.encoder.encode(data))

    def _encode_request_output(self, obj: RequestOutput) -> Any:
        """Encode vLLM RequestOutput."""
        # RequestOutput is complex - we'll use msgspec's default handling
        # but wrap it in an extension type for identification
        return msgspec.msgpack.Ext(CUSTOM_TYPE_REQUEST_OUTPUT,
                                    self.encoder.encode(obj))

    def _encode_omni_text_prompt(self, obj: OmniTextPrompt) -> Any:
        """Encode OmniTextPrompt (TypedDict)."""
        return msgspec.msgpack.Ext(CUSTOM_TYPE_OMNI_TEXT_PROMPT,
                                    self.encoder.encode(dict(obj)))

    def _encode_omni_tokens_prompt(self, obj: OmniTokensPrompt) -> Any:
        """Encode OmniTokensPrompt (TypedDict)."""
        return msgspec.msgpack.Ext(CUSTOM_TYPE_OMNI_TOKENS_PROMPT,
                                    self.encoder.encode(dict(obj)))

    def _encode_omni_token_inputs(self, obj: OmniTokenInputs) -> Any:
        """Encode OmniTokenInputs (TypedDict)."""
        return msgspec.msgpack.Ext(CUSTOM_TYPE_OMNI_TOKEN_INPUTS,
                                    self.encoder.encode(dict(obj)))

    def _encode_omni_embeds_prompt(self, obj: OmniEmbedsPrompt) -> Any:
        """Encode OmniEmbedsPrompt (TypedDict)."""
        return msgspec.msgpack.Ext(CUSTOM_TYPE_OMNI_EMBEDS_PROMPT,
                                    self.encoder.encode(dict(obj)))

    def _encode_pil_image(self, obj: Image.Image) -> Any:
        """Encode PIL Image to bytes."""
        buf = io.BytesIO()
        obj.save(buf, format="PNG")
        return msgspec.msgpack.Ext(CUSTOM_TYPE_PIL_IMAGE, buf.getvalue())

    def _encode_lora_request(self, obj: LoRARequest) -> Any:
        """Encode LoRARequest."""
        data = {
            "lora_name": obj.lora_name,
            "lora_int_id": obj.lora_int_id,
            "lora_path": obj.lora_path,
            "lora_local_path": getattr(obj, "lora_local_path", None),
        }
        return msgspec.msgpack.Ext(CUSTOM_TYPE_LORA_REQUEST,
                                    self.encoder.encode(data))


class StageMsgpackDecoder(VllmMsgpackDecoder):
    """Msgpack decoder for vLLM-Omni stage communication.

    Extends vLLM's MsgpackDecoder to handle vLLM-Omni specific types.
    Supports zero-copy tensor deserialization via shared memory.
    """

    def ext_hook(self, code: int, data: memoryview) -> Any:
        """Custom extension hook for vLLM-Omni types."""
        if code == CUSTOM_TYPE_OMNI_REQUEST_OUTPUT:
            return self._decode_omni_request_output(data)

        if code == CUSTOM_TYPE_OMNI_DIFFUSION_SAMPLING_PARAMS:
            return self._decode_omni_diffusion_sampling_params(data)

        if code == CUSTOM_TYPE_SAMPLING_PARAMS:
            return self._decode_sampling_params(data)

        if code == CUSTOM_TYPE_REQUEST_OUTPUT:
            return self._decode_request_output(data)

        if code == CUSTOM_TYPE_OMNI_TEXT_PROMPT:
            return self._decode_omni_text_prompt(data)

        if code == CUSTOM_TYPE_OMNI_TOKENS_PROMPT:
            return self._decode_omni_tokens_prompt(data)

        if code == CUSTOM_TYPE_OMNI_TOKEN_INPUTS:
            return self._decode_omni_token_inputs(data)

        if code == CUSTOM_TYPE_OMNI_EMBEDS_PROMPT:
            return self._decode_omni_embeds_prompt(data)

        if code == CUSTOM_TYPE_PIL_IMAGE:
            return self._decode_pil_image(data)

        if code == CUSTOM_TYPE_LORA_REQUEST:
            return self._decode_lora_request(data)

        # Fall back to parent decoder
        return super().ext_hook(code, data)

    def _decode_omni_request_output(self, data: memoryview) -> OmniRequestOutput:
        """Decode OmniRequestOutput."""
        obj_dict = self.decoder.decode(data)
        return OmniRequestOutput(
            request_id=obj_dict["request_id"],
            finished=obj_dict["finished"],
            stage_id=obj_dict["stage_id"],
            final_output_type=obj_dict["final_output_type"],
            request_output=obj_dict["request_output"],
            images=obj_dict["images"],
            prompt=obj_dict["prompt"],
            latents=obj_dict["latents"],
            metrics=obj_dict["metrics"],
            multimodal_output=obj_dict["multimodal_output"],
        )

    def _decode_omni_diffusion_sampling_params(self, data: memoryview) -> OmniDiffusionSamplingParams:
        """Decode OmniDiffusionSamplingParams."""
        obj_dict = self.decoder.decode(data)
        return OmniDiffusionSamplingParams(**obj_dict)

    def _decode_sampling_params(self, data: memoryview) -> SamplingParams:
        """Decode vLLM SamplingParams."""
        obj_dict = self.decoder.decode(data)
        # Use vLLM's from_dict if available
        if hasattr(SamplingParams, "from_dict"):
            return SamplingParams.from_dict(obj_dict)
        else:
            return SamplingParams(**obj_dict)

    def _decode_request_output(self, data: memoryview) -> RequestOutput:
        """Decode vLLM RequestOutput."""
        return self.decoder.decode(data)

    def _decode_omni_text_prompt(self, data: memoryview) -> OmniTextPrompt:
        """Decode OmniTextPrompt."""
        return OmniTextPrompt(self.decoder.decode(data))

    def _decode_omni_tokens_prompt(self, data: memoryview) -> OmniTokensPrompt:
        """Decode OmniTokensPrompt."""
        return OmniTokensPrompt(self.decoder.decode(data))

    def _decode_omni_token_inputs(self, data: memoryview) -> OmniTokenInputs:
        """Decode OmniTokenInputs."""
        return OmniTokenInputs(self.decoder.decode(data))

    def _decode_omni_embeds_prompt(self, data: memoryview) -> OmniEmbedsPrompt:
        """Decode OmniEmbedsPrompt."""
        return OmniEmbedsPrompt(self.decoder.decode(data))

    def _decode_pil_image(self, data: memoryview) -> Image.Image:
        """Decode PIL Image from bytes."""
        buf = io.BytesIO(bytes(data))
        return Image.open(buf)

    def _decode_lora_request(self, data: memoryview) -> LoRARequest:
        """Decode LoRARequest."""
        obj_dict = self.decoder.decode(data)
        return LoRARequest(
            lora_name=obj_dict["lora_name"],
            lora_int_id=obj_dict["lora_int_id"],
            lora_path=obj_dict["lora_path"],
            lora_local_path=obj_dict.get("lora_local_path"),
        )
