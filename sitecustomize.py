import torch


def _patch_gptq_hf_device_map() -> None:
    try:
        from transformers.quantizers.quantizer_gptq import GptqHfQuantizer
    except Exception:
        return

    if getattr(GptqHfQuantizer, "_patched_hf_device_map", False):
        return

    original = GptqHfQuantizer._process_model_after_weight_loading

    def wrapped(self, model, **kwargs):
        if not hasattr(model, "hf_device_map"):
            if torch.cuda.is_available():
                model.hf_device_map = {"": 0}
            else:
                model.hf_device_map = {"": "cpu"}
        return original(self, model, **kwargs)

    GptqHfQuantizer._process_model_after_weight_loading = wrapped
    GptqHfQuantizer._patched_hf_device_map = True


_patch_gptq_hf_device_map()
