import torch
from comfy.model_detection import count_blocks
from comfy.utils import flux_to_diffusers

def convert_diffusers_flux_lora(state_dict, output_prefix=""):
    out_sd = {}

    state_dict = {key.replace('transformer.', ''): value for key, value in state_dict.items()}

    new_sd = {}
    for lora_a in state_dict.keys():
        if "lora_A" not in lora_a:
            continue
        lora_b = lora_a.replace("lora_A", "lora_B")
        key = lora_a.replace("lora_A.", "")
        assert "lora_A" in lora_a and "lora_B" in lora_b, f"Invalid LoRA checkpoint. {lora_a} {lora_b}"

        new_sd[key] = state_dict[lora_b] @ state_dict[lora_a]

    state_dict = new_sd

    depth = count_blocks(state_dict, 'transformer_blocks.{}.')
    depth_single_blocks = count_blocks(state_dict, 'single_transformer_blocks.{}.')
    hidden_size = state_dict["x_embedder.weight"].shape[0]
    sd_map = flux_to_diffusers({"depth": depth, "depth_single_blocks": depth_single_blocks, "hidden_size": hidden_size}, output_prefix=output_prefix)

    for k in sd_map:
        weight = state_dict.get(k, None)
        if weight is not None:
            t = sd_map[k]

            if not isinstance(t, str):
                if len(t) > 2:
                    fun = t[2]
                else:
                    fun = lambda a: a
                offset = t[1]
                if offset is not None:
                    old_weight = out_sd.get(t[0], None)
                    if old_weight is None:
                        old_weight = torch.empty_like(weight)
                    if old_weight.shape[offset[0]] < offset[1] + offset[2]:
                        exp = list(weight.shape)
                        exp[offset[0]] = offset[1] + offset[2]
                        new = torch.empty(exp, device=weight.device, dtype=weight.dtype)
                        new[:old_weight.shape[0]] = old_weight
                        old_weight = new

                    w = old_weight.narrow(offset[0], offset[1], offset[2])
                else:
                    old_weight = weight
                    w = weight
                w[:] = fun(weight)
                t = t[0]
                out_sd[t] = old_weight
            else:
                out_sd[t] = weight
            state_dict.pop(k)

    return out_sd