import os
import numpy as np

import torch
from torchvision import transforms
from .pipeline_flux_fill import FluxFillPipeline

import comfy.model_management as mm


script_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class LoadCatvtonFlux:
    
    RETURN_TYPES = ("CatvtonFluxModel",)
    FUNCTION = "load_catvton_flux"
    CATEGORY = "CatvtonFluxWrapper"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
        },
    } 
    
    def load_catvton_flux(self):

        load_device = mm.text_encoder_device()
        offload_device = mm.text_encoder_offload_device()

        print("Start loading LoRA weights")
        state_dict, network_alphas = FluxFillPipeline.lora_state_dict(
            pretrained_model_name_or_path_or_dict="xiaozaa/catvton-flux-lora-alpha",     ## The tryon Lora weights
            weight_name="pytorch_lora_weights.safetensors",
            return_alphas=True
        )
        is_correct_format = all("lora" in key or "dora_scale" in key for key in state_dict.keys())
        if not is_correct_format:
            raise ValueError("Invalid LoRA checkpoint.")
        print('Loading diffusion model ...')
        pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            torch_dtype=torch.bfloat16
        ).to(load_device)
        FluxFillPipeline.load_lora_into_transformer(
            state_dict=state_dict,
            network_alphas=network_alphas,
            transformer=pipe.transformer,
        )
        pipe.transformer.to(torch.bfloat16)
        print('Loading Finished!')

        model = {"pipe": pipe}
        
        return (model,)


class CatvtonFluxSampler:
    
    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("TryonResult", "GarmentResult",)

    FUNCTION = "sample"
    CATEGORY = "CatvtonFluxWrapper"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "CatvtonFluxModel": ("CatvtonFluxModel",),
                "prompt": ("STRING",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "garment": ("IMAGE",),
                "steps": ("INT", {"default": 30}),
                "guidance_scale": ("FLOAT", {"default": 30.0}),
                "seed": ("INT", {"default": 0}),
                "width": ("INT", {"default": 768}),
                "height": ("INT", {"default": 1024}),
        },
    } 
    
    def sample(self, CatvtonFluxModel, prompt, image, mask, garment, steps=30, guidance_scale=30.0, seed=-1, width=768, height=1024):
        load_device = mm.text_encoder_device()
        offload_device = mm.text_encoder_offload_device()

        pipe = CatvtonFluxModel["pipe"]
        pipe.transformer.to(load_device)

        size=(width, height)

        # Add transform
        transform = transforms.Compose([
            transforms.Normalize([0.5], [0.5])  # For RGB images
        ])

        image = image.permute(0, 3, 1, 2)
        mask = mask[:, None, ...]
        garment = garment.permute(0, 3, 1, 2)

        # Transform images using the new preprocessing
        image = transform(image)
        garment = transform(garment)

        # Create concatenated images
        inpaint_image = torch.cat([garment, image], dim=3)  # Concatenate along width
        garment_mask = torch.zeros_like(mask)
        extended_mask = torch.cat([garment_mask, mask], dim=3)

        result = pipe(
            height=size[1],
            width=size[0] * 2,
            image=inpaint_image[0],
            mask_image=extended_mask[0],
            num_inference_steps=steps,
            generator=torch.Generator(device=load_device).manual_seed(seed),
            max_sequence_length=512,
            guidance_scale=guidance_scale,
            prompt=prompt,
        ).images[0]

        pipe.transformer.to(offload_device)

        # Split and save results
        width = size[0]
        garment_result = result.crop((0, 0, width, size[1]))
        tryon_result = result.crop((width, 0, width * 2, size[1]))

        tryon_result = torch.tensor(
            np.array(tryon_result) / 255.0, dtype=torch.float32
        ).unsqueeze(0)
        garment_result = torch.tensor(
            np.array(garment_result) / 255.0, dtype=torch.float32
        ).unsqueeze(0)

        return (tryon_result, garment_result,)
