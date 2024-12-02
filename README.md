# ComfyUI-CatvtonFluxWrapper
ComfyUI wrapper of [catvton-flux](https://github.com/nftblackmagic/catvton-flux)

## Install
Search `ComfyUI-CatvtonFluxWrapper` in the ComfyUI manager, if cannot find it update ComfyUI-manager node then search again.

## workflow preview

The workflows can be found in the `examples` folder.

If you'd like more flexibility like using fp8 flux fill model, use `CatvtonLoRA-SAM2.json` or `CatvtonLoRA-Draw.json`. If you'd like to automatically download models with diffusers, use `CatvtonFluxWrapper-SAM2.json` or `CatvtonFluxWrapper-draw.json` (Not recommended though because of slow downloading of large models).

### SAM2 segment

- CatvtonFluxWrapper-SAM2.json
<img width="960" alt="preview" src="https://github.com/user-attachments/assets/5d3cb124-1988-433e-b2fa-0a6102a7ed89">

- CatvtonLoRA-SAM2.json
<img width="960" alt="preview" src="https://github.com/user-attachments/assets/6b08f124-6b19-41a2-bc5c-fbcf1badcc4c">


### Manually segment

- CatvtonFluxWrapper-draw.json
<img width="960" alt="preview" src="https://github.com/user-attachments/assets/8c334427-81d5-4efe-ba11-2dc477b0fc18">

- CatvtonLoRA-Draw.json
<img width="960" alt="preview" src="https://github.com/user-attachments/assets/9bbfaaa5-fca3-495f-8e36-d29fea3e8314">

