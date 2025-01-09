import torch
from diffusers import FluxPipeline

class FluxAIModel:
    def __init__(self, model_id="black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16):
        self.pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
        self.pipe.enable_model_cpu_offload()  # Offload to CPU if VRAM is limited

    def generate_image(self, prompt, height=1024, width=1024, guidance_scale=3.5, 
                       num_inference_steps=50, max_sequence_length=512, seed=0):
        generator = torch.Generator("cpu").manual_seed(seed)
        image = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            generator=generator
        ).images[0]
        return image

    def save_image(self, image, path="output.png"):
        image.save(path)

