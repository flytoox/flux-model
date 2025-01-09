from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import torch
from io import BytesIO
from flux_model import FluxAIModel

app = FastAPI()
model = FluxAIModel()

class InputData(BaseModel):
    prompt: str
    height: int = 1024
    width: int = 1024
    guidance_scale: float = 3.5
    num_inference_steps: int = 50
    max_sequence_length: int = 512
    seed: int = 0

@app.post("/predict")
async def predict(data: InputData):
    try:
        with torch.no_grad():
            image = model.generate_image(
                prompt=data.prompt,
                height=data.height,
                width=data.width,
                guidance_scale=data.guidance_scale,
                num_inference_steps=data.num_inference_steps,
                max_sequence_length=data.max_sequence_length,
                seed=data.seed
            )

            # Convert image to bytes
            img_bytes = BytesIO()
            image.save(img_bytes, format="PNG")
            img_bytes.seek(0)

        return StreamingResponse(img_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

