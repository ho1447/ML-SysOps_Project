from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel, Field
import torch
import numpy as np
import soundfile as sf
import onnxruntime as ort
from transformers import Wav2Vec2Processor
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title="Speech Recognition API",
    description="API for recognizing speech",
    version="1.0.0"
)

# Define the request and response models
class Request(BaseModel):
    file: str  

class PredictionResponse(BaseModel):
    prediction: str

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
ort_session = ort.InferenceSession("test.onnx")


@app.post("/predict")
def predict(request: Request):
    try:
        speech_array, sr = sf.read(request.file)
        features = processor(speech_array, sampling_rate=16000, return_tensors="pt")
        input_values = features.input_values
        
        # Run inference
        with torch.no_grad():
              onnx_outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: input_values.numpy()})[0]
              prediction = np.argmax(onnx_outputs, axis=-1)

        return PredictionResponse(prediction=processor.decode(prediction.squeeze().tolist()))

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {str(e)}")


Instrumentator().instrument(app).expose(app)