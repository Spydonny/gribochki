from fastapi import FastAPI, File, UploadFile
from torchvision import transforms
from PIL import Image
import torch
import io

from model import SimpleCNN

app = FastAPI()

# Загружаем модель
model = SimpleCNN(num_classes=12)
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu"), weights_only=True))
model.eval()

# Преобразование входного изображения
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)  # [1, 3, 32, 32]

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    return {"predicted_class": predicted.item()}
