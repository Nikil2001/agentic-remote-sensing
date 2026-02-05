import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from unet_model import load_unet_model

# Paths
eval_dir = "evaluation_images"
output_dir = "predicted_masks"
os.makedirs(output_dir, exist_ok=True)

# Load model
model = load_unet_model(num_classes=7)
device = torch.device("cpu")
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def predict_and_save(image_path, output_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)[0]
        mask = torch.argmax(output, dim=0).cpu().numpy()

    Image.fromarray(mask.astype(np.uint8)).save(output_path)

# Predict all
for file in os.listdir(eval_dir):
    if file.endswith(".jpg") or file.endswith(".png"):
        image_path = os.path.join(eval_dir, file)
        base = os.path.splitext(file)[0].replace("_sat", "")
        out_path = os.path.join(output_dir, f"{base}_pred.png")
        predict_and_save(image_path, out_path)

print("✅ All predictions saved to predicted_masks/")
