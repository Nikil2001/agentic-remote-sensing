import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from unet_model import load_unet_model
import os
import pandas as pd
import matplotlib.pyplot as plt
from llm_summarizer import generate_summary_from_land_cover  # ✅ Gemini-based summarizer

# ====== CONFIG ======
IMAGE_PATH = "test_images/satellite.jpg"   # 🔁 Replace with your test image
CLASS_CSV = "data/class_dict.csv"
IMG_SIZE = 128
DEVICE = torch.device("cpu")

# ====== Load Class Mapping ======
def load_class_map(csv_path):
    df = pd.read_csv(csv_path)
    index_to_class = {i: row['name'] for i, row in df.iterrows()}
    return index_to_class

# ====== Load and Preprocess Image ======
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # [1, 3, H, W]

# ====== Predict Segmentation Mask ======
def predict_mask(image_tensor, model):
    with torch.no_grad():
        output = model(image_tensor.to(DEVICE))[0]  # [num_classes, H, W]
        predicted = torch.argmax(output, dim=0).cpu().numpy()  # [H, W]
    return predicted

# ====== Count Class Percentages ======
def calculate_percentages(mask):
    total = mask.size
    unique, counts = np.unique(mask, return_counts=True)
    percentages = {int(cls): round(100 * count / total, 2) for cls, count in zip(unique, counts)}
    return percentages

# ====== Visualize Predicted Mask ======
def visualize_mask(mask):
    plt.imshow(mask, cmap="tab20")
    plt.title("Predicted Segmentation Mask")
    plt.axis("off")
    plt.show()

# ====== MAIN ======
def main():
    print("🔍 Loading model...")
    class_map = load_class_map(CLASS_CSV)
    model = load_unet_model(num_classes=len(class_map))

    print("🖼️ Preprocessing image...")
    image_tensor = preprocess_image(IMAGE_PATH)

    print("🧠 Predicting mask...")
    predicted_mask = predict_mask(image_tensor, model)

    print("📊 Calculating class percentages...")
    percentages_raw = calculate_percentages(predicted_mask)

    # Convert class indices to names
    percentages_named = {
        class_map[k]: v for k, v in percentages_raw.items()
    }

    print("\n📋 Land Cover Report:")
    for cls_name, percent in percentages_named.items():
        print(f"  {cls_name}: {percent}%")

    # ✅ Use Gemini to summarize based on land cover + image
    summary = generate_summary_from_land_cover(percentages_named, IMAGE_PATH)
    print("\n🧠 Gemini Summary:")
    print(summary)

    visualize_mask(predicted_mask)

if __name__ == "__main__":
    main()
