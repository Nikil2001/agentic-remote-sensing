from flask import Flask, render_template, request
from PIL import Image
import os

from image_utils import (
    preprocess_image,
    predict_mask,
    calculate_land_cover_percentages,
    save_mask_image
)

from llm_summarizer import (
    generate_summary_from_land_cover,
    generate_change_summary
)

from unet_model import load_unet_model

app = Flask(__name__)
UPLOAD_FOLDER = "static"
MODEL = load_unet_model(num_classes=7)


# Home Page → home.html
@app.route("/")
def home():
    return render_template("home.html")


# Single Image Analysis
@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    if request.method == "POST":
        image = request.files.get("image")
        question = request.form.get("question")
        action = request.form.get("action")

        if image:
            img_path = os.path.join(UPLOAD_FOLDER, "uploaded.png")
            image.save(img_path)

            img = Image.open(img_path).convert("RGB")
            tensor = preprocess_image(img)
            mask = predict_mask(MODEL, tensor)
            save_mask_image(mask, os.path.join(UPLOAD_FOLDER, "mask.png"))

            percentages = calculate_land_cover_percentages(mask)
            report_lines = [f"{k.replace('_', ' ').title()}: {v:.2f}%" for k, v in percentages.items()]

            summary = generate_summary_from_land_cover(percentages, img_path, question if action == "ask" else None)

            return render_template("index.html", report=report_lines, summary=summary, answer=summary if action == "ask" else None)

    return render_template("index.html")


# Change Detection Analysis
@app.route("/compare", methods=["GET", "POST"])
def compare():
    if request.method == "POST":
        image1 = request.files.get("image1")
        image2 = request.files.get("image2")
        question = request.form.get("question")

        if image1 and image2:
            path1 = os.path.join(UPLOAD_FOLDER, "img1.png")
            path2 = os.path.join(UPLOAD_FOLDER, "img2.png")
            image1.save(path1)
            image2.save(path2)

            img1 = Image.open(path1).convert("RGB")
            img2 = Image.open(path2).convert("RGB")

            tensor1 = preprocess_image(img1)
            tensor2 = preprocess_image(img2)

            mask1 = predict_mask(MODEL, tensor1)
            mask2 = predict_mask(MODEL, tensor2)

            save_mask_image(mask2, os.path.join(UPLOAD_FOLDER, "change_mask.png"))

            report1 = calculate_land_cover_percentages(mask1)
            report2 = calculate_land_cover_percentages(mask2)

            summary = generate_change_summary(report1, report2)

            return render_template("compare.html", summary=summary)

    return render_template("compare.html")


if __name__ == "__main__":
    app.run(debug=True)
