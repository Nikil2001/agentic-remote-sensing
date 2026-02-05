# 🛰️ Query-Driven Remote Sensing Data Interpretation

An AI-powered satellite image analysis platform that combines deep learning-based semantic segmentation with Large Language Model (LLM) intelligence. Upload satellite imagery, automatically classify land cover types, and receive AI-generated summaries and answers to natural language queries.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Web_Framework-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red.svg)
![Gemini](https://img.shields.io/badge/Google_Gemini-LLM-orange.svg)

---

## ✨ Features

- **🗺️ Land Cover Segmentation**: U-Net model classifies satellite pixels into 7 land cover categories
- **💬 Query-Driven Analysis**: Ask natural language questions about satellite images
- **📊 Change Detection**: Compare two temporal images to identify land use changes
- **🤖 LLM-Powered Summaries**: Google Gemini generates intelligent analysis reports
- **🌐 Web Interface**: Flask-based portal with intuitive upload and visualization

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Web Layer                                │
│  Flask App  →  Templates (Jinja2)  →  Static Assets (CSS)       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Processing Layer                            │
│  Image Utils  →  Preprocessing  →  Mask Generation  →  Stats    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         AI Layer                                 │
│      U-Net Model (Segmentation)  ←→  Gemini LLM (Summarization) │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
├── app.py                    # Flask web application
├── unet_model.py             # U-Net architecture definition
├── image_utils.py            # Image preprocessing utilities
├── llm_summarizer.py         # Gemini LLM integration
├── train_unet.py             # Model training script
├── unet_weights.pth          # Pre-trained model weights
├── templates/                # HTML templates
│   ├── home.html
│   ├── index.html
│   └── compare.html
├── static/                   # CSS and images
├── data/                     # Training data
└── developer-guide.md        # Detailed documentation
```

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone git@github.com:Nikil2001/agentic-remote-sensing.git
cd agentic-remote-sensing
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Gemini API Key

1. Get API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create `.env` file:

```bash
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

### 5. Run Application

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

## 🎯 Land Cover Classes

| Class       | Color      | Description     |
| ----------- | ---------- | --------------- |
| Urban Land  | 🔵 Cyan    | Built-up areas  |
| Agriculture | 🟡 Yellow  | Farming regions |
| Rangeland   | 🟣 Magenta | Grasslands      |
| Forest      | 🟢 Green   | Tree cover      |
| Water       | 🔵 Blue    | Water bodies    |
| Barren Land | ⚪ White   | Exposed soil    |
| Unknown     | ⬛ Black   | Unclassified    |

## 📖 Usage

### Single Image Analysis

1. Navigate to **Single Image Analysis**
2. Upload a satellite image (JPEG/PNG)
3. Optionally enter a question (e.g., "What percentage is forest?")
4. View land cover report and AI summary

### Change Detection

1. Navigate to **Change Detection**
2. Upload two temporal images (before/after)
3. View comparative analysis of land cover changes

## 🛠️ Technology Stack

| Category         | Technology              |
| ---------------- | ----------------------- |
| Backend          | Flask (Python)          |
| Deep Learning    | PyTorch, torchvision    |
| Neural Network   | U-Net (custom)          |
| LLM              | Google Gemini 2.0 Flash |
| Image Processing | Pillow, OpenCV, NumPy   |

## 📄 Documentation

See [developer-guide.md](developer-guide.md) for detailed:

- Architecture diagrams
- Component documentation
- API reference
- Debugging guide
- Extension guide

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m "Add feature"`
4. Push to branch: `git push origin feature-name`
5. Open a Pull Request

## 📝 License

This project is developed for **M.Tech Major Project** - Query-Driven Remote Sensing Data Interpretation.

---

_Built with Flask, PyTorch, U-Net, and Google Gemini AI_
