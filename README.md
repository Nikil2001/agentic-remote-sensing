# Query-Driven Remote Sensing Data Interpretation

An AI-powered satellite image analysis platform that combines deep learning-based semantic segmentation with Large Language Model (LLM) intelligence. Upload satellite imagery, automatically classify land cover types, and receive AI-generated summaries and answers to natural language queries.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Web_Framework-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red.svg)
![Gemini](https://img.shields.io/badge/Google_Gemini-LLM-orange.svg)

---

## Features

- **Land Cover Segmentation**: U-Net model classifies satellite pixels into 7 land cover categories
- **Query-Driven Analysis**: Ask natural language questions about satellite images
- **Change Detection**: Compare two temporal images to identify land use changes
- **LLM-Powered Summaries**: Google Gemini 2.0 Flash generates intelligent analysis reports
- **Web Interface**: Flask-based portal with intuitive upload and visualization

## Architecture

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

## Project Structure

```
├── app.py                    # Flask web application (routes + orchestration)
├── unet_model.py             # U-Net architecture + model loader
├── image_utils.py            # Image preprocessing, mask generation, stats
├── llm_summarizer.py         # Gemini 2.0 Flash integration
├── train_unet.py             # Model training script
├── unet_weights.pth          # Pre-trained model weights (7.1 MB)
├── requirements.txt          # Python dependencies
├── .env                      # API key (not committed — see setup)
├── templates/
│   ├── home.html             # Landing page
│   ├── index.html            # Single image analysis
│   └── compare.html          # Change detection
├── static/                   # CSS, logo, and generated output images
├── test_images/              # Sample satellite images for testing
└── developer-guide.md        # Detailed architecture and API docs
```

## Quick Start

### 1. Clone and enter the repository

```bash
git clone git@github.com:Nikil2001/agentic-remote-sensing.git
cd agentic-remote-sensing
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` installs: Flask, PyTorch, torchvision, OpenCV, Pillow, NumPy, Transformers, google-generativeai, python-dotenv.

### 4. Configure Gemini API key

Get a free key from [Google AI Studio](https://aistudio.google.com/app/apikey), then create a `.env` file in the project root:

```bash
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

The `.env` file is listed in `.gitignore` and will not be committed.

### 5. Run the application

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

> **Note:** On macOS, port 5000 may be occupied by AirPlay Receiver. Disable it in **System Settings → General → AirDrop & Handoff**, or run on a different port: `flask run --port 5001`.

---

## Testing

### Manual testing via browser

| Route | Method | What to test |
|---|---|---|
| `http://127.0.0.1:5000/` | GET | Landing page loads with navigation links |
| `http://127.0.0.1:5000/analyze` | GET | Analysis form renders correctly |
| `http://127.0.0.1:5000/compare` | GET | Change detection form renders correctly |

**Single image analysis:**
1. Go to `/analyze`
2. Upload any satellite JPEG/PNG (samples in `test_images/`)
3. Click **Analyze** — the land cover report and AI summary should appear
4. Try again with a question in the text field (e.g. `What percentage is forest cover?`) and click **Ask**

**Change detection:**
1. Go to `/compare`
2. Upload `test_images/before.jpg` as Image 1 and `test_images/after.jpg` as Image 2
3. Click **Compare** — a summary of changes between the two images should appear

### Quick curl smoke tests

Verify all routes respond before opening the browser:

```bash
# GET routes — all should return 200
curl -o /dev/null -w "%{http_code}\n" http://127.0.0.1:5000/
curl -o /dev/null -w "%{http_code}\n" http://127.0.0.1:5000/analyze
curl -o /dev/null -w "%{http_code}\n" http://127.0.0.1:5000/compare

# POST /analyze with a test image
curl -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:5000/analyze \
  -F "image=@test_images/satellite.jpg"

# POST /analyze with a natural language question
curl -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:5000/analyze \
  -F "image=@test_images/satellite.jpg" \
  -F "question=What percentage is forest?" \
  -F "action=ask"

# POST /compare with before/after images
curl -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:5000/compare \
  -F "image1=@test_images/before.jpg" \
  -F "image2=@test_images/after.jpg"
```

All POST routes should return `200` when the API key is valid and has quota.

---

## Land Cover Classes

| Class | Color | Description |
|---|---|---|
| Urban Land | Cyan | Built-up areas |
| Agriculture | Yellow | Farming regions |
| Rangeland | Magenta | Grasslands |
| Forest | Green | Tree cover |
| Water | Blue | Water bodies |
| Barren Land | White | Exposed soil |
| Unknown | Black | Unclassified |

---

## Technology Stack

| Category | Technology |
|---|---|
| Backend | Flask (Python) |
| Deep Learning | PyTorch, torchvision |
| Neural Network | U-Net (custom) |
| LLM | Google Gemini 2.0 Flash (`gemini-2.0-flash`) |
| Image Processing | Pillow, OpenCV, NumPy |

---

## Troubleshooting

**`500` error on POST /analyze or /compare**
- Check that `.env` exists and contains a valid `GEMINI_API_KEY`
- The free tier has rate limits — if you see `429 ResourceExhausted` in the logs, the key has hit its quota. Wait a minute or use a different key.

**`Address already in use` on port 5000**
- macOS AirPlay Receiver uses port 5000. Disable it in System Settings or use `flask run --port 5001`.
- Kill an existing process: `kill $(lsof -ti:5000)`

**`ModuleNotFoundError: No module named 'dotenv'` or `'google.generativeai'`**
- Re-run `pip install -r requirements.txt` inside the activated virtual environment.

**Model weights not found (`unet_weights.pth`)**
- The weights file (7.1 MB) must be present in the project root. It is committed to this repository and is not in `.gitignore`.

---

## Documentation

See [developer-guide.md](developer-guide.md) for detailed architecture diagrams, component documentation, API reference, and the training pipeline.

---

## License

Developed for **M.Tech Major Project** — Query-Driven Remote Sensing Data Interpretation.

_Built with Flask, PyTorch, U-Net, and Google Gemini AI_
