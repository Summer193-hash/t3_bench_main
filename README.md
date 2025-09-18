# Tekken Vision Analysis Pipeline & Benchmark

This project provides a **complete computer vision pipeline** to extract game state data (health, timer) from **Tekken 3** gameplay videos.  
It also includes a **benchmarking suite** to compare the vision-based data against ground truth data, analyzing metrics for realism, accuracy, and gameplay outcomes.

---

## Project Structure

```
tekken-vision-pipeline/
├── benchmarking/                   
│   ├── run_tekken3_benchmark.py    # Main script to generate reports and charts
│   └── reports/                    # All output charts and summaries are saved here
│
├── data/                           
│   ├── 01_raw_videos/              # Input gameplay videos (.mp4, .avi)
│   ├── 02_npz_ground_truth/        # Ground truth .npz files
│   └── 03_processed_csVs/          
│       ├── ground_truth_output/
│       └── vision_output/
│
├── tekken_vision/                  
│   ├── health_detector.py          # CV functions for health detection
│   └── run_analysis.py             # Main engine for processing videos into CSVs
│
└── tools/
    ├── prepare_ground_truth.py     # Converts .npz files into CSVs
    └── run_setup.py                # Interactive calibration tool
```

---

## Setup

```bash
# Clone repo
git clone <your-repo-url>
cd tekken-vision-pipeline

# Create and activate venv
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> Note: `easyocr` and `torch` are large. First run may take time to download OCR models.

---

## Usage

### Step 1: Add Data
- Place raw videos into `data/01_raw_videos/`
- Place ground truth `.npz` files into `data/02_npz_ground_truth/`

### Step 2: Prepare Ground Truth
```bash
python3 tools/prepare_ground_truth.py
```
➡ Output: `data/03_processed_csVs/ground_truth_output/`

### Step 3: Calibrate Vision
```bash
python3 tools/run_setup.py
```
➡ Creates `config.json`

### Step 4: Run Vision Analysis
```bash
# Full analysis
python3 tekken_vision/run_analysis.py

# Fast win-rate only
python3 tekken_vision/run_analysis.py --mode win_rate
```
➡ Output: `data/03_processed_csVs/vision_output/`

### Step 5: Run Benchmark
```bash
python3 benchmarking/run_tekken3_benchmark.py
```
➡ Output: `benchmarking/reports/`

---

## Outputs
- Ground Truth CSVs → `data/03_processed_csVs/ground_truth_output/`
- Vision Extracted CSVs → `data/03_processed_csVs/vision_output/`
- Reports & Charts → `benchmarking/reports/`

---

## Acknowledgements
- OpenCV
- EasyOCR
- NumPy & Pandas