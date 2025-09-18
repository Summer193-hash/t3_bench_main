Tekken Vision Analysis Pipeline & Benchmark
This project provides a complete computer vision pipeline to extract game state data (health, timer) from Tekken 3 gameplay videos. It also includes a comprehensive benchmarking suite to compare the vision-based data against ground truth data, analyzing metrics for realism, accuracy, and gameplay outcomes.

Project Structure
The project is organized into distinct modules for data generation, analysis, and benchmarking.

tekken-vision-pipeline/
├── benchmarking/                   # Scripts for analyzing and comparing data
│   ├── run_tekken3_benchmark.py    # Main script to generate reports and charts
│   └── reports/                    # All output charts and summaries are saved here
│
├── data/                           # All data files (IGNORED BY GIT)
│   ├── 01_raw_videos/              # Place input videos (.mp4, .avi) here
│   ├── 02_npz_ground_truth/        # Place ground truth .npz files here
│   └── 03_processed_csVs/          # All generated CSVs are stored here
│       ├── ground_truth_output/
│       └── vision_output/
│
├── tekken_vision/                  # The core Python source code for the pipeline
│   ├── health_detector.py          # Core CV functions for health detection
│   └── run_analysis.py             # Main engine that processes videos into CSVs
│
└── tools/
    ├── prepare_ground_truth.py     # Processes .npz files into clean ground truth CSVs
    └── run_setup.py                # Interactive tool to calibrate vision settings

Setup Instructions
Clone the repository:

git clone <your-repo-url>
cd tekken-vision-pipeline

Install Dependencies: It is highly recommended to use a Python virtual environment to avoid conflicts with other projects.

# Create and activate the virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install all required libraries
pip install -r requirements.txt

Note: easyocr and its dependency torch can be large. The first run of the analysis script may take some time to download the necessary OCR models.

How to Run the Full Pipeline
Follow these steps in order. All commands should be run from the main tekken-vision-pipeline/ root directory.

Step 1: Prepare Your Data
Place your raw gameplay videos (.mp4, .avi) into the data/01_raw_videos/ folder.

Place your ground truth .npz files into the data/02_npz_ground_truth/ folder.

Step 2: Prepare the Ground Truth
This script processes all .npz files into clean, processed CSVs and their corresponding videos.

python3 tools/prepare_ground_truth.py

Output will be saved to data/03_processed_csVs/ground_truth_output/.

Step 3: Calibrate the Vision Settings
This is a one-time interactive setup. A window will pop up asking you to select a video and use your mouse to define the health bars and timer region of interest (ROI).

python3 tools/run_setup.py

This will create a config.json file in the project root.

Step 4: Run the Vision Analysis
This script processes all videos in data/01_raw_videos/ using your config.json settings. It has two modes for efficiency.

Option A: Full Analysis (Default)
This is the standard, detailed analysis. It processes every frame to generate data for all benchmark metrics.

python3 tekken_vision/run_analysis.py

Option B: Fast Win-Rate Analysis
Use this mode if you ONLY need to calculate the win rate. It's much faster as it only analyzes the last frame of each video.

python3 tekken_vision/run_analysis.py --mode win_rate

Output for both modes will be saved to data/03_processed_csVs/vision_output/.

Step 5: Run the Benchmark
Finally, this script compares the ground truth data against the vision-based data and generates your final report with charts and summary statistics.

python3 benchmarking/run_tekken3_benchmark.py

Output charts and summary CSVs will be saved to benchmarking/reports/.