# Game Analyzer - Video Game Footage Analysis Tool

## Overview
Game Analyzer is a Windows application that analyzes video game footage for Helldivers 2 and Undisputed Boxing. It provides tactical analysis, statistical overlays, and highlight generation.

## Features

### Helldivers 2 Analysis
- Friendly fire detection and analysis
- Stratagem usage tracking
- Squad synergy scoring
- Objective completion analysis
- Tactical suggestions for improvement

### Undisputed Boxing Analysis
- Punch accuracy analysis
- Combo detection and effectiveness
- Head movement scoring
- Defense analysis
- Fight rhythm and pacing analysis

### General Features
- Video processing (MP4 format)
- Statistical overlays and reports
- Highlight reel generation
- Performance profiling
- Comprehensive GUI interface

## Installation

### Option 1: Automatic Installation (Recommended)
1. Run `install.bat` as Administrator
2. Wait for installation to complete
3. Run `run.bat` to start the application

### Option 2: Manual Installation
1. Install Python 3.8 or later from https://python.org
2. Open Command Prompt as Administrator
3. Navigate to this directory
4. Run: `python -m venv game_analyzer_env`
5. Run: `game_analyzer_env\Scripts\activate.bat`
6. Run: `pip install -r requirements.txt`
7. Run: `python src\main_gui.py`

## System Requirements
- Windows 10/11 (64-bit)
- Python 3.8 or later
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- Graphics card with OpenGL support

## Usage
1. Launch the application using `run.bat` or manually
2. Select your game type (Helldivers 2 or Undisputed Boxing)
3. Load an MP4 video file of your gameplay
4. Click "Start Analysis" to begin processing
5. View results in the analysis panel
6. Export reports and highlights as needed

## Supported Video Formats
- MP4 (recommended)
- AVI
- MOV
- MKV

## Troubleshooting

### Common Issues
1. **"Python not found"**: Install Python from python.org and ensure it's in your PATH
2. **"Module not found"**: Run the installation script again or manually install requirements
3. **Video won't load**: Ensure video is in a supported format and not corrupted
4. **Analysis is slow**: Close other applications and ensure sufficient RAM is available

### Performance Tips
- Use videos with resolution 1920x1080 or lower for best performance
- Close unnecessary applications during analysis
- Ensure sufficient disk space for temporary files

## Technical Details
- Built with Python 3.11
- Uses OpenCV for computer vision
- PyQt5 for GUI interface
- TensorFlow for machine learning analysis
- MoviePy for video processing

## Support
For issues and questions, please check the documentation in the `docs` folder.

## Version
Version 1.0 - Initial Release

## License
This software is provided as-is for educational and analysis purposes.
