# Document Scanner Project

This project is a Document Scanner application that allows users to upload, scan, and extract text from documents. It uses OpenCV for image processing, Tesseract OCR for text extraction, and Streamlit for a user-friendly web interface.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Introduction

The Document Scanner project aims to provide a simple yet effective solution for digitizing printed documents. By leveraging OpenCV and Tesseract OCR, the application processes and extracts text from uploaded images, presenting it in an easy-to-use web interface built with Streamlit.

## Features

- **Upload Documents**: Upload images of documents in PNG format.
- **Image Processing**: Enhance and preprocess the uploaded images using OpenCV.
- **Text Extraction**: Extract text from the processed images using Tesseract OCR.
- **Interactive UI**: User-friendly interface developed with Streamlit.

## Requirements

- Python 3.7 or higher
- OpenCV
- Tesseract OCR
- Streamlit
- NumPy

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/sathvik-ng-07/Document-Scanner.git
   cd document-scanner
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Tesseract OCR:**
   - Download and install Tesseract OCR from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).

5. **Update Tesseract Path:**
   - Update the Tesseract path in `app.py` to the installed location.

## Usage

1. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

2. **Upload and Scan Document:**
   - Open your browser and go to `http://localhost:8501`.
   - Use the interface to upload an image in PNG format.
   - View the scanned document and extracted text.


## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.
