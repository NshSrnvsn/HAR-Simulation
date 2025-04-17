# HAR-Simulation 🏃 
## A Real-Time Activity Recognition Dashboard

Simulates real-time predictions using the UCI HAR Dataset and a trained Random Forest model. Built with Python and Streamlit.

## Features
- Live stream simulation of sensor data
- ML model trained to classify 6 human activities
- "Safe" vs "Potential Risk" indicators
- Activity distribution chart updated in real-time
- Export predictions to CSV
- Built with: Python, Streamlit, scikit-learn, pandas

## 🎥 Demo
![Demo](assets/demo.gif)

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
