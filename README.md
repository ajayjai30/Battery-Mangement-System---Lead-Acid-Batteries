# 🔋 AI Battery Management System (BMS) with ThingSpeak Integration

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![IoT](https://img.shields.io/badge/IoT-ThingSpeak-green)](https://thingspeak.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📌 Project Overview

This project implements an **Artificial Intelligence–based Battery Management System (BMS)** designed to predict the **State of Charge (SOC)** of batteries in real time.

Unlike traditional BMS solutions that rely on simple voltage lookup tables (which are inaccurate under load), this system uses a **Long Short-Term Memory (LSTM)** neural network. It analyzes the *temporal history* of **Voltage**, **Current**, and **Temperature**, capturing battery hysteresis and internal resistance effects to achieve **high-accuracy SOC prediction (>99% correlation)**.

### 🔄 Hybrid IoT Architecture

1. **Sensors** push raw data to the **ThingSpeak Cloud**
2. **AI Engine** fetches data in real time
3. **LSTM Model** predicts SOC and logs it for monitoring

---

## 🚀 Key Features

- **Deep Learning Core**
  - Stacked LSTM network (128 + 64 units)
  - Trained on extensive charge/discharge datasets

- **Physics-Based Feature Engineering**
  - Power calculation: `P = V × I`
  - Voltage slope (`dV/dt`) for rapid voltage drop detection
  - Moving averages to reduce sensor noise

- **Universal Deployment**
  - Full Keras model for server/GPU inference
  - Quantized `.tflite` model for Edge devices (ESP32 / Raspberry Pi)

- **Robust Inference Engine**
  - Automatic buffering
  - Feature scaling
  - Missing-data handling

---

## 📂 Repository Structure

```text
/
├── models/
│   ├── bms_model_best.keras       # Trained LSTM Model (Server/GPU)
│   └── bms_model_quantized.tflite # Quantized Model (Edge/IoT)
├── scalers/
│   ├── scaler_X.pkl               # Input Feature Scaler
│   └── scaler_y.pkl               # Target SOC Scaler
├── bms_predictor.py               # 🧠 Main Inference Engine
├── thingspeak_bridge.py           # ☁️ ThingSpeak Integration
├── training_pipeline.ipynb        # Model Training Notebook
├── requirements.txt               # Python Dependencies
└── README.md                      # Project Documentation
````

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/ai-bms-system.git
cd ai-bms-system
```

### 2️⃣ Install Dependencies

```bash
pip install tensorflow numpy scikit-learn joblib pandas requests
```

> **Note:** For GPU deployment, ensure NVIDIA CUDA and cuDNN are properly installed.

---

## ☁️ Usage: Running with ThingSpeak

Use the `thingspeak_bridge.py` script to connect the AI engine to live sensor data.

### 1️⃣ ThingSpeak Channel Configuration

Ensure your ThingSpeak channel fields are mapped as follows:

* **Field 1:** Voltage (V)
* **Field 2:** Current (A)
* **Field 3:** Temperature (°C)

### 2️⃣ Configure API Credentials

Edit `thingspeak_bridge.py`:

```python
CHANNEL_ID = "YOUR_CHANNEL_ID"
READ_API_KEY = "YOUR_READ_API_KEY"
```

### 3️⃣ Start the AI Engine

```bash
python thingspeak_bridge.py
```

### 📟 Sample Output

```text
🔋 Initializing AI BMS Engine...
✅ AI Engine Ready. Listening to ThingSpeak...
--------------------------------------------------
📊 Sensor Reading: 12.6V | 2.5A | 34.0°C
🔋 PREDICTED SOC: 98.45%
--------------------------------------------------
```

---

## 🧠 Code Usage (Custom Integration)

You can use the AI predictor directly without ThingSpeak.

```python
from bms_predictor import BMS_Predictor

# Initialize predictor
predictor = BMS_Predictor(
    model_path='models/bms_model_best.keras',
    scaler_x_path='scalers/scaler_X.pkl',
    scaler_y_path='scalers/scaler_y.pkl'
)

# Real-time prediction (streamed input)
soc = predictor.predict_realtime(12.4, 1.5, 25.0)

if soc is not None:
    print(f"Current Battery SOC: {soc}%")
else:
    print("Buffering data (waiting for 10 samples)...")
```

---

## 📊 Model Performance

* **Training Hardware:** Tesla P100 GPU
* **Metric:** Mean Absolute Error (MAE)
* **MAE:** ~0.07 SOC
* **Correlation:** > 0.99 compared to Coulomb Counting
---

## 📜 License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for details.

