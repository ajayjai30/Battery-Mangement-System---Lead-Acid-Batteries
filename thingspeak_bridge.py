"""
ThingSpeak Bridge - AI Model Cloud Connector
==============================================
This script connects the BMS AI model to ThingSpeak cloud platform,
continuously fetching sensor data and providing real-time SOC predictions.

Author: BMS Project Team
Date: 2026
"""

import requests
import time
from datetime import datetime
from bms_predictor import BMS_Predictor

# ============================================================================
# CONFIGURATION
# ============================================================================

# ThingSpeak Channel Configuration
THINGSPEAK_CHANNEL_ID = 'YOUR_CHANNEL_ID'  # Replace with your actual Channel ID
READ_API_KEY = 'YOUR_READ_API_KEY'          # Replace with your actual Read API Key

# Polling Configuration
POLL_INTERVAL = 16  # Seconds (ThingSpeak allows 1 request per 15s, use 16s to be safe)

# Model and Scaler Paths
MODEL_PATH = 'models/bms_model_best.keras'
SCALER_X_PATH = 'scalers/scaler_X.pkl'
SCALER_Y_PATH = 'scalers/scaler_y.pkl'

# ThingSpeak API Endpoint
THINGSPEAK_URL = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds/last.json?api_key={READ_API_KEY}"

# ============================================================================
# INITIALIZATION
# ============================================================================

print("="*80)
print("🌐 THINGSPEAK BRIDGE - AI MODEL CLOUD CONNECTOR")
print("="*80)
print(f"Channel ID: {THINGSPEAK_CHANNEL_ID}")
print(f"Poll Interval: {POLL_INTERVAL} seconds")
print(f"API Endpoint: {THINGSPEAK_URL}")
print("="*80)

# Initialize the BMS Predictor
print("\n[INIT] Initializing AI Engine...")
try:
    predictor = BMS_Predictor(
        model_path=MODEL_PATH,
        scaler_x_path=SCALER_X_PATH,
        scaler_y_path=SCALER_Y_PATH
    )
    print("\n✅ 🔋 AI Engine Initialized. Waiting for data...")
except Exception as e:
    print(f"\n❌ ERROR: Failed to initialize AI Engine: {e}")
    print("\nPlease ensure:")
    print("  1. Model file exists: models/bms_model_best.keras")
    print("  2. Scaler files exist: scalers/scaler_X.pkl, scalers/scaler_y.pkl")
    print("  3. All dependencies are installed (tensorflow, joblib, etc.)")
    exit(1)

print("="*80)
print("🚀 BRIDGE ACTIVE - LISTENING FOR SENSOR DATA...")
print("="*80)

# ============================================================================
# MAIN LOOP - REAL-TIME DATA FETCHING AND PREDICTION
# ============================================================================

# Counter for tracking buffer status
data_count = 0
error_count = 0
prediction_count = 0

try:
    while True:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            # ================================================================
            # STEP 1: Fetch latest data from ThingSpeak
            # ================================================================
            response = requests.get(THINGSPEAK_URL, timeout=10)
            
            # Check if request was successful
            if response.status_code != 200:
                print(f"[{current_time}] ⚠️  HTTP Error {response.status_code} - Retrying...")
                error_count += 1
                time.sleep(POLL_INTERVAL)
                continue
            
            # Parse JSON response
            data = response.json()
            
            # ================================================================
            # STEP 2: Extract sensor values
            # ================================================================
            # Check if data contains the required fields
            if 'field1' not in data or 'field2' not in data or 'field3' not in data:
                print(f"[{current_time}] ⚠️  Incomplete data received - Waiting for sensors...")
                time.sleep(POLL_INTERVAL)
                continue
            
            # Extract values and convert to float
            voltage = float(data['field1'])
            current = float(data['field2'])
            temp = float(data['field3'])
            
            # Increment data counter
            data_count += 1
            
            # ================================================================
            # STEP 3: Run AI prediction
            # ================================================================
            soc = predictor.predict_realtime(voltage, current, temp, verbose=False)
            
            # Get buffer status
            status = predictor.get_buffer_status()
            
            # ================================================================
            # STEP 4: Display results
            # ================================================================
            if soc is None:
                # Still buffering data
                print(f"[{current_time}] 📊 Sensor: {voltage:.2f}V, {current:.2f}A, {temp:.1f}°C | "
                      f"⏳ Buffering data... ({status['current_size']}/10)")
            else:
                # Prediction ready
                prediction_count += 1
                print(f"[{current_time}] 📊 Sensor: {voltage:.2f}V, {current:.2f}A, {temp:.1f}°C | "
                      f"🔋 AI SOC: {soc:.2f}%")
                
                # Optional: Log statistics every 10 predictions
                if prediction_count % 10 == 0:
                    print(f"\n📈 Statistics: {prediction_count} predictions made, "
                          f"{error_count} errors encountered")
            
            # Reset error counter on successful fetch
            error_count = 0
            
        except requests.exceptions.Timeout:
            print(f"[{current_time}] ⚠️  Request timeout - Check internet connection")
            error_count += 1
            
        except requests.exceptions.ConnectionError:
            print(f"[{current_time}] ⚠️  Connection error - Is internet available?")
            error_count += 1
            
        except ValueError as e:
            print(f"[{current_time}] ⚠️  Data parsing error: {e}")
            error_count += 1
            
        except KeyError as e:
            print(f"[{current_time}] ⚠️  Missing field in response: {e}")
            error_count += 1
            
        except Exception as e:
            print(f"[{current_time}] ❌ Unexpected error: {e}")
            error_count += 1
        
        # ================================================================
        # STEP 5: Wait before next poll
        # ================================================================
        time.sleep(POLL_INTERVAL)

except KeyboardInterrupt:
    # Graceful shutdown on Ctrl+C
    print("\n" + "="*80)
    print("🛑 BRIDGE SHUTDOWN REQUESTED")
    print("="*80)
    print(f"📊 Final Statistics:")
    print(f"  • Total data points received: {data_count}")
    print(f"  • Total predictions made: {prediction_count}")
    print(f"  • Total errors encountered: {error_count}")
    print(f"  • Uptime: Bridge terminated by user")
    print("\n✅ Bridge stopped gracefully. Goodbye!")
    print("="*80)

except Exception as e:
    # Handle any unexpected errors
    print("\n" + "="*80)
    print("❌ CRITICAL ERROR - BRIDGE SHUTDOWN")
    print("="*80)
    print(f"Error: {e}")
    print(f"Last successful data count: {data_count}")
    print(f"Last successful prediction count: {prediction_count}")
    print("="*80)
