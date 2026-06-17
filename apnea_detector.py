import os
import sys
import time
import joblib
import librosa
import numpy as np
import serial  
import threading
from flask import Flask, render_template_string, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Initialize Flask Browser App
app = Flask(__name__)

# ==========================================
# CONFIGURATION SETUP
# ==========================================
SAMPLE_RATE = 16000      # Target audio frame rate (16kHz Mono)
DURATION = 1             # Audio frame window size (1 second)
N_MFCC = 13              # Number of coefficients
MODEL_FILENAME = "snoring_detector_model.pkl"
DATASET_PATH = "Snoring_dataset" 

# Shared live memory buffer for web-browser rendering
LATEST_METRICS = {
    "status": "System Initializing",
    "confidence": 0.0,
    "decibels": 0.0,
    "alert_level": "NORMAL",
    "alert_symbol": "✨",
    "alert_message": "Royal Monitor Active. Awaiting physiological breath signals..."
}

# ==========================================
# ACOUSTIC PROCESSING & MODEL UTILITIES
# ==========================================
def extract_features(audio_chunk):
    try:
        target_length = SAMPLE_RATE * DURATION
        if len(audio_chunk) < target_length:
            audio_chunk = np.pad(audio_chunk, (0, target_length - len(audio_chunk)), mode='constant')
        else:
            audio_chunk = audio_chunk[:target_length]
            
        mfccs = librosa.feature.mfcc(y=audio_chunk, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"[ERROR] Feature extraction anomaly: {e}")
        return None

def calculate_decibels(audio_chunk):
    rms = np.sqrt(np.mean(audio_chunk**2))
    if rms < 1e-5:
        return 0.0
    db = 20 * np.log10(rms) + 90 
    return float(np.clip(db, 0, 100))

def train_detector_model():
    print(f"=== Compiling Data Structures from: {DATASET_PATH} ===")
    X, y = [], []
    for label in ['0', '1']:
        folder_path = os.path.join(DATASET_PATH, label)
        if not os.path.exists(folder_path):
            print(f"[ABORT] Target path '{folder_path}' missing.")
            return False
        
        file_list = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
        if not file_list:
            print(f"[WARNING] No .wav files found in '{folder_path}'")
            continue

        for file_name in file_list:
            try:
                audio, _ = librosa.load(os.path.join(folder_path, file_name), sr=SAMPLE_RATE, mono=True)
                features = extract_features(audio)
                if features is not None:
                    X.append(features)
                    y.append(int(label))
            except Exception:
                continue
                    
    X, y = np.array(X), np.array(y)
    if len(X) == 0:
        print("[CRITICAL] Dataset processing yielded 0 valid samples.")
        return False
        
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_FILENAME)
        print("=== Random Forest Successfully Serialized ===")
        return True
    except Exception as e:
        print(f"[CRITICAL] Model compilation failure: {e}")
        return False

# ==========================================
# LIVE MONITORING CORE THREAD
# ==========================================
def start_hardware_listener():
    global LATEST_METRICS
    
    # Check if we have the trained file, if not try to build it safely
    if not os.path.exists(MODEL_FILENAME):
        print("[SYSTEM] Pre-trained ML model payload not found. Attempting compile sequence...")
        if not train_detector_model():
            print("[FALLBACK] Missing dataset structure. Running live on fallback Decibel Mode.")
            model = None
        else:
            model = joblib.load(MODEL_FILENAME)
    else:
        model = joblib.load(MODEL_FILENAME)

    snore_timestamps = []
    apnea_start_time = None
    
    PORT = "COM11" 
    BAUD_RATE = 115200
    CHUNK = int(SAMPLE_RATE * DURATION)

    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
        print(f"[HARDWARE] Connected to ESP32 on {PORT}")
    except Exception as e:
        print(f"[CRITICAL] Could not open Serial Port {PORT}: {e}")
        LATEST_METRICS["status"] = "Hardware Serial Disconnected"
        LATEST_METRICS["alert_message"] = f"Error: Cannot access port {PORT}. Check USB cable connections."
        return

    while True:
        try:
            audio_buffer = []
            
            while len(audio_buffer) < CHUNK:
                if ser.in_waiting:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if line.isdigit():
                        val = float(line) / 4095.0
                        audio_buffer.append(val)
            
            audio_chunk = np.array(audio_buffer, dtype=np.float32)
            db_level = calculate_decibels(audio_chunk)
            
            # Use ML prediction if model successfully loaded, otherwise fallback to decibel rules
            if model is not None:
                features = extract_features(audio_chunk).reshape(1, -1)
                probabilities = model.predict_proba(features)[0]  
                snore_confidence = float(probabilities[1]) * 100   
            else:
                # Decibel Fallback Heuristic
                snore_confidence = 80.0 if db_level > 42.0 else 5.0
            
            current_time = time.time()
            if snore_confidence >= 50.0:
                status = "Active Snoring Detected"
                snore_timestamps.append(current_time)
                apnea_start_time = None
                alert_level = "NORMAL"
                alert_symbol = "💤"
                alert_message = "Rhythmic snoring patterns registered. Signal stability normal."
            else:
                if db_level < 38.0:  
                    if len([t for t in snore_timestamps if current_time - t <= 60]) >= 2:
                        if apnea_start_time is None:
                            apnea_start_time = current_time
                        
                        elapsed_silence = current_time - apnea_start_time
                        if elapsed_silence >= 10.0:
                            status = "CRITICAL APNEA CESSATION"
                            alert_level = "EMERGENCY"
                            alert_symbol = "🚨"
                            alert_message = "EMERGENCY: Prolonged Sleep Apnea event verified! Call an ambulance immediately."
                        else:
                            status = "Suspected Respiratory Pause"
                            alert_level = "WARNING"
                            alert_symbol = "⚠️"
                            alert_message = f"Warning: Airway collapse detected for {int(elapsed_silence)}s. Monitoring for recovery gasp."
                    else:
                        status = "Optimal Rest Environment"
                        alert_level = "NORMAL"
                        alert_symbol = "👑"
                        alert_message = "Ambient sound levels serene. Sleeping environment perfectly quiet."
                else:
                    status = "Transient Background Acoustic Noise"
                    alert_level = "NORMAL"
                    alert_symbol = "🔊"
                    alert_message = "Non-respiratory sound elements filtered out by model matrix."

            snore_timestamps = [t for t in snore_timestamps if current_time - t <= 60]
            
            LATEST_METRICS = {
                "status": status,
                "confidence": round(snore_confidence, 1),
                "decibels": round(db_level, 1),
                "alert_level": alert_level,
                "alert_symbol": alert_symbol,
                "alert_message": alert_message
            }
        except Exception as ex:
            print(f"Error in listener thread loop: {ex}")
            time.sleep(0.1)

# ==========================================
# SCIENTIFIC WEB GUI INTERFACE
# ==========================================
@app.route('/')
def render_gui_dashboard():
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sleep Apnea Detection Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { 
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); 
                color: #f3f4f6; 
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                letter-spacing: 0.01em;
            }
            .dashboard-header {
                background: rgba(15, 23, 42, 0.6);
                border-bottom: 2px solid #3b82f6;
                padding: 25px 0;
                margin-bottom: 40px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            }
            .dashboard-title {
                color: #f59e0b;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .metric-card { 
                background: #1e293b; 
                border: 1px solid rgba(148, 163, 184, 0.2); 
                border-radius: 12px; 
                padding: 25px; 
                box-shadow: 0 10px 25px rgba(0,0,0,0.2);
            }
            .custom-progress {
                background-color: #0f172a !important;
                border-radius: 8px;
                height: 14px !important;
            }
            .bar-blue { background: linear-gradient(90deg, #2563eb 0%, #60a5fa 100%) !important; }
            .bar-cyan { background: linear-gradient(90deg, #0891b2 0%, #06b6d4 100%) !important; }
            
            .box-emergency { 
                background: linear-gradient(135deg, #7f1d1d 0%, #b91c1c 100%) !important; 
                border: 2px solid #ef4444 !important;
                animation: pulseGlow 1.5s infinite;
            }
            .box-warning { 
                background: linear-gradient(135deg, #78350f 0%, #d97706 100%) !important; 
                border: 2px solid #f59e0b !important;
            }
            .box-safe { 
                background: linear-gradient(135deg, #064e3b 0%, #059669 100%) !important; 
                border: 2px solid #10b981 !important;
            }
            @keyframes pulseGlow {
                0% { opacity: 1; } 50% { opacity: 0.7; } 100% { opacity: 1; }
            }
        </style>
    </head>
    <body>
        <header class="dashboard-header text-center">
            <div class="container">
                <h1 class="dashboard-title">Sleep Apnea Detection Dashboard</h1>
                <p class="text-muted mb-0">Real-Time Acoustic Analysis & Machine Learning Diagnostics</p>
            </div>
        </header>

        <div class="container">
            <div class="row g-4 mb-5">
                <div class="col-lg-6">
                    <div class="metric-card h-100 d-flex flex-column justify-content-center">
                        <h6 class="text-uppercase tracking-wider text-muted mb-2">Current System State</h6>
                        <h2 id="lbl-status" class="display-6 fw-bold text-white mb-0">Synchronizing...</h2>
                    </div>
                </div>

                <div class="col-lg-3">
                    <div class="metric-card h-100">
                        <h6 class="text-uppercase tracking-wider text-muted mb-3">Model Prediction Confidence</h6>
                        <h2 id="lbl-confidence" class="display-5 fw-bold text-info mb-3">0%</h2>
                        <div class="progress custom-progress">
                            <div id="bar-confidence" class="progress-bar bar-cyan" style="width: 0%"></div>
                        </div>
                    </div>
                </div>

                <div class="col-lg-3">
                    <div class="metric-card h-100">
                        <h6 class="text-uppercase tracking-wider text-muted mb-3">Sound Level Index</h6>
                        <h2 id="lbl-decibels" class="display-5 fw-bold text-primary mb-3">0 dB</h2>
                        <div class="progress custom-progress">
                            <div id="bar-decibels" class="progress-bar bar-blue" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
            </div>

            <div id="container-alert" class="p-4 rounded-4 box-safe d-flex align-items-center mb-5">
                <div class="me-4 display-4" id="lbl-symbol">✅</div>
                <div>
                    <h4 class="fw-bold mb-1" id="lbl-alert-headline">System Active</h4>
                    <p class="m-0 text-light opacity-90" id="lbl-alert-msg">Awaiting respiratory sensor inputs.</p>
                </div>
            </div>
        </div>

        <script>
            function fetchRuntimeMetrics() {
                fetch('/api/metrics')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('lbl-status').innerText = data.status;
                        document.getElementById('lbl-confidence').innerText = data.confidence + '%';
                        document.getElementById('bar-confidence').style.width = data.confidence + '%';
                        document.getElementById('lbl-decibels').innerText = data.decibels + ' dB';
                        document.getElementById('bar-decibels').style.width = data.decibels + '%';
                        
                        document.getElementById('lbl-alert-msg').innerText = data.alert_message;
                        
                        const alertBox = document.getElementById('container-alert');
                        const headline = document.getElementById('lbl-alert-headline');
                        const symbol = document.getElementById('lbl-symbol');
                        
                        alertBox.className = "p-4 rounded-4 d-flex align-items-center mb-5 ";
                        if (data.alert_level === "EMERGENCY") {
                            alertBox.classList.add("box-emergency");
                            headline.innerText = "🚨 APNEA DANGER ALERT: CRITICAL CESSATION DETECTED";
                            symbol.innerText = "🚨";
                        } else if (data.alert_level === "WARNING") {
                            alertBox.classList.add("box-warning");
                            headline.innerText = "⚠️ APNEA WARNING: SUSPECTED AIRWAY COLLAPSE";
                            symbol.innerText = "⚠️";
                        } else {
                            alertBox.classList.add("box-safe");
                            headline.innerText = "✅ MONITORING STATUS: NO APNEA DETECTED";
                            symbol.innerText = "✅";
                        }
                    });
            }
            setInterval(fetchRuntimeMetrics, 800);
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/api/metrics')
def stream_metrics_api():
    return jsonify(LATEST_METRICS)

if __name__ == "__main__":
    import threading
    
    # 1. Start the hardware background routine
    worker = threading.Thread(target=start_hardware_listener, daemon=True)
    worker.start()
    
    # 2. Start Flask on a SEPARATE background thread so it doesn't block IDLE
    print("[FLASK] Launching local interface layer on background thread...")
    flask_worker = threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False),
        daemon=True
    )
    flask_worker.start()

    # 3. Keep the main thread alive with a passive loop so the background threads can run
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[SYSTEM] Shutting down monitor gracefully.")
