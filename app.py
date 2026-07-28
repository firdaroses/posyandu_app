from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# ============================================================
# LOAD MODEL DAN SCALER SAAT SERVER START
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    with open(os.path.join(BASE_DIR, 'knn_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    print("✅ Model KNN berhasil dimuat!")
except FileNotFoundError:
    raise RuntimeError("❌ File 'knn_model.pkl' tidak ditemukan! Pastikan sudah diupload ke folder flask_api/")

try:
    with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    print("✅ StandardScaler berhasil dimuat!")
except FileNotFoundError:
    raise RuntimeError("❌ File 'scaler.pkl' tidak ditemukan! Pastikan sudah diupload ke folder flask_api/")

# ============================================================
# MAPPING LABEL NUMERIK → TEKS STATUS
# Normal=0, Severely Stunted=1, Stunted=2
# ============================================================
LABEL_MAP = {
    0: "Normal",
    1: "Severely Stunted",
    2: "Stunted"
}


# ============================================================
# ROUTE: GET / — Cek status server
# ============================================================
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "KNN Stunting Prediction API is running!",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /predict"
        }
    }), 200


# ============================================================
# ROUTE: POST /predict — Prediksi status stunting
# ============================================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Body request kosong atau bukan JSON"}), 400

        # --- Validasi field yang wajib ada ---
        required_fields = ['umur', 'berat', 'tinggi', 'jenis_kelamin']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Field '{field}' tidak ditemukan dalam request"}), 400

        # --- Parse dan validasi nilai ---
        umur = float(data['umur'])       # usia dalam bulan
        berat = float(data['berat'])     # berat badan dalam kg
        tinggi = float(data['tinggi'])   # tinggi badan dalam cm
        jenis_kelamin_str = str(data['jenis_kelamin']).strip()

        # --- Encode jenis kelamin: Laki-laki=0, Perempuan=1 ---
        jenis_kelamin_lower = jenis_kelamin_str.lower()
        if jenis_kelamin_lower in ['laki-laki', 'laki laki', 'l', 'male', 'boy']:
            jenis_kelamin_encoded = 0
        else:
            jenis_kelamin_encoded = 1  # Perempuan / Female

        # --- Susun array fitur sesuai urutan saat training ---
        # Urutan: [umur, berat_badan, tinggi_badan, jenis_kelamin]
        features = np.array([[umur, berat, tinggi, jenis_kelamin_encoded]])

        # --- Normalisasi menggunakan StandardScaler yang sama dengan training ---
        features_scaled = scaler.transform(features)

        # --- Prediksi dengan model KNN ---
        prediction_raw = model.predict(features_scaled)[0]

        # --- Ambil confidence score (probabilitas tertinggi dari KNN) ---
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = float(max(probabilities))

        # --- Map angka ke label teks ---
        status = LABEL_MAP.get(int(prediction_raw), "Unknown")

        return jsonify({
            "status": status,
            "label_numerik": int(prediction_raw),
            "confidence": round(confidence, 4),
            "confidence_persen": f"{round(confidence * 100, 1)}%"
        }), 200

    except ValueError as ve:
        return jsonify({"error": f"Nilai input tidak valid: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan server: {str(e)}"}), 500


# ============================================================
# JALANKAN SERVER
# ============================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
