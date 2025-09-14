# 🏍️ Motorcycle AI Training System

Sistem machine learning untuk diagnosis masalah motor berdasarkan dataset expanded_motorcycle_dataset.json.

## 🚀 Quick Start

```bash
# 1. Install dependencies
python install_dependencies.py

# 2. Run full pipeline
python run_full_pipeline.py

# 3. Test the trained AI
python demo_ai_system.py
```

## 📁 Struktur Folder

```
AI_Training/
├── expanded_motorcycle_dataset.json    # Dataset utama (3355 entri)
├── data_preprocessing.py               # Script preprocessing data
├── model_training.py                   # Script training ML models
├── model_evaluation.py                 # Script evaluasi model
├── model_inference.py                  # Script inference/prediksi
├── demo_ai_system.py                   # Demo interaktif sistem AI
├── run_full_pipeline.py                # Script menjalankan full pipeline
├── install_dependencies.py             # Script instalasi dependencies
├── requirements.txt                    # Daftar dependencies
├── README.md                           # Dokumentasi ini
├── preprocessed_data.pkl               # Data yang sudah diproses (generated)
└── models/                             # Folder model terlatih (generated)
    ├── category_*.pkl                  # Model prediksi kategori
    └── severity_*.pkl                  # Model prediksi severity
```

## 🔧 Setup & Instalasi

### 1. Install Dependencies

```bash
# Otomatis install semua dependencies
python install_dependencies.py

# Atau manual
pip install -r requirements.txt
```

### 2. Verifikasi Dataset

Pastikan file `expanded_motorcycle_dataset.json` ada di folder ini.

## 🎯 Pipeline Training

### Step 1: Data Preprocessing

```bash
python data_preprocessing.py
```

**Output:**
- `preprocessed_data.pkl` - Data yang sudah diproses
- Feature extraction (text, numerical, categorical)
- Target encoding untuk kategori dan severity

### Step 2: Model Training

```bash
python model_training.py
```

**Features:**
- 6 algoritma ML: Random Forest, Gradient Boosting, XGBoost, LightGBM, SVM, Logistic Regression
- 9 kombinasi features: text_only, numerical_only, categorical_only, dll.
- Cross-validation untuk setiap model
- Automatic model selection berdasarkan performa

**Output:**
- `models/` folder dengan model terbaik
- Training logs dengan performa setiap model

### Step 3: Model Evaluation

```bash
python model_evaluation.py
```

**Output:**
- `evaluation_report.json` - Laporan performa detail
- `model_comparison.png` - Visualisasi perbandingan model
- Confusion matrix untuk setiap model

### Step 4: Testing & Inference

```bash
python model_inference.py --interactive
```

**Features:**
- Interactive testing
- Batch prediction
- Confidence scores
- Feature importance

## 🎮 Demo System

```bash
python demo_ai_system.py
```

**Features:**
- Demo cases dengan masalah motor umum
- Interactive mode untuk testing manual
- System information dan model stats
- Accuracy testing

## 🎯 Tasks Prediksi

### 1. Category Prediction
Memprediksi kategori masalah motor:
- sistem_starter
- sistem_pendingin
- sistem_rem
- sistem_kelistrikan
- transmisi
- sistem_pelumasan
- sistem_bahan_bakar
- suspensi
- dan 76 kategori lainnya

### 2. Severity Prediction
Memprediksi tingkat keparahan:
- ringan
- sedang
- berat
- kritis
- dan level lainnya

## 🔬 Feature Engineering

### Text Features
- TF-IDF vectorization (1000 dimensions)
- Problem description analysis
- Symptom text processing
- Combined text features

### Numerical Features
- Problem length
- Word count
- Symptom count
- Cost statistics (min, max, avg, range)
- Keyword count

### Categorical Features
- Category encoding
- Severity encoding
- One-hot encoding untuk features kategorikal

## 📊 Model Performance

**Expected Performance:**
- Category Prediction: ~95%+ accuracy
- Severity Prediction: ~90%+ accuracy
- Cross-validation scores dengan confidence intervals

## 💡 Usage Examples

### Prediksi Tunggal

```python
from model_inference import MotorcycleInferenceEngine

engine = MotorcycleInferenceEngine()
result = engine.predict_single("Motor susah hidup pagi hari")
print(f"Category: {result['predicted_category']}")
print(f"Severity: {result['predicted_severity']}")
```

### Batch Prediction

```python
problems = [
    "Rem blong tidak menggigit",
    "Mesin overheat suhu tinggi",
    "Lampu mati aki tekor"
]

results = engine.predict_batch(problems)
for i, result in enumerate(results):
    print(f"Problem {i+1}: {result['predicted_category']}")
```

## 🔧 Tips & Improvement

### Meningkatkan Akurasi
1. **Tambah data training** - Lebih banyak contoh untuk kategori yang jarang
2. **Feature engineering** - Ekstrak features baru dari teks
3. **Hyperparameter tuning** - Optimize parameter model
4. **Ensemble methods** - Kombinasi multiple models

### Optimasi Performance
1. **Feature selection** - Pilih features yang paling informatif
2. **Model compression** - Reduce model size untuk deployment
3. **Caching** - Cache predictions untuk input yang sama

## 🐛 Troubleshooting

### Error: "No module named 'sklearn'"
```bash
python install_dependencies.py
```

### Error: "preprocessed_data.pkl not found"
```bash
python data_preprocessing.py
```

### Error: "No trained models found"
```bash
python model_training.py
```

### Low Accuracy
- Check data quality
- Verify feature engineering
- Try different algorithms
- Increase training data

## 🚀 Advanced Usage

### Custom Feature Engineering

Edit `data_preprocessing.py` untuk menambah features baru:

```python
def extract_custom_features(self, data):
    # Add your custom features here
    features = []
    # ... custom logic
    return features
```

### Custom Models

Tambah algoritma baru di `model_training.py`:

```python
from sklearn.ensemble import ExtraTreesClassifier

models = {
    'extra_trees': ExtraTreesClassifier(n_estimators=100, random_state=42)
}
```

### Integration dengan Aplikasi

```python
# Untuk integrasi dengan aplikasi web/mobile
from model_inference import MotorcycleInferenceEngine

class MotorcycleAPI:
    def __init__(self):
        self.engine = MotorcycleInferenceEngine()
    
    def diagnose(self, problem_description):
        return self.engine.predict_single(problem_description)
```

## 📈 Production Deployment

### Model Serving
1. **Flask/FastAPI** - REST API untuk model serving
2. **Docker** - Containerization untuk deployment
3. **Model versioning** - Track model versions
4. **Monitoring** - Monitor model performance

### Scaling
1. **Load balancing** - Multiple model instances
2. **Caching** - Redis untuk cache predictions
3. **Async processing** - Queue untuk batch predictions

## 📞 Support

Jika ada pertanyaan atau issues:
1. Check troubleshooting section
2. Review logs untuk error details
3. Verify all dependencies installed
4. Check dataset format dan completeness

---

**Happy Training! 🏍️🤖**