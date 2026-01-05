# Question Answering API

API Backend untuk sistem Question Answering menggunakan Named Entity Recognition (NER) untuk mendeteksi hama dan penyakit padi.

## Instalasi

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Pastikan file model dan vocabulary ada:
   - `bilstm_crf.pth` (model file)
   - `vocab.json` (vocabulary file)

## Menjalankan API

### Cara 1: Menggunakan script run_api.py
```bash
python run_api.py
```

### Cara 2: Langsung menggunakan uvicorn
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

### Cara 3: Dari dalam src/
```bash
cd src
python api.py
```

API akan berjalan di: `http://localhost:8000`

## Dokumentasi API

Setelah server berjalan, dokumentasi API otomatis tersedia di:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Endpoints

### 1. Health Check
```
GET /health
GET /
```
Mengecek status API dan apakah model sudah dimuat.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu",
  "vocab_size": 5000,
  "tag_size": 20
}
```

### 2. Predict NER (Single)
```
POST /predict
```
Prediksi NER untuk satu kalimat.

**Request:**
```json
{
  "text": "Daun padi menguning dan muncul bercak coklat"
}
```

**Response:**
```json
{
  "text": "Daun padi menguning dan muncul bercak coklat",
  "token_tags": [
    {"token": "daun", "tag": "B-BAGIAN_TANAMAN"},
    {"token": "padi", "tag": "I-BAGIAN_TANAMAN"},
    {"token": "menguning", "tag": "B-GEJALA"},
    ...
  ],
  "entities": [
    {
      "entity_type": "BAGIAN_TANAMAN",
      "entities": ["daun padi"]
    },
    {
      "entity_type": "GEJALA",
      "entities": ["menguning", "bercak coklat"]
    }
  ]
}
```

### 3. Predict NER (Batch)
```
POST /predict/batch
```
Prediksi NER untuk beberapa kalimat sekaligus (maksimal 100).

**Request:**
```json
{
  "texts": [
    "Daun padi menguning",
    "Apa itu penyakit blas?",
    "Batang padi berlubang"
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "text": "Daun padi menguning",
      "token_tags": [...],
      "entities": [...]
    },
    ...
  ]
}
```

### 4. Extract Entities Only
```
POST /extract-entities
```
Hanya mengekstrak entitas tanpa detail token-tag.

**Request:**
```json
{
  "text": "Daun padi menguning dan muncul bercak coklat"
}
```

**Response:**
```json
{
  "text": "Daun padi menguning dan muncul bercak coklat",
  "entities": {
    "BAGIAN_TANAMAN": ["daun padi"],
    "GEJALA": ["menguning", "bercak coklat"]
  }
}
```

## Contoh Penggunaan

### Menggunakan curl

```bash
# Health check
curl http://localhost:8000/health

# Predict single
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Daun padi menguning dan muncul bercak coklat"}'

# Extract entities only
curl -X POST "http://localhost:8000/extract-entities" \
  -H "Content-Type: application/json" \
  -d '{"text": "Apa itu penyakit blas?"}'
```

### Menggunakan Python requests

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Predict
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Daun padi menguning dan muncul bercak coklat"}
)
print(response.json())

# Batch predict
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={
        "texts": [
            "Daun padi menguning",
            "Apa itu penyakit blas?",
            "Batang padi berlubang"
        ]
    }
)
print(response.json())
```

### Menggunakan JavaScript (fetch)

```javascript
// Predict
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text: 'Daun padi menguning dan muncul bercak coklat'
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

## CORS

API sudah dikonfigurasi untuk menerima request dari semua origin (untuk development). 
Untuk production, ubah di `src/api.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Ganti dengan domain spesifik
    ...
)
```

## Production Deployment

Untuk production, gunakan uvicorn dengan workers:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
```

Atau gunakan gunicorn dengan uvicorn workers:

```bash
gunicorn src.api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Troubleshooting

1. **Model not loaded error**: Pastikan file `bilstm_crf.pth` dan `vocab.json` ada di root directory
2. **Import error**: Pastikan virtual environment sudah diaktifkan
3. **Port already in use**: Ganti port di `run_api.py` atau gunakan flag `--port` dengan uvicorn

