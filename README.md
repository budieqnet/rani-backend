# Rani Backend — Penasihat Hukum Digital Indonesia

**Rani** adalah sistem *Legal AI Assistant* untuk hukum perdata & Islam Indonesia. Menggunakan arsitektur **RAG (Retrieval-Augmented Generation)** dengan *hybrid search* (BM25 + dense embeddings) dan *dual-model pipeline*: embedding lokal via **Ollama (bge-m3)** + generasi jawaban via **NVIDIA Nemotron-3-Ultra**.

---

## 🏗️ Arsitektur & Tech Stack

| Layer | Teknologi |
|-------|-----------|
| **API Server** | Flask (Python 3.10+) |
| **Embedding** | Ollama — `bge-m3` (1024-dim, multilingual) |
| **Generation** | NVIDIA Nemotron-3-Ultra-550B (via OpenAI-compatible API) |
| **Vector Search** | NumPy cosine similarity + BM25Okapi (rank_bm25) |
| **Cache** | SQLite (`cache_jawaban.db`) — TTL 24 jam |
| **Embedding Cache** | `.npy` + JSON (`embeddings_rani.npy`, `embed_cache.json`) |
| **File Hash Tracking** | SHA256 per file (`file_hashes.json`) |
| **Logging** | CSV (`logs/interactions.csv`) |
| **PDF Parsing** | PyMuPDF (fitz) — pasal-based chunking |
| **Env Config** | python-dotenv |

---

## ✨ Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| **Hybrid Retrieval** | BM25 (keyword) + Cosine Similarity (semantic) — gabung & rerank |
| **Pasal-based Chunking** | Split PDF per "Pasal X" + fallback paragraph splitting |
| **OCR Normalization** | Fix "Pasa1" → "Pasal", "Pasal12" → "Pasal 12" |
| **Multi-source Corpus** | 26 dokumen hukum (UU, PERMA, PP, KHI, KUHAP, dll) |
| **Answer Caching** | SQLite cache dengan TTL 24 jam (SHA256 normalized query) |
| **Incremental Embedding** | File-hash tracking — hanya re-embed file yang berubah |
| **Session Chat History** | SQLite per `session_id` (role: user/assistant) |
| **Interaction Logging** | CSV log: timestamp, source, session_id, pertanyaan, jawaban |
| **CORS Ready** | Flask-CORS untuk integrasi frontend |
| **Graceful Shutdown** | Signal handler (SIGINT/SIGTERM) |

---

## 📚 Corpus Hukum (26 Dokumen)

| Kategori | Dokumen |
|----------|---------|
| **UU Perkawinan** | UU 1/1974, UU 16/2019 |
| **Peradilan Agama** | UU 7/1989, UU 3/2006, UU 50/2009 |
| **PERMA** | 1/2014 (Prodeo), 1/2016 (Mediasi), 1/2019 (Elektonik), 2/2015, 3/2017, 4/2019, 5/2019, 7/2022, 14/2016 |
| **PP** | PP 9/1975, PP 10/1983, PP 45/1990 (Izin Kawin PNS) |
| **Hukum Islam** | KHI (Kompilasi Hukum Islam) |
| **Lainnya** | UU 18/2003 (Advokat), UU 19/2008 (SBSN), UU 21/2008 (Perbankan Syariah), UU 41/2004 (Wakaf), UU 48/2009 (Kekuasaan Hakim), Syarat Perkara PA Medan |

---

## 🚀 Cara Menjalankan

### Prasyarat
- Python 3.10+
- **Ollama** running dengan model `bge-m3`: `ollama pull bge-m3`
- **NVIDIA API Key** untuk Nemotron-3-Ultra (daftar di https://integrate.api.nvidia.com)

### 1. Clone & Install
```bash
git clone https://github.com/budieqnet/rani-backend.git
cd rani-backend

# Buat virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Konfigurasi Environment
```bash
cp .env.example .env  # atau buat manual
```

Edit `.env`:
```env
# Ollama Embedding
OLLAMA_EMBED_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=bge-m3

# NVIDIA Generation
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_GEN_MODEL=nvidia/nemotron-3-ultra-550b-a55b
NVIDIA_API_KEY_1=your_nvidia_api_key_here
NVIDIA_API_KEY_2=optional_second_key
```

### 3. Generate Embeddings (Offline Pipeline)
```bash
# Pertama kali / force rebuild
python rani-embed.py --force

# Atau incremental (hanya file yang berubah)
python rani-embed.py

# Cek status
python rani-embed.py --status
```

> Output: `embeddings_rani.npy`, `embed_cache.json`, `embed_progress.json`, `file_hashes.json`

### 4. Jalankan API Server
```bash
python rani-api.py
```
Server berjalan di `http://localhost:5000` (default Flask).

---

## 📡 API Endpoints

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| `POST` | `/api/chat` | Chat utama — body: `{ "message": "...", "session_id": "..." }` |
| `GET` | `/api/history/<session_id>` | Riwayat chat per session |
| `POST` | `/api/clear-history` | Hapus riwayat session |
| `GET` | `/api/health` | Health check |

### Contoh Request
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Apa syarat nikah di bawah umur menurut UU 16/2019?", "session_id": "user-123"}'
```

### Contoh Response
```json
{
  "answer": "Berdasarkan UU 16/2019 Pasal 7...",
  "sources": ["[UU 16/2019 (Perkawinan)] Pasal 7", "..."],
  "session_id": "user-123"
}
```

---

## 📁 Struktur Proyek

```
rani-backend/
├── rani-api.py          # Flask API server (entry point)
├── rani-embed.py        # Offline embedding pipeline (CLI)
├── rani-monitor.py      # Monitoring utilities
├── config.json          # Daftar sumber & cache config
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (gitignored)
├── .gitignore
├── README.md            # (file ini)
├── cache_jawaban.db     # SQLite answer cache (gitignored)
├── embeddings_rani.npy  # Embedding vectors (gitignored)
├── embed_cache.json     # Embedding metadata (gitignored)
├── embed_progress.json  # Progress tracking (gitignored)
├── file_hashes.json     # File SHA256 hashes (gitignored)
├── logs/
│   └── interactions.csv # Interaction logs (gitignored)
└── sumber/              # Corpus PDF/TXT (26 dokumen)
    ├── khi-id.pdf
    ├── sumber2.txt
    ├── perma-1-2014-prodeo.pdf
    ├── perma-1-2016-prosedur-mediasi.pdf
    ├── perma-1-2019-perkara-elektronik-a.pdf
    ├── perma-14-2016-penyelesaian-ekonomi-syariah.pdf
    ├── perma-2-2015-gugatan-sederhana-a.pdf
    ├── perma-3-2017-perkara-perempuan.pdf
    ├── perma-4-2019-gugatan-sederhana-b.pdf
    ├── perma-5-2019-dispensasi-nikah.pdf
    ├── perma-7-2022-perkara-elektronik-b.pdf
    ├── pp-10-1983-izin-kawin-pns-a.pdf
    ├── pp-45-1990-izin-kawin-pns-b.pdf
    ├── pp-9-1975-pelaksana-perkawinan.pdf
    ├── uu-1-1974-perkawinan-a.pdf
    ├── uu-16-2019-perkawinan-b.pdf
    ├── uu-18-2003-advokat.pdf
    ├── uu-19-2008-surat-berharga-syariah-negara.pdf
    ├── uu-21-2008-perbankan-syariah.pdf
    ├── uu-3-2006-peradilan-agama-b.pdf
    ├── uu-41-2004-wakaf.pdf
    ├── uu-48-2009-kekuasaan-hakim.pdf
    ├── uu-50-2009-peradilan-agama-c.pdf
    └── uu-7-1989-peradilan-agama-a.pdf
```

---

## ⚙️ Konfigurasi Penting (`config.json`)

```json
{
  "DAFTAR_SUMBER": [
    "sumber/khi-id.pdf",
    "sumber/sumber2.txt",
    "sumber/perma-1-2014-prodeo.pdf",
    ...
  ],
  "CACHE_FILENAME": "embeddings_rani.npy"
}
```

- `DAFTAR_SUMBER`: Daftar path relatif ke file corpus
- `CACHE_FILENAME`: Nama file embedding cache (`.npy`)

---

## 🔧 Embedding Pipeline (`rani-embed.py`)

```bash
# Full rebuild (hapus cache lama, embed ulang semua)
python rani-embed.py --force

# Incremental (bandingkan file_hashes.json, hanya embed yang berubah)
python rani-embed.py

# Cek status progress
python rani-embed.py --status
```

**Strategi:**
1. Load `file_hashes.json` → bandingkan SHA256 file saat ini
2. File baru/berubah → parse PDF → chunk per pasal → embed via Ollama
3. Simpan ke `embeddings_rani.npy` + `embed_cache.json`
4. Update `file_hashes.json` + `embed_progress.json`

---

## 🗄️ Cache & Database Schema

### `cache_jawaban.db` — Answer Cache
```sql
CREATE TABLE query_cache (
    hash_key TEXT PRIMARY KEY,      -- SHA256(normalized_query)
    pertanyaan TEXT NOT NULL,
    jawaban TEXT NOT NULL,
    timestamp TEXT NOT NULL         -- ISO8601 UTC
);
```

### `chat_history` — Session History
```sql
CREATE TABLE chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,             -- 'user' | 'assistant'
    message TEXT NOT NULL,
    timestamp TEXT NOT NULL
);
CREATE INDEX idx_session ON chat_history(session_id);
```

---

## 📊 Monitoring (`rani-monitor.py`)

```bash
python rani-monitor.py
```
Menampilkan statistik: total chunks, embedding dim, cache size, log entries, dll.

---

## 🛡️ Keamanan & Best Practices

- **API Keys**: Simpan di `.env` (gitignored), jangan hardcode
- **Rate Limiting**: NVIDIA API key rotation (2 keys, round-robin)
- **Input Validation**: Normalisasi query (lowercase, strip, collapse whitespace)
- **CORS**: Dikonfigurasi via Flask-CORS
- **Graceful Shutdown**: SIGINT/SIGTERM handler untuk cleanup

---

## 📝 Requirements (`requirements.txt`)

```
numpy
flask
flask-cors
requests
python-dotenv
pymupdf
rank-bm25
```

Install: `pip install -r requirements.txt`

---

## 📄 License

Proyek ini open-source di bawah lisensi **MIT**. Silakan gunakan, modifikasi, dan distribusikan sesuai kebutuhan.

---

## 🤝 Kontribusi

1. Fork repository
2. Buat branch fitur (`git checkout -b fitur-baru`)
3. Commit perubahan (`git commit -m 'Tambah fitur X'`)
4. Push ke branch (`git push origin fitur-baru`)
5. Buat Pull Request

---

**Dibangun dengan ❤️ untuk akses hukum Indonesia yang lebih mudah** — *Rani: Penasihat Hukum Digital*