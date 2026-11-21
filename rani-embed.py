#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# RANI Embed Engine v1.0.0 — Offline PDF → embeddings pipeline
# Usage: ./venv/bin/python rani-embed.py [--force] [--status]

import os
import sys
import re
import json
import time
import hashlib
import signal
import argparse
import numpy as np
import requests
import fitz
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
DOT_ENV_FILE = os.path.join(BASE_DIR, ".env")

EMBED_DIM = 1024
BATCH_SIZE = 10
MAX_RETRIES = 9
SLEEP_SEC = 0.5

OLLAMA_EMBED_BASE_URL = "http://localhost:11434"
OLLAMA_EMBED_MODEL = "bge-m3"

FILE_HASHES_FILE = os.path.join(BASE_DIR, "file_hashes.json")
EMBED_CACHE_FILE = os.path.join(BASE_DIR, "embed_cache.json")
EMBED_PROGRESS_FILE = os.path.join(BASE_DIR, "embed_progress.json")

CONFIG = {}
DAFTAR_SUMBER = []
CACHE_FILENAME = ""


def signal_handler(signum, frame):
    print("\n⚠️ Proses embedding dihentikan. Progress disimpan.", flush=True)
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ========== CONFIG & ENV ==========

def load_config():
    global CONFIG, DAFTAR_SUMBER, CACHE_FILENAME
    if not os.path.exists(CONFIG_FILE):
        print(f"❌ ERROR: config.json tidak ditemukan di {CONFIG_FILE}")
        sys.exit(1)
    with open(CONFIG_FILE, 'r') as f:
        CONFIG = json.load(f)
    DAFTAR_SUMBER = CONFIG.get("DAFTAR_SUMBER", [])
    raw = CONFIG.get("CACHE_FILENAME", "embeddings_rani_v6.npy")
    CACHE_FILENAME = os.path.join(BASE_DIR, raw) if not os.path.isabs(raw) else raw


def load_ollama_config():
    global OLLAMA_EMBED_BASE_URL, OLLAMA_EMBED_MODEL
    load_dotenv(DOT_ENV_FILE)
    OLLAMA_EMBED_BASE_URL = os.environ.get("OLLAMA_EMBED_BASE_URL", "http://localhost:11434").rstrip("/")
    OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3").strip()
    print(f"🔗 Ollama Embed: {OLLAMA_EMBED_BASE_URL} | Model: {OLLAMA_EMBED_MODEL}", flush=True)


# ========== PDF PARSER (PASAL-BASED) ==========

def resolve_alias(filename):
    basename = os.path.basename(filename).lower().replace('.pdf', '').replace('.txt', '')
    if 'perma-14-2016' in basename:
        return 'PERMA 14/2016 (Ekonomi Syariah)'
    if 'perma-7-2022' in basename:
        return 'PERMA 7/2022 (Perkara Elektronik)'
    if 'perma-5-2019' in basename:
        return 'PERMA 5/2019 (Dispensasi Nikah)'
    if 'perma-4-2019' in basename:
        return 'PERMA 4/2019 (Gugatan Sederhana)'
    if 'perma-3-2017' in basename:
        return 'PERMA 3/2017 (Perkara Perempuan)'
    if 'perma-2-2015' in basename:
        return 'PERMA 2/2015 (Gugatan Sederhana)'
    if 'perma-1-2019' in basename:
        return 'PERMA 1/2019 (Perkara Elektronik)'
    if 'perma-1-2016' in basename:
        return 'PERMA 1/2016 (Prosedur Mediasi)'
    if 'perma-1-2014' in basename:
        return 'PERMA 1/2014 (Prodeo)'
    if 'pp-45-1990' in basename:
        return 'PP 45/1990 (Izin Kawin PNS)'
    if 'pp-10-1983' in basename:
        return 'PP 10/1983 (Izin Kawin PNS)'
    if 'pp-9-1975' in basename:
        return 'PP 9/1975 (Pelaksanaan Perkawinan)'
    if 'uu-50-2009' in basename:
        return 'UU 50/2009 (Peradilan Agama)'
    if 'uu-48-2009' in basename:
        return 'UU 48/2009 (Kekuasaan Hakim)'
    if 'uu-41-2004' in basename:
        return 'UU 41/2004 (Wakaf)'
    if 'uu-21-2008' in basename:
        return 'UU 21/2008 (Perbankan Syariah)'
    if 'uu-19-2008' in basename:
        return 'UU 19/2008 (Surat Berharga Syariah Negara)'
    if 'uu-18-2003' in basename:
        return 'UU 18/2003 (Advokat)'
    if 'uu-16-2019' in basename:
        return 'UU 16/2019 (Perkawinan)'
    if 'uu-7-1989' in basename:
        return 'UU 7/1989 (Peradilan Agama)'
    if 'uu-3-2006' in basename:
        return 'UU 3/2006 (Peradilan Agama)'
    if 'uu-1-1974' in basename:
        return 'UU 1/1974 (Perkawinan)'
    if 'sumber2' in basename:
        return 'Syarat Perkara PA Medan'
    if 'khi-id' in basename:
        return 'KHI (Kompilasi Hukum Islam)'
    return os.path.basename(filename).replace('.pdf', '').upper()


def parse_pdf_pasal(filename):
    chunks = []
    seen_hashes = set()
    try:
        doc = fitz.open(filename)
        full_text = ""
        for page in doc:
            text = page.get_text("text")
            lines = text.split('\n')
            cleaned = [l for l in lines if not re.search(r'(---\s*PAGE|SK\s*No|SALINAN|PRESIDEN|REPUBLIK|INDONESIA)', l, re.I)]
            full_text += "\n".join(cleaned) + "\n"
        doc.close()

        alias = resolve_alias(filename)

        def normalize_ocr(text):
            text = re.sub(r'Pasa1(\s*)(\d+)', r'Pasal \2', text)
            text = re.sub(r'Pasal(\d+)', r'Pasal \1', text)
            return text
        full_text = normalize_ocr(full_text)

        # Strategy A: split by pasal headers at line-start (handles Roman numerals, quoted pasal)
        pasal_pat = r'\n["\']?(Pasal\s+(?:\d+[a-zA-Z]?|[IVXLCDM]+))\s*\n'
        parts = re.split(pasal_pat, '\n' + full_text.strip(), flags=re.IGNORECASE)

        pasal_chunks = []
        if len(parts) >= 3:
            if parts[0].strip() and len(parts[0].strip()) > 100:
                pasal_chunks.append(("[%s] Pembukaan" % alias, parts[0].strip()))

            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    header = parts[i].strip()
                    isi = parts[i+1].strip()
                    if len(isi) < 50:
                        continue
                    if "cukup jelas" in isi.lower() and len(isi) < 100:
                        continue
                    if header and isi:
                        pasal_chunks.append(("[%s] %s" % (alias, header), isi))

        # Strategy B: split by paragraphs (double newlines)
        para_chunks = []
        blocks = re.split(r'\n\s*\n', full_text.strip())
        for i, block in enumerate(blocks, 1):
            block = block.strip()
            if len(block) >= 50:
                para_chunks.append(("[%s] Bagian %d" % (alias, i), block))

        # Merge strategies: pasal chunks preferred, supplemented by para chunks
        def content_hash(text):
            return hashlib.md5(text.encode()).hexdigest()[:16]

        def split_large_chunk(label, isi, max_size=1500):
            """Split very large chunks into smaller pieces by double-newlines."""
            if len(isi) <= max_size:
                return [(label, isi)]
            sub_blocks = re.split(r'\n\s*\n', isi)
            result = []
            current = ""
            sub_idx = 1
            for block in sub_blocks:
                block = block.strip()
                if not block:
                    continue
                if len(current) + len(block) > max_size and current:
                    result.append(("%s (bag %d)" % (label, sub_idx), current))
                    sub_idx += 1
                    current = block
                else:
                    current = (current + "\n\n" + block) if current else block
            if current:
                result.append(("%s (bag %d)" % (label, sub_idx), current))
            return result

        def sanitize_text(text):
            text = re.sub(r'[.]{3,}', '...', text)
            text = re.sub(r'[ ]{4,}', ' ', text)
            text = re.sub(r'[_]{3,}', '___', text)
            return text

        for label, isi in pasal_chunks:
            for sub_label, sub_isi in split_large_chunk(label, isi):
                sub_isi = sanitize_text(sub_isi)
                if not sub_isi.strip():
                    continue
                h = content_hash(sub_isi)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    chunks.append("%s\n%s" % (sub_label, sub_isi))

        # Supplement with para chunks if pasal chunks are too few
        doc_char_count = len(full_text)
        expected_min = max(10, doc_char_count // 500)
        if len(pasal_chunks) < expected_min:
            for label, isi in para_chunks:
                for sub_label, sub_isi in split_large_chunk(label, isi):
                    sub_isi = sanitize_text(sub_isi)
                    if not sub_isi.strip():
                        continue
                    h = content_hash(sub_isi)
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        chunks.append("%s\n%s" % (sub_label, sub_isi))

        if alias and len(pasal_chunks) > 0:
            chunks.insert(0, "[%s] Metadata: %d pasal ditemukan dalam dokumen ini" % (alias, len(pasal_chunks)))
    except Exception as e:
        print(f"⚠️ Gagal parse PDF {filename}: {e}", flush=True)
    return chunks

def parse_txt(filepath):
    chunks = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        alias = resolve_alias(filepath)
        blocks = re.split(r'\n\s*\n', text.strip())
        for i, block in enumerate(blocks, 1):
            block = block.strip()
            if block:
                chunks.append(f"[{alias}] Bagian {i}\n{block}")
        if alias and len(chunks) > 0:
            chunks.insert(0, "[%s] Metadata: %d bagian ditemukan dalam dokumen ini" % (alias, len(chunks)))
    except Exception as e:
        print(f"⚠️ Gagal parse TXT {filepath}: {e}", flush=True)
    return chunks

def ingest_dokumen(sumber_list, file_filter=None):
    all_chunks = []
    file_map = []
    for filepath in sumber_list:
        if not os.path.exists(filepath):
            print(f"⚠️ Melewati {filepath}: File tidak ditemukan.", flush=True)
            continue

        if file_filter and filepath not in file_filter:
            print(f"📂 Melewati {filepath} (tidak berubah)...", flush=True)
            continue

        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.pdf':
            print(f"📂 Memproses {filepath} (PASAL-BASED)...", flush=True)
            chunks = parse_pdf_pasal(filepath)
            print(f"   -> {len(chunks)} Pasal ditemukan", flush=True)
        elif ext == '.txt':
            print(f"📂 Memproses {filepath} (TXT-BASED)...", flush=True)
            chunks = parse_txt(filepath)
            print(f"   -> {len(chunks)} bagian ditemukan", flush=True)
        else:
            print(f"⚠️ Format tidak didukung: {ext}. Hanya PDF/TXT yang diterima.", flush=True)
            continue

        all_chunks.extend(chunks)
        file_map.extend([filepath] * len(chunks))

    return all_chunks, file_map


# ========== FILE HASH MANAGEMENT ==========

def compute_file_hash(filepath):
    if not os.path.exists(filepath):
        return None
    h = hashlib.md5()
    h.update(str(os.path.getmtime(filepath)).encode())
    h.update(str(os.path.getsize(filepath)).encode())
    return h.hexdigest()[:12]


def compute_all_hashes():
    hashes = {}
    for src in DAFTAR_SUMBER:
        h = compute_file_hash(src)
        if h:
            hashes[src] = h
    return hashes


def load_file_hashes():
    if os.path.exists(FILE_HASHES_FILE):
        try:
            with open(FILE_HASHES_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_file_hashes(hashes):
    with open(FILE_HASHES_FILE, 'w') as f:
        json.dump(hashes, f, indent=2)


def get_file_status():
    current_hashes = compute_all_hashes()
    saved_hashes = load_file_hashes()
    new_files = []
    changed_files = []
    unchanged_files = []
    for src in DAFTAR_SUMBER:
        if src not in current_hashes:
            continue
        if src not in saved_hashes:
            new_files.append(src)
        elif current_hashes[src] != saved_hashes[src]:
            changed_files.append(src)
        else:
            unchanged_files.append(src)
    return {"new": new_files, "changed": changed_files, "unchanged": unchanged_files, "all": current_hashes}


# ========== EMBEDDING ENGINE ==========

def load_embed_cache():
    if os.path.exists(EMBED_CACHE_FILE):
        try:
            with open(EMBED_CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_embed_cache(cache):
    with open(EMBED_CACHE_FILE, 'w') as f:
        json.dump(cache, f)


def load_embed_progress():
    if os.path.exists(EMBED_PROGRESS_FILE):
        try:
            with open(EMBED_PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_embed_progress(progress):
    with open(EMBED_PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)


def embed_chunks(chunks, file_name):
    if not chunks:
        return []

    progress = load_embed_progress()
    embeddings = []

    if file_name in progress:
        existing = progress[file_name]
        if len(existing) >= len(chunks):
            print(f"      ✅ {file_name} sudah selesai di-cache", flush=True)
            return [np.array(e, dtype=np.float32) for e in existing]
        embeddings = [np.array(e, dtype=np.float32) for e in existing]
        print(f"      🔄 Melanjutkan progress parsial: {len(embeddings)}/{len(chunks)}", flush=True)

    start_idx = len(embeddings)
    for i in range(start_idx, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        total_retries = 0
        success = False

        while not success and total_retries < MAX_RETRIES:
            try:
                time.sleep(SLEEP_SEC)
                resp = requests.post(
                    f"{OLLAMA_EMBED_BASE_URL}/api/embed",
                    json={"model": OLLAMA_EMBED_MODEL, "input": batch, "keep_alive": "10m"},
                    timeout=300
                )
                resp.raise_for_status()
                data = resp.json()
                batch_embeddings = data.get("embeddings", [])
                embeddings.extend([np.array(e, dtype=np.float32) for e in batch_embeddings])
                print(f"      Progress: {len(embeddings)}/{len(chunks)}", flush=True)

                progress[file_name] = [e.tolist() for e in embeddings]
                save_embed_progress(progress)

                success = True
            except Exception as e:
                err = str(e).lower()
                if "429" in err or "quota" in err or "resource exhausted" in err or "rate limit" in err:
                    total_retries += 1
                    print(f"      ⚠️ Rate limit (percobaan {total_retries}/{MAX_RETRIES})", flush=True)
                    if total_retries >= MAX_RETRIES:
                        print(f"      ⚠️ Semua percobaan habis. Mengisi sisa dengan vektor nol.", flush=True)
                        embeddings.extend([np.zeros(EMBED_DIM, dtype=np.float32) for _ in batch])
                        progress[file_name] = [e.tolist() for e in embeddings]
                        save_embed_progress(progress)
                        success = True
                        break
                    time.sleep(2)
                    continue
                elif "connection" in err or "timeout" in err:
                    total_retries += 1
                    print(f"      ⚠️ Koneksi gagal (percobaan {total_retries}/{MAX_RETRIES}): {e}", flush=True)
                    if total_retries >= MAX_RETRIES:
                        embeddings.extend([np.zeros(EMBED_DIM, dtype=np.float32) for _ in batch])
                        progress[file_name] = [e.tolist() for e in embeddings]
                        save_embed_progress(progress)
                        success = True
                    time.sleep(2)
                else:
                    print(f"      ⚠️ Error: {e}", flush=True)
                    total_retries += 1
                    if total_retries >= MAX_RETRIES:
                        embeddings.extend([np.zeros(EMBED_DIM, dtype=np.float32) for _ in batch])
                        progress[file_name] = [e.tolist() for e in embeddings]
                        save_embed_progress(progress)
                        success = True

    if file_name in progress:
        del progress[file_name]
        save_embed_progress(progress)

    return [np.array(e, dtype=np.float32) for e in embeddings]


# ========== VECTOR DB BUILDER ==========

def compile_vector_db(force_rebuild=False):
    if force_rebuild:
        print(f"\n🗑️ Menghapus cache embeddings karena --force...", flush=True)
        for fpath in [EMBED_CACHE_FILE, EMBED_PROGRESS_FILE, FILE_HASHES_FILE, CACHE_FILENAME]:
            if os.path.exists(fpath):
                os.remove(fpath)
                print(f"   - Hapus {os.path.basename(fpath)}", flush=True)

    file_status = get_file_status()
    current_hashes = file_status["all"]

    # Fast-path: muat .npy langsung jika tidak ada perubahan
    if not force_rebuild and os.path.exists(CACHE_FILENAME):
        saved_hashes = load_file_hashes()
        if saved_hashes and saved_hashes == current_hashes:
            print(f"⚡ Memuat embeddings dari {CACHE_FILENAME}...", flush=True)
            vectors = np.load(CACHE_FILENAME)
            print(f"✅ {len(vectors)} vectors dimuat dari cache (skip embed)", flush=True)
            return True

    print(f"\n📊 Status file sumber:")
    print(f"   - {len(file_status['new'])} baru")
    print(f"   - {len(file_status['changed'])} berubah")
    print(f"   - {len(file_status['unchanged'])} tidak berubah (gunakan cache)")

    all_chunks, file_map = ingest_dokumen(DAFTAR_SUMBER)
    if not all_chunks:
        print("❌ Tidak ada pasal yang berhasil diekstrak.")
        return False

    print(f"\n📚 Total pasal: {len(all_chunks)}")

    embed_cache = load_embed_cache()
    final_vectors = []
    current_file = None
    current_file_chunks = []

    def _process(src_file, file_chunks):
        nonlocal final_vectors
        needs_embed = src_file in file_status['new'] or src_file in file_status['changed'] or src_file not in embed_cache
        if needs_embed:
            print(f"⚡ Embedding: {src_file}...", flush=True)
            vecs = embed_chunks(file_chunks, src_file)
        else:
            print(f"💾 Cache: {src_file}", flush=True)
            vecs = [np.array(e, dtype=np.float32) for e in embed_cache[src_file]]
        embed_cache[src_file] = [v.tolist() for v in vecs]
        final_vectors.extend(vecs)

    for chunk, src_file in zip(all_chunks, file_map):
        if src_file != current_file:
            if current_file and current_file_chunks:
                _process(current_file, current_file_chunks)
            current_file = src_file
            current_file_chunks = [chunk]
        else:
            current_file_chunks.append(chunk)

    if current_file and current_file_chunks:
        _process(current_file, current_file_chunks)

    save_embed_cache(embed_cache)
    save_file_hashes(current_hashes)

    vectors = np.array(final_vectors, dtype=np.float32)
    np.save(CACHE_FILENAME, vectors)
    print(f"\n✅ Selesai: {len(vectors)} vectors tersimpan ke {os.path.basename(CACHE_FILENAME)}", flush=True)
    return True


# ========== STATUS CHECKER ==========

def show_status():
    print(f"\n{'=' * 50}")
    print("📊 STATUS EMBEDDING ENGINE")
    print(f"{'=' * 50}")

    # Ollama config
    print(f"\n🔗 Ollama Embed: {OLLAMA_EMBED_BASE_URL}")
    print(f"   Model: {OLLAMA_EMBED_MODEL}")

    # Config
    print(f"\n📄 Config: {len(DAFTAR_SUMBER)} sumber, cache → {os.path.basename(CACHE_FILENAME)}")

    # File status
    file_status = get_file_status()
    print(f"\n📂 Status file sumber:")
    for src in DAFTAR_SUMBER:
        if os.path.exists(src):
            size_mb = os.path.getsize(src) / (1024 * 1024)
            status_str = f"✅ Ada ({size_mb:.2f} MB)"
            if src in file_status["new"]:
                status_str += " [BARU]"
            elif src in file_status["changed"]:
                status_str += " [BERUBAH]"
            else:
                status_str += " [CACHE OK]"
        else:
            status_str = "❌ TIDAK DITEMUKAN"
        print(f"   - {src}: {status_str}")

    # .npy file
    print(f"\n💾 File embeddings:")
    npy_ok = os.path.exists(CACHE_FILENAME)
    if npy_ok:
        vecs = np.load(CACHE_FILENAME)
        print(f"   - {os.path.basename(CACHE_FILENAME)}: ✅ {len(vecs)} vectors, dim {vecs.shape[1]}")
    else:
        print(f"   - {os.path.basename(CACHE_FILENAME)}: ❌ Belum ada")

    # Cache files
    print(f"   - embed_cache.json: {'✅ Ada' if os.path.exists(EMBED_CACHE_FILE) else '❌ Belum ada'}")
    print(f"   - file_hashes.json: {'✅ Ada' if os.path.exists(FILE_HASHES_FILE) else '❌ Belum ada'}")
    print(f"{'=' * 50}\n")


# ========== MAIN ==========

def main():
    load_config()
    load_ollama_config()

    parser = argparse.ArgumentParser(
        description="RANI Embed Engine — Offline PDF → embeddings pipeline",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--force", action="store_true",
                        help="Hapus semua cache dan rebuild dari awal")
    parser.add_argument("--status", action="store_true",
                        help="Tampilkan status embedding engine tanpa memproses")
    args = parser.parse_args()

    if args.status:
        show_status()
        sys.exit(0)

    print(f"\n⚖️ RANI Embed Engine v1.0.0")
    print("=" * 40)

    success = compile_vector_db(force_rebuild=args.force)
    if not success:
        sys.exit(1)

    print(f"\n🚀 Embedding selesai. {os.path.basename(CACHE_FILENAME)} siap digunakan rani-api.py.\n")


if __name__ == "__main__":
    main()
