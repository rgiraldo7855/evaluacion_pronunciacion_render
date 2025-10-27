import os
import json
import subprocess
import re
import unicodedata
import wave
import datetime
import pandas as pd
import tempfile
import glob
import shutil
import atexit
import zipfile
from vosk import Model, KaldiRecognizer

# =============================================================
# ⚙️ CONFIGURACIÓN BASE
# =============================================================
MODEL_DIR = os.path.join("models", "vosk-model-small-en-us-0.15")
MODEL_ZIP = os.path.join("models", "vosk-model-small-en-us-0.15.zip")
RESULTS_CSV = "results.csv"
RESULT_TXT = "resultado.txt"

BONIFICACION_FINAL = 7.0  # %
TOLERANCIA_NIVEL = {
    "A1": 0.40, "A2": 0.50, "B1": 0.60,
    "B2": 0.70, "C1": 0.80, "C2": 0.90
}

# =============================================================
# 📦 DESCOMPRESIÓN AUTOMÁTICA DEL MODELO
# =============================================================
if os.path.exists(MODEL_ZIP) and not os.path.exists(MODEL_DIR):
    print("📦 Descomprimiendo modelo Vosk (esto ocurre solo una vez)...")
    try:
        with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
            zip_ref.extractall("models")
        print("✅ Modelo descomprimido correctamente.")
    except Exception as e:
        print(f"❌ Error al descomprimir el modelo: {e}")

# =============================================================
# 🧹 LIMPIEZA AUTOMÁTICA (sin borrar caché de Vosk)
# =============================================================
def limpiar_temporales():
    temp_dirs = [tempfile.gettempdir(), "results", "audios", "uploads"]
    patrones = ["*.wav", "*.json", "*.txt"]
    for carpeta in temp_dirs:
        if os.path.exists(carpeta):
            for patron in patrones:
                for archivo in glob.glob(os.path.join(carpeta, patron)):
                    try:
                        os.remove(archivo)
                    except Exception:
                        pass
    print("🧹 Limpieza completada (caché Vosk preservada).")

atexit.register(limpiar_temporales)

# =============================================================
# 🧠 CARGA Y PRECALENTAMIENTO DEL MODELO
# =============================================================
if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(f"Vosk model not found in {MODEL_DIR} or {MODEL_ZIP}")

print("🧠 Cargando modelo Vosk...")
MODEL = Model(MODEL_DIR)
print("✅ Modelo cargado en memoria.")

# =============================================================
# 🔤 NORMALIZACIÓN DE TEXTO
# =============================================================
def normalize_contractions(text: str) -> str:
    if not text:
        return text
    text = text.replace("’", "'").replace("‘", "'")
    contractions = {
        "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
        "it's": "it is", "that's": "that is", "there's": "there is",
        "what's": "what is", "who's": "who is", "let's": "let us",
        "can't": "cannot", "won't": "will not", "don't": "do not",
        "doesn't": "does not", "didn't": "did not", "haven't": "have not",
        "hasn't": "has not", "hadn't": "had not", "isn't": "is not",
        "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        "shouldn't": "should not", "wouldn't": "would not", "couldn't": "could not",
        "mustn't": "must not", "i've": "i have", "we've": "we have", "they've": "they have",
        "you've": "you have", "i'll": "i will", "he'll": "he will", "she'll": "she will",
        "they'll": "they will", "we'll": "we will", "you'll": "you will",
        "i'd": "i would", "he'd": "he would", "she'd": "she would",
        "they'd": "they would", "we'd": "we would", "you'd": "you would"
    }
    for k, v in contractions.items():
        text = re.sub(rf"\b{k}\b", v, text, flags=re.IGNORECASE)
    return text


def safe_normalize_text(text: str) -> str:
    if text is None:
        return ""
    t = text.lower().strip()
    t = normalize_contractions(t)
    t = ''.join(c for c in unicodedata.normalize('NFD', t) if unicodedata.category(c) != 'Mn')
    t = re.sub(r"[^a-z0-9'\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def equivalence_normalize(text: str) -> str:
    if not text:
        return text
    eq_pairs = [
        (r"\bit is\b", "its"),
        (r"\blet us\b", "lets"),
        (r"\bthey are\b", "theyre"),
        (r"\bwe are\b", "were"),
        (r"\byou are\b", "youre"),
    ]
    for pattern, repl in eq_pairs:
        text = re.sub(pattern, repl, text)
    return text

# =============================================================
# 🎧 CONVERSIÓN Y TRANSCRIPCIÓN
# =============================================================
def convert_to_wav_16k(in_path: str, out_path: str):
    # 🔹 Mantén 16 kHz (modelo small fue entrenado a esa frecuencia)
    cmd = ['ffmpeg', '-y', '-i', in_path, '-ac', '1', '-ar', '16000', '-sample_fmt', 's16', out_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(out_path):
        raise RuntimeError("ffmpeg conversion failed")


def transcribe_wav(wav_path: str) -> str:
    """Transcribe el audio en bloques de 5 s para mayor rapidez."""
    wf = wave.open(wav_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000]:
        wf.close()
        raise RuntimeError("WAV must be mono 16kHz PCM")

    rec = KaldiRecognizer(MODEL, wf.getframerate())
    rec.SetWords(False)
    transcript = ""

    CHUNK = wf.getframerate() * 5  # 5 segundos por bloque
    while True:
        data = wf.readframes(CHUNK)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            j = json.loads(rec.Result())
            transcript += " " + j.get("text", "")
    j = json.loads(rec.FinalResult())
    transcript += " " + j.get("text", "")
    wf.close()
    return transcript.strip()

# =============================================================
# 🔍 COMPARACIÓN DE TEXTOS
# =============================================================
def compare_texts(reference: str, transcription: str, nivel_mcer: str):
    ref_n = equivalence_normalize(safe_normalize_text(reference))
    tr_n = equivalence_normalize(safe_normalize_text(transcription))

    ref_words = ref_n.split()
    tr_words = tr_n.split()

    nivel_mcer = nivel_mcer.upper().strip() if nivel_mcer else "B1"
    umbral = TOLERANCIA_NIVEL.get(nivel_mcer, 0.6)

    missing = [w for w in ref_words if w not in tr_words]

    total = len(ref_words) or 1
    correct = total - len(missing)
    base_percent = (correct / total) * 100
    final_percent = min(base_percent + BONIFICACION_FINAL, 100.0)
    grade_0_5 = round((final_percent / 100) * 5.0, 1)

    return {
        "nivel": nivel_mcer,
        "umbral": umbral,
        "missing": missing,
        "base_percent": base_percent,
        "final_percent": final_percent,
        "grade_0_5": grade_0_5,
        "ref_words_count": len(ref_words),
        "tr_words_count": len(tr_words),
        "ref_norm": ref_n,
        "tr_norm": tr_n
    }

# =============================================================
# 🧩 EVALUACIÓN PRINCIPAL
# =============================================================
def evaluate_audio_file(uploaded_path, reference_text: str, nivel_mcer: str = "B1"):
    """
    Evalúa audio recibido desde Gradio (ruta o dict numpy) y genera transcripción.
    Compatible con Gradio 4.44+ (audio en memoria).
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_input = f"input_{timestamp}.wav"
    tmp_wav = f"tmp_{timestamp}.wav"

    # --- Detectar tipo de entrada ---
    if isinstance(uploaded_path, dict):
        # Gradio >= 4.44 entrega audio en memoria
        data = uploaded_path.get("data")
        sample_rate = uploaded_path.get("sample_rate", 16000)
        if data is None:
            raise ValueError("Audio data missing in input dict.")
        import numpy as np
        from scipy.io.wavfile import write
        data = np.asarray(data, dtype=np.float32)
        write(tmp_input, int(sample_rate), (data * 32767.0).astype(np.int16))
        in_path = tmp_input
        print(f"🎙️ Audio recibido como numpy: {len(data)} muestras @ {sample_rate} Hz")
    elif isinstance(uploaded_path, str) and os.path.exists(uploaded_path):
        in_path = uploaded_path
        print(f"🎙️ Audio recibido como archivo físico: {uploaded_path}")
    else:
        raise ValueError(f"Unsupported audio input type: {type(uploaded_path)}")

    try:
        # --- Convertir a 16 kHz (solo si es necesario) ---
        convert_to_wav_16k(in_path, tmp_wav)
        if not os.path.exists(tmp_wav):
            raise RuntimeError("ffmpeg conversion failed or output missing.")

        print(f"🎧 Iniciando transcripción: {tmp_wav}")
        transcript = transcribe_wav(tmp_wav)
        print("✅ Transcripción completada.")
        comp = compare_texts(reference_text, transcript, nivel_mcer)

        # --- Guardar resultado ---
        with open(RESULT_TXT, "w", encoding="utf-8") as f:
            f.write(f"NIVEL MCER: {comp['nivel']}  (Umbral: {comp['umbral']*100:.0f}%)\n")
            f.write(f"FINAL GRADE % : {comp['final_percent']:.1f}\n")
            f.write(f"FINAL GRADE (0-5): {comp['grade_0_5']:.1f}\n")
            f.write("MISSING WORDS:\n")
            f.write(", ".join(comp['missing'][:200]) + "\n")
            f.write(f"\n{'='*50}\nREFERENCE TEXT:\n{reference_text}\n")
            f.write(f"\n{'='*50}\nTRANSCRIPTION:\n{comp['tr_norm']}\n")
            f.write(f"\nTOTAL WORDS REF: {comp['ref_words_count']}\n")
            f.write(f"WORDS DETECTED: {comp['tr_words_count']}\n")
            f.write(f"WORDS MISSING: {len(comp['missing'])}\n")

        return {
            "nivel": comp["nivel"],
            "transcript": transcript,
            "transcript_norm": comp["tr_norm"],
            "missing": comp["missing"],
            "base_percent": comp["base_percent"],
            "final_percent": comp["final_percent"],
            "grade_0_5": comp["grade_0_5"],
            "result_txt_path": RESULT_TXT
        }

    finally:
        # Limpieza de temporales
        for f in [tmp_input, tmp_wav]:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass

# =============================================================
# 💾 GUARDADO DE RESULTADOS
# =============================================================
def save_result_record(student_name: str, reference_text: str,
                       final_percent: float, grade: float, missing_words: list):
    row = {
        "student": student_name,
        "datetime": datetime.datetime.now().isoformat(sep=' ', timespec='seconds'),
        "reference": reference_text,
        "score_percent": round(final_percent, 1),
        "grade_0_5": grade,
        "missing": "; ".join(missing_words[:200])
    }
    if os.path.exists(RESULTS_CSV):
        try:
            df = pd.read_csv(RESULTS_CSV)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        except Exception:
            df = pd.DataFrame([row])
    else:
        df = pd.DataFrame([row])
    df.to_csv(RESULTS_CSV, index=False, encoding="utf-8")
    return RESULTS_CSV