# app.py
import os
import glob
import shutil
import tempfile
import traceback
import mimetypes
import stat
import json
from datetime import datetime, timedelta, timezone

import gradio as gr
import numpy as np
import psutil
from scipy.io.wavfile import write
from gtts import gTTS

from analyzer import evaluate_audio_file, save_result_record, RESULT_TXT

# =============================================================
# 🔒 PROTECCIÓN DE ARCHIVOS INTERNOS (Hugging Face Spaces)
# =============================================================
MODO_PROTEGIDO = False
DATA_DIR = ".internal_data"
TEXTS_DIR = "texts"  # 🔹 vuelve a ser accesible públicamente
USUARIOS_FILE = os.path.join(DATA_DIR, "usuarios.json")

def proteger_archivos_internos():
    """Crea carpeta interna segura para datos confidenciales."""
    global MODO_PROTEGIDO

    # Asegurar carpetas necesarias
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(TEXTS_DIR, exist_ok=True)

    # Detectar ejecución en Hugging Face Spaces
    if os.getenv("SPACE_ID"):
        MODO_PROTEGIDO = True
        print("🔐 Activando modo protegido en Hugging Face Spaces...")

        # Solo proteger los archivos internos sensibles
        carpetas_sensibles = [DATA_DIR, "models", "__pycache__"]
        for c in carpetas_sensibles:
            if os.path.exists(c):
                try:
                    os.chmod(c, 0o700)  # acceso solo al propietario
                    print(f"✔️ Protegida: {c}")
                except Exception as e:
                    print(f"⚠️ No se pudo proteger {c}: {e}")

        print("✅ Modo protegido activado (textos accesibles).")
    else:
        print("ℹ️ Ejecución local: modo protegido desactivado.")

proteger_archivos_internos()


# =============================================================
# 🧠 DETECCIÓN DEL HARDWARE
# =============================================================
def detectar_modo_hardware():
    def _read(path):
        try:
            with open(path, "r") as f:
                return f.read().strip()
        except Exception:
            return None

    mem_gb = None
    mem_v2 = _read("/sys/fs/cgroup/memory.max")
    if mem_v2 and mem_v2 != "max":
        try:
            mem_gb = round(int(mem_v2) / (1024 ** 3))
        except Exception:
            mem_gb = None
    if mem_gb is None:
        mem_v1 = _read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        if mem_v1 and mem_v1 not in ("max",):
            try:
                mem_gb = round(int(mem_v1) / (1024 ** 3))
            except Exception:
                mem_gb = None
    if mem_gb is None:
        mem_gb = round(psutil.virtual_memory().total / (1024 ** 3))

    cpu_limit = None
    cpu_v2 = _read("/sys/fs/cgroup/cpu.max")
    if cpu_v2 and " " in cpu_v2:
        quota, period = cpu_v2.split()
        if quota != "max":
            try:
                cpu_limit = float(quota) / float(period)
            except Exception:
                cpu_limit = None
    if cpu_limit is None:
        q = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        p = _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        try:
            if q and p and int(q) > 0:
                cpu_limit = int(q) / int(p)
        except Exception:
            cpu_limit = None
    if cpu_limit is None:
        cpu_limit = float(psutil.cpu_count(logical=True) or 0)

    if mem_gb >= 28 or cpu_limit >= 6:
        mode = "🟢 Running on CPU upgrade (Pro)"
        color_text = "white"
        color_bg = "#2e7d32"
    elif mem_gb >= 12 or cpu_limit >= 2:
        mode = "⚪ Running on CPU basic (Free)"
        color_text = "black"
        color_bg = "#cccccc"
    else:
        mode = "🟠 Running on custom hardware"
        color_text = "black"
        color_bg = "#f9a825"
    return mode, color_text, color_bg, cpu_limit, mem_gb


# =============================================================
# 🕒 HORA LOCAL DE COLOMBIA + BANNER
# =============================================================
COL_TZ = timezone(timedelta(hours=-5))
hora_colombia = datetime.now(COL_TZ).strftime("%Y-%m-%d %H:%M:%S")
mode_text, color_text, color_bg, cpu_val, mem_val = detectar_modo_hardware()
hardware_status = (
    f"{mode_text} — limits: ~{int(round(cpu_val))} vCPU / {mem_val} GB RAM — "
    f"Started at {hora_colombia} (Colombia time)"
)

# =============================================================
# 👤 SISTEMA DE USUARIOS
# =============================================================
def cargar_usuarios():
    if not os.path.exists(USUARIOS_FILE):
        admin_default = [{"usuario": "Rubén", "clave": "123", "rol": "admin"}]
        with open(USUARIOS_FILE, "w", encoding="utf-8") as f:
            json.dump(admin_default, f, ensure_ascii=False, indent=2)
        return admin_default
    with open(USUARIOS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def guardar_usuarios(data):
    with open(USUARIOS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def registrar_usuario(usuario, clave, rol="student"):
    usuario = usuario.strip()
    if not usuario or not clave:
        return False, "❌ Usuario y contraseña son obligatorios."
    if usuario.lower() in ["admin", "rubén", "ruben"]:
        return False, "❌ Este nombre está reservado. Usa otro."
    data = cargar_usuarios()
    if any(u["usuario"].lower() == usuario.lower() for u in data):
        return False, "❌ El usuario ya existe."
    data.append({"usuario": usuario, "clave": clave, "rol": rol})
    guardar_usuarios(data)
    return True, f"✅ Usuario '{usuario}' registrado correctamente."

def verificar_login(usuario, clave):
    data = cargar_usuarios()
    for u in data:
        if u["usuario"].lower() == usuario.lower() and u["clave"] == clave:
            return True, u["rol"]
    return False, ""


# =============================================================
# ⚙️ UTILIDADES
# =============================================================
ALLOWED_EXTS = (".wav", ".mp3", ".m4a")
MAX_SECONDS = 60
MAX_FILE_MB = 25

def list_texts():
    files = sorted(glob.glob(os.path.join(TEXTS_DIR, "*.txt")))
    return [os.path.basename(f) for f in files]

def load_text(filename):
    if not filename:
        return ""
    with open(os.path.join(TEXTS_DIR, filename), "r", encoding="utf-8") as f:
        return f.read()

def tts_play(text):
    if not text or not text.strip():
        return None
    t = gTTS(text=text, lang="en", tld="com", slow=False)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    t.save(tmp.name)
    return tmp.name

def _validate_filepath(path: str):
    if not isinstance(path, str) or not os.path.exists(path):
        raise ValueError("Audio path inválido.")
    ext = os.path.splitext(path)[1].lower()
    if ext not in ALLOWED_EXTS:
        raise ValueError(f"Formato no permitido ({ext}). Usa: {', '.join(ALLOWED_EXTS)}")
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise ValueError(f"Archivo demasiado grande ({size_mb:.1f} MB). Máximo {MAX_FILE_MB} MB.")

def _duration_seconds_from_array(sample_rate: int, data: np.ndarray) -> float:
    if sample_rate and isinstance(sample_rate, (int, float)) and sample_rate > 0:
        return float(len(data)) / float(sample_rate)
    return 0.0


# =============================================================
# 🔹 EVALUACIÓN
# =============================================================
def process_evaluation(student_name, reference, audio_input, nivel_mcer):
    tmp_wav_created = None
    try:
        if isinstance(audio_input, str):
            _validate_filepath(audio_input)
            audio_path = audio_input
        elif isinstance(audio_input, dict):
            sample_rate = audio_input.get("sample_rate")
            data = np.asarray(audio_input.get("data"), dtype=np.float32)
            dur = _duration_seconds_from_array(sample_rate, data)
            if dur > MAX_SECONDS:
                raise ValueError(f"El audio excede {MAX_SECONDS} s.")
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            write(tmp.name, int(sample_rate), (data * 32767.0).astype(np.int16))
            audio_path = tmp.name
            tmp_wav_created = tmp.name
        else:
            raise ValueError("Formato de audio no reconocido.")

        reference = (reference or "").strip()
        if not reference:
            raise ValueError("No hay texto de referencia.")
        res = evaluate_audio_file(audio_path, reference, nivel_mcer=nivel_mcer)
        res["reference"] = reference
        save_result_record(
            student_name,
            reference,
            res.get("final_percent", 0),
            res.get("grade_0_5", 0),
            res.get("missing", [])
        )
        safe_out = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        shutil.copyfile(res.get("result_txt_path", RESULT_TXT), safe_out.name)
        res["result_txt_path"] = safe_out.name
        if tmp_wav_created and os.path.exists(tmp_wav_created):
            os.remove(tmp_wav_created)
        return res
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "final_percent": "ERROR", "grade_0_5": "ERROR", "missing": "ERROR"}


# =============================================================
# 🎨 INTERFAZ GRADIO CON LOGIN Y REGISTRO
# =============================================================
with gr.Blocks(
    title="English Pronunciation Checker (Vosk)",
    css=f"""
    .hardware-status {{
        background-color: {color_bg};
        color: {color_text};
        font-weight: bold;
        text-align: center;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
        font-size: 16px;
    }}
    .protected-mode {{
        background-color: #1565c0;
        color: white;
        font-weight: bold;
        text-align: center;
        padding: 8px;
        border-radius: 8px;
        margin-bottom: 5px;
        font-size: 15px;
    }}
    """
) as demo:
    login_state = gr.State(False)
    current_user = gr.State("")
    current_role = gr.State("")

    # --- Login ---
    with gr.Group(visible=True) as login_group:
        gr.Markdown("## 🔐 Iniciar sesión")
        username = gr.Textbox(label="Usuario")
        password = gr.Textbox(label="Contraseña", type="password")
        login_btn = gr.Button("Entrar", variant="primary")
        login_msg = gr.Markdown("")
        with gr.Accordion("¿Eres nuevo? Regístrate aquí"):
            reg_user = gr.Textbox(label="Nuevo usuario")
            reg_pass = gr.Textbox(label="Nueva contraseña", type="password")
            reg_btn = gr.Button("Registrar usuario")
            reg_msg = gr.Markdown("")

    # --- App principal ---
    with gr.Group(visible=False) as app_group:
        protected_alert = gr.Markdown(
            "<div class='protected-mode'>🛡️ Modo protegido activado en Hugging Face</div>",
            visible=False
        )
        gr.Markdown(f"<div class='hardware-status'>{hardware_status}</div>")
        gr.Markdown("# 🗣️ English Pronunciation Checker")
        with gr.Row():
            with gr.Column(scale=2):
                student_name = gr.Textbox(label="Student name")
                nivel_mcer = gr.Dropdown(["A1","A2","B1","B2","C1","C2"], label="CEFR level", value="B1")
                choice = gr.Dropdown(choices=list_texts(), label="Select predefined text")
                load_btn = gr.Button("Load selected text")
                custom_text = gr.Textbox(label="Or write your own text", lines=6)
                with gr.Row():
                    tts_btn = gr.Button("🔊 Listen native pronunciation")
                    record = gr.Audio(
                        label=f"🎙️ Record or upload your voice (max. {MAX_SECONDS} seconds)",
                        type="filepath",
                        max_length=MAX_SECONDS,
                        streaming=False
                    )
                eval_btn = gr.Button("✅ Evaluate pronunciation")
                clear_btn = gr.Button("🧹 Clear all fields")
            with gr.Column(scale=1):
                ref_box = gr.Textbox(label="Reference", lines=6, interactive=False)
                result_percent = gr.Textbox(label="Final %", interactive=False)
                result_grade = gr.Textbox(label="Grade (0–5)", interactive=False)
                missing_box = gr.Textbox(label="Missing words", lines=4, interactive=False)
                trans_box = gr.Textbox(label="Transcription", lines=4, interactive=False)
                result_link = gr.File(label="Download result (.txt)")

    # --- Funciones ---
    def do_login(usuario, clave):
        ok, rol = verificar_login(usuario, clave)
        if ok:
            print(f"✅ Usuario '{usuario}' inició sesión como {rol}")
            return True, usuario, rol, gr.update(visible=False), gr.update(visible=True), gr.update(
                visible=(rol == "admin" and MODO_PROTEGIDO)
            ), f"✅ Bienvenido {usuario} ({rol})"
        return False, "", "", gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), "❌ Credenciales inválidas."

    login_btn.click(
        fn=do_login,
        inputs=[username, password],
        outputs=[
            login_state, current_user, current_role,
            login_group, app_group, protected_alert, login_msg
        ]
    )

    def do_register(usuario, clave):
        ok, msg = registrar_usuario(usuario, clave, "student")
        return msg

    reg_btn.click(fn=do_register, inputs=[reg_user, reg_pass], outputs=reg_msg)

    def do_tts(filename, custom):
        text = (custom or "").strip() if (custom or "").strip() else load_text(filename)
        return tts_play(text)

    load_btn.click(fn=load_text, inputs=choice, outputs=custom_text)
    tts_btn.click(fn=do_tts, inputs=[choice, custom_text], outputs=gr.Audio(label="Native Audio"))

    def do_eval(name, filename, custom, audio, nivel):
        r = process_evaluation(name, custom or filename, audio, nivel)
        if r.get("error"):
            return "", "", "", "", "", "", None
        ref_text = (custom or "").strip() if (custom or "").strip() else load_text(filename)
        missing_text = ", ".join(r["missing"]) if isinstance(r["missing"], list) else str(r["missing"])
        return (
            ref_text,
            f"{r['final_percent']:.2f}",
            f"{r['grade_0_5']:.2f}",
            missing_text,
            r.get("transcript_norm", ""),
            "",
            r["result_txt_path"]
        )

    eval_btn.click(
        fn=do_eval,
        inputs=[student_name, choice, custom_text, record, nivel_mcer],
        outputs=[ref_box, result_percent, result_grade, missing_box, trans_box, gr.Textbox(), result_link]
    )

    def clear_all():
        return "", "", "", "", "", "", None

    clear_btn.click(fn=clear_all,
        inputs=None,
        outputs=[ref_box, result_percent, result_grade, missing_box, trans_box, custom_text, result_link])

# =============================================================
# 🚀 EJECUCIÓN COMPATIBLE CON RENDER (Gradio 4.x)
# =============================================================
if __name__ == "__main__":
    import os

    port = int(os.getenv("PORT", 7860))
    host = "0.0.0.0"

    print(f"🌍 Ejecutando en entorno Render en {host}:{port} ...")

    demo.launch(
        server_name=host,
        server_port=port,
        share=True,          # Necesario en Render; crea enlace interno pero lo ignora si ya hay proxy
        show_error=True,
        debug=False,
        inbrowser=False,     # No intenta abrir navegador
        prevent_thread_lock=True  # Evita bloqueo ASGI
    )