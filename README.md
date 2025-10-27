# Evaluación de Pronunciación – Despliegue en Render

## Cómo desplegar

1. Ve a [https://render.com](https://render.com)
2. Inicia sesión y haz clic en **New → Web Service**
3. Elige **Manual Deploy → Upload .zip file**
4. Sube este paquete `evaluacion_pronunciacion_render.zip`
5. Configura los siguientes parámetros:
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
6. Haz clic en **Create Web Service** y espera 3–5 minutos.

✅ Cuando el estado cambie a “Live”, Render te dará una URL tipo:
https://evaluacion-pronunciacion.onrender.com

Usa esa URL en Moodle (como “Recurso URL – Ventana emergente”).
