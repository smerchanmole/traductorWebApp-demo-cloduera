![Traductor](traductor.png)

# 🎙️ Cloudera GenAI Demo: Traductor y Lector Universal (100% Offline)

Esta aplicación es una demostración de Inteligencia Artificial Generativa diseñada para ejecutarse íntegramente de forma local (**Air-gapped / 100% Offline**) dentro de **Cloudera Machine Learning (CML)**. 

La herramienta permite grabar un audio con el micrófono, transcribirlo, traducirlo a múltiples idiomas y, finalmente, leer la traducción en voz alta generando un audio sintético de alta calidad. 

Para lograrlo respetando límites de memoria estrictos (entornos de 8GB RAM + GPU V100), la aplicación encadena tres modelos fundacionales ligeros:
1. **OpenAI Whisper (Small):** Transcripción de audio a texto.
2. **Meta NLLB (Distilled 600M):** Traducción neuronal masiva.
3. **Meta MMS-TTS:** Síntesis de voz (Text-to-Speech) multilingüe.

---

## 📂 Estructura del Proyecto

El proyecto se compone estrictamente de 4 ficheros que deben estar en la raíz de tu proyecto en CML:

* `app.py`: El código principal de la interfaz gráfica desarrollada con Streamlit y la lógica de inferencia de PyTorch.
* `lanzador.py`: El script de arranque de CML. Se encarga de instalar las dependencias previas (`pip install -r requirements.txt`) y de levantar el servidor web de Streamlit en el puerto correcto.
* `requirements.txt`: Listado de librerías de Python necesarias y sus versiones fijadas.
* `traductor.png`: Imagen de cabecera que se muestra en la interfaz gráfica.

---

## ⚠️ IMPORTANTE: El Infierno de las Dependencias (Dependency Pinning)

Para garantizar que esta aplicación funcione siempre, independientemente de la fecha en la que se despliegue, **es crítico respetar las versiones del fichero `requirements.txt`**.

En concreto, la librería **`transformers` DEBE ser exactamente la versión `4.57.6`**. 
Versiones superiores de Hugging Face introdujeron validadores estrictos en la clase `pipeline` que rompen la inicialización del modelo de traducción NLLB, causando el error *`Invalid translation task`*.

Tu archivo `requirements.txt` debe contener, como mínimo, lo siguiente:

    streamlit
    torch
    transformers==4.57.6
    scipy


---

## 🚀 Guía de Despliegue en Cloudera AI (CML)

Sigue estos pasos para crear y levantar la aplicación de forma correcta en tu entorno corporativo:

### Paso 1: Crear el Proyecto y subir los ficheros
1. Accede a tu entorno de **Cloudera Machine Learning**.
2. Haz clic en **New Project** (Nuevo Proyecto) y dale un nombre (ej. *GenAI Traductor Offline*).
3. Una vez dentro del proyecto, ve a la sección **Files** y sube los 4 ficheros mencionados anteriormente (`app.py`, `lanzador.py`, `requirements.txt`, `traductor.png`).

### Paso 2: Crear la "Application" (Interfaz Web)
1. En el menú izquierdo de tu proyecto, haz clic en **Applications**.
2. Haz clic en el botón azul **New Application**.
3. Rellena el formulario de configuración:
   * **Name:** Traductor AI
   * **Subdomain:** traductor-ai (o el que prefieras, será parte de la URL).
   * **Script:** Escribe `lanzador.py` (Este es el archivo que CML ejecutará al arrancar).
   * **Environment Variables:** No es necesario configurar Proxys ya que los modelos funcionarán offline una vez descargados en la caché local.

### Paso 3: Asignar Recursos (Resource Profile)
Es vital elegir un perfil (Runtime) que soporte ejecución por GPU y tenga la memoria suficiente para la descarga inicial:
* **Editor / Runtime:** Selecciona un Runtime nativo de PyTorch o un *Standard Engine* con soporte para GPU y Python 3.9 o superior.
* **CPU:** 2 o 4 vCPUs.
* **Memoria RAM:** Mínimo **8 GB**.
* **GPU:** **1** (Ej. Nvidia V100, T4, L40 o similar).

### Paso 4: Lanzar y Disfrutar
1. Haz clic en **Create Application**.
2. Cloudera levantará el contenedor, ejecutará el script `lanzador.py` (que instalará las dependencias respetando el `requirements.txt`) y mostrará el enlace a tu aplicación Streamlit.
3. *Nota:* La primera vez que selecciones un idioma, el sistema tardará unos segundos extra en descargar los modelos ligeros de la IA a la caché de Cloudera. En las siguientes ejecuciones, la respuesta será inmediata.

---

## 🛠️ Troubleshooting (Solución de problemas frecuentes)

* **Error "NoneType object has no attribute 'replace'":** Este error ocurre si la memoria RAM colapsó en un despliegue anterior y corrompió el archivo del modelo. Para solucionarlo, borra la carpeta oculta de Hugging Face en CML accediendo por terminal y ejecutando: `rm -rf ~/.cache/huggingface/hub`.
* **La aplicación no carga o el micrófono no responde:** Asegúrate de que tu token JWT (sesión de Cloudera) no ha caducado. Refresca toda la pestaña de tu navegador (F5).