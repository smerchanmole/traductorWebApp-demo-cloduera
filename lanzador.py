import os
import subprocess

# Obtenemos el puerto que CML asigna dinámicamente a la aplicación
port = os.environ.get("CDSW_APP_PORT", "8100")

# Definimos el comando exacto de Streamlit
# Ajustamos la dirección a 127.0.0.1 y el puerto correcto
cmd = [
    "streamlit", 
    "run", 
    "app_hf.py", 
    "--server.port", port, 
    "--server.address", "127.0.0.1"
]

print(f"Iniciando Streamlit en el puerto {port}...")

# Ejecutamos el comando como si fuera una terminal
subprocess.call(cmd)
