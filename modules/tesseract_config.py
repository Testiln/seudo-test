import os
import platform
import shutil
import pytesseract

# Rutas predeterminadas para Tesseract en diferentes sistemas operativos
DEFAULT_TESSERACT_PATHS = {
    "Windows": [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
    ],
    "Linux": [
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract"
    ],
    "Darwin": [  # macOS
        '/opt/homebrew/bin/tesseract',
        '/usr/local/bin/tesseract'
    ]
}

# Mensajes de error
_ERROR_MESSAGES = {
    "NOT_FOUND": (
        "ADVERTENCIA: No se pudo encontrar el ejecutable de Tesseract para {os_name}.\n"
        "  {install_hint}\n"
        "  Asegúrese de que Tesseract OCR esté instalado y añadido al PATH del sistema,\n"
        "  o proporcione una ruta personalizada si está en una ubicación no estándar."
    ),
    "PYTESSERACT_ERROR": (
        "ERROR: Pytesseract no pudo ejecutar Tesseract correctamente usando la ruta '{path}'.\n"
        "  Esto puede ocurrir si la ruta es correcta pero la instalación de Tesseract está corrupta, es incompleta,\n"
        "  o si el archivo en la ruta no es el ejecutable de Tesseract.\n"
        "  Detalles de Pytesseract: {details}"
    ),
    "UNEXPECTED_ERROR": (
        "ERROR: Ocurrió un error inesperado al intentar verificar Tesseract con la ruta '{path}': {details}"
    )
}

def _find_tesseract_executable(os_name, custom_paths_for_os=None):
    """
    Busca el ejecutable de Tesseract.
    Verifica rutas personalizadas, luego rutas predeterminadas, y finalmente el PATH del sistema.

    Args:
        os_name (str): Nombre del sistema operativo ("Windows", "Linux", "Darwin").
        custom_paths_for_os (list, optional): Lista de rutas personalizadas para el SO actual.

    Returns:
        str or None: La ruta al ejecutable de Tesseract si se encuentra, de lo contrario None.
    """
    paths_to_check = []

    # Verificar si existen rutas personalizadas
    if custom_paths_for_os:
        paths_to_check.extend(custom_paths_for_os)
    
    # Rutas predeterminadas
    paths_to_check.extend(DEFAULT_TESSERACT_PATHS.get(os_name, []))

    for path in paths_to_check:
        if os.path.exists(path):
            return path  # Encontrado en rutas predefinidas/personalizadas

    # Si no se encuentra, intentar con shutil.which (verifica el PATH del sistema)
    executable_name = "tesseract.exe" if os_name == "Windows" else "tesseract"
    found_in_system_path = shutil.which(executable_name)
    if found_in_system_path:
        return found_in_system_path  # Encontrado en el PATH del sistema
    
    return None  # No encontrado

def configure_tesseract(custom_os_paths_config=None):
    """
    Configura la ruta al ejecutable de Tesseract OCR y la verifica.
    Imprime mensajes informativos o de error directamente.

    Args:
        custom_os_paths_config (dict, optional): Un diccionario donde las claves son nombres de SO
                                             ("Windows", "Linux", "Darwin") y los valores son
                                             listas de rutas personalizadas para ese SO.
                                             Ejemplo: {"Windows": [r"D:\Tesseract\tesseract.exe"]}

    Returns:
        bool: True si Tesseract se configuró y verificó correctamente, False en caso contrario.
    """
    current_os = platform.system()
    
    specific_custom_paths = None
    if custom_os_paths_config and current_os in custom_os_paths_config:
        specific_custom_paths = custom_os_paths_config[current_os]

    tesseract_exe_path = _find_tesseract_executable(current_os, custom_paths_for_os=specific_custom_paths)

    if tesseract_exe_path:
        print(f"INFO: Ejecutable de Tesseract encontrado en: '{tesseract_exe_path}'")
        pytesseract.pytesseract.tesseract_cmd = tesseract_exe_path
        
        try:
            version = pytesseract.get_tesseract_version()
            print(f"INFO: Versión de Tesseract OCR ({version}) detectada y configurada correctamente.")
            return True
        except pytesseract.TesseractNotFoundError as e:
            # Indica que tesseract_cmd está configurado, pero Tesseract no funcionó.
            message = _ERROR_MESSAGES["PYTESSERACT_ERROR"].format(path=tesseract_exe_path, details=str(e))
            print(message)
            return False
        except Exception as e:
            # Otros errores durante la verificación de Tesseract.
            message = _ERROR_MESSAGES["UNEXPECTED_ERROR"].format(path=tesseract_exe_path, details=str(e))
            print(message)
            return False
    else:
        # El ejecutable de Tesseract no fue encontrado.
        install_message = "Ejecutable no encontrado. Revise la documentación del proyecto para solucionar el error."
        message = _ERROR_MESSAGES["NOT_FOUND"].format(os_name=current_os, install_hint=install_message)
        print(message)
        return False

# Implemnetación del modulo
if __name__ == "__main__":
    if configure_tesseract():
        print("==> Configuración de Tesseract: ÉXITO.")
    else:
        print("==> Configuración de Tesseract: FALLIDA. Revisa los mensajes anteriores.")