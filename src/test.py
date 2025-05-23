import tkinter as tk
from tkinter import ttk
from tkinter import font as tkFont

# --- Variables Globales y de Estilo ---
secciones_info = []
COLOR_BOTON_ACCION_PRINCIPAL_AZUL = "#0078D4" # Un azul estándar (Windows blue)
COLOR_TEXTO_BOTON_AZUL = "white"
COLOR_TEXTO_EXITO = "green"
COLOR_TEXTO_ADVERTENCIA = "#cc5500"
COLOR_GRIS_TEXTO = "#595959" # Gris para texto secundario
COLOR_BOTON_MINIMALISTA_BG = "#f0f0f0" # Similar al fondo para minimalista
COLOR_BOTON_MINIMALISTA_FG = "#333333"
COLOR_BOTON_MINIMALISTA_ACTIVE_BG = "#e0e0e0"

# --- Funciones Auxiliares para Secciones Desplegables ---

def _actualizar_estado_visual_seccion(section_data):
    """Actualiza la UI de la sección (flecha y visibilidad del contenido)."""
    if section_data['is_expanded']:
        section_data['frame_contenido'].pack(fill=tk.X, padx=5, pady=(0, 5), ipady=5)
        section_data['btn_toggle'].config(text=f"▲ {section_data['titulo_texto']}")
    else:
        section_data['frame_contenido'].pack_forget()
        section_data['btn_toggle'].config(text=f"▼ {section_data['titulo_texto']}")

def toggle_contenido(section_data):
    """Maneja el evento de clic en el botón de despliegue/repliegue."""
    section_data['is_expanded'] = not section_data['is_expanded']
    _actualizar_estado_visual_seccion(section_data)

def accion_principal_paso(section_data):
    """
    Acción ejecutada por el botón principal de un paso.
    Deshabilita el botón, repliega la sección actual y abre la siguiente.
    """
    action_btn = section_data.get('action_button_principal')
    if action_btn:
        action_btn.config(state="disabled")

    if section_data['is_expanded']:
        section_data['is_expanded'] = False
        _actualizar_estado_visual_seccion(section_data)

    # Auto-abrir la siguiente sección
    try:
        current_index = secciones_info.index(section_data)
        # Solo auto-abrir si este paso es uno de los que tiene 'action_button_principal'
        # (definido en pasos 1, 2, 3) y no es el último paso en la lista global `secciones_info`.
        if section_data.get('action_button_principal') and (current_index + 1 < len(secciones_info)):
            next_section_data = secciones_info[current_index + 1]
            if not next_section_data['is_expanded']:
                next_section_data['is_expanded'] = True
                _actualizar_estado_visual_seccion(next_section_data)
    except ValueError:
        # Esto no debería ocurrir si section_data siempre proviene de secciones_info
        print("Error: section_data no encontrada en secciones_info durante la progresión.")
        pass


def crear_seccion_desplegable(master, titulo_texto, contenido_callback, estado_inicial=False):
    """Crea una sección desplegable."""
    frame_externo = ttk.Frame(master) # Contenedor principal para un "Paso"
    frame_externo.pack(fill=tk.X, pady=(0, 10), padx=10)

    current_section_data = {} # Diccionario para widgets e info de esta sección

    # Botón para desplegar/replegar, usando el estilo "Toggle.TButton"
    btn_toggle = ttk.Button(
        frame_externo,
        command=lambda: toggle_contenido(current_section_data),
        style="Toggle.TButton"
    )
    btn_toggle.pack(fill=tk.X, ipady=4) # ipady para altura del botón

    # Contenedor para el contenido del paso (la "caja con borde")
    frame_contenido = ttk.Frame(
        frame_externo,
        relief="groove", # Estilo de borde
        borderwidth=2,
        padding=(15, 12, 15, 12) # Margen interno (left, top, right, bottom)
    )

    current_section_data.update({
        'frame_externo': frame_externo,
        'btn_toggle': btn_toggle,
        'frame_contenido': frame_contenido,
        'titulo_texto': titulo_texto,
        'is_expanded': estado_inicial,
        'action_button_principal': None # Referencia al botón principal del paso
    })

    contenido_callback(frame_contenido, current_section_data) # Poblar el contenido
    _actualizar_estado_visual_seccion(current_section_data) # Estado visual inicial

    secciones_info.append(current_section_data)
    return current_section_data

# --- Funciones de Contenido para Cada Paso ---

def contenido_paso1(frame, section_data):
    # "Imagen ... agregada con éxito" - Centrado
    ttk.Label(frame, text="Imagen: \"EJERCICIO 14-MM1K.PNG\" agregada con éxito.",
              foreground=COLOR_TEXTO_EXITO, wraplength=600, justify="center").pack(pady=(0,10), fill=tk.X)

    # Botón "AGREGAR IMAGEN" - Estilo minimalista
    btn_agregar = tk.Button(frame, text="AGREGAR IMAGEN",
                            relief="flat", borderwidth=1,
                            bg=COLOR_BOTON_MINIMALISTA_BG, fg=COLOR_BOTON_MINIMALISTA_FG,
                            activebackground=COLOR_BOTON_MINIMALISTA_ACTIVE_BG, activeforeground=COLOR_BOTON_MINIMALISTA_FG,
                            highlightthickness=0, pady=5) # pady para altura interna
    btn_agregar.pack(pady=(0,2)) # Espacio vertical, centrado por defecto

    # Label "Formato soportado..." - Gris y centrado
    ttk.Label(frame, text="Formato soportado: jpeg, jpg, png, tiff, bmp, webp",
              foreground=COLOR_GRIS_TEXTO, justify="center").pack(pady=(0,10), fill=tk.X)

    # Botón principal del paso - Azul y más ancho
    btn_procesar = tk.Button(frame, text="PROCESAR IMAGEN",
                             relief="raised", borderwidth=1,
                             bg=COLOR_BOTON_ACCION_PRINCIPAL_AZUL, fg=COLOR_TEXTO_BOTON_AZUL,
                             highlightthickness=0, font=('Helvetica', 9, 'bold'))
    btn_procesar.pack(pady=(10,5), fill=tk.X, padx=20, ipady=4) # fill=tk.X, padx para márgenes laterales, ipady para altura

    section_data['action_button_principal'] = btn_procesar
    btn_procesar.config(command=lambda: accion_principal_paso(section_data))

def contenido_paso2(frame, section_data):
    ttk.Label(frame, text="ENUNCIADO 1", font=("Helvetica", 11, "bold")).pack(anchor="w", pady=(0,5))
    texto = ("Los pacientes llegan a la clínica de un médico de acuerdo con una distribución de Poisson a razón de 20 "
             "pacientes por hora. La sala de espera no puede acomodar más de 14 pacientes. El tiempo de consulta por "
             "paciente es exponencial, con una media de 8 minutos.")
    ttk.Label(frame, text=texto, wraplength=600, justify="left").pack(anchor="w", pady=(0,10), fill=tk.X)
    ttk.Label(frame, text="⚠ Si detecta errores en la transcripción del texto, intente subir la imagen en un formato de mejor resolución.",
              foreground=COLOR_TEXTO_ADVERTENCIA, wraplength=600, justify="left").pack(anchor="w", pady=(0,10), fill=tk.X)

    # Botón principal del paso - Azul y más ancho
    btn_extraer = tk.Button(frame, text="EXTRAER PARÁMETROS",
                            relief="raised", borderwidth=1,
                            bg=COLOR_BOTON_ACCION_PRINCIPAL_AZUL, fg=COLOR_TEXTO_BOTON_AZUL,
                            highlightthickness=0, font=('Helvetica', 9, 'bold'))
    btn_extraer.pack(pady=5, fill=tk.X, padx=20, ipady=4)

    section_data['action_button_principal'] = btn_extraer
    btn_extraer.config(command=lambda: accion_principal_paso(section_data))


def contenido_paso3(frame, section_data):
    # Tasa de llegadas
    frame_lambda = ttk.Frame(frame)
    frame_lambda.pack(fill=tk.X, pady=3)
    ttk.Label(frame_lambda, text="Tasa de llegadas (lambda):").pack(side="left", padx=(0,5))
    entry_lambda = ttk.Entry(frame_lambda, width=12) # Un poco más ancho para "Ingrese..."
    entry_lambda.pack(side="left", padx=5)
    entry_lambda.insert(0, "20") # Valor de ejemplo
    combo_unidad_lambda = ttk.Combobox(frame_lambda, values=["pacientes/hora", "pacientes/minuto", "pacientes/día"], width=15, state="readonly")
    combo_unidad_lambda.pack(side="left", padx=5)
    combo_unidad_lambda.current(0)


    # Tiempo de servicio
    frame_mu = ttk.Frame(frame)
    frame_mu.pack(fill=tk.X, pady=3)
    ttk.Label(frame_mu, text="Tiempo de servicio (mu):").pack(side="left", padx=(0,5))
    entry_mu_valor = ttk.Entry(frame_mu, width=12)
    entry_mu_valor.insert(0, "Ingrese...")
    entry_mu_valor.pack(side="left", padx=5)
    combo_unidad_mu = ttk.Combobox(frame_mu, values=["pacientes/hora", "pacientes/minuto", "minutos/paciente", "horas/paciente"], width=15, state="readonly")
    combo_unidad_mu.pack(side="left", padx=5)
    combo_unidad_mu.insert(0, "Unidad") # Placeholder

    # Capacidad del sistema
    frame_k = ttk.Frame(frame)
    frame_k.pack(fill=tk.X, pady=3)
    ttk.Label(frame_k, text="Capacidad del sistema (K):").pack(side="left", padx=(0,5))
    entry_k = ttk.Entry(frame_k, width=15) # Más ancho para "Ingrese valor"
    entry_k.insert(0, "Ingrese valor")
    entry_k.pack(side="left", padx=5)

    # Servidores en paralelo
    frame_s = ttk.Frame(frame)
    frame_s.pack(fill=tk.X, pady=3)
    ttk.Label(frame_s, text="Servidores en paralelo (s):").pack(side="left", padx=(0,5))
    entry_s = ttk.Entry(frame_s, width=15)
    entry_s.insert(0, "Ingrese valor")
    entry_s.pack(side="left", padx=5)

    # Botón principal del paso - Azul y más ancho
    btn_ingresar_modelo = tk.Button(frame, text="INGRESAR DATOS AL MODELO",
                                   relief="raised", borderwidth=1,
                                   bg=COLOR_BOTON_ACCION_PRINCIPAL_AZUL, fg=COLOR_TEXTO_BOTON_AZUL,
                                   highlightthickness=0, font=('Helvetica', 9, 'bold'))
    btn_ingresar_modelo.pack(pady=(15,5), fill=tk.X, padx=20, ipady=4)

    section_data['action_button_principal'] = btn_ingresar_modelo
    btn_ingresar_modelo.config(command=lambda: accion_principal_paso(section_data))

def contenido_paso4(frame, section_data):
    # "Los datos corresponden..." - Título más grande
    ttk.Label(frame, text="Los datos corresponden a un modelo M/M/1/K (rho != 1)",
              foreground=COLOR_TEXTO_EXITO, font=("Helvetica", 12, "bold"), justify="center").pack(pady=(0,10), fill=tk.X)
    ttk.Label(frame, text="Seleccione que desea calcular:", justify="center").pack(pady=(0,8), fill=tk.X)

    frame_botones_calculo = ttk.Frame(frame) # Frame para centrar los botones
    frame_botones_calculo.pack(pady=5)

    # Botones de acción del Paso 4 (no azules por defecto, pueden personalizarse si se desea)
    btn_medidas = tk.Button(frame_botones_calculo, text="Medidas de desempeño",
                            relief="raised", borderwidth=1, pady=3, padx=10,
                            bg=COLOR_BOTON_MINIMALISTA_BG, fg=COLOR_BOTON_MINIMALISTA_FG)
    btn_medidas.pack(side="left", padx=5)

    btn_probabilidad = tk.Button(frame_botones_calculo, text="Probabilidad de \"n\" clientes",
                                 relief="raised", borderwidth=1, pady=3, padx=10,
                                 bg=COLOR_BOTON_MINIMALISTA_BG, fg=COLOR_BOTON_MINIMALISTA_FG)
    btn_probabilidad.pack(side="left", padx=5)

# --- Crear Ventana Principal y Estilos ---
root = tk.Tk()
root.title("TITULO DEL PROYECTO")
root.minsize(700, 750) # Tamaño mínimo para la ventana
root.configure(bg="#f0f0f0") # Color de fondo general

# Estilos ttk
style = ttk.Style()
# Estilo para los botones de toggle (títulos de Paso X)
# Incluye padding (izquierdo para separar flecha), anchor para alinear texto a la izquierda, y fuente.
default_font_family = tkFont.nametofont("TkDefaultFont").actual()["family"]
default_font_size = tkFont.nametofont("TkDefaultFont").actual()["size"]
style.configure("Toggle.TButton",
                padding=(10, 6, 6, 6), # (left, top, right, bottom)
                anchor="w", # Alinea el texto (con la flecha) a la izquierda (West)
                font=(default_font_family, default_font_size + 2, "bold"))

# Título principal del proyecto
titulo_font_size_grande = default_font_size + 6
titulo_label = ttk.Label(root, text="TITULO DEL PROYECTO",
                         font=(default_font_family, titulo_font_size_grande, "bold"),
                         background="#f0f0f0") # Mismo fondo que root
titulo_label.pack(pady=(15, 25))

# --- Crear Pasos Desplegables ---
crear_seccion_desplegable(root, "Paso 1. Adjuntar imagen.", contenido_paso1, estado_inicial=True)
crear_seccion_desplegable(root, "Paso 2. Vista previa del documento WORD #1 generado.", contenido_paso2)
crear_seccion_desplegable(root, "Paso 3. Datos extraídos.", contenido_paso3)
crear_seccion_desplegable(root, "Paso 4. Modelo de teoría de colas.", contenido_paso4)

root.mainloop()