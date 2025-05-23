# src/nlp_pipeline.py

import spacy
from spacy.matcher import Matcher
from sentence_transformers import SentenceTransformer
import json
import os
import re

# --- Definición de Frases Clave para Parámetros ---
FRASES_CLAVE_PARAMETROS = {
    "llegada": [
        "tasa de llegada", "llegan clientes", "arriban unidades", "frecuencia de entrada",
        "tiempo entre llegadas", "llega un cliente cada", "intervalo entre arribos"
    ],
    "servicio": [
        "tasa de servicio", "capacidad de atención", "tiempo de servicio", "demora en atender",
        "procesa unidades", "atiende clientes", "tarda en procesar"
    ],
    "servidores": [
        "número de servidores", "cantidad de cajeros", "puestos de atención", "operarios disponibles",
        "cuántas máquinas hay", "dispone de servidores"
    ],
    "capacidad_sistema": [
        "capacidad del sistema", "tamaño de la cola", "límite de clientes en el sistema",
        "espacio en la sala de espera", "cuántos caben como máximo", "capacidad de k",
        "capacidad es de", "capacidad máxima", "rechaza clientes si hay más de",
        "solo se permiten", "buffer de tamaño"
    ],
    "disciplina_cola": [
        "disciplina de la cola", "orden de atención", "tipo de cola", "cómo se atiende",
        "FIFO", "LIFO", "primero en llegar primero en ser servido", "atención por prioridad",
        "orden de llegada"
    ]
}

# --- Model Loading ---
NLP_SPACY = None
MODEL_SENTENCE_TRANSFORMERS = None
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def cargar_modelos():
    global NLP_SPACY, MODEL_SENTENCE_TRANSFORMERS
    print("Cargando modelos NLP...")
    try:
        NLP_SPACY = spacy.load("es_core_news_sm")
        MODEL_SENTENCE_TRANSFORMERS = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("Modelos cargados exitosamente.")
    except Exception as e:
        print(f"Error al cargar modelos: {e}")
        NLP_SPACY = None
        MODEL_SENTENCE_TRANSFORMERS = None

def inicializar_estructura_salida():
    return {
        "texto_original": None,
        "parametros_extraidos": {
            "tasa_llegada": {"valor": None, "unidades": None, "distribucion": None, "fragmento_texto": None},
            "tiempo_entre_llegadas": {"valor": None, "unidades": None, "distribucion": None, "fragmento_texto": None},
            "tasa_servicio_por_servidor": {"valor": None, "unidades": None, "distribucion": None, "fragmento_texto": None},
            "tiempo_servicio_por_servidor": {"valor": None, "unidades": None, "distribucion": None, "fragmento_texto": None},
            "cantidad_servidores": {"valor": None, "fragmento_texto": None},
            "capacidad_sistema": {"valor": None, "fragmento_texto": None},
            "disciplina_cola": {"valor": None, "fragmento_texto": None} 
        },
        "errores": []
    }

def procesar_texto_basico(texto_entrada):
    if NLP_SPACY is None:
        resultado = inicializar_estructura_salida()
        resultado["errores"].append("El modelo de spaCy no está cargado.")
        return None, resultado
    doc_spacy = NLP_SPACY(texto_entrada)
    resultado = inicializar_estructura_salida()
    resultado["texto_original"] = texto_entrada
    return doc_spacy, resultado

# --- Módulos de Extracción Específicos ---

def extraer_numero_servidores(doc_spacy, resultado_parcial):
    if NLP_SPACY is None: 
        resultado_parcial["errores"].append("Modelo spaCy no cargado en extraer_numero_servidores.")
        return resultado_parcial

    matcher = Matcher(NLP_SPACY.vocab)
    keywords_servidor = ["servidor", "cajero", "máquina", "operador", "ventanilla", "puesto", "estación", "médico", "doctor", "enfermero", "consultorio", "línea de ensamblaje", "caja"]

    pattern1 = [{"LIKE_NUM": True}, {"LEMMA": {"IN": keywords_servidor}}]
    pattern2 = [{"LOWER": {"IN": ["un", "una"]}}, {"LEMMA": {"IN": keywords_servidor}}]
    pattern3 = [
        {"LEMMA": {"IN": ["haber", "existir", "tener", "contar con", "disponer de", "operar con"]}},
        {"LIKE_NUM": True, "OP": "?"}, 
        {"LOWER": {"IN": ["un", "una"]}, "OP": "?"},
        {"LEMMA": {"IN": keywords_servidor}}
    ]
    
    matcher.add("NUM_SERVIDORES_PATTERN1", [pattern1])
    matcher.add("NUM_SERVIDORES_PATTERN2", [pattern2])
    matcher.add("NUM_SERVIDORES_PATTERN3", [pattern3])

    matches = matcher(doc_spacy)
    found_values = [] 

    for match_id, start, end in matches:
        span = doc_spacy[start:end]
        num_val = None
        potential_num_token = None
        
        if NLP_SPACY.vocab.strings[match_id] == "NUM_SERVIDORES_PATTERN2":
            num_val = 1
            potential_num_token = span[0] 
        else:
            for token_idx_in_span, token in enumerate(span):
                if token.like_num:
                    num_map = {"uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5} 
                    if token.text.lower() in num_map:
                        num_val = num_map[token.text.lower()]
                    else:
                        try:
                            num_val = int(float(token.text.replace(",", ".")))
                        except ValueError:
                            continue 
                    potential_num_token = token
                    break 
                elif token.lower_ in ["un", "una"] and NLP_SPACY.vocab.strings[match_id] == "NUM_SERVIDORES_PATTERN3":
                    is_quantifier = False
                    for next_token_in_span_idx in range(token_idx_in_span + 1, len(span)):
                        if span[next_token_in_span_idx].lemma_ in keywords_servidor:
                            is_quantifier = True
                            break
                    if is_quantifier:
                        num_val = 1
                        potential_num_token = token
                        break

        if num_val is not None and num_val > 0:
            fragmento = potential_num_token.sent.text if potential_num_token else span.sent.text
            found_values.append({"valor": num_val, "fragmento": fragmento, "span_text": span.text})

    if not found_values:
        print("INFO: No se encontró información explícita sobre el número de servidores.")
    else:
        primary_value = found_values[0]["valor"]
        primary_fragment = found_values[0]["fragmento"]
        
        resultado_parcial["parametros_extraidos"]["cantidad_servidores"]["valor"] = primary_value
        resultado_parcial["parametros_extraidos"]["cantidad_servidores"]["fragmento_texto"] = primary_fragment
        print(f"INFO: Número de servidores extraído: {primary_value} (Fuente: '{primary_fragment}')")

        unique_numeric_values = set(item["valor"] for item in found_values)
        if len(unique_numeric_values) > 1:
            advertencia = (f"Advertencia: Se encontraron múltiples valores diferentes para el número de servidores. "
                           f"Se utilizó el primero ({primary_value} de '{primary_fragment}'). Otros valores encontrados: ")
            otros_valores_str_list = [f"{item['valor']} (en \"{item['span_text']}\" de la oración \"{item['fragmento']}\")" 
                                      for item in found_values if item["valor"] != primary_value]
            advertencia += "; ".join(otros_valores_str_list)
            resultado_parcial["errores"].append(advertencia)
            print(f"ADVERTENCIA DETALLADA: {advertencia}")
    return resultado_parcial

def extraer_capacidad_sistema(doc_spacy, resultado_parcial):
    if NLP_SPACY is None:
        resultado_parcial["errores"].append("Modelo spaCy no cargado en extraer_capacidad_sistema.")
        return resultado_parcial

    matcher = Matcher(NLP_SPACY.vocab)
    keywords_cap = ["capacidad", "límite", "tamaño", "espacio", "máximo", "buffer"]
    keywords_entidad = ["cliente", "persona", "unidad", "auto", "puesto", "elemento", "paciente", "trabajo", "item"]
    keywords_lugar_espera = ["cola", "sala de espera", "almacén", "buffer"]

    pattern_cap_num1 = [
        {"LEMMA": {"IN": keywords_cap}},
        {"LOWER": {"IN": ["del", "de la"]}, "OP": "?"},
        {"LOWER": {"IN": ["sistema", "total"] + keywords_lugar_espera}, "OP": "?"},
        {"LOWER": "es", "OP": "?"},
        {"LOWER": "de", "OP": "?"},
        {"LOWER": {"IN": ["para", "en"]}, "OP": "?"},
        {"LIKE_NUM": True},
        {"LEMMA": {"IN": keywords_entidad}, "OP": "?"}
    ]
    pattern_cap_num2 = [
        {"LIKE_NUM": True},
        {"LEMMA": {"IN": keywords_entidad}, "OP": "+"}, 
        {"LOWER": {"IN": ["de", "en", "para"]}, "OP": "+"}, 
        {"LEMMA": {"IN": ["capacidad"] + keywords_lugar_espera + ["sistema"]}} 
    ]
    pattern_caben = [
        {"LEMMA": {"IN": ["caber", "acomodar"]}},
        {"LIKE_NUM": True},
        {"LEMMA": {"IN": keywords_entidad}, "OP": "?"},
        {"LOWER": "en", "OP": "?"},
        {"LOWER": {"IN": ["la", "el"]}, "OP": "?"},
        {"LEMMA": {"IN": keywords_lugar_espera}}
    ]
    pattern_no_mas_de = [
        {"LOWER": "no", "OP": "?"},
        {"LEMMA": {"IN": ["poder", "aceptar", "admitir", "acomodar"]}, "OP": "+"},
        {"LOWER": "más"},
        {"LOWER": "de"},
        {"LIKE_NUM": True},
        {"LEMMA": {"IN": keywords_entidad}, "OP": "?"}
    ]
    pattern_k_igual = [{"TEXT": {"IN": ["K", "k"]}}, {"LOWER": {"IN": ["=", "es"]}, "OP": "?"}, {"LIKE_NUM": True}]
    pattern_rechazo = [
        {"LEMMA": {"IN": ["rechazar", "no admitir", "no aceptar", "dejar de aceptar"]}},
        {"IS_PUNCT": False, "OP": "{0,5}"}, 
        {"LOWER": "si"},
        {"LOWER": "hay"},
        {"LOWER": {"IN": ["más", "mas"]}},
        {"LOWER": "de"},
        {"LIKE_NUM": True},
        {"LEMMA": {"IN": keywords_entidad}, "OP": "?"}
    ]
    pattern_solo_permiten = [
        {"LOWER": {"IN": ["solo", "solamente", "únicamente"]}},
        {"LOWER": {"IN": ["se", "puede", "pueden"]}, "OP": "?"},
        {"LEMMA": {"IN": ["permitir", "caber", "almacenar", "haber", "tener", "aceptar"]}, "OP": "+"},
        {"LIKE_NUM": True},
        {"LEMMA": {"IN": keywords_entidad}, "OP": "?"}
    ]
    pattern_infinita_directa = [{"LOWER": {"IN": ["capacidad", "límite"]}, "OP": "?"}, {"LOWER": {"IN": ["infinita", "ilimitada"]}}]
    pattern_no_limite = [{"LOWER": {"IN": ["no", "sin"]}}, {"LOWER": {"IN": ["hay", "tiene", "existe"]}, "OP": "?"}, {"LEMMA": {"IN": ["límite", "restricción"]}}]

    matcher.add("CAP_NUM1", [pattern_cap_num1])
    matcher.add("CAP_NUM2", [pattern_cap_num2])
    matcher.add("CAP_CABEN", [pattern_caben])
    matcher.add("CAP_NO_MAS_DE", [pattern_no_mas_de])
    matcher.add("CAP_K_IGUAL", [pattern_k_igual])
    matcher.add("CAP_RECHAZO", [pattern_rechazo])
    matcher.add("CAP_SOLO_PERMITEN", [pattern_solo_permiten])
    matcher.add("CAP_INFINITA", [pattern_infinita_directa])
    matcher.add("CAP_NO_LIMITE", [pattern_no_limite])

    matches = matcher(doc_spacy)
    found_capacities = []

    for match_id, start, end in matches:
        span = doc_spacy[start:end]
        match_name = NLP_SPACY.vocab.strings[match_id]
        current_value = None
        current_fragment = span.sent.text 
        match_type = "general_numeric" 

        if match_name in ["CAP_INFINITA", "CAP_NO_LIMITE"]:
            current_value = "infinita"
            match_type = "infinite"
        else: 
            for token in span:
                if token.like_num:
                    try:
                        num_map = {"uno": 1, "dos": 2, "tres": 3}
                        if token.text.lower() in num_map:
                            current_value = num_map[token.text.lower()]
                        else:
                            current_value = int(float(token.text.replace(",", ".")))
                        break 
                    except ValueError:
                        continue
            
            if current_value is not None:
                sentence_text_lower = current_fragment.lower()
                if match_name in ["CAP_RECHAZO", "CAP_SOLO_PERMITEN", "CAP_K_IGUAL", "CAP_NO_MAS_DE"]:
                    match_type = "system_direct_k" 
                    if any(kw_lugar in sentence_text_lower for kw_lugar in keywords_lugar_espera) and \
                       not any(kw_sys in sentence_text_lower for kw_sys in ["sistema", "total"]):
                            match_type = "queue_explicit" 
                elif match_name in ["CAP_NUM1", "CAP_NUM2", "CAP_CABEN"]:
                    if any(kw in sentence_text_lower for kw in ["sistema", "total"]):
                        match_type = "system_explicit"
                    elif any(kw_lugar in sentence_text_lower for kw_lugar in keywords_lugar_espera) or match_name == "CAP_CABEN":
                        match_type = "queue_explicit"
        
        if current_value is not None:
            if isinstance(current_value, (int, float)) and current_value <= 0:
                continue
            found_capacities.append({
                "valor": current_value, 
                "fragmento": current_fragment, 
                "span_text": span.text, 
                "match_name": match_name,
                "match_type": match_type
            })

    if not found_capacities:
        print("INFO: No se encontró info de capacidad. Asumiendo infinita por defecto.")
        resultado_parcial["parametros_extraidos"]["capacidad_sistema"]["valor"] = "infinita"
        resultado_parcial["parametros_extraidos"]["capacidad_sistema"]["fragmento_texto"] = "Asumida infinita (no encontrada explícitamente)."
    else:
        chosen_cap = None
        priority_order = ["system_direct_k", "system_explicit", "general_numeric", "queue_explicit", "infinite"]
        
        for p_type in priority_order:
            caps_of_type = [fc for fc in found_capacities if fc["match_type"] == p_type]
            if caps_of_type:
                chosen_cap = caps_of_type[0] 
                break
        
        if chosen_cap:
            primary_value = chosen_cap["valor"]
            primary_fragment = chosen_cap["fragmento"]
            resultado_parcial["parametros_extraidos"]["capacidad_sistema"]["valor"] = primary_value
            resultado_parcial["parametros_extraidos"]["capacidad_sistema"]["fragmento_texto"] = primary_fragment
            print(f"INFO: Capacidad del sistema extraída: {primary_value} (Fuente: '{primary_fragment}', Patrón: {chosen_cap['match_name']}, Tipo: {chosen_cap['match_type']})")

            numeric_caps_all = [fc for fc in found_capacities if isinstance(fc["valor"], (int, float))]
            all_found_numeric_values = set(nc["valor"] for nc in numeric_caps_all)

            if len(all_found_numeric_values) > 1 and isinstance(primary_value, (int, float)):
                adv = (f"Advertencia: Múltiples valores numéricos para capacidad. Usado: {primary_value} (Tipo: {chosen_cap['match_type']}). "
                       f"Otros detectados: {[f'{nc["valor"]} (Tipo: {nc["match_type"]})' for nc in found_capacities if isinstance(nc['valor'], (int,float)) and nc['valor'] != primary_value]}")
                resultado_parcial["errores"].append(adv)
                print(f"ADVERTENCIA: {adv}")
            
            if numeric_caps_all and any(fc["match_type"] == "infinite" for fc in found_capacities) and isinstance(primary_value, (int, float)):
                adv = (f"Advertencia: Se encontró capacidad numérica ({primary_value}) y mención de 'infinita'. "
                       f"Se priorizó el valor numérico basado en el tipo de patrón ({chosen_cap['match_type']}).")
                resultado_parcial["errores"].append(adv)
                print(f"ADVERTENCIA: {adv}")
        else: 
            print("INFO: No se pudo determinar la capacidad explícita con prioridades. Asumiendo infinita por defecto.")
            resultado_parcial["parametros_extraidos"]["capacidad_sistema"]["valor"] = "infinita"
            resultado_parcial["parametros_extraidos"]["capacidad_sistema"]["fragmento_texto"] = "Asumida infinita (lógica de selección)."
            
    return resultado_parcial

def extraer_disciplina_cola(doc_spacy, resultado_parcial):
    """
    Extrae la disciplina de la cola del texto.
    Busca siglas y frases completas en español e inglés.
    Si no se encuentra, el valor será None.
    """
    if NLP_SPACY is None:
        resultado_parcial["errores"].append("Modelo spaCy no cargado en extraer_disciplina_cola.")
        return resultado_parcial

    matcher = Matcher(NLP_SPACY.vocab)
    disciplinas_encontradas = []

    # FIFO
    pattern_fifo_sigla = [{"TEXT": "FIFO"}]
    pattern_fifo_frase_es = [
        {"LOWER": "primero"}, {"LOWER": "en"}, {"LOWER": "llegar"}, {"IS_PUNCT": True, "OP": "?"},
        {"LOWER": "primero"}, {"LOWER": "en"}, {"LOWER": "ser"}, {"LOWER": "servido"}
    ]
    pattern_fifo_frase_es_alt = [ 
        {"LOWER": "primero"}, {"LOWER": "que"}, {"LOWER": "llega"},
        {"LOWER": "es"}, {"LOWER": "el"}, {"LOWER": "primero"}, {"LOWER": "que"}, {"LOWER": "se"}, {"LOWER": "atiende"}
    ]
    pattern_fifo_fcfs = [{"TEXT": "FCFS"}] 
    pattern_orden_llegada = [ # Para "orden de llegada", "en el orden en que van llegando/ingresando"
        {"LOWER": "orden"}, {"LOWER": "de"}, {"LOWER": "llegada"}
    ]
    pattern_orden_ingreso = [
        {"LOWER": "orden"}, {"LOWER": "en"}, {"LOWER": "que"}, 
        {"LEMMA": {"IN": ["ir", "llegar", "ingresar"]}} # "van llegando", "van ingresando"
    ]


    # LIFO
    pattern_lifo_sigla = [{"TEXT": "LIFO"}]
    pattern_lifo_frase_es = [
        {"LOWER": "último"}, {"LOWER": "en"}, {"LOWER": "llegar"}, {"IS_PUNCT": True, "OP": "?"},
        {"LOWER": "primero"}, {"LOWER": "en"}, {"LOWER": "ser"}, {"LOWER": "servido"}
    ]
    pattern_lifo_lcfs = [{"TEXT": "LCFS"}]

    # SIRO 
    pattern_siro_sigla = [{"TEXT": "SIRO"}]
    pattern_siro_frase_es = [
        {"LOWER": "servicio"}, {"LOWER": "en"}, {"LOWER": "orden"}, {"LOWER": "aleatorio"}
    ]
    pattern_siro_random = [
        {"LOWER": "atención"}, {"LOWER": "aleatoria"}
    ]
    
    # Prioridad (PRI)
    pattern_prioridad_sigla = [{"TEXT": "PRI"}]
    pattern_prioridad_frase = [
        {"LEMMA": {"IN": ["prioridad", "prioritario"]}} 
    ]
    pattern_prioridad_clases = [ 
        {"LEMMA": {"IN": ["clase", "tipo", "nivel"]}},
        {"LOWER": "de", "OP": "?"},
        {"LEMMA": "prioridad"}
    ]

    matcher.add("DISC_FIFO_SIGLA", [pattern_fifo_sigla])
    matcher.add("DISC_FIFO_FRASE_ES", [pattern_fifo_frase_es])
    matcher.add("DISC_FIFO_FRASE_ES_ALT", [pattern_fifo_frase_es_alt])
    matcher.add("DISC_FIFO_FCFS", [pattern_fifo_fcfs])
    matcher.add("DISC_ORDEN_LLEGADA", [pattern_orden_llegada])
    matcher.add("DISC_ORDEN_INGRESO", [pattern_orden_ingreso])


    matcher.add("DISC_LIFO_SIGLA", [pattern_lifo_sigla])
    matcher.add("DISC_LIFO_FRASE_ES", [pattern_lifo_frase_es])
    matcher.add("DISC_LIFO_LCFS", [pattern_lifo_lcfs])

    matcher.add("DISC_SIRO_SIGLA", [pattern_siro_sigla])
    matcher.add("DISC_SIRO_FRASE_ES", [pattern_siro_frase_es])
    matcher.add("DISC_SIRO_RANDOM", [pattern_siro_random])
    
    matcher.add("DISC_PRIORIDAD_SIGLA", [pattern_prioridad_sigla])
    matcher.add("DISC_PRIORIDAD_FRASE", [pattern_prioridad_frase])
    matcher.add("DISC_PRIORIDAD_CLASES", [pattern_prioridad_clases])

    matches = matcher(doc_spacy)

    for match_id, start, end in matches:
        span = doc_spacy[start:end]
        rule_id_str = NLP_SPACY.vocab.strings[match_id]
        disciplina_detectada = None

        if "FIFO" in rule_id_str or "FCFS" in rule_id_str or "ORDEN_LLEGADA" in rule_id_str or "ORDEN_INGRESO" in rule_id_str:
            disciplina_detectada = "FIFO"
        elif "LIFO" in rule_id_str or "LCFS" in rule_id_str:
            disciplina_detectada = "LIFO"
        elif "SIRO" in rule_id_str or "RANDOM" in rule_id_str:
            disciplina_detectada = "SIRO"
        elif "PRIORIDAD" in rule_id_str:
            disciplina_detectada = "Prioridad"
        
        if disciplina_detectada:
            disciplinas_encontradas.append({
                "valor": disciplina_detectada,
                "fragmento": span.sent.text,
                "span_text": span.text
            })

    if not disciplinas_encontradas:
        print("INFO: No se encontró disciplina de cola explícita. Se establece como no especificado.")
        resultado_parcial["parametros_extraidos"]["disciplina_cola"]["valor"] = None # O "No especificado"
        resultado_parcial["parametros_extraidos"]["disciplina_cola"]["fragmento_texto"] = None
    else:
        chosen_discipline = disciplinas_encontradas[0] 
        primary_value = chosen_discipline["valor"]
        primary_fragment = chosen_discipline["fragmento"]

        resultado_parcial["parametros_extraidos"]["disciplina_cola"]["valor"] = primary_value
        resultado_parcial["parametros_extraidos"]["disciplina_cola"]["fragmento_texto"] = primary_fragment
        print(f"INFO: Disciplina de cola extraída: {primary_value} (Fuente: '{primary_fragment}')")

        unique_disciplines = set(d["valor"] for d in disciplinas_encontradas)
        if len(unique_disciplines) > 1:
            advertencia = (f"Advertencia: Se encontraron múltiples menciones de disciplinas de cola diferentes. "
                           f"Se utilizó la primera detectada ('{primary_value}' de '{primary_fragment}'). "
                           f"Otras detectadas: {[f'{d["valor"]} (en \"{d["span_text"]}\")' for d in disciplinas_encontradas if d["valor"] != primary_value]}")
            resultado_parcial["errores"].append(advertencia)
            print(f"ADVERTENCIA DETALLADA: {advertencia}")
            
    return resultado_parcial


# --- Función Principal de Extracción ---
def extraer_parametros_colas(texto_entrada):
    if NLP_SPACY is None or MODEL_SENTENCE_TRANSFORMERS is None:
        cargar_modelos() 
        if NLP_SPACY is None or MODEL_SENTENCE_TRANSFORMERS is None:
            res_error = inicializar_estructura_salida()
            res_error["errores"].append("Fallo crítico al cargar modelos NLP.")
            return res_error

    doc_spacy, resultado_parcial = procesar_texto_basico(texto_entrada)
    if doc_spacy is None: return resultado_parcial 

    resultado_parcial = extraer_numero_servidores(doc_spacy, resultado_parcial)
    resultado_parcial = extraer_capacidad_sistema(doc_spacy, resultado_parcial) 
    resultado_parcial = extraer_disciplina_cola(doc_spacy, resultado_parcial)
    return resultado_parcial

if __name__ == "__main__":
    cargar_modelos() 

    textos_prueba_disciplina = [
        "Los clientes son atendidos según la disciplina FIFO.",
        "El sistema opera con una política LIFO.",
        "Se sigue un orden de primero en llegar, primero en ser servido.",
        "La atención se da al último en llegar, primero en ser servido.",
        "El servicio es en orden aleatorio (SIRO).",
        "Los trabajos urgentes tienen prioridad.",
        "Existen diferentes clases de prioridad para los usuarios.",
        "Se utiliza FCFS para la gestión de la cola.",
        "No se especifica el orden, pero los clientes VIP son prioritarios sobre el resto que sigue FIFO.",
        "El medico atiende a los pacientes en el orden en que van ingresando.", # Nuevo caso de prueba
        "La política de atención es por orden de llegada." # Nuevo caso de prueba
    ]

    for i, texto_test in enumerate(textos_prueba_disciplina):
        print(f"\n--- Procesando Texto de Prueba Disciplina #{i+1}: '{texto_test}' ---")
        if NLP_SPACY and MODEL_SENTENCE_TRANSFORMERS: 
            info_extraida = extraer_parametros_colas(texto_test)
            print("\n--- Resultado Extracción (JSON) ---")
            print(json.dumps(info_extraida, indent=4, ensure_ascii=False))
            print("-----------------------------------\n")
        else:
            print("Modelos no cargados, saltando prueba.")
    
    print("\n--- Prueba con archivo ejemplo1.txt (si existe) ---")
    ruta_ejemplo = os.path.join(DATA_DIR, "ejemplo1.txt")
    # Contenido sugerido para data/ejemplo1.txt:
    # "Los pacientes llegan a la clínica de un médico a razón de 20 por hora. La sala de espera no puede acomodar más de 14 pacientes. Hay un solo médico. El medico atiende a los pacientes en el orden en que van ingresando."
    try:
        with open(ruta_ejemplo, 'r', encoding='utf-8') as f:
            texto_ejemplo_regresion = f.read()
        print(f"--- Procesando: {ruta_ejemplo} ---")
        print(texto_ejemplo_regresion)
        print("------------------------------------")
        if NLP_SPACY and MODEL_SENTENCE_TRANSFORMERS:
            info_extraida_regresion = extraer_parametros_colas(texto_ejemplo_regresion)
            print("\n--- Resultado (JSON) ---")
            print(json.dumps(info_extraida_regresion, indent=4, ensure_ascii=False))
            print("--------------------------\n")
    except FileNotFoundError:
        print(f"Archivo {ruta_ejemplo} no encontrado. Crea uno con ejemplos para probar todos los parámetros.")
    except Exception as e:
        print(f"Error con {ruta_ejemplo}: {e}")






# src/nlp_pipeline.py

import spacy
from spacy.matcher import Matcher
from sentence_transformers import SentenceTransformer, util 
import json
import os
import re

# --- Definición de Frases Clave para Parámetros (Iteración v7) ---
FRASES_CLAVE_PARAMETROS = {
    "llegada": [
        # Énfasis en "tasa de llegada" y variantes directas
        "tasa de llegada", "la tasa de llegada es", "tasa de llegada de clientes", "tasa de llegada de unidades",
        "tasa media de llegada", "la tasa media de llegada es", "tasa esperada de llegada", "la tasa esperada de llegada es",
        "razón de llegada", "la razón de llegada es", "a razón de llegadas", "llegan a razón de",
        "frecuencia de llegada", "la frecuencia de llegada es", "frecuencia de arribo", "tasa de arribo",
        "llegan clientes a una tasa de", "arriban unidades con una frecuencia de",
        "pacientes se presentan a razón de", "el flujo de llegada es de",
        "llegan en promedio X por Y", "las llegadas ocurren a razón de",
        "se registran llegadas con una tasa de", "arribos por hora", "llegadas por minuto",
        "clientes por hora llegan", "unidades por día entran",
        "llegan según Poisson a razón de", "llegadas Poisson con media de",
        "tasa de llegadas de Poisson", "proceso de llegada Poisson con tasa",
        "llegan de acuerdo con una distribución de Poisson a razón de", 
        "la razón de arribo es", "la frecuencia de entrada es", "tasa de entrada",
        "patrón de llegadas", "patrón de arribos",
        # Énfasis en "tiempo entre llegadas" y variantes
        "tiempo entre llegadas", "el tiempo entre llegadas es", "tiempo promedio entre llegadas",
        "intervalo promedio entre arribos", "tiempo esperado entre llegadas",
        "tiempo medio entre llegadas", "intervalo de llegada",
        "llega un cliente cada", "los clientes ingresan cada", "unidades arriban cada",
        "se espera que arriben unidades cada", 
        "un nuevo pedido entra al sistema cada", 
        "un arribo cada", "la entrada de clientes es cada", "tiempo entre arribos de",
        "intervalo entre llegadas exponencial con media de", "tiempo entre llegadas sigue una exponencial",
        "el tiempo entre llegadas sigue una distribución de Poisson", 
        "tiempo entre llegadas distribuido exponencialmente con media de",
        # Generales de llegada
        "entran al sistema", "se presentan en la cola", "la llegada de clientes sigue una distribución",
        "llegan según un proceso de Poisson", "las llegadas ocurren", "se registran llegadas",
        "llegan al sistema", "ingresan a la cola", "se espera la llegada de",
        "nuevos pedidos entran", "entrada de pedidos", "se reciben pedidos", "flujo de entrada"
    ],
    "servicio": [
        # Énfasis en "tasa de servicio" y variantes directas
        "tasa de servicio", "la tasa de servicio es", "tasa de servicio por servidor",
        "tasa media de servicio", "la tasa media de servicio es", "tasa esperada de servicio", "la tasa esperada de servicio es",
        "razón de servicio", "la razón de servicio es", "a razón de servicios", "atiende a razón de",
        "capacidad de atención", "velocidad de servicio", "ritmo de servicio", "tasa de proceso",
        "frecuencia de servicio", "la frecuencia de servicio es", "tasa de atención por servidor",
        "procesa unidades a una tasa de", "el servidor atiende a una tasa de",
        "atiende en promedio X por Y", "clientes atendidos por hora", "unidades procesadas por minuto",
        "frecuencia con que se procesa", "tasa de servicio exponencial",
        "el servicio es Poisson con tasa de", "tasa de servicio del servidor",
        "el servidor puede procesar a una tasa de", "tasa de procesamiento del servidor",
        # Énfasis en "tiempo de servicio" y variantes
        "tiempo de servicio", "el tiempo de servicio es", "tiempo promedio de servicio",
        "tiempo esperado de servicio", "la duración del servicio es", "tiempo promedio para atender",
        "tiempo de procesamiento", "el tiempo de procesamiento es",
        "demora en atender", "atiende clientes cada", "se procesan ítems cada",
        "el servidor tarda en procesar", "el tiempo de atención sigue una distribución",
        "tiempo de consulta por paciente", "la duración media del servicio es",
        "tiempo para completar una tarea", "el servicio toma en promedio",
        "tiempo de ocupación del servidor", "tiempo de servicio exponencial con media de",
        "el tiempo de servicio sigue una exponencial de",
        "el tiempo de servicio sigue una distribución de Poisson",
        "tiempo de servicio distribuido exponencialmente con media de",
        # Generales de servicio
        "el sistema procesa", "patrón de servicio", "el servidor puede procesar", "capacidad de procesamiento"
    ],
    "servidores": ["número de servidores", "cantidad de cajeros", "puestos de atención", "operarios disponibles", "cuántas máquinas hay", "dispone de servidores"],
    "capacidad_sistema": ["capacidad del sistema", "tamaño de la cola", "límite de clientes en el sistema", "espacio en la sala de espera", "cuántos caben como máximo", "capacidad de k", "capacidad es de", "capacidad máxima", "rechaza clientes si hay más de", "solo se permiten", "buffer de tamaño"],
    "disciplina_cola": ["disciplina de la cola", "orden de atención", "tipo de cola", "cómo se atiende", "FIFO", "LIFO", "primero en llegar primero en ser servido", "atención por prioridad", "orden de llegada"]
}

# --- Model Loading ---
NLP_SPACY = None
MODEL_SENTENCE_TRANSFORMERS = None
EMBEDDINGS_FRASES_CLAVE = {}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# --- Diccionario para números en palabras (expandible) ---
NUMEROS_EN_PALABRAS_MAP = {
    "cero": 0, "un": 1, "una": 1, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12,
    "trece": 13, "catorce": 14, "quince": 15, "dieciséis": 16, "diecisiete": 17,
    "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiun": 21, "veintiuno": 21,
    "veintidós": 22, "veintitrés": 23, "veinticuatro": 24, "veinticinco": 25,
    "veintiséis": 26, "veintisiete": 27, "veintiocho": 28, "veintinueve": 29,
    "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60,
    "setenta": 70, "ochenta": 80, "noventa": 90, "cien": 100, "ciento": 100,
}
PALABRAS_NUMERO_REGEX = r"\b(" + "|".join(NUMEROS_EN_PALABRAS_MAP.keys()) + r")\b"
DIGITAL_NUMERO_REGEX = r"\b\d+([.,]\d+)?\b"


def cargar_modelos_y_precalcular_embeddings():
    global NLP_SPACY, MODEL_SENTENCE_TRANSFORMERS, EMBEDDINGS_FRASES_CLAVE
    if NLP_SPACY and MODEL_SENTENCE_TRANSFORMERS and \
       EMBEDDINGS_FRASES_CLAVE.get("llegada") is not None and \
       EMBEDDINGS_FRASES_CLAVE.get("servicio") is not None:
        return
    print("Cargando modelos NLP y precalculando embeddings de frases clave...")
    try:
        if NLP_SPACY is None: NLP_SPACY = spacy.load("es_core_news_sm")
        if MODEL_SENTENCE_TRANSFORMERS is None: MODEL_SENTENCE_TRANSFORMERS = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        if not EMBEDDINGS_FRASES_CLAVE or EMBEDDINGS_FRASES_CLAVE.get("llegada") is None or EMBEDDINGS_FRASES_CLAVE.get("servicio") is None:
            EMBEDDINGS_FRASES_CLAVE = {} 
            for cat, frases in FRASES_CLAVE_PARAMETROS.items():
                if cat in ["llegada", "servicio"] and frases: 
                    EMBEDDINGS_FRASES_CLAVE[cat] = MODEL_SENTENCE_TRANSFORMERS.encode(frases, convert_to_tensor=True)
                elif cat in ["llegada", "servicio"]: EMBEDDINGS_FRASES_CLAVE[cat] = None
            if EMBEDDINGS_FRASES_CLAVE.get("llegada") is not None and EMBEDDINGS_FRASES_CLAVE.get("servicio") is not None: print("Embeddings precalculados.")
            else: print("Advertencia: No se generaron embeddings para llegada/servicio.")
        else: print("Embeddings ya precalculados.")
        print("Modelos NLP listos.")
    except Exception as e:
        print(f"Error cargando modelos/embeddings: {e}")
        NLP_SPACY, MODEL_SENTENCE_TRANSFORMERS, EMBEDDINGS_FRASES_CLAVE = None, None, {}

def inicializar_estructura_salida():
    return {
        "texto_original": None,
        "parametros_extraidos": {
            "tasa_llegada": {"valor": None, "unidades": None, "distribucion": None, "fragmento_texto": None},
            "tiempo_entre_llegadas": {"valor": None, "unidades": None, "distribucion": None, "fragmento_texto": None},
            "tasa_servicio_por_servidor": {"valor": None, "unidades": None, "distribucion": None, "fragmento_texto": None},
            "tiempo_servicio_por_servidor": {"valor": None, "unidades": None, "distribucion": None, "fragmento_texto": None},
            "cantidad_servidores": {"valor": None, "fragmento_texto": None},
            "capacidad_sistema": {"valor": None, "fragmento_texto": None},
            "disciplina_cola": {"valor": None, "fragmento_texto": None} 
        },
        "oraciones_candidatas_debug": {"llegada": [], "servicio": []}, "errores": []
    }

def procesar_texto_basico(texto_entrada):
    if NLP_SPACY is None:
        resultado = inicializar_estructura_salida(); resultado["errores"].append("spaCy no cargado."); return None, resultado
    doc_spacy = NLP_SPACY(texto_entrada)
    resultado = inicializar_estructura_salida(); resultado["texto_original"] = texto_entrada
    return doc_spacy, resultado

def extraer_valor_y_unidad_de_oracion(texto_oracion_candidata):
    unidades_tiempo_singular = r"(segundo|minuto|hora|día|semana|mes|año)"
    unidades_tiempo_plural = r"(segundos|minutos|horas|días|semanas|meses|años)"
    unidades_tiempo_regex = rf"\b({unidades_tiempo_singular}|{unidades_tiempo_plural})\b"
    entidades_comunes = r"(clientes?|pacientes?|unidades?|ítems?|items?|trabajos?|pedidos?|vehículos?|tareas?|consultas?|llamadas?|operaciones?)"
    tasa_regex_1 = rf"({entidades_comunes})\s*(por|al)\s*({unidades_tiempo_singular})" 
    tasa_regex_2 = rf"({entidades_comunes})\s*/\s*({unidades_tiempo_singular})" 
    tasa_regex_3 = rf"({entidades_comunes})\s*(por|al)\s*({unidades_tiempo_plural})" 
    tiempo_por_entidad_regex_1 = rf"({unidades_tiempo_regex})\s*(por|al)\s*({entidades_comunes})"
    tiempo_por_entidad_regex_2 = rf"({unidades_tiempo_regex})\s*/\s*({entidades_comunes})"
    tiempo_cada_regex = rf"cada\s+(?:{PALABRAS_NUMERO_REGEX}|{DIGITAL_NUMERO_REGEX})\s*({unidades_tiempo_regex})"
    patrones_unidad = [
        (tasa_regex_1, "tasa"), (tasa_regex_2, "tasa"), (tasa_regex_3, "tasa"),
        (tiempo_por_entidad_regex_1, "tiempo"), (tiempo_por_entidad_regex_2, "tiempo"),
        (unidades_tiempo_regex, "tiempo") 
    ]
    posibles_numeros = []
    for match_palabra in re.finditer(PALABRAS_NUMERO_REGEX, texto_oracion_candidata, re.IGNORECASE):
        palabra = match_palabra.group(1).lower()
        if palabra in NUMEROS_EN_PALABRAS_MAP:
            posibles_numeros.append({"valor": float(NUMEROS_EN_PALABRAS_MAP[palabra]), "texto": palabra, "pos": match_palabra.span(), "es_palabra": True})
    for match_digital in re.finditer(DIGITAL_NUMERO_REGEX, texto_oracion_candidata):
        texto_num = match_digital.group(0)
        try:
            posibles_numeros.append({"valor": float(texto_num.replace(",", ".")), "texto": texto_num, "pos": match_digital.span(), "es_palabra": False})
        except ValueError: continue
    if not posibles_numeros: return None
    posibles_numeros.sort(key=lambda x: x["pos"][0])
    for num_info in posibles_numeros:
        texto_despues_num = texto_oracion_candidata[num_info["pos"][1]:].lstrip()
        match_tiempo_cada = re.search(tiempo_cada_regex, texto_oracion_candidata, re.IGNORECASE)
        if match_tiempo_cada and num_info["pos"][0] > texto_oracion_candidata.lower().find("cada"):
            unidad_match_texto = match_tiempo_cada.group(1) 
            return {"valor": num_info["valor"], "valor_texto": num_info["texto"], "unidad_texto": unidad_match_texto, "tipo_parametro": "tiempo", "posicion_valor": num_info["pos"], "posicion_unidad": (match_tiempo_cada.start(1), match_tiempo_cada.end(1))}
        for unidad_regex, tipo_param in patrones_unidad:
            match_unidad = re.search(unidad_regex, texto_despues_num, re.IGNORECASE)
            if match_unidad:
                pos_unidad_inicio = num_info["pos"][1] + texto_despues_num.find(match_unidad.group(0))
                pos_unidad_fin = pos_unidad_inicio + len(match_unidad.group(0))
                return {"valor": num_info["valor"], "valor_texto": num_info["texto"], "unidad_texto": match_unidad.group(0).strip(), "tipo_parametro": tipo_param, "posicion_valor": num_info["pos"], "posicion_unidad": (pos_unidad_inicio, pos_unidad_fin)}
    return None

def identificar_oraciones_candidatas(doc_spacy, umbral_similitud=0.6, debug_specific_sentence_part=None):
    candidatas = {"llegada": [], "servicio": []}
    if not MODEL_SENTENCE_TRANSFORMERS or not EMBEDDINGS_FRASES_CLAVE or \
       EMBEDDINGS_FRASES_CLAVE.get("llegada") is None or \
       EMBEDDINGS_FRASES_CLAVE.get("servicio") is None:
        print("ERROR: Modelo SentenceTransformer o embeddings de frases clave para llegada/servicio no cargados en identificar_oraciones_candidatas.")
        return candidatas
    for sent in doc_spacy.sents:
        sent_text = sent.text
        if not sent_text.strip(): continue
        embedding_oracion = MODEL_SENTENCE_TRANSFORMERS.encode(sent_text, convert_to_tensor=True)
        similitudes_llegada = util.cos_sim(embedding_oracion, EMBEDDINGS_FRASES_CLAVE["llegada"])
        max_sim_llegada = 0.0; idx_max_llegada = -1
        if similitudes_llegada.numel() > 0: max_sim_llegada = similitudes_llegada.max().item(); idx_max_llegada = similitudes_llegada.argmax().item()
        if max_sim_llegada >= umbral_similitud:
            candidatas["llegada"].append({"oracion_texto": sent_text, "similitud": round(max_sim_llegada, 4)})
        similitudes_servicio = util.cos_sim(embedding_oracion, EMBEDDINGS_FRASES_CLAVE["servicio"])
        max_sim_servicio = 0.0; idx_max_servicio = -1
        if similitudes_servicio.numel() > 0: max_sim_servicio = similitudes_servicio.max().item(); idx_max_servicio = similitudes_servicio.argmax().item()
        if max_sim_servicio >= umbral_similitud:
            candidatas["servicio"].append({"oracion_texto": sent_text, "similitud": round(max_sim_servicio, 4)})
        if debug_specific_sentence_part and debug_specific_sentence_part in sent_text:
            print(f"\nDEBUG para oración: \"{sent_text}\"")
            if idx_max_llegada != -1 and idx_max_llegada < len(FRASES_CLAVE_PARAMETROS['llegada']): print(f"  Max Sim Llegada: {max_sim_llegada:.4f} (con frase clave: '{FRASES_CLAVE_PARAMETROS['llegada'][idx_max_llegada]}')")
            else: print(f"  Max Sim Llegada: {max_sim_llegada:.4f} (sin match de frase clave o índice fuera de rango)")
            if idx_max_servicio != -1 and idx_max_servicio < len(FRASES_CLAVE_PARAMETROS['servicio']): print(f"  Max Sim Servicio: {max_sim_servicio:.4f} (con frase clave: '{FRASES_CLAVE_PARAMETROS['servicio'][idx_max_servicio]}')")
            else: print(f"  Max Sim Servicio: {max_sim_servicio:.4f} (sin match de frase clave o índice fuera de rango)")
    for categoria in candidatas:
        unique_candidatas = []; seen_oraciones = set()
        for cand in sorted(candidatas[categoria], key=lambda x: x["similitud"], reverse=True):
            if cand["oracion_texto"] not in seen_oraciones: unique_candidatas.append(cand); seen_oraciones.add(cand["oracion_texto"])
        candidatas[categoria] = unique_candidatas
    return candidatas

def extraer_numero_servidores(doc_spacy, resultado_parcial):
    # ... (código sin cambios)
    if NLP_SPACY is None: 
        resultado_parcial["errores"].append("Modelo spaCy no cargado en extraer_numero_servidores.")
        return resultado_parcial
    matcher = Matcher(NLP_SPACY.vocab)
    keywords_servidor = ["servidor", "cajero", "máquina", "operador", "ventanilla", "puesto", "estación", "médico", "doctor", "enfermero", "consultorio", "línea de ensamblaje", "caja"]
    pattern1 = [{"LIKE_NUM": True}, {"LEMMA": {"IN": keywords_servidor}}]
    pattern2 = [{"LOWER": {"IN": ["un", "una"]}}, {"LEMMA": {"IN": keywords_servidor}}]
    pattern3 = [{"LEMMA": {"IN": ["haber", "existir", "tener", "contar con", "disponer de", "operar con"]}}, {"LIKE_NUM": True, "OP": "?"}, {"LOWER": {"IN": ["un", "una"]}, "OP": "?"}, {"LEMMA": {"IN": keywords_servidor}}]
    matcher.add("NUM_SERVIDORES_PATTERN1", [pattern1]); matcher.add("NUM_SERVIDORES_PATTERN2", [pattern2]); matcher.add("NUM_SERVIDORES_PATTERN3", [pattern3])
    matches = matcher(doc_spacy); found_values = []
    for match_id, start, end in matches:
        span = doc_spacy[start:end]; num_val = None; potential_num_token = None
        if NLP_SPACY.vocab.strings[match_id] == "NUM_SERVIDORES_PATTERN2": num_val = 1; potential_num_token = span[0] 
        else:
            for token_idx_in_span, token in enumerate(span):
                if token.like_num:
                    num_map = {"uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5} 
                    if token.text.lower() in num_map: num_val = num_map[token.text.lower()]
                    else:
                        try: num_val = int(float(token.text.replace(",", ".")))
                        except ValueError: continue 
                    potential_num_token = token; break 
                elif token.lower_ in ["un", "una"] and NLP_SPACY.vocab.strings[match_id] == "NUM_SERVIDORES_PATTERN3":
                    is_quantifier = False
                    for next_token_in_span_idx in range(token_idx_in_span + 1, len(span)):
                        if span[next_token_in_span_idx].lemma_ in keywords_servidor: is_quantifier = True; break
                    if is_quantifier: num_val = 1; potential_num_token = token; break
        if num_val is not None and num_val > 0:
            fragmento = potential_num_token.sent.text if potential_num_token else span.sent.text
            found_values.append({"valor": num_val, "fragmento": fragmento, "span_text": span.text})
    if not found_values: print("INFO: No se encontró información explícita sobre el número de servidores.")
    else:
        primary_value = found_values[0]["valor"]; primary_fragment = found_values[0]["fragmento"]
        resultado_parcial["parametros_extraidos"]["cantidad_servidores"]["valor"] = primary_value
        resultado_parcial["parametros_extraidos"]["cantidad_servidores"]["fragmento_texto"] = primary_fragment
        print(f"INFO: Número de servidores extraído: {primary_value} (Fuente: '{primary_fragment}')")
        unique_numeric_values = set(item["valor"] for item in found_values)
        if len(unique_numeric_values) > 1:
            advertencia = (f"Advertencia: Se encontraron múltiples valores diferentes para el número de servidores. Se utilizó el primero ({primary_value} de '{primary_fragment}'). Otros valores encontrados: ")
            otros_valores_str_list = [f"{item['valor']} (en \"{item['span_text']}\" de la oración \"{item['fragmento']}\")" for item in found_values if item["valor"] != primary_value]
            advertencia += "; ".join(otros_valores_str_list); resultado_parcial["errores"].append(advertencia); print(f"ADVERTENCIA DETALLADA: {advertencia}")
    return resultado_parcial

def extraer_capacidad_sistema(doc_spacy, resultado_parcial):
    # ... (código sin cambios)
    if NLP_SPACY is None: resultado_parcial["errores"].append("Modelo spaCy no cargado en extraer_capacidad_sistema."); return resultado_parcial
    matcher = Matcher(NLP_SPACY.vocab)
    keywords_cap = ["capacidad", "límite", "tamaño", "espacio", "máximo", "buffer"]; keywords_entidad = ["cliente", "persona", "unidad", "auto", "puesto", "elemento", "paciente", "trabajo", "item"]; keywords_lugar_espera = ["cola", "sala de espera", "almacén", "buffer"]
    pattern_cap_num1 = [{"LEMMA": {"IN": keywords_cap}}, {"LOWER": {"IN": ["del", "de la"]}, "OP": "?"}, {"LOWER": {"IN": ["sistema", "total"] + keywords_lugar_espera}, "OP": "?"}, {"LOWER": "es", "OP": "?"}, {"LOWER": "de", "OP": "?"}, {"LOWER": {"IN": ["para", "en"]}, "OP": "?"}, {"LIKE_NUM": True}, {"LEMMA": {"IN": keywords_entidad}, "OP": "?"}]
    pattern_cap_num2 = [{"LIKE_NUM": True}, {"LEMMA": {"IN": keywords_entidad}, "OP": "+"}, {"LOWER": {"IN": ["de", "en", "para"]}, "OP": "+"}, {"LEMMA": {"IN": ["capacidad"] + keywords_lugar_espera + ["sistema"]}}]
    pattern_caben = [{"LEMMA": {"IN": ["caber", "acomodar"]}}, {"LIKE_NUM": True}, {"LEMMA": {"IN": keywords_entidad}, "OP": "?"}, {"LOWER": "en", "OP": "?"}, {"LOWER": {"IN": ["la", "el"]}, "OP": "?"}, {"LEMMA": {"IN": keywords_lugar_espera}}]
    pattern_no_mas_de = [{"LOWER": "no", "OP": "?"}, {"LEMMA": {"IN": ["poder", "aceptar", "admitir", "acomodar"]}, "OP": "+"}, {"LOWER": "más"}, {"LOWER": "de"}, {"LIKE_NUM": True}, {"LEMMA": {"IN": keywords_entidad}, "OP": "?"}]
    pattern_k_igual = [{"TEXT": {"IN": ["K", "k"]}}, {"LOWER": {"IN": ["=", "es"]}, "OP": "?"}, {"LIKE_NUM": True}]
    pattern_rechazo = [{"LEMMA": {"IN": ["rechazar", "no admitir", "no aceptar", "dejar de aceptar"]}}, {"IS_PUNCT": False, "OP": "{0,5}"}, {"LOWER": "si"}, {"LOWER": "hay"}, {"LOWER": {"IN": ["más", "mas"]}}, {"LOWER": "de"}, {"LIKE_NUM": True}, {"LEMMA": {"IN": keywords_entidad}, "OP": "?"}]
    pattern_solo_permiten = [{"LOWER": {"IN": ["solo", "solamente", "únicamente"]}}, {"LOWER": {"IN": ["se", "puede", "pueden"]}, "OP": "?"}, {"LEMMA": {"IN": ["permitir", "caber", "almacenar", "haber", "tener", "aceptar"]}, "OP": "+"}, {"LIKE_NUM": True}, {"LEMMA": {"IN": keywords_entidad}, "OP": "?"}]
    pattern_infinita_directa = [{"LOWER": {"IN": ["capacidad", "límite"]}, "OP": "?"}, {"LOWER": {"IN": ["infinita", "ilimitada"]}}]; pattern_no_limite = [{"LOWER": {"IN": ["no", "sin"]}}, {"LOWER": {"IN": ["hay", "tiene", "existe"]}, "OP": "?"}, {"LEMMA": {"IN": ["límite", "restricción"]}}]
    matcher.add("CAP_NUM1", [pattern_cap_num1]); matcher.add("CAP_NUM2", [pattern_cap_num2]); matcher.add("CAP_CABEN", [pattern_caben]); matcher.add("CAP_NO_MAS_DE", [pattern_no_mas_de]); matcher.add("CAP_K_IGUAL", [pattern_k_igual]); matcher.add("CAP_RECHAZO", [pattern_rechazo]); matcher.add("CAP_SOLO_PERMITEN", [pattern_solo_permiten]); matcher.add("CAP_INFINITA", [pattern_infinita_directa]); matcher.add("CAP_NO_LIMITE", [pattern_no_limite])
    matches = matcher(doc_spacy); found_capacities = []
    for match_id, start, end in matches:
        span = doc_spacy[start:end]; match_name = NLP_SPACY.vocab.strings[match_id]; current_value = None; current_fragment = span.sent.text; match_type = "general_numeric"
        if match_name in ["CAP_INFINITA", "CAP_NO_LIMITE"]: current_value = "infinita"; match_type = "infinite"
        else:
            for token in span:
                if token.like_num:
                    try: num_map = {"uno": 1, "dos": 2, "tres": 3}; current_value = num_map[token.text.lower()] if token.text.lower() in num_map else int(float(token.text.replace(",", "."))); break
                    except ValueError: continue
            if current_value is not None:
                sentence_text_lower = current_fragment.lower()
                if match_name in ["CAP_RECHAZO", "CAP_SOLO_PERMITEN", "CAP_K_IGUAL", "CAP_NO_MAS_DE"]: match_type = "system_direct_k"; 
                if any(kw_lugar in sentence_text_lower for kw_lugar in keywords_lugar_espera) and not any(kw_sys in sentence_text_lower for kw_sys in ["sistema", "total"]): match_type = "queue_explicit" 
                elif match_name in ["CAP_NUM1", "CAP_NUM2", "CAP_CABEN"]:
                    if any(kw in sentence_text_lower for kw in ["sistema", "total"]): match_type = "system_explicit"
                    elif any(kw_lugar in sentence_text_lower for kw_lugar in keywords_lugar_espera) or match_name == "CAP_CABEN": match_type = "queue_explicit"
        if current_value is not None:
            if isinstance(current_value, (int, float)) and current_value <= 0: continue
            found_capacities.append({"valor": current_value, "fragmento": current_fragment, "span_text": span.text, "match_name": match_name, "match_type": match_type})
    if not found_capacities:
        print("INFO: No se encontró info de capacidad. Asumiendo infinita por defecto."); resultado_parcial["parametros_extraidos"]["capacidad_sistema"]["valor"] = "infinita"; resultado_parcial["parametros_extraidos"]["capacidad_sistema"]["fragmento_texto"] = "Asumida infinita (no encontrada explícitamente)."
    else:
        chosen_cap = None; priority_order = ["system_direct_k", "system_explicit", "general_numeric", "queue_explicit", "infinite"]
        for p_type in priority_order:
            caps_of_type = [fc for fc in found_capacities if fc["match_type"] == p_type]
            if caps_of_type: chosen_cap = caps_of_type[0]; break
        if chosen_cap:
            primary_value = chosen_cap["valor"]; primary_fragment = chosen_cap["fragmento"]
            resultado_parcial["parametros_extraidos"]["capacidad_sistema"]["valor"] = primary_value; resultado_parcial["parametros_extraidos"]["capacidad_sistema"]["fragmento_texto"] = primary_fragment
            print(f"INFO: Capacidad del sistema extraída: {primary_value} (Fuente: '{primary_fragment}', Patrón: {chosen_cap['match_name']}, Tipo: {chosen_cap['match_type']})")
            numeric_caps_all = [fc for fc in found_capacities if isinstance(fc["valor"], (int, float))]; all_found_numeric_values = set(nc["valor"] for nc in numeric_caps_all)
            if len(all_found_numeric_values) > 1 and isinstance(primary_value, (int, float)):
                adv = (f"Advertencia: Múltiples valores numéricos para capacidad. Usado: {primary_value} (Tipo: {chosen_cap['match_type']}). Otros detectados: {[f'{nc["valor"]} (Tipo: {nc["match_type"]})' for nc in found_capacities if isinstance(nc['valor'], (int,float)) and nc['valor'] != primary_value]}")
                resultado_parcial["errores"].append(adv); print(f"ADVERTENCIA: {adv}")
            if numeric_caps_all and any(fc["match_type"] == "infinite" for fc in found_capacities) and isinstance(primary_value, (int, float)):
                adv = (f"Advertencia: Se encontró capacidad numérica ({primary_value}) y mención de 'infinita'. Se priorizó el valor numérico basado en el tipo de patrón ({chosen_cap['match_type']}).")
                resultado_parcial["errores"].append(adv); print(f"ADVERTENCIA: {adv}")
        else: print("INFO: No se pudo determinar la capacidad explícita con prioridades. Asumiendo infinita por defecto."); resultado_parcial["parametros_extraidos"]["capacidad_sistema"]["valor"] = "infinita"; resultado_parcial["parametros_extraidos"]["capacidad_sistema"]["fragmento_texto"] = "Asumida infinita (lógica de selección)."
    return resultado_parcial

def extraer_disciplina_cola(doc_spacy, resultado_parcial):
    # ... (código sin cambios)
    if NLP_SPACY is None: resultado_parcial["errores"].append("Modelo spaCy no cargado en extraer_disciplina_cola."); return resultado_parcial
    matcher = Matcher(NLP_SPACY.vocab); disciplinas_encontradas = []
    pattern_fifo_sigla = [{"TEXT": "FIFO"}]; pattern_fifo_frase_es = [{"LOWER": "primero"}, {"LOWER": "en"}, {"LOWER": "llegar"}, {"IS_PUNCT": True, "OP": "?"}, {"LOWER": "primero"}, {"LOWER": "en"}, {"LOWER": "ser"}, {"LOWER": "servido"}]; pattern_fifo_frase_es_alt = [{"LOWER": "primero"}, {"LOWER": "que"}, {"LOWER": "llega"}, {"LOWER": "es"}, {"LOWER": "el"}, {"LOWER": "primero"}, {"LOWER": "que"}, {"LOWER": "se"}, {"LOWER": "atiende"}]; pattern_fifo_fcfs = [{"TEXT": "FCFS"}]; pattern_orden_llegada = [{"LOWER": "orden"}, {"LOWER": "de"}, {"LOWER": "llegada"}]; pattern_orden_ingreso = [{"LOWER": "orden"}, {"LOWER": "en"}, {"LOWER": "que"}, {"LEMMA": {"IN": ["ir", "llegar", "ingresar"]}}]
    pattern_lifo_sigla = [{"TEXT": "LIFO"}]; pattern_lifo_frase_es = [{"LOWER": "último"}, {"LOWER": "en"}, {"LOWER": "llegar"}, {"IS_PUNCT": True, "OP": "?"}, {"LOWER": "primero"}, {"LOWER": "en"}, {"LOWER": "ser"}, {"LOWER": "servido"}]; pattern_lifo_lcfs = [{"TEXT": "LCFS"}]
    pattern_siro_sigla = [{"TEXT": "SIRO"}]; pattern_siro_frase_es = [{"LOWER": "servicio"}, {"LOWER": "en"}, {"LOWER": "orden"}, {"LOWER": "aleatorio"}]; pattern_siro_random = [{"LOWER": "atención"}, {"LOWER": "aleatoria"}]
    pattern_prioridad_sigla = [{"TEXT": "PRI"}]; pattern_prioridad_frase = [{"LEMMA": {"IN": ["prioridad", "prioritario"]}}]; pattern_prioridad_clases = [{"LEMMA": {"IN": ["clase", "tipo", "nivel"]}}, {"LOWER": "de", "OP": "?"}, {"LEMMA": "prioridad"}]
    matcher.add("DISC_FIFO_SIGLA", [pattern_fifo_sigla]); matcher.add("DISC_FIFO_FRASE_ES", [pattern_fifo_frase_es]); matcher.add("DISC_FIFO_FRASE_ES_ALT", [pattern_fifo_frase_es_alt]); matcher.add("DISC_FIFO_FCFS", [pattern_fifo_fcfs]); matcher.add("DISC_ORDEN_LLEGADA", [pattern_orden_llegada]); matcher.add("DISC_ORDEN_INGRESO", [pattern_orden_ingreso])
    matcher.add("DISC_LIFO_SIGLA", [pattern_lifo_sigla]); matcher.add("DISC_LIFO_FRASE_ES", [pattern_lifo_frase_es]); matcher.add("DISC_LIFO_LCFS", [pattern_lifo_lcfs])
    matcher.add("DISC_SIRO_SIGLA", [pattern_siro_sigla]); matcher.add("DISC_SIRO_FRASE_ES", [pattern_siro_frase_es]); matcher.add("DISC_SIRO_RANDOM", [pattern_siro_random])
    matcher.add("DISC_PRIORIDAD_SIGLA", [pattern_prioridad_sigla]); matcher.add("DISC_PRIORIDAD_FRASE", [pattern_prioridad_frase]); matcher.add("DISC_PRIORIDAD_CLASES", [pattern_prioridad_clases])
    matches = matcher(doc_spacy)
    for match_id, start, end in matches:
        span = doc_spacy[start:end]; rule_id_str = NLP_SPACY.vocab.strings[match_id]; disciplina_detectada = None
        if "FIFO" in rule_id_str or "FCFS" in rule_id_str or "ORDEN_LLEGADA" in rule_id_str or "ORDEN_INGRESO" in rule_id_str: disciplina_detectada = "FIFO"
        elif "LIFO" in rule_id_str or "LCFS" in rule_id_str: disciplina_detectada = "LIFO"
        elif "SIRO" in rule_id_str or "RANDOM" in rule_id_str: disciplina_detectada = "SIRO"
        elif "PRIORIDAD" in rule_id_str: disciplina_detectada = "Prioridad"
        if disciplina_detectada: disciplinas_encontradas.append({"valor": disciplina_detectada, "fragmento": span.sent.text, "span_text": span.text})
    if not disciplinas_encontradas: print("INFO: No se encontró disciplina de cola explícita. Se establece como no especificado."); resultado_parcial["parametros_extraidos"]["disciplina_cola"]["valor"] = None; resultado_parcial["parametros_extraidos"]["disciplina_cola"]["fragmento_texto"] = None
    else:
        chosen_discipline = disciplinas_encontradas[0]; primary_value = chosen_discipline["valor"]; primary_fragment = chosen_discipline["fragmento"]
        resultado_parcial["parametros_extraidos"]["disciplina_cola"]["valor"] = primary_value; resultado_parcial["parametros_extraidos"]["disciplina_cola"]["fragmento_texto"] = primary_fragment
        print(f"INFO: Disciplina de cola extraída: {primary_value} (Fuente: '{primary_fragment}')")
        unique_disciplines = set(d["valor"] for d in disciplinas_encontradas)
        if len(unique_disciplines) > 1:
            advertencia = (f"Advertencia: Se encontraron múltiples menciones de disciplinas de cola diferentes. Se utilizó la primera detectada ('{primary_value}' de '{primary_fragment}'). Otras detectadas: {[f'{d["valor"]} (en \"{d["span_text"]}\")' for d in disciplinas_encontradas if d["valor"] != primary_value]}")
            resultado_parcial["errores"].append(advertencia); print(f"ADVERTENCIA DETALLADA: {advertencia}")
    return resultado_parcial

# --- Función Principal de Extracción ---
def extraer_parametros_colas(texto_entrada, umbral_similitud_candidatas=0.6, debug_specific_sentence_part=None):
    if NLP_SPACY is None or MODEL_SENTENCE_TRANSFORMERS is None or \
       not EMBEDDINGS_FRASES_CLAVE or \
       EMBEDDINGS_FRASES_CLAVE.get("llegada") is None or \
       EMBEDDINGS_FRASES_CLAVE.get("servicio") is None:
        print("Intentando recargar modelos y/o embeddings...")
        cargar_modelos_y_precalcular_embeddings() 
        if NLP_SPACY is None or MODEL_SENTENCE_TRANSFORMERS is None or \
           not EMBEDDINGS_FRASES_CLAVE or \
           EMBEDDINGS_FRASES_CLAVE.get("llegada") is None or \
           EMBEDDINGS_FRASES_CLAVE.get("servicio") is None:
            res_error = inicializar_estructura_salida()
            res_error["errores"].append("Fallo crítico al cargar modelos NLP o embeddings de frases clave.")
            return res_error

    doc_spacy, resultado_parcial = procesar_texto_basico(texto_entrada)
    if doc_spacy is None: return resultado_parcial 

    resultado_parcial = extraer_numero_servidores(doc_spacy, resultado_parcial)
    resultado_parcial = extraer_capacidad_sistema(doc_spacy, resultado_parcial) 
    resultado_parcial = extraer_disciplina_cola(doc_spacy, resultado_parcial)

    oraciones_candidatas = identificar_oraciones_candidatas(doc_spacy, umbral_similitud_candidatas, debug_specific_sentence_part=debug_specific_sentence_part) # Corregido aquí
    resultado_parcial["oraciones_candidatas_debug"] = oraciones_candidatas 

    print("\n--- Oraciones Candidatas Detectadas (Debug) ---")
    if oraciones_candidatas.get("llegada"): 
        print("Llegada:")
        for cand in oraciones_candidatas["llegada"]:
            print(f"  - Sim: {cand['similitud']:.4f}, Oración: \"{cand['oracion_texto']}\"")
    if oraciones_candidatas.get("servicio"):
        print("Servicio:")
        for cand in oraciones_candidatas["servicio"]:
            print(f"  - Sim: {cand['similitud']:.4f}, Oración: \"{cand['oracion_texto']}\"")
    print("--------------------------------------------")
    
    # --- Tarea 2.3 y 2.4 (Poblado de tasas/tiempos) ---
    # Usar las oraciones candidatas y extraer_valor_y_unidad_de_oracion
    
    # Procesar candidatas de LLEGADA
    if resultado_parcial["oraciones_candidatas_debug"]["llegada"]:
        # Por ahora, tomamos la primera oración candidata de llegada con mayor similitud
        # En el futuro, podríamos iterar o tener lógica más compleja si hay múltiples.
        oracion_llegada_principal = resultado_parcial["oraciones_candidatas_debug"]["llegada"][0]["oracion_texto"]
        datos_llegada = extraer_valor_y_unidad_de_oracion(oracion_llegada_principal)
        if datos_llegada:
            print(f"DEBUG - Datos LLEGADA extraídos de '{oracion_llegada_principal}': {datos_llegada}")
            if datos_llegada["tipo_parametro"] == "tasa":
                resultado_parcial["parametros_extraidos"]["tasa_llegada"]["valor"] = datos_llegada["valor"]
                resultado_parcial["parametros_extraidos"]["tasa_llegada"]["unidades"] = datos_llegada["unidad_texto"]
                resultado_parcial["parametros_extraidos"]["tasa_llegada"]["fragmento_texto"] = oracion_llegada_principal
            elif datos_llegada["tipo_parametro"] == "tiempo":
                resultado_parcial["parametros_extraidos"]["tiempo_entre_llegadas"]["valor"] = datos_llegada["valor"]
                resultado_parcial["parametros_extraidos"]["tiempo_entre_llegadas"]["unidades"] = datos_llegada["unidad_texto"]
                resultado_parcial["parametros_extraidos"]["tiempo_entre_llegadas"]["fragmento_texto"] = oracion_llegada_principal
            # Aquí podríamos intentar extraer la distribución si se menciona en la oración
            if "poisson" in oracion_llegada_principal.lower():
                 resultado_parcial["parametros_extraidos"]["tasa_llegada"]["distribucion"] = "Poisson"
                 resultado_parcial["parametros_extraidos"]["tiempo_entre_llegadas"]["distribucion"] = "Exponencial" # Si llegadas son Poisson, tiempo entre es Exponencial
            elif "exponencial" in oracion_llegada_principal.lower():
                 resultado_parcial["parametros_extraidos"]["tasa_llegada"]["distribucion"] = "Poisson" # Asumiendo que si el tiempo es exp, la tasa es Poisson
                 resultado_parcial["parametros_extraidos"]["tiempo_entre_llegadas"]["distribucion"] = "Exponencial"


    # Procesar candidatas de SERVICIO
    if resultado_parcial["oraciones_candidatas_debug"]["servicio"]:
        oracion_servicio_principal = resultado_parcial["oraciones_candidatas_debug"]["servicio"][0]["oracion_texto"]
        datos_servicio = extraer_valor_y_unidad_de_oracion(oracion_servicio_principal)
        if datos_servicio:
            print(f"DEBUG - Datos SERVICIO extraídos de '{oracion_servicio_principal}': {datos_servicio}")
            if datos_servicio["tipo_parametro"] == "tasa":
                resultado_parcial["parametros_extraidos"]["tasa_servicio_por_servidor"]["valor"] = datos_servicio["valor"]
                resultado_parcial["parametros_extraidos"]["tasa_servicio_por_servidor"]["unidades"] = datos_servicio["unidad_texto"]
                resultado_parcial["parametros_extraidos"]["tasa_servicio_por_servidor"]["fragmento_texto"] = oracion_servicio_principal
            elif datos_servicio["tipo_parametro"] == "tiempo":
                resultado_parcial["parametros_extraidos"]["tiempo_servicio_por_servidor"]["valor"] = datos_servicio["valor"]
                resultado_parcial["parametros_extraidos"]["tiempo_servicio_por_servidor"]["unidades"] = datos_servicio["unidad_texto"]
                resultado_parcial["parametros_extraidos"]["tiempo_servicio_por_servidor"]["fragmento_texto"] = oracion_servicio_principal
            
            if "poisson" in oracion_servicio_principal.lower():
                resultado_parcial["parametros_extraidos"]["tasa_servicio_por_servidor"]["distribucion"] = "Poisson" # O Exponencial si se refiere al tiempo
                resultado_parcial["parametros_extraidos"]["tiempo_servicio_por_servidor"]["distribucion"] = "Exponencial"
            elif "exponencial" in oracion_servicio_principal.lower():
                resultado_parcial["parametros_extraidos"]["tasa_servicio_por_servidor"]["distribucion"] = "Poisson"
                resultado_parcial["parametros_extraidos"]["tiempo_servicio_por_servidor"]["distribucion"] = "Exponencial"
    return resultado_parcial

if __name__ == "__main__":
    cargar_modelos_y_precalcular_embeddings() 

    textos_prueba_tasas_tiempos = [
        "Los clientes llegan a una tasa de 10 por hora.",
        "El tiempo promedio de servicio es de 5 minutos por cliente.",
        "Se espera que arriben unidades cada 2 minutos.", 
        "La capacidad de atención del sistema es de 15 trabajos por día.",
        "Un nuevo pedido entra al sistema aproximadamente cada 30 segundos.", 
        "El servidor procesa los ítems con una frecuencia de 60 ítems/hora.", 
        "La tasa esperada de llegada de pacientes es de 3 por hora.", 
        "El tiempo medio entre llegadas de vehículos es de 120 segundos.",
        "La tasa media de servicio por cajero es de 25 clientes/hora.", 
        "El tiempo esperado de servicio para cada tarea es de 10 minutos.",
        "Los pacientes llegan a la clínica de un médico de acuerdo con una distribución de Poisson a razón de treinta pacientes por hora." # Prueba con número en palabra
    ]

    for i, texto_test in enumerate(textos_prueba_tasas_tiempos):
        print(f"\n--- Procesando Texto de Prueba Tasas/Tiempos #{i+1}: '{texto_test}' ---")
        if NLP_SPACY and MODEL_SENTENCE_TRANSFORMERS and \
           EMBEDDINGS_FRASES_CLAVE.get("llegada") is not None and \
           EMBEDDINGS_FRASES_CLAVE.get("servicio") is not None: 
            info_extraida = extraer_parametros_colas(texto_test, umbral_similitud_candidatas=0.55) 
            print("\n--- Resultado Extracción (JSON) ---") 
            print(json.dumps(info_extraida, indent=4, ensure_ascii=False))
            print("-----------------------------------\n")
        else:
            print("Modelos o embeddings de frases clave no cargados, saltando prueba.")
    
    print("\n--- Prueba con archivo ejemplo1.txt (si existe) ---")
    ruta_ejemplo = os.path.join(DATA_DIR, "ejemplo1.txt")
    texto_ejemplo_usuario = "Los pacientes llegan a la clínica de un médico de acuerdo con una distribución de Poisson a razón de 20 pacientes por hora. La sala de espera no puede acomodar más de 14 pacientes. El tiempo de consulta por paciente es exponencial, con una media de 8 minutos."
    
    try:
        print(f"\n--- Procesando Texto Específico del Usuario ---")
        print(texto_ejemplo_usuario)
        print("------------------------------------")
        if NLP_SPACY and MODEL_SENTENCE_TRANSFORMERS and \
           EMBEDDINGS_FRASES_CLAVE.get("llegada") is not None and \
           EMBEDDINGS_FRASES_CLAVE.get("servicio") is not None:
            info_extraida_usuario = extraer_parametros_colas(
                texto_ejemplo_usuario, 
                umbral_similitud_candidatas=0.50, 
                debug_specific_sentence_part="a razón de 20 pacientes por hora" 
            ) 
            print("\n--- Resultado (JSON) - Texto Usuario ---")
            print(json.dumps(info_extraida_usuario, indent=4, ensure_ascii=False))
            print("--------------------------\n")
        else:
            print("Modelos o embeddings de frases clave no cargados, saltando prueba del usuario.")

        if os.path.exists(ruta_ejemplo):
            with open(ruta_ejemplo, 'r', encoding='utf-8') as f:
                texto_ejemplo_regresion = f.read() 
            print(f"--- Procesando Archivo de Regresión: {ruta_ejemplo} ---")
            print(texto_ejemplo_regresion) 
            print("------------------------------------")
            if NLP_SPACY and MODEL_SENTENCE_TRANSFORMERS and \
               EMBEDDINGS_FRASES_CLAVE.get("llegada") is not None and \
               EMBEDDINGS_FRASES_CLAVE.get("servicio") is not None:
                info_extraida_regresion = extraer_parametros_colas(texto_ejemplo_regresion, umbral_similitud_candidatas=0.50, debug_specific_sentence_part="a razón de 20 pacientes por hora") 
                print("\n--- Resultado (JSON) - Archivo Regresión ---")
                print(json.dumps(info_extraida_regresion, indent=4, ensure_ascii=False))
                print("--------------------------\n")
        else:
            print(f"Archivo de regresión {ruta_ejemplo} no encontrado.")

    except Exception as e:
        print(f"Error durante las pruebas: {e}")



# src/nlp_pipeline.py

import spacy
from spacy.matcher import Matcher
from sentence_transformers import SentenceTransformer, util 
import json
import os
import re

# --- Definición de Frases Clave para Parámetros (Iteración v7) ---
FRASES_CLAVE_PARAMETROS = {
    "llegada": [
        # Énfasis en "tasa de llegada" y variantes directas
        "tasa de llegada", "la tasa de llegada es", "tasa de llegada de clientes", "tasa de llegada de unidades",
        "tasa media de llegada", "la tasa media de llegada es", "tasa esperada de llegada", "la tasa esperada de llegada es",
        "razón de llegada", "la razón de llegada es", "a razón de llegadas", "llegan a razón de",
        "frecuencia de llegada", "la frecuencia de llegada es", "frecuencia de arribo", "tasa de arribo",
        "llegan clientes a una tasa de", "arriban unidades con una frecuencia de",
        "pacientes se presentan a razón de", "el flujo de llegada es de",
        "llegan en promedio X por Y", "las llegadas ocurren a razón de",
        "se registran llegadas con una tasa de", "arribos por hora", "llegadas por minuto",
        "clientes por hora llegan", "unidades por día entran",
        
        "llegan según Poisson a razón de", "llegadas Poisson con media de",
        "tasa de llegadas de Poisson", "proceso de llegada Poisson con tasa",
        "llegan de acuerdo con una distribución de Poisson a razón de", 
        "las llegadas siguen una distribución de Poisson con tasa de",
        "la razón de arribo es", "la frecuencia de entrada es", "tasa de entrada",
        "patrón de llegadas", "patrón de arribos",

        # Énfasis en "tiempo entre llegadas" y variantes
        "tiempo entre llegadas", "el tiempo entre llegadas es", "tiempo promedio entre llegadas",
        "intervalo promedio entre arribos", "tiempo esperado entre llegadas",
        "tiempo medio entre llegadas", "intervalo de llegada",
        "llega un cliente cada", "los clientes ingresan cada", "unidades arriban cada",
        "se espera que arriben unidades cada", 
        "un nuevo pedido entra al sistema cada", 
        "un arribo cada", "la entrada de clientes es cada", "tiempo entre arribos de",
        "intervalo entre llegadas exponencial con media de", "tiempo entre llegadas sigue una exponencial",
        "el tiempo entre llegadas sigue una distribución de Poisson", 
        "tiempo entre llegadas distribuido exponencialmente con media de",

        # Generales de llegada
        "entran al sistema", "se presentan en la cola", "la llegada de clientes sigue una distribución",
        "llegan según un proceso de Poisson", "las llegadas ocurren", "se registran llegadas",
        "llegan al sistema", "ingresan a la cola", "se espera la llegada de",
        "nuevos pedidos entran", "entrada de pedidos", "se reciben pedidos", "flujo de entrada"
    ],
    "servicio": [
        # Énfasis en "tasa de servicio" y variantes directas
        "tasa de servicio", "la tasa de servicio es", "tasa de servicio por servidor",
        "tasa media de servicio", "la tasa media de servicio es", "tasa esperada de servicio", "la tasa esperada de servicio es",
        "razón de servicio", "la razón de servicio es", "a razón de servicios", "atiende a razón de",
        "capacidad de atención", "velocidad de servicio", "ritmo de servicio", "tasa de proceso",
        "frecuencia de servicio", "la frecuencia de servicio es", "tasa de atención por servidor",
        "procesa unidades a una tasa de", "el servidor atiende a una tasa de",
        "atiende en promedio X por Y", "clientes atendidos por hora", "unidades procesadas por minuto",
        "frecuencia con que se procesa", "tasa de servicio exponencial",
        "el servicio es Poisson con tasa de", "tasa de servicio del servidor",
        "el servidor puede procesar a una tasa de", "tasa de procesamiento del servidor",

        # Énfasis en "tiempo de servicio" y variantes
        "tiempo de servicio", "el tiempo de servicio es", "tiempo promedio de servicio",
        "tiempo esperado de servicio", "la duración del servicio es", "tiempo promedio para atender",
        "tiempo de procesamiento", "el tiempo de procesamiento es",
        "demora en atender", "atiende clientes cada", "se procesan ítems cada",
        "el servidor tarda en procesar", "el tiempo de atención sigue una distribución",
        "tiempo de consulta por paciente", "la duración media del servicio es",
        "tiempo para completar una tarea", "el servicio toma en promedio",
        "tiempo de ocupación del servidor", "tiempo de servicio exponencial con media de",
        "el tiempo de servicio sigue una exponencial de",
        "el tiempo de servicio sigue una distribución de Poisson",
        "tiempo de servicio distribuido exponencialmente con media de",
        # Generales de servicio
        "el sistema procesa", "patrón de servicio", "el servidor puede procesar", "capacidad de procesamiento"
    ],
    "servidores": ["número de servidores", "cantidad de cajeros", "puestos de atención", "operarios disponibles", "cuántas máquinas hay", "dispone de servidores"],
    "capacidad_sistema": ["capacidad del sistema", "tamaño de la cola", "límite de clientes en el sistema", "espacio en la sala de espera", "cuántos caben como máximo", "capacidad de k", "capacidad es de", "capacidad máxima", "rechaza clientes si hay más de", "solo se permiten", "buffer de tamaño"],
    "disciplina_cola": ["disciplina de la cola", "orden de atención", "tipo de cola", "cómo se atiende", "FIFO", "LIFO", "primero en llegar primero en ser servido", "atención por prioridad", "orden de llegada"]
}

# --- Model Loading ---
NLP_SPACY = None
MODEL_SENTENCE_TRANSFORMERS = None
EMBEDDINGS_FRASES_CLAVE = {}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# --- Diccionario para números en palabras (MODIFICADO POR USUARIO) ---
NUMEROS_EN_PALABRAS_MAP = {
    "cero": 0, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, # Eliminado "un", "una", "uno"
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12,
    "trece": 13, "catorce": 14, "quince": 15, "dieciséis": 16, "diecisiete": 17,
    "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiun": 21, "veintiuno": 21,
    "veintidós": 22, "veintitrés": 23, "veinticuatro": 24, "veinticinco": 25,
    "veintiséis": 26, "veintisiete": 27, "veintiocho": 28, "veintinueve": 29,
    "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60,
    "setenta": 70, "ochenta": 80, "noventa": 90, "cien": 100, "ciento": 100,
}
PALABRAS_NUMERO_REGEX = r"(?<!\w)(" + "|".join(NUMEROS_EN_PALABRAS_MAP.keys()) + r")(?!\w)"
DIGITAL_NUMERO_REGEX = r"\b\d+([.,]\d+)?\b"


def cargar_modelos_y_precalcular_embeddings():
    global NLP_SPACY, MODEL_SENTENCE_TRANSFORMERS, EMBEDDINGS_FRASES_CLAVE
    if NLP_SPACY and MODEL_SENTENCE_TRANSFORMERS and \
       EMBEDDINGS_FRASES_CLAVE.get("llegada") is not None and \
       EMBEDDINGS_FRASES_CLAVE.get("servicio") is not None:
        return
    print("Cargando modelos NLP y precalculando embeddings de frases clave...")
    try:
        if NLP_SPACY is None: NLP_SPACY = spacy.load("es_core_news_sm")
        if MODEL_SENTENCE_TRANSFORMERS is None: MODEL_SENTENCE_TRANSFORMERS = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        if not EMBEDDINGS_FRASES_CLAVE or EMBEDDINGS_FRASES_CLAVE.get("llegada") is None or EMBEDDINGS_FRASES_CLAVE.get("servicio") is None:
            EMBEDDINGS_FRASES_CLAVE = {} 
            for cat, frases in FRASES_CLAVE_PARAMETROS.items():
                if cat in ["llegada", "servicio"] and frases: 
                    EMBEDDINGS_FRASES_CLAVE[cat] = MODEL_SENTENCE_TRANSFORMERS.encode(frases, convert_to_tensor=True)
                elif cat in ["llegada", "servicio"]: EMBEDDINGS_FRASES_CLAVE[cat] = None
            if EMBEDDINGS_FRASES_CLAVE.get("llegada") is not None and EMBEDDINGS_FRASES_CLAVE.get("servicio") is not None: print("Embeddings precalculados.")
            else: print("Advertencia: No se generaron embeddings para llegada/servicio.")
        else: print("Embeddings ya precalculados.")
        print("Modelos NLP listos.")
    except Exception as e:
        print(f"Error cargando modelos/embeddings: {e}")
        NLP_SPACY, MODEL_SENTENCE_TRANSFORMERS, EMBEDDINGS_FRASES_CLAVE = None, None, {}

def inicializar_estructura_salida():
    return {
        "texto_original": None,
        "parametros_extraidos": {
            "tasa_llegada": {"valor": None, "unidades": None, "distribucion": None, "fragmento_texto": None},
            "tiempo_entre_llegadas": {"valor": None, "unidades": None, "distribucion": None, "fragmento_texto": None},
            "tasa_servicio_por_servidor": {"valor": None, "unidades": None, "distribucion": None, "fragmento_texto": None},
            "tiempo_servicio_por_servidor": {"valor": None, "unidades": None, "distribucion": None, "fragmento_texto": None},
            "cantidad_servidores": {"valor": None, "fragmento_texto": None},
            "capacidad_sistema": {"valor": None, "fragmento_texto": None},
            "disciplina_cola": {"valor": None, "fragmento_texto": None} 
        },
        "oraciones_candidatas_debug": {"llegada": [], "servicio": []}, "errores": []
    }

def procesar_texto_basico(texto_entrada):
    if NLP_SPACY is None:
        resultado = inicializar_estructura_salida(); resultado["errores"].append("spaCy no cargado."); return None, resultado
    doc_spacy = NLP_SPACY(texto_entrada)
    resultado = inicializar_estructura_salida(); resultado["texto_original"] = texto_entrada
    return doc_spacy, resultado

# --- Tarea 2.2: Extracción de Valor Numérico y Unidades ---
def extraer_valor_y_unidad_de_oracion(texto_oracion_candidata):
    """
    Extrae un valor numérico (digital o palabra) y sus unidades asociadas (tasa o tiempo)
    de una oración candidata.
    Aplica lógica contextual para tasas si la entidad está separada.
    """
    unidades_tiempo_singular = r"(segundo|minuto|hora|día|semana|mes|año)"
    unidades_tiempo_plural = r"(segundos|minutos|horas|días|semanas|meses|años)"
    unidades_tiempo_regex_str = rf"\b({unidades_tiempo_singular}|{unidades_tiempo_plural})\b" 
    
    entidades_comunes_str = r"(clientes?|pacientes?|unidades?|ítems?|items?|trabajos?|pedidos?|vehículos?|tareas?|consultas?|llamadas?|operaciones?|eventos?|personas?)"
    
    # Patrones para identificar el tipo de unidad y extraer componentes
    # El orden es importante: de más específico a más general.
    
    # 1. Tasa explícita: [NUMERO] [ENTIDAD] por/al/[UNIDAD_TIEMPO] o [NUMERO] [ENTIDAD]/[UNIDAD_TIEMPO]
    patron_tasa_directa = rf"({DIGITAL_NUMERO_REGEX}|{PALABRAS_NUMERO_REGEX})\s+({entidades_comunes_str})\s*(?:por|al|/)\s*({unidades_tiempo_singular}|{unidades_tiempo_plural})"
    
    # 2. Tiempo por entidad: [NUMERO] [UNIDAD_TIEMPO] por/al/[ENTIDAD]
    patron_tiempo_por_entidad = rf"({DIGITAL_NUMERO_REGEX}|{PALABRAS_NUMERO_REGEX})\s+({unidades_tiempo_regex_str})\s*(?:por|al|/)\s*({entidades_comunes_str})"
    
    # 3. Tiempo con "cada": cada [NUMERO] [UNIDAD_TIEMPO] (el número es parte del patrón)
    patron_tiempo_cada = rf"cada\s+({DIGITAL_NUMERO_REGEX}|{PALABRAS_NUMERO_REGEX})\s*({unidades_tiempo_regex_str})"
    
    # 4. Tiempo simple (o posible tasa con entidad separada): [NUMERO] (por/al)? [UNIDAD_TIEMPO]
    #    Grupo 2 es (por|al) (opcional), Grupo 3 es la unidad de tiempo
    patron_tiempo_simple_o_tasa_potencial = rf"({DIGITAL_NUMERO_REGEX}|{PALABRAS_NUMERO_REGEX})\s*(?:(por|al)\s*)?({unidades_tiempo_regex_str})"

    compiled_patterns_ordered = [
        (re.compile(patron_tasa_directa, re.IGNORECASE), "tasa", 1, 2, 3),
        (re.compile(patron_tiempo_por_entidad, re.IGNORECASE), "tiempo", 1, 3, 2), 
        (re.compile(patron_tiempo_cada, re.IGNORECASE), "tiempo", 1, None, 2),
        (re.compile(patron_tiempo_simple_o_tasa_potencial, re.IGNORECASE), "tiempo_potencial_tasa", 1, None, 3) 
    ]
    
    extracted_results = []

    for compiled_regex, tipo_param_inicial, num_g, entidad_g, unidad_t_g in compiled_patterns_ordered:
        for match in compiled_regex.finditer(texto_oracion_candidata):
            num_str = match.group(num_g)
            valor_num = None
            if num_str.lower() in NUMEROS_EN_PALABRAS_MAP:
                valor_num = float(NUMEROS_EN_PALABRAS_MAP[num_str.lower()])
            else:
                try: valor_num = float(num_str.replace(",","."))
                except ValueError: continue
            
            unidad_texto_final = ""
            tipo_param_final = tipo_param_inicial
            
            ent_str_local = match.group(entidad_g) if entidad_g and len(match.groups()) >= entidad_g and match.group(entidad_g) is not None else None
            ut_str_local = match.group(unidad_t_g) if unidad_t_g and len(match.groups()) >= unidad_t_g and match.group(unidad_t_g) is not None else None

            if tipo_param_inicial == "tasa":
                if ent_str_local and ut_str_local:
                    unidad_texto_final = f"{ent_str_local.lower()}/{ut_str_local.lower()}"
                else: continue
            elif tipo_param_inicial == "tiempo":
                if not ut_str_local: continue
                if ent_str_local: 
                    unidad_texto_final = f"{ut_str_local.lower()}/{ent_str_local.lower()}"
                else: 
                    unidad_texto_final = ut_str_local.lower()
            elif tipo_param_inicial == "tiempo_potencial_tasa": 
                if not ut_str_local: continue
                tipo_param_final = "tiempo" # Por defecto es tiempo
                unidad_texto_final = ut_str_local.lower()
                
                # --- INICIO DE LÓGICA CONTEXTUAL PARA RECLASIFICAR A TASA ---
                keywords_tasa_contexto_regex = r"\b(tasa|frecuencia|razón\s+de)\b"
                condicion1_tasa_en_oracion = re.search(keywords_tasa_contexto_regex, texto_oracion_candidata, re.IGNORECASE)
                
                # El grupo 2 de patron_tiempo_simple_o_tasa_potencial es (por|al)
                # Verificar si el grupo existe y tiene contenido
                por_al_capturado = None
                if len(match.groups()) >= 2 : # Asegurar que el grupo 2 (para "por" o "al") existe
                    por_al_capturado = match.group(2) 
                condicion2_por_al_presente = por_al_capturado is not None

                if condicion1_tasa_en_oracion or condicion2_por_al_presente:
                    texto_antes_num_completo = texto_oracion_candidata[:match.start(num_g)]
                    entidades_halladas_antes = list(re.finditer(rf"({entidades_comunes_str})\b", texto_antes_num_completo, re.IGNORECASE))
                    if entidades_halladas_antes:
                        entidad_contextual = entidades_halladas_antes[-1].group(1).lower() 
                        tipo_param_final = "tasa"
                        unidad_texto_final = f"{entidad_contextual}/{ut_str_local.lower()}"
                        # print(f"DEBUG: Reclasificado a TASA por contexto: {unidad_texto_final} para num {valor_num} en '{texto_oracion_candidata}'")
                    elif condicion1_tasa_en_oracion: # Si hay keyword de tasa pero no entidad antes, usar "eventos" como default
                        tipo_param_final = "tasa"
                        unidad_texto_final = f"eventos/{ut_str_local.lower()}"
                        # print(f"DEBUG: Reclasificado a TASA (default entidad) por contexto: {unidad_texto_final} para num {valor_num} en '{texto_oracion_candidata}'")
                # --- FIN DE LÓGICA CONTEXTUAL ---
            
            if not unidad_texto_final: continue

            prioridad_val = 3 
            if tipo_param_final == "tasa": prioridad_val = 0 
            elif tipo_param_final == "tiempo":
                # Ajustar prioridad para "cada"
                if compiled_regex.pattern == patron_tiempo_cada: prioridad_val = 1 
                elif entidad_g and ent_str_local: prioridad_val = 2 
            
            extracted_results.append({
                "valor": valor_num, "valor_texto": num_str,
                "unidad_texto": unidad_texto_final, "tipo_parametro": tipo_param_final,
                "posicion_valor": match.span(num_g), 
                "posicion_unidad_compleja": match.span(), 
                "prioridad": prioridad_val,
                "match_text": match.group(0) 
            })

    if not extracted_results: return None
    extracted_results.sort(key=lambda x: (x["prioridad"], x["posicion_unidad_compleja"][0]))
    return extracted_results[0]


def identificar_oraciones_candidatas(doc_spacy, umbral_similitud=0.6, debug_specific_sentence_part=None):
    # ... (código sin cambios, igual al que me proporcionaste)
    candidatas = {"llegada": [], "servicio": []}
    if not MODEL_SENTENCE_TRANSFORMERS or not EMBEDDINGS_FRASES_CLAVE or \
       EMBEDDINGS_FRASES_CLAVE.get("llegada") is None or \
       EMBEDDINGS_FRASES_CLAVE.get("servicio") is None:
        print("ERROR: Modelo SentenceTransformer o embeddings de frases clave para llegada/servicio no cargados en identificar_oraciones_candidatas.")
        return candidatas
    for sent in doc_spacy.sents:
        sent_text = sent.text
        if not sent_text.strip(): continue
        embedding_oracion = MODEL_SENTENCE_TRANSFORMERS.encode(sent_text, convert_to_tensor=True)
        similitudes_llegada = util.cos_sim(embedding_oracion, EMBEDDINGS_FRASES_CLAVE["llegada"])
        max_sim_llegada = 0.0; idx_max_llegada = -1
        if similitudes_llegada.numel() > 0: max_sim_llegada = similitudes_llegada.max().item(); idx_max_llegada = similitudes_llegada.argmax().item()
        if max_sim_llegada >= umbral_similitud:
            candidatas["llegada"].append({"oracion_texto": sent_text, "similitud": round(max_sim_llegada, 4)})
        similitudes_servicio = util.cos_sim(embedding_oracion, EMBEDDINGS_FRASES_CLAVE["servicio"])
        max_sim_servicio = 0.0; idx_max_servicio = -1
        if similitudes_servicio.numel() > 0: max_sim_servicio = similitudes_servicio.max().item(); idx_max_servicio = similitudes_servicio.argmax().item()
        if max_sim_servicio >= umbral_similitud:
            candidatas["servicio"].append({"oracion_texto": sent_text, "similitud": round(max_sim_servicio, 4)})
        if debug_specific_sentence_part and debug_specific_sentence_part in sent_text:
            print(f"\nDEBUG para oración: \"{sent_text}\"")
            if idx_max_llegada != -1 and idx_max_llegada < len(FRASES_CLAVE_PARAMETROS['llegada']): print(f"  Max Sim Llegada: {max_sim_llegada:.4f} (con frase clave: '{FRASES_CLAVE_PARAMETROS['llegada'][idx_max_llegada]}')")
            else: print(f"  Max Sim Llegada: {max_sim_llegada:.4f} (sin match de frase clave o índice fuera de rango)")
            if idx_max_servicio != -1 and idx_max_servicio < len(FRASES_CLAVE_PARAMETROS['servicio']): print(f"  Max Sim Servicio: {max_sim_servicio:.4f} (con frase clave: '{FRASES_CLAVE_PARAMETROS['servicio'][idx_max_servicio]}')")
            else: print(f"  Max Sim Servicio: {max_sim_servicio:.4f} (sin match de frase clave o índice fuera de rango)")
    for categoria in candidatas:
        unique_candidatas = []; seen_oraciones = set()
        for cand in sorted(candidatas[categoria], key=lambda x: x["similitud"], reverse=True):
            if cand["oracion_texto"] not in seen_oraciones: unique_candidatas.append(cand); seen_oraciones.add(cand["oracion_texto"])
        candidatas[categoria] = unique_candidatas
    return candidatas

def extraer_numero_servidores(doc_spacy, resultado_parcial):
    # ... (código sin cambios, igual al que me proporcionaste)
    if NLP_SPACY is None: 
        resultado_parcial["errores"].append("Modelo spaCy no cargado en extraer_numero_servidores.")
        return resultado_parcial
    matcher = Matcher(NLP_SPACY.vocab)
    keywords_servidor = ["servidor", "cajero", "máquina", "operador", "ventanilla", "puesto", "estación", "médico", "doctor", "enfermero", "consultorio", "línea de ensamblaje", "caja"]
    pattern1 = [{"LIKE_NUM": True}, {"LEMMA": {"IN": keywords_servidor}}]
    pattern2 = [{"LOWER": {"IN": ["un", "una"]}}, {"LEMMA": {"IN": keywords_servidor}}] # "un/una" ya no están en NUMEROS_EN_PALABRAS_MAP
    pattern3 = [{"LEMMA": {"IN": ["haber", "existir", "tener", "contar con", "disponer de", "operar con"]}}, {"LIKE_NUM": True, "OP": "?"}, {"LOWER": {"IN": ["un", "una"]}, "OP": "?"}, {"LEMMA": {"IN": keywords_servidor}}]
    matcher.add("NUM_SERVIDORES_PATTERN1", [pattern1]); matcher.add("NUM_SERVIDORES_PATTERN2", [pattern2]); matcher.add("NUM_SERVIDORES_PATTERN3", [pattern3])
    matches = matcher(doc_spacy); found_values = []
    for match_id, start, end in matches:
        span = doc_spacy[start:end]; num_val = None; potential_num_token = None
        if NLP_SPACY.vocab.strings[match_id] == "NUM_SERVIDORES_PATTERN2": num_val = 1; potential_num_token = span[0] 
        else:
            for token_idx_in_span, token in enumerate(span):
                if token.like_num:
                    # "un/una" ya no están en el map, así que no hay riesgo de tomarlos aquí si son seguidos por un digital
                    if token.text.lower() in NUMEROS_EN_PALABRAS_MAP: 
                        num_val = NUMEROS_EN_PALABRAS_MAP[token.text.lower()]
                    else:
                        try: num_val = int(float(token.text.replace(",", ".")))
                        except ValueError: continue 
                    potential_num_token = token; break 
                elif token.lower_ in ["un", "una"] and NLP_SPACY.vocab.strings[match_id] == "NUM_SERVIDORES_PATTERN3":
                    is_quantifier = False
                    for next_token_in_span_idx in range(token_idx_in_span + 1, len(span)):
                        if span[next_token_in_span_idx].lemma_ in keywords_servidor: is_quantifier = True; break
                    if is_quantifier: num_val = 1; potential_num_token = token; break
        if num_val is not None and num_val > 0:
            fragmento = potential_num_token.sent.text if potential_num_token else span.sent.text
            found_values.append({"valor": num_val, "fragmento": fragmento, "span_text": span.text})
    if not found_values: print("INFO: No se encontró información explícita sobre el número de servidores.")
    else:
        primary_value = found_values[0]["valor"]; primary_fragment = found_values[0]["fragmento"]
        resultado_parcial["parametros_extraidos"]["cantidad_servidores"]["valor"] = primary_value
        resultado_parcial["parametros_extraidos"]["cantidad_servidores"]["fragmento_texto"] = primary_fragment
        print(f"INFO: Número de servidores extraído: {primary_value} (Fuente: '{primary_fragment}')")
        unique_numeric_values = set(item["valor"] for item in found_values)
        if len(unique_numeric_values) > 1:
            advertencia = (f"Advertencia: Se encontraron múltiples valores diferentes para el número de servidores. Se utilizó el primero ({primary_value} de '{primary_fragment}'). Otros valores encontrados: ")
            otros_valores_str_list = [f"{item['valor']} (en \"{item['span_text']}\" de la oración \"{item['fragmento']}\")" for item in found_values if item["valor"] != primary_value]
            advertencia += "; ".join(otros_valores_str_list); resultado_parcial["errores"].append(advertencia); print(f"ADVERTENCIA DETALLADA: {advertencia}")
    return resultado_parcial

def extraer_capacidad_sistema(doc_spacy, resultado_parcial):
    # ... (código sin cambios, igual al que me proporcionaste)
    if NLP_SPACY is None: resultado_parcial["errores"].append("Modelo spaCy no cargado en extraer_capacidad_sistema."); return resultado_parcial
    matcher = Matcher(NLP_SPACY.vocab)
    keywords_cap = ["capacidad", "límite", "tamaño", "espacio", "máximo", "buffer"]; keywords_entidad = ["cliente", "persona", "unidad", "auto", "puesto", "elemento", "paciente", "trabajo", "item"]; keywords_lugar_espera = ["cola", "sala de espera", "almacén", "buffer"]
    pattern_cap_num1 = [{"LEMMA": {"IN": keywords_cap}}, {"LOWER": {"IN": ["del", "de la"]}, "OP": "?"}, {"LOWER": {"IN": ["sistema", "total"] + keywords_lugar_espera}, "OP": "?"}, {"LOWER": "es", "OP": "?"}, {"LOWER": "de", "OP": "?"}, {"LOWER": {"IN": ["para", "en"]}, "OP": "?"}, {"LIKE_NUM": True}, {"LEMMA": {"IN": keywords_entidad}, "OP": "?"}]
    pattern_cap_num2 = [{"LIKE_NUM": True}, {"LEMMA": {"IN": keywords_entidad}, "OP": "+"}, {"LOWER": {"IN": ["de", "en", "para"]}, "OP": "+"}, {"LEMMA": {"IN": ["capacidad"] + keywords_lugar_espera + ["sistema"]}}]
    pattern_caben = [{"LEMMA": {"IN": ["caber", "acomodar"]}}, {"LIKE_NUM": True}, {"LEMMA": {"IN": keywords_entidad}, "OP": "?"}, {"LOWER": "en", "OP": "?"}, {"LOWER": {"IN": ["la", "el"]}, "OP": "?"}, {"LEMMA": {"IN": keywords_lugar_espera}}]
    pattern_no_mas_de = [{"LOWER": "no", "OP": "?"}, {"LEMMA": {"IN": ["poder", "aceptar", "admitir", "acomodar"]}, "OP": "+"}, {"LOWER": "más"}, {"LOWER": "de"}, {"LIKE_NUM": True}, {"LEMMA": {"IN": keywords_entidad}, "OP": "?"}]
    pattern_k_igual = [{"TEXT": {"IN": ["K", "k"]}}, {"LOWER": {"IN": ["=", "es"]}, "OP": "?"}, {"LIKE_NUM": True}]
    pattern_rechazo = [{"LEMMA": {"IN": ["rechazar", "no admitir", "no aceptar", "dejar de aceptar"]}}, {"IS_PUNCT": False, "OP": "{0,5}"}, {"LOWER": "si"}, {"LOWER": "hay"}, {"LOWER": {"IN": ["más", "mas"]}}, {"LOWER": "de"}, {"LIKE_NUM": True}, {"LEMMA": {"IN": keywords_entidad}, "OP": "?"}]
    pattern_solo_permiten = [{"LOWER": {"IN": ["solo", "solamente", "únicamente"]}}, {"LOWER": {"IN": ["se", "puede", "pueden"]}, "OP": "?"}, {"LEMMA": {"IN": ["permitir", "caber", "almacenar", "haber", "tener", "aceptar"]}, "OP": "+"}, {"LIKE_NUM": True}, {"LEMMA": {"IN": keywords_entidad}, "OP": "?"}]
    pattern_infinita_directa = [{"LOWER": {"IN": ["capacidad", "límite"]}, "OP": "?"}, {"LOWER": {"IN": ["infinita", "ilimitada"]}}]; pattern_no_limite = [{"LOWER": {"IN": ["no", "sin"]}}, {"LOWER": {"IN": ["hay", "tiene", "existe"]}, "OP": "?"}, {"LEMMA": {"IN": ["límite", "restricción"]}}]
    matcher.add("CAP_NUM1", [pattern_cap_num1]); matcher.add("CAP_NUM2", [pattern_cap_num2]); matcher.add("CAP_CABEN", [pattern_caben]); matcher.add("CAP_NO_MAS_DE", [pattern_no_mas_de]); matcher.add("CAP_K_IGUAL", [pattern_k_igual]); matcher.add("CAP_RECHAZO", [pattern_rechazo]); matcher.add("CAP_SOLO_PERMITEN", [pattern_solo_permiten]); matcher.add("CAP_INFINITA", [pattern_infinita_directa]); matcher.add("CAP_NO_LIMITE", [pattern_no_limite])
    matches = matcher(doc_spacy); found_capacities = []
    for match_id, start, end in matches:
        span = doc_spacy[start:end]; match_name = NLP_SPACY.vocab.strings[match_id]; current_value = None; current_fragment = span.sent.text; match_type = "general_numeric"
        if match_name in ["CAP_INFINITA", "CAP_NO_LIMITE"]: current_value = "infinita"; match_type = "infinite"
        else:
            for token in span:
                if token.like_num:
                    # "un/una" ya no están en el map
                    if token.text.lower() in NUMEROS_EN_PALABRAS_MAP: 
                        current_value = NUMEROS_EN_PALABRAS_MAP[token.text.lower()]
                    else:
                        try: current_value = int(float(token.text.replace(",", ".")))
                        except ValueError: continue
                    break 
            if current_value is not None:
                sentence_text_lower = current_fragment.lower()
                if match_name in ["CAP_RECHAZO", "CAP_SOLO_PERMITEN", "CAP_K_IGUAL", "CAP_NO_MAS_DE"]: match_type = "system_direct_k"; 
                if any(kw_lugar in sentence_text_lower for kw_lugar in keywords_lugar_espera) and not any(kw_sys in sentence_text_lower for kw_sys in ["sistema", "total"]): match_type = "queue_explicit" 
                elif match_name in ["CAP_NUM1", "CAP_NUM2", "CAP_CABEN"]:
                    if any(kw in sentence_text_lower for kw in ["sistema", "total"]): match_type = "system_explicit"
                    elif any(kw_lugar in sentence_text_lower for kw_lugar in keywords_lugar_espera) or match_name == "CAP_CABEN": match_type = "queue_explicit"
        if current_value is not None:
            if isinstance(current_value, (int, float)) and current_value <= 0: continue
            found_capacities.append({"valor": current_value, "fragmento": current_fragment, "span_text": span.text, "match_name": match_name, "match_type": match_type})
    if not found_capacities:
        print("INFO: No se encontró info de capacidad. Asumiendo infinita por defecto."); resultado_parcial["parametros_extraidos"]["capacidad_sistema"]["valor"] = "infinita"; resultado_parcial["parametros_extraidos"]["capacidad_sistema"]["fragmento_texto"] = "Asumida infinita (no encontrada explícitamente)."
    else:
        chosen_cap = None; priority_order = ["system_direct_k", "system_explicit", "general_numeric", "queue_explicit", "infinite"]
        for p_type in priority_order:
            caps_of_type = [fc for fc in found_capacities if fc["match_type"] == p_type]
            if caps_of_type: chosen_cap = caps_of_type[0]; break
        if chosen_cap:
            primary_value = chosen_cap["valor"]; primary_fragment = chosen_cap["fragmento"]
            resultado_parcial["parametros_extraidos"]["capacidad_sistema"]["valor"] = primary_value; resultado_parcial["parametros_extraidos"]["capacidad_sistema"]["fragmento_texto"] = primary_fragment
            print(f"INFO: Capacidad del sistema extraída: {primary_value} (Fuente: '{primary_fragment}', Patrón: {chosen_cap['match_name']}, Tipo: {chosen_cap['match_type']})")
            numeric_caps_all = [fc for fc in found_capacities if isinstance(fc["valor"], (int, float))]; all_found_numeric_values = set(nc["valor"] for nc in numeric_caps_all)
            if len(all_found_numeric_values) > 1 and isinstance(primary_value, (int, float)):
                adv = (f"Advertencia: Múltiples valores numéricos para capacidad. Usado: {primary_value} (Tipo: {chosen_cap['match_type']}). Otros detectados: {[f'{nc["valor"]} (Tipo: {nc["match_type"]})' for nc in found_capacities if isinstance(nc['valor'], (int,float)) and nc['valor'] != primary_value]}")
                resultado_parcial["errores"].append(adv); print(f"ADVERTENCIA: {adv}")
            if numeric_caps_all and any(fc["match_type"] == "infinite" for fc in found_capacities) and isinstance(primary_value, (int, float)):
                adv = (f"Advertencia: Se encontró capacidad numérica ({primary_value}) y mención de 'infinita'. Se priorizó el valor numérico basado en el tipo de patrón ({chosen_cap['match_type']}).")
                resultado_parcial["errores"].append(adv); print(f"ADVERTENCIA: {adv}")
        else: print("INFO: No se pudo determinar la capacidad explícita con prioridades. Asumiendo infinita por defecto."); resultado_parcial["parametros_extraidos"]["capacidad_sistema"]["valor"] = "infinita"; resultado_parcial["parametros_extraidos"]["capacidad_sistema"]["fragmento_texto"] = "Asumida infinita (lógica de selección)."
    return resultado_parcial

def extraer_disciplina_cola(doc_spacy, resultado_parcial):
    # ... (código sin cambios, igual al que me proporcionaste)
    if NLP_SPACY is None: resultado_parcial["errores"].append("Modelo spaCy no cargado en extraer_disciplina_cola."); return resultado_parcial
    matcher = Matcher(NLP_SPACY.vocab); disciplinas_encontradas = []
    pattern_fifo_sigla = [{"TEXT": "FIFO"}]; pattern_fifo_frase_es = [{"LOWER": "primero"}, {"LOWER": "en"}, {"LOWER": "llegar"}, {"IS_PUNCT": True, "OP": "?"}, {"LOWER": "primero"}, {"LOWER": "en"}, {"LOWER": "ser"}, {"LOWER": "servido"}]; pattern_fifo_frase_es_alt = [{"LOWER": "primero"}, {"LOWER": "que"}, {"LOWER": "llega"}, {"LOWER": "es"}, {"LOWER": "el"}, {"LOWER": "primero"}, {"LOWER": "que"}, {"LOWER": "se"}, {"LOWER": "atiende"}]; pattern_fifo_fcfs = [{"TEXT": "FCFS"}]; pattern_orden_llegada = [{"LOWER": "orden"}, {"LOWER": "de"}, {"LOWER": "llegada"}]; pattern_orden_ingreso = [{"LOWER": "orden"}, {"LOWER": "en"}, {"LOWER": "que"}, {"LEMMA": {"IN": ["ir", "llegar", "ingresar"]}}]
    pattern_lifo_sigla = [{"TEXT": "LIFO"}]; pattern_lifo_frase_es = [{"LOWER": "último"}, {"LOWER": "en"}, {"LOWER": "llegar"}, {"IS_PUNCT": True, "OP": "?"}, {"LOWER": "primero"}, {"LOWER": "en"}, {"LOWER": "ser"}, {"LOWER": "servido"}]; pattern_lifo_lcfs = [{"TEXT": "LCFS"}]
    pattern_siro_sigla = [{"TEXT": "SIRO"}]; pattern_siro_frase_es = [{"LOWER": "servicio"}, {"LOWER": "en"}, {"LOWER": "orden"}, {"LOWER": "aleatorio"}]; pattern_siro_random = [{"LOWER": "atención"}, {"LOWER": "aleatoria"}]
    pattern_prioridad_sigla = [{"TEXT": "PRI"}]; pattern_prioridad_frase = [{"LEMMA": {"IN": ["prioridad", "prioritario"]}}]; pattern_prioridad_clases = [{"LEMMA": {"IN": ["clase", "tipo", "nivel"]}}, {"LOWER": "de", "OP": "?"}, {"LEMMA": "prioridad"}]
    matcher.add("DISC_FIFO_SIGLA", [pattern_fifo_sigla]); matcher.add("DISC_FIFO_FRASE_ES", [pattern_fifo_frase_es]); matcher.add("DISC_FIFO_FRASE_ES_ALT", [pattern_fifo_frase_es_alt]); matcher.add("DISC_FIFO_FCFS", [pattern_fifo_fcfs]); matcher.add("DISC_ORDEN_LLEGADA", [pattern_orden_llegada]); matcher.add("DISC_ORDEN_INGRESO", [pattern_orden_ingreso])
    matcher.add("DISC_LIFO_SIGLA", [pattern_lifo_sigla]); matcher.add("DISC_LIFO_FRASE_ES", [pattern_lifo_frase_es]); matcher.add("DISC_LIFO_LCFS", [pattern_lifo_lcfs])
    matcher.add("DISC_SIRO_SIGLA", [pattern_siro_sigla]); matcher.add("DISC_SIRO_FRASE_ES", [pattern_siro_frase_es]); matcher.add("DISC_SIRO_RANDOM", [pattern_siro_random])
    matcher.add("DISC_PRIORIDAD_SIGLA", [pattern_prioridad_sigla]); matcher.add("DISC_PRIORIDAD_FRASE", [pattern_prioridad_frase]); matcher.add("DISC_PRIORIDAD_CLASES", [pattern_prioridad_clases])
    matches = matcher(doc_spacy)
    for match_id, start, end in matches:
        span = doc_spacy[start:end]; rule_id_str = NLP_SPACY.vocab.strings[match_id]; disciplina_detectada = None
        if "FIFO" in rule_id_str or "FCFS" in rule_id_str or "ORDEN_LLEGADA" in rule_id_str or "ORDEN_INGRESO" in rule_id_str: disciplina_detectada = "FIFO"
        elif "LIFO" in rule_id_str or "LCFS" in rule_id_str: disciplina_detectada = "LIFO"
        elif "SIRO" in rule_id_str or "RANDOM" in rule_id_str: disciplina_detectada = "SIRO"
        elif "PRIORIDAD" in rule_id_str: disciplina_detectada = "Prioridad"
        if disciplina_detectada: disciplinas_encontradas.append({"valor": disciplina_detectada, "fragmento": span.sent.text, "span_text": span.text})
    if not disciplinas_encontradas: print("INFO: No se encontró disciplina de cola explícita. Se establece como no especificado."); resultado_parcial["parametros_extraidos"]["disciplina_cola"]["valor"] = None; resultado_parcial["parametros_extraidos"]["disciplina_cola"]["fragmento_texto"] = None
    else:
        chosen_discipline = disciplinas_encontradas[0]; primary_value = chosen_discipline["valor"]; primary_fragment = chosen_discipline["fragmento"]
        resultado_parcial["parametros_extraidos"]["disciplina_cola"]["valor"] = primary_value; resultado_parcial["parametros_extraidos"]["disciplina_cola"]["fragmento_texto"] = primary_fragment
        print(f"INFO: Disciplina de cola extraída: {primary_value} (Fuente: '{primary_fragment}')")
        unique_disciplines = set(d["valor"] for d in disciplinas_encontradas)
        if len(unique_disciplines) > 1:
            advertencia = (f"Advertencia: Se encontraron múltiples menciones de disciplinas de cola diferentes. Se utilizó la primera detectada ('{primary_value}' de '{primary_fragment}'). Otras detectadas: {[f'{d["valor"]} (en \"{d["span_text"]}\")' for d in disciplinas_encontradas if d["valor"] != primary_value]}")
            resultado_parcial["errores"].append(advertencia); print(f"ADVERTENCIA DETALLADA: {advertencia}")
    return resultado_parcial

# --- Función Principal de Extracción ---
def extraer_parametros_colas(texto_entrada, umbral_similitud_candidatas=0.6, debug_specific_sentence_part=None):
    if NLP_SPACY is None or MODEL_SENTENCE_TRANSFORMERS is None or \
       not EMBEDDINGS_FRASES_CLAVE or \
       EMBEDDINGS_FRASES_CLAVE.get("llegada") is None or \
       EMBEDDINGS_FRASES_CLAVE.get("servicio") is None:
        print("Intentando recargar modelos y/o embeddings...")
        cargar_modelos_y_precalcular_embeddings() 
        if NLP_SPACY is None or MODEL_SENTENCE_TRANSFORMERS is None or \
           not EMBEDDINGS_FRASES_CLAVE or \
           EMBEDDINGS_FRASES_CLAVE.get("llegada") is None or \
           EMBEDDINGS_FRASES_CLAVE.get("servicio") is None:
            res_error = inicializar_estructura_salida()
            res_error["errores"].append("Fallo crítico al cargar modelos NLP o embeddings de frases clave.")
            return res_error

    doc_spacy, resultado_parcial = procesar_texto_basico(texto_entrada)
    if doc_spacy is None: return resultado_parcial 

    resultado_parcial = extraer_numero_servidores(doc_spacy, resultado_parcial)
    resultado_parcial = extraer_capacidad_sistema(doc_spacy, resultado_parcial) 
    resultado_parcial = extraer_disciplina_cola(doc_spacy, resultado_parcial)

    oraciones_candidatas = identificar_oraciones_candidatas(doc_spacy, umbral_similitud_candidatas, debug_specific_sentence_part=debug_specific_sentence_part) 
    resultado_parcial["oraciones_candidatas_debug"] = oraciones_candidatas 

    print("\n--- Oraciones Candidatas Detectadas (Debug) ---")
    if oraciones_candidatas.get("llegada"): 
        print("Llegada:")
        for cand in oraciones_candidatas["llegada"]:
            print(f"  - Sim: {cand['similitud']:.4f}, Oración: \"{cand['oracion_texto']}\"")
    if oraciones_candidatas.get("servicio"):
        print("Servicio:")
        for cand in oraciones_candidatas["servicio"]:
            print(f"  - Sim: {cand['similitud']:.4f}, Oración: \"{cand['oracion_texto']}\"")
    print("--------------------------------------------")
    
    # --- Tarea 2.3 y 2.4 (Poblado de tasas/tiempos) ---
    processed_sentences_for_llegada = set()
    if resultado_parcial["oraciones_candidatas_debug"]["llegada"]:
        candidatas_llegada_sorted = sorted(resultado_parcial["oraciones_candidatas_debug"]["llegada"], key=lambda x: x["similitud"], reverse=True)
        
        for cand_info_llegada in candidatas_llegada_sorted:
            oracion_llegada_principal = cand_info_llegada["oracion_texto"]
            if oracion_llegada_principal in processed_sentences_for_llegada:
                continue 

            servicio_cand_match = next((s_cand for s_cand in resultado_parcial["oraciones_candidatas_debug"].get("servicio", []) if s_cand["oracion_texto"] == oracion_llegada_principal), None)
            if servicio_cand_match and \
               (resultado_parcial["parametros_extraidos"]["tasa_servicio_por_servidor"]["fragmento_texto"] == oracion_llegada_principal or \
                resultado_parcial["parametros_extraidos"]["tiempo_servicio_por_servidor"]["fragmento_texto"] == oracion_llegada_principal) and \
               servicio_cand_match["similitud"] > cand_info_llegada["similitud"] + 0.05: 
                print(f"INFO: Oración '{oracion_llegada_principal}' ya asignada a servicio con mayor similitud o ya procesada. Omitiendo para llegada.")
                continue 
            
            datos_llegada = extraer_valor_y_unidad_de_oracion(oracion_llegada_principal)
            if datos_llegada:
                print(f"DEBUG - Datos LLEGADA extraídos de '{oracion_llegada_principal}': {datos_llegada}")
                if resultado_parcial["parametros_extraidos"]["tasa_llegada"]["valor"] is None and \
                   resultado_parcial["parametros_extraidos"]["tiempo_entre_llegadas"]["valor"] is None:
                    if datos_llegada["tipo_parametro"] == "tasa":
                        resultado_parcial["parametros_extraidos"]["tasa_llegada"]["valor"] = datos_llegada["valor"]
                        resultado_parcial["parametros_extraidos"]["tasa_llegada"]["unidades"] = datos_llegada["unidad_texto"]
                        resultado_parcial["parametros_extraidos"]["tasa_llegada"]["fragmento_texto"] = oracion_llegada_principal
                    elif datos_llegada["tipo_parametro"] == "tiempo":
                        resultado_parcial["parametros_extraidos"]["tiempo_entre_llegadas"]["valor"] = datos_llegada["valor"]
                        resultado_parcial["parametros_extraidos"]["tiempo_entre_llegadas"]["unidades"] = datos_llegada["unidad_texto"]
                        resultado_parcial["parametros_extraidos"]["tiempo_entre_llegadas"]["fragmento_texto"] = oracion_llegada_principal
                    
                    if "poisson" in oracion_llegada_principal.lower():
                         resultado_parcial["parametros_extraidos"]["tasa_llegada"]["distribucion"] = "Poisson"
                         resultado_parcial["parametros_extraidos"]["tiempo_entre_llegadas"]["distribucion"] = "Exponencial" 
                    elif "exponencial" in oracion_llegada_principal.lower():
                         resultado_parcial["parametros_extraidos"]["tasa_llegada"]["distribucion"] = "Poisson" 
                         resultado_parcial["parametros_extraidos"]["tiempo_entre_llegadas"]["distribucion"] = "Exponencial"
                    processed_sentences_for_llegada.add(oracion_llegada_principal) 

    processed_sentences_for_servicio = set()
    if resultado_parcial["oraciones_candidatas_debug"]["servicio"]:
        candidatas_servicio_sorted = sorted(resultado_parcial["oraciones_candidatas_debug"]["servicio"], key=lambda x: x["similitud"], reverse=True)

        for cand_info_servicio in candidatas_servicio_sorted:
            oracion_servicio_principal = cand_info_servicio["oracion_texto"]
            if oracion_servicio_principal in processed_sentences_for_servicio:
                continue
            
            if oracion_servicio_principal == resultado_parcial["parametros_extraidos"]["tasa_llegada"]["fragmento_texto"] or \
               oracion_servicio_principal == resultado_parcial["parametros_extraidos"]["tiempo_entre_llegadas"]["fragmento_texto"]:
                llegada_cand_match = next((l_cand for l_cand in resultado_parcial["oraciones_candidatas_debug"].get("llegada", []) if l_cand["oracion_texto"] == oracion_servicio_principal), None)
                if llegada_cand_match and llegada_cand_match["similitud"] >= cand_info_servicio["similitud"] - 0.05:
                    print(f"INFO: Oración '{oracion_servicio_principal}' ya asignada a llegada o con similitud no dominante para servicio. Omitiendo para servicio.")
                    continue
            
            datos_servicio = extraer_valor_y_unidad_de_oracion(oracion_servicio_principal)
            if datos_servicio:
                print(f"DEBUG - Datos SERVICIO extraídos de '{oracion_servicio_principal}': {datos_servicio}")
                if resultado_parcial["parametros_extraidos"]["tasa_servicio_por_servidor"]["valor"] is None and \
                   resultado_parcial["parametros_extraidos"]["tiempo_servicio_por_servidor"]["valor"] is None:
                    if datos_servicio["tipo_parametro"] == "tasa":
                        resultado_parcial["parametros_extraidos"]["tasa_servicio_por_servidor"]["valor"] = datos_servicio["valor"]
                        resultado_parcial["parametros_extraidos"]["tasa_servicio_por_servidor"]["unidades"] = datos_servicio["unidad_texto"]
                        resultado_parcial["parametros_extraidos"]["tasa_servicio_por_servidor"]["fragmento_texto"] = oracion_servicio_principal
                    elif datos_servicio["tipo_parametro"] == "tiempo":
                        resultado_parcial["parametros_extraidos"]["tiempo_servicio_por_servidor"]["valor"] = datos_servicio["valor"]
                        resultado_parcial["parametros_extraidos"]["tiempo_servicio_por_servidor"]["unidades"] = datos_servicio["unidad_texto"]
                        resultado_parcial["parametros_extraidos"]["tiempo_servicio_por_servidor"]["fragmento_texto"] = oracion_servicio_principal
                    
                    if "poisson" in oracion_servicio_principal.lower():
                        resultado_parcial["parametros_extraidos"]["tasa_servicio_por_servidor"]["distribucion"] = "Poisson" 
                        resultado_parcial["parametros_extraidos"]["tiempo_servicio_por_servidor"]["distribucion"] = "Exponencial"
                    elif "exponencial" in oracion_servicio_principal.lower():
                        resultado_parcial["parametros_extraidos"]["tasa_servicio_por_servidor"]["distribucion"] = "Poisson"
                        resultado_parcial["parametros_extraidos"]["tiempo_servicio_por_servidor"]["distribucion"] = "Exponencial"
                    processed_sentences_for_servicio.add(oracion_servicio_principal)
    return resultado_parcial

if __name__ == "__main__":
    cargar_modelos_y_precalcular_embeddings() 

    textos_prueba_tasas_tiempos = [
        "Los clientes llegan a una tasa de 10 por hora.",
        "El tiempo promedio de servicio es de 5 minutos por cliente.",
        "Se espera que arriben unidades cada 2 minutos.", 
        "La capacidad de atención del sistema es de 15 trabajos por día.",
        "Un nuevo pedido entra al sistema aproximadamente cada 30 segundos.", 
        "El servidor procesa los ítems con una frecuencia de 60 ítems/hora.", 
        "La tasa esperada de llegada de pacientes es de 3 por hora.", 
        "El tiempo medio entre llegadas de vehículos es de 120 segundos.",
        "La tasa media de servicio por cajero es de 25 clientes/hora.", 
        "El tiempo esperado de servicio para cada tarea es de 10 minutos.",
        "Los pacientes llegan a la clínica de un médico de acuerdo con una distribución de Poisson a razón de treinta pacientes por hora." 
    ]

    for i, texto_test in enumerate(textos_prueba_tasas_tiempos):
        print(f"\n--- Procesando Texto de Prueba Tasas/Tiempos #{i+1}: '{texto_test}' ---")
        if NLP_SPACY and MODEL_SENTENCE_TRANSFORMERS and \
           EMBEDDINGS_FRASES_CLAVE.get("llegada") is not None and \
           EMBEDDINGS_FRASES_CLAVE.get("servicio") is not None: 
            info_extraida = extraer_parametros_colas(texto_test, umbral_similitud_candidatas=0.55) 
            print("\n--- Resultado Extracción (JSON) ---") 
            print(json.dumps(info_extraida, indent=4, ensure_ascii=False))
            print("-----------------------------------\n")
        else:
            print("Modelos o embeddings de frases clave no cargados, saltando prueba.")
    
    print("\n--- Prueba con archivo ejemplo1.txt (si existe) ---")
    ruta_ejemplo = os.path.join(DATA_DIR, "ejemplo1.txt")
    texto_ejemplo_usuario = "Los pacientes llegan a la clínica de un médico de acuerdo con una distribución de Poisson a razón de 20 pacientes por hora. La sala de espera no puede acomodar más de 14 pacientes. El tiempo de consulta por paciente es exponencial, con una media de 8 minutos."
    
    try:
        print(f"\n--- Procesando Texto Específico del Usuario ---")
        print(texto_ejemplo_usuario)
        print("------------------------------------")
        if NLP_SPACY and MODEL_SENTENCE_TRANSFORMERS and \
           EMBEDDINGS_FRASES_CLAVE.get("llegada") is not None and \
           EMBEDDINGS_FRASES_CLAVE.get("servicio") is not None:
            info_extraida_usuario = extraer_parametros_colas(
                texto_ejemplo_usuario, 
                umbral_similitud_candidatas=0.50, 
                debug_specific_sentence_part="a razón de 20 pacientes por hora" 
            ) 
            print("\n--- Resultado (JSON) - Texto Usuario ---")
            print(json.dumps(info_extraida_usuario, indent=4, ensure_ascii=False))
            print("--------------------------\n")
        else:
            print("Modelos o embeddings de frases clave no cargados, saltando prueba del usuario.")

        if os.path.exists(ruta_ejemplo):
            # Para asegurar que el archivo de regresión tiene el contenido deseado para esta prueba
            with open(ruta_ejemplo, 'w', encoding='utf-8') as f_write: 
                f_write.write(texto_ejemplo_usuario) 
            
            with open(ruta_ejemplo, 'r', encoding='utf-8') as f:
                texto_ejemplo_regresion = f.read() 
            print(f"--- Procesando Archivo de Regresión: {ruta_ejemplo} ---")
            print(texto_ejemplo_regresion) 
            print("------------------------------------")
            if NLP_SPACY and MODEL_SENTENCE_TRANSFORMERS and \
               EMBEDDINGS_FRASES_CLAVE.get("llegada") is not None and \
               EMBEDDINGS_FRASES_CLAVE.get("servicio") is not None:
                info_extraida_regresion = extraer_parametros_colas(texto_ejemplo_regresion, umbral_similitud_candidatas=0.50, debug_specific_sentence_part="a razón de 20 pacientes por hora") 
                print("\n--- Resultado (JSON) - Archivo Regresión ---")
                print(json.dumps(info_extraida_regresion, indent=4, ensure_ascii=False))
                print("--------------------------\n")
        else:
            print(f"Archivo de regresión {ruta_ejemplo} no encontrado.")

    except Exception as e:
        print(f"Error durante las pruebas: {e}")


#el de 500 y algo funciona y este tambien:

# src/nlp_pipeline.py

import spacy
from spacy.matcher import Matcher
from sentence_transformers import SentenceTransformer, util
import json
import os
import re

# --- Definición de Frases Clave para Parámetros (Iteración v7) ---
FRASES_CLAVE_PARAMETROS = {
    "llegada": [
        # Énfasis en "tasa de llegada" y variantes directas
        "tasa de llegada", "la tasa de llegada es", "tasa de llegada de clientes", "tasa de llegada de unidades",
        "tasa media de llegada", "la tasa media de llegada es", "tasa esperada de llegada", "la tasa esperada de llegada es",
        "razón de llegada", "la razón de llegada es", "a razón de llegadas", "llegan a razón de",
        "frecuencia de llegada", "la frecuencia de llegada es", "frecuencia de arribo", "tasa de arribo",
        "llegan clientes a una tasa de", "arriban unidades con una frecuencia de",
        "pacientes se presentan a razón de", "el flujo de llegada es de",
        "llegan en promedio X por Y", "las llegadas ocurren a razón de",
        "se registran llegadas con una tasa de", "arribos por hora", "llegadas por minuto",
        "clientes por hora llegan", "unidades por día entran",

        "llegan según Poisson a razón de", "llegadas Poisson con media de",
        "tasa de llegadas de Poisson", "proceso de llegada Poisson con tasa",
        "llegan de acuerdo con una distribución de Poisson a razón de",
        "las llegadas siguen una distribución de Poisson con tasa de",
        "la razón de arribo es", "la frecuencia de entrada es", "tasa de entrada",
        "patrón de llegadas", "patrón de arribos",

        # Énfasis en "tiempo entre llegadas" y variantes
        "tiempo entre llegadas", "el tiempo entre llegadas es", "tiempo promedio entre llegadas",
        "intervalo promedio entre arribos", "tiempo esperado entre llegadas",
        "tiempo medio entre llegadas", "intervalo de llegada",
        "llega un cliente cada", "los clientes ingresan cada", "unidades arriban cada",
        "se espera que arriben unidades cada",
        "un nuevo pedido entra al sistema cada",
        "un arribo cada", "la entrada de clientes es cada", "tiempo entre arribos de",
        "intervalo entre llegadas exponencial con media de", "tiempo entre llegadas sigue una exponencial",
        "el tiempo entre llegadas sigue una distribución de Poisson", # Frase potencialmente conflictiva, se manejará
        "tiempo entre llegadas distribuido exponencialmente con media de",

        # Generales de llegada
        "entran al sistema", "se presentan en la cola", "la llegada de clientes sigue una distribución",
        "llegan según un proceso de Poisson", "las llegadas ocurren", "se registran llegadas",
        "llegan al sistema", "ingresan a la cola", "se espera la llegada de",
        "nuevos pedidos entran", "entrada de pedidos", "se reciben pedidos", "flujo de entrada"
    ],
    "servicio": [
        # Énfasis en "tasa de servicio" y variantes directas
        "tasa de servicio", "la tasa de servicio es", "tasa de servicio por servidor",
        "tasa media de servicio", "la tasa media de servicio es", "tasa esperada de servicio", "la tasa esperada de servicio es",
        "razón de servicio", "la razón de servicio es", "a razón de servicios", "atiende a razón de",
        "capacidad de atención", "velocidad de servicio", "ritmo de servicio", "tasa de proceso",
        "frecuencia de servicio", "la frecuencia de servicio es", "tasa de atención por servidor",
        "procesa unidades a una tasa de", "el servidor atiende a una tasa de",
        "atiende en promedio X por Y", "clientes atendidos por hora", "unidades procesadas por minuto",
        "frecuencia con que se procesa", "tasa de servicio exponencial",
        "el servicio es Poisson con tasa de", "tasa de servicio del servidor",
        "el servidor puede procesar a una tasa de", "tasa de procesamiento del servidor",

        # Énfasis en "tiempo de servicio" y variantes
        "tiempo de servicio", "el tiempo de servicio es", "tiempo promedio de servicio",
        "tiempo esperado de servicio", "la duración del servicio es", "tiempo promedio para atender",
        "tiempo de procesamiento", "el tiempo de procesamiento es",
        "demora en atender", "atiende clientes cada", "se procesan ítems cada",
        "el servidor tarda en procesar", "el tiempo de atención sigue una distribución",
        "tiempo de consulta por paciente", "la duración media del servicio es",
        "tiempo para completar una tarea", "el servicio toma en promedio",
        "tiempo de ocupación del servidor", "tiempo de servicio exponencial con media de",
        "el tiempo de servicio sigue una exponencial de",
        "el tiempo de servicio sigue una distribución de Poisson", # Frase potencialmente conflictiva
        "tiempo de servicio distribuido exponencialmente con media de",
        # Generales de servicio
        "el sistema procesa", "patrón de servicio", "el servidor puede procesar", "capacidad de procesamiento"
    ],
    "servidores": ["número de servidores", "cantidad de cajeros", "puestos de atención", "operarios disponibles", "cuántas máquinas hay", "dispone de servidores"],
    "capacidad_sistema": ["capacidad del sistema", "tamaño de la cola", "límite de clientes en el sistema", "espacio en la sala de espera", "cuántos caben como máximo", "capacidad de k", "capacidad es de", "capacidad máxima", "rechaza clientes si hay más de", "solo se permiten", "buffer de tamaño"],
    "disciplina_cola": ["disciplina de la cola", "orden de atención", "tipo de cola", "cómo se atiende", "FIFO", "LIFO", "primero en llegar primero en ser servido", "atención por prioridad", "orden de llegada"]
}

# --- Model Loading ---
NLP_SPACY = None
MODEL_SENTENCE_TRANSFORMERS = None
EMBEDDINGS_FRASES_CLAVE = {}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# --- Diccionario para números en palabras (MODIFICADO POR USUARIO) ---
NUMEROS_EN_PALABRAS_MAP = {
    "cero": 0, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, # Eliminado "un", "una", "uno"
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12,
    "trece": 13, "catorce": 14, "quince": 15, "dieciséis": 16, "diecisiete": 17,
    "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiun": 21, "veintiuno": 21,
    "veintidós": 22, "veintitrés": 23, "veinticuatro": 24, "veinticinco": 25,
    "veintiséis": 26, "veintisiete": 27, "veintiocho": 28, "veintinueve": 29,
    "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60,
    "setenta": 70, "ochenta": 80, "noventa": 90, "cien": 100, "ciento": 100,
}
PALABRAS_NUMERO_REGEX = r"(?<!\w)(" + "|".join(NUMEROS_EN_PALABRAS_MAP.keys()) + r")(?!\w)"
DIGITAL_NUMERO_REGEX = r"\b\d+([.,]\d+)?\b"


def cargar_modelos_y_precalcular_embeddings():
    global NLP_SPACY, MODEL_SENTENCE_TRANSFORMERS, EMBEDDINGS_FRASES_CLAVE
    if NLP_SPACY and MODEL_SENTENCE_TRANSFORMERS and \
        EMBEDDINGS_FRASES_CLAVE.get("llegada") is not None and \
        EMBEDDINGS_FRASES_CLAVE.get("servicio") is not None:
        return
    print("Cargando modelos NLP y precalculando embeddings de frases clave...")
    try:
        if NLP_SPACY is None: NLP_SPACY = spacy.load("es_core_news_sm")
        if MODEL_SENTENCE_TRANSFORMERS is None: MODEL_SENTENCE_TRANSFORMERS = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        if not EMBEDDINGS_FRASES_CLAVE or EMBEDDINGS_FRASES_CLAVE.get("llegada") is None or EMBEDDINGS_FRASES_CLAVE.get("servicio") is None:
            EMBEDDINGS_FRASES_CLAVE = {}
            for cat, frases in FRASES_CLAVE_PARAMETROS.items():
                if cat in ["llegada", "servicio"] and frases:
                    EMBEDDINGS_FRASES_CLAVE[cat] = MODEL_SENTENCE_TRANSFORMERS.encode(frases, convert_to_tensor=True)
                elif cat in ["llegada", "servicio"]: EMBEDDINGS_FRASES_CLAVE[cat] = None
            if EMBEDDINGS_FRASES_CLAVE.get("llegada") is not None and EMBEDDINGS_FRASES_CLAVE.get("servicio") is not None: print("Embeddings precalculados.")
            else: print("Advertencia: No se generaron embeddings para llegada/servicio.")
        else: print("Embeddings ya precalculados.")
        print("Modelos NLP listos.")
    except Exception as e:
        print(f"Error cargando modelos/embeddings: {e}")
        NLP_SPACY, MODEL_SENTENCE_TRANSFORMERS, EMBEDDINGS_FRASES_CLAVE = None, None, {}

def inicializar_estructura_salida():
    return {
        "texto_original": None,
        "parametros_extraidos": {
            "tasa_llegada": {"valor": None, "unidades": None, "distribucion": None, "fragmento_texto": None},
            "tiempo_entre_llegadas": {"valor": None, "unidades": None, "distribucion": None, "fragmento_texto": None},
            "tasa_servicio_por_servidor": {"valor": None, "unidades": None, "distribucion": None, "fragmento_texto": None},
            "tiempo_servicio_por_servidor": {"valor": None, "unidades": None, "distribucion": None, "fragmento_texto": None},
            "cantidad_servidores": {"valor": None, "fragmento_texto": None},
            "capacidad_sistema": {"valor": None, "fragmento_texto": None},
            "disciplina_cola": {"valor": None, "fragmento_texto": None}
        },
        "oraciones_candidatas_debug": {"llegada": [], "servicio": []}, "errores": []
    }

def procesar_texto_basico(texto_entrada):
    if NLP_SPACY is None:
        resultado = inicializar_estructura_salida(); resultado["errores"].append("spaCy no cargado."); return None, resultado
    doc_spacy = NLP_SPACY(texto_entrada)
    resultado = inicializar_estructura_salida(); resultado["texto_original"] = texto_entrada
    return doc_spacy, resultado

# --- Tarea 2.2: Extracción de Valor Numérico y Unidades ---
def extraer_valor_y_unidad_de_oracion(texto_oracion_candidata):
    """
    Extrae un valor numérico (digital o palabra) y sus unidades asociadas (tasa o tiempo)
    de una oración candidata.
    Aplica lógica contextual para tasas si la entidad está separada.
    """
    unidades_tiempo_singular_list = ["segundo", "minuto", "hora", "día", "semana", "mes", "año"]
    unidades_tiempo_plural_list = ["segundos", "minutos", "horas", "días", "semanas", "meses", "años"]
    known_time_units_set = set(unidades_tiempo_singular_list + unidades_tiempo_plural_list)

    # Regex para una unidad de tiempo (singular o plural). Captura la unidad de tiempo en group(1).
    unidades_tiempo_regex_str = rf"\b({'|'.join(unidades_tiempo_singular_list + unidades_tiempo_plural_list)})\b"

    entidades_comunes_list = ["cliente", "paciente", "unidad", "ítem", "item", "trabajo", "pedido", "vehículo", "tarea", "consulta", "llamada", "operacion", "evento"]
    # Regex para una entidad común (singular o plural usando 's?'). Captura la entidad en group(1).
    entidades_comunes_regex_str = r"\b(" + "|".join([e + "s?" for e in entidades_comunes_list]) + r")\b"
    
    # Patrones:
    # Patrón 1: [ENTIDAD] (por|al|/) [UNIDAD_TIEMPO] -> group(1) es entidad, group(2) es unidad de tiempo
    patron_tasa_completa = rf"({entidades_comunes_regex_str})\s*(?:por|al|/)\s*({unidades_tiempo_regex_str})"
    # Patrón 2: [UNIDAD_TIEMPO] (por|al|/) [ENTIDAD] -> group(1) es unidad de tiempo, group(2) es entidad
    patron_tiempo_por_entidad = rf"({unidades_tiempo_regex_str})\s*(?:por|al|/)\s*({entidades_comunes_regex_str})"
    # Patrón 3: (por|al) [UNIDAD_TIEMPO] -> group(1) es unidad de tiempo
    patron_por_unidad_tiempo = rf"(?:por|al)\s*({unidades_tiempo_regex_str})"
    # Patrón 4: [UNIDAD_TIEMPO] (sola) -> group(1) es unidad de tiempo
    patron_unidad_tiempo_sola = unidades_tiempo_regex_str 


    posibles_numeros = []
    for match_palabra in re.finditer(PALABRAS_NUMERO_REGEX, texto_oracion_candidata, re.IGNORECASE):
        palabra = match_palabra.group(1).lower()
        if palabra in NUMEROS_EN_PALABRAS_MAP:
            posibles_numeros.append({
                "valor": float(NUMEROS_EN_PALABRAS_MAP[palabra]), "texto": palabra,
                "pos": match_palabra.span(), "es_palabra": True
            })
    for match_digital in re.finditer(DIGITAL_NUMERO_REGEX, texto_oracion_candidata):
        texto_num = match_digital.group(0)
        try:
            posibles_numeros.append({
                "valor": float(texto_num.replace(",", ".")), "texto": texto_num,
                "pos": match_digital.span(), "es_palabra": False
            })
        except ValueError: continue

    if not posibles_numeros: return None
    posibles_numeros.sort(key=lambda x: x["pos"][0])

    for num_info in posibles_numeros:
        num_start_pos, num_end_pos = num_info["pos"]
        texto_despues_num = texto_oracion_candidata[num_end_pos:].lstrip()

        # Prioridad 0: "cada [NUM] [UNIDAD_TIEMPO]"
        texto_antes_num_para_cada = texto_oracion_candidata[:num_start_pos].lower().rstrip()
        if texto_antes_num_para_cada.endswith("cada"):
            match_unidad_cada = re.match(patron_unidad_tiempo_sola, texto_despues_num, re.IGNORECASE)
            if match_unidad_cada:
                unidad_texto = match_unidad_cada.group(1).strip().lower()
                if unidad_texto in known_time_units_set:
                    # Posición de la unidad de tiempo después de "cada NUM"
                    pos_unidad_rel_start = texto_despues_num.find(match_unidad_cada.group(0))
                    pos_unidad_abs_start = num_end_pos + pos_unidad_rel_start
                    pos_unidad_abs_end = pos_unidad_abs_start + len(match_unidad_cada.group(0))
                    return {
                        "valor": num_info["valor"], "valor_texto": num_info["texto"],
                        "unidad_texto": unidad_texto, "tipo_parametro": "tiempo",
                        "posicion_valor": num_info["pos"],
                        "posicion_unidad": (pos_unidad_abs_start, pos_unidad_abs_end)
                    }
        
        # Prioridad 1: Tasa explícita ([NUM] [ENTIDAD] por/al// [UNIDAD_TIEMPO])
        match_tasa = re.match(patron_tasa_completa, texto_despues_num, re.IGNORECASE)
        if match_tasa:
            entidad_match = match_tasa.group(1).lower()
            unidad_tiempo_match = match_tasa.group(2).lower() 

            if unidad_tiempo_match in known_time_units_set and (entidad_match not in known_time_units_set or entidad_match == "entidad"): # Permitir "entidad" genérica
                unidad_final = f"{entidad_match}/{unidad_tiempo_match}"
                pos_unidad_inicio = num_end_pos + texto_despues_num.find(match_tasa.group(0))
                pos_unidad_fin = pos_unidad_inicio + len(match_tasa.group(0))
                return {"valor": num_info["valor"], "valor_texto": num_info["texto"], "unidad_texto": unidad_final,
                        "tipo_parametro": "tasa", "posicion_valor": num_info["pos"], "posicion_unidad": (pos_unidad_inicio, pos_unidad_fin)}
            # else:
                # print(f"DEBUG: patron_tasa_completa: entidad='{entidad_match}', unidad_tiempo='{unidad_tiempo_match}'. Descartado. Texto: '{texto_despues_num}'")
        
        # Prioridad 2: Tiempo por entidad ([NUM] [UNIDAD_TIEMPO] por/al// [ENTIDAD])
        match_tiempo_entidad = re.match(patron_tiempo_por_entidad, texto_despues_num, re.IGNORECASE)
        if match_tiempo_entidad:
            unidad_tiempo_match = match_tiempo_entidad.group(1).lower()
            entidad_match = match_tiempo_entidad.group(2).lower()

            if unidad_tiempo_match in known_time_units_set and (entidad_match not in known_time_units_set or entidad_match == "entidad"):
                unidad_final = f"{unidad_tiempo_match}/{entidad_match}"
                pos_unidad_inicio = num_end_pos + texto_despues_num.find(match_tiempo_entidad.group(0))
                pos_unidad_fin = pos_unidad_inicio + len(match_tiempo_entidad.group(0))
                return {"valor": num_info["valor"], "valor_texto": num_info["texto"], "unidad_texto": unidad_final,
                        "tipo_parametro": "tiempo", "posicion_valor": num_info["pos"], "posicion_unidad": (pos_unidad_inicio, pos_unidad_fin)}
            # else:
            #     print(f"DEBUG: patron_tiempo_por_entidad: unidad_tiempo='{unidad_tiempo_match}', entidad='{entidad_match}'. Descartado. Texto: '{texto_despues_num}'")

        # Prioridad 3: Tiempo simple ([NUM] [UNIDAD_TIEMPO]) o Tasa contextual 
        match_p_ut = re.match(patron_por_unidad_tiempo, texto_despues_num, re.IGNORECASE) # e.g., "por hora"
        match_ut_sola = re.match(patron_unidad_tiempo_sola, texto_despues_num, re.IGNORECASE) # e.g., "hora"
        
        current_match_obj_p3 = None
        ut_str_local_p3 = None

        if match_p_ut:
            potential_time_unit = match_p_ut.group(1).lower()
            if potential_time_unit in known_time_units_set:
                current_match_obj_p3 = match_p_ut
                ut_str_local_p3 = potential_time_unit
        
        if not current_match_obj_p3 and match_ut_sola:
            potential_time_unit = match_ut_sola.group(1).lower()
            if potential_time_unit in known_time_units_set:
                current_match_obj_p3 = match_ut_sola
                ut_str_local_p3 = potential_time_unit
        
        if current_match_obj_p3 and ut_str_local_p3:
            tipo_param_final_p3 = "tiempo"
            unidad_texto_final_p3 = ut_str_local_p3
            
            keywords_tasa_contexto_regex = r"\b(tasa|frecuencia|razón\s+de|ritmo\s+de|velocidad\s+de)\b"
            
            is_contextual_rate = False
            if re.search(keywords_tasa_contexto_regex, texto_oracion_candidata, re.IGNORECASE):
                is_contextual_rate = True
            if match_p_ut and current_match_obj_p3 == match_p_ut : # Estructura "por/al UNIDAD_TIEMPO"
                is_contextual_rate = True

            if is_contextual_rate:
                tipo_param_final_p3 = "tasa"
                entidad_contextual_p3 = None
                
                texto_antes_num_completo = texto_oracion_candidata[:num_start_pos].strip()
                entidades_halladas_antes = list(re.finditer(rf"({entidades_comunes_regex_str})\b", texto_antes_num_completo, re.IGNORECASE))
                if entidades_halladas_antes:
                    temp_ent = entidades_halladas_antes[-1].group(1).lower()
                    if temp_ent not in known_time_units_set:
                        entidad_contextual_p3 = temp_ent
                
                if not entidad_contextual_p3 and (match_p_ut and current_match_obj_p3 == match_p_ut):
                    # Busca entidad entre el número y "por/al unidad" ej: "60 clientes por hora"
                    # texto_despues_num es "clientes por hora", current_match_obj_p3.group(0) es "por hora"
                    # current_match_obj_p3.start() es el inicio de "por hora" en texto_despues_num
                    text_between_num_and_por_al_unit = texto_despues_num[:current_match_obj_p3.start()].strip()
                    if text_between_num_and_por_al_unit:
                        # Usar re.fullmatch para asegurar que toda la cadena intermedia sea la entidad
                        match_entidad_intermedia = re.fullmatch(entidades_comunes_regex_str, text_between_num_and_por_al_unit, re.IGNORECASE)
                        if match_entidad_intermedia:
                            temp_ent = match_entidad_intermedia.group(1).lower()
                            if temp_ent not in known_time_units_set:
                                entidad_contextual_p3 = temp_ent
                
                if entidad_contextual_p3:
                    unidad_texto_final_p3 = f"{entidad_contextual_p3}/{ut_str_local_p3}"
                else: # Es una tasa pero no se encontró entidad explícita, usar placeholder
                    unidad_texto_final_p3 = f"entidad/{ut_str_local_p3}" 
            
            pos_unidad_inicio_p3 = num_end_pos + texto_despues_num.find(current_match_obj_p3.group(0))
            pos_unidad_fin_p3 = pos_unidad_inicio_p3 + len(current_match_obj_p3.group(0))

            return {"valor": num_info["valor"], "valor_texto": num_info["texto"],
                    "unidad_texto": unidad_texto_final_p3, "tipo_parametro": tipo_param_final_p3,
                    "posicion_valor": num_info["pos"], "posicion_unidad": (pos_unidad_inicio_p3, pos_unidad_fin_p3)}
    return None


def identificar_oraciones_candidatas(doc_spacy, umbral_similitud=0.6, debug_specific_sentence_part=None):
    # ... (código sin cambios, igual al proporcionado por el usuario en la v8) ...
    candidatas = {"llegada": [], "servicio": []}
    if not MODEL_SENTENCE_TRANSFORMERS or not EMBEDDINGS_FRASES_CLAVE or \
        EMBEDDINGS_FRASES_CLAVE.get("llegada") is None or \
        EMBEDDINGS_FRASES_CLAVE.get("servicio") is None:
        print("ERROR: Modelo SentenceTransformer o embeddings de frases clave para llegada/servicio no cargados en identificar_oraciones_candidatas.")
        return candidatas
    
    for sent in doc_spacy.sents:
        sent_text = sent.text
        if not sent_text.strip(): continue
        embedding_oracion = MODEL_SENTENCE_TRANSFORMERS.encode(sent_text, convert_to_tensor=True)
        
        similitudes_llegada = util.cos_sim(embedding_oracion, EMBEDDINGS_FRASES_CLAVE["llegada"])
        max_sim_llegada = 0.0; idx_max_llegada = -1
        if similitudes_llegada.numel() > 0: max_sim_llegada = similitudes_llegada.max().item(); idx_max_llegada = similitudes_llegada.argmax().item()
        if max_sim_llegada >= umbral_similitud:
            candidatas["llegada"].append({"oracion_texto": sent_text, "similitud": round(max_sim_llegada, 4)})

        similitudes_servicio = util.cos_sim(embedding_oracion, EMBEDDINGS_FRASES_CLAVE["servicio"])
        max_sim_servicio = 0.0; idx_max_servicio = -1
        if similitudes_servicio.numel() > 0: max_sim_servicio = similitudes_servicio.max().item(); idx_max_servicio = similitudes_servicio.argmax().item()
        if max_sim_servicio >= umbral_similitud:
            candidatas["servicio"].append({"oracion_texto": sent_text, "similitud": round(max_sim_servicio, 4)})
        
        if debug_specific_sentence_part and debug_specific_sentence_part in sent_text:
            print(f"\nDEBUG para oración: \"{sent_text}\"")
            if idx_max_llegada != -1 and idx_max_llegada < len(FRASES_CLAVE_PARAMETROS['llegada']): print(f"  Max Sim Llegada: {max_sim_llegada:.4f} (con frase clave: '{FRASES_CLAVE_PARAMETROS['llegada'][idx_max_llegada]}')")
            else: print(f"  Max Sim Llegada: {max_sim_llegada:.4f} (sin match de frase clave o índice fuera de rango)")
            if idx_max_servicio != -1 and idx_max_servicio < len(FRASES_CLAVE_PARAMETROS['servicio']): print(f"  Max Sim Servicio: {max_sim_servicio:.4f} (con frase clave: '{FRASES_CLAVE_PARAMETROS['servicio'][idx_max_servicio]}')")
            else: print(f"  Max Sim Servicio: {max_sim_servicio:.4f} (sin match de frase clave o índice fuera de rango)")

    for categoria in candidatas:
        unique_candidatas = []; seen_oraciones = set()
        for cand in sorted(candidatas[categoria], key=lambda x: x["similitud"], reverse=True):
            if cand["oracion_texto"] not in seen_oraciones: unique_candidatas.append(cand); seen_oraciones.add(cand["oracion_texto"])
        candidatas[categoria] = unique_candidatas
    return candidatas

def extraer_numero_servidores(doc_spacy, resultado_parcial):
    # ... (código sin cambios, igual al proporcionado por el usuario en la v8) ...
    if NLP_SPACY is None:
        resultado_parcial["errores"].append("Modelo spaCy no cargado en extraer_numero_servidores.")
        return resultado_parcial
    matcher = Matcher(NLP_SPACY.vocab)
    keywords_servidor = ["servidor", "cajero", "máquina", "operador", "ventanilla", "puesto", "estación", "médico", "doctor", "enfermero", "consultorio", "línea de ensamblaje", "caja"]
    pattern1 = [{"LIKE_NUM": True}, {"LEMMA": {"IN": keywords_servidor}}]
    pattern2 = [{"LOWER": {"IN": ["un", "una"]}}, {"LEMMA": {"IN": keywords_servidor}}]
    pattern3 = [{"LEMMA": {"IN": ["haber", "existir", "tener", "contar con", "disponer de", "operar con"]}}, {"LIKE_NUM": True, "OP": "?"}, {"LOWER": {"IN": ["un", "una"]}, "OP": "?"}, {"LEMMA": {"IN": keywords_servidor}}]
    matcher.add("NUM_SERVIDORES_PATTERN1", [pattern1]); matcher.add("NUM_SERVIDORES_PATTERN2", [pattern2]); matcher.add("NUM_SERVIDORES_PATTERN3", [pattern3])
    matches = matcher(doc_spacy); found_values = []
    for match_id, start, end in matches:
        span = doc_spacy[start:end]; num_val = None; potential_num_token = None
        if NLP_SPACY.vocab.strings[match_id] == "NUM_SERVIDORES_PATTERN2": num_val = 1; potential_num_token = span[0]
        else:
            for token_idx_in_span, token in enumerate(span):
                if token.like_num:
                    if token.text.lower() in NUMEROS_EN_PALABRAS_MAP:
                        num_val = NUMEROS_EN_PALABRAS_MAP[token.text.lower()]
                    else:
                        try: num_val = int(float(token.text.replace(",", ".")))
                        except ValueError: continue
                    potential_num_token = token; break
                elif token.lower_ in ["un", "una"] and NLP_SPACY.vocab.strings[match_id] == "NUM_SERVIDORES_PATTERN3":
                    is_quantifier = False
                    for next_token_in_span_idx in range(token_idx_in_span + 1, len(span)):
                        if span[next_token_in_span_idx].lemma_ in keywords_servidor: is_quantifier = True; break
                    if is_quantifier: num_val = 1; potential_num_token = token; break
        if num_val is not None and num_val > 0:
            fragmento = potential_num_token.sent.text if potential_num_token else span.sent.text
            found_values.append({"valor": num_val, "fragmento": fragmento, "span_text": span.text})
    if not found_values: print("INFO: No se encontró información explícita sobre el número de servidores.")
    else:
        primary_value = found_values[0]["valor"]; primary_fragment = found_values[0]["fragmento"]
        resultado_parcial["parametros_extraidos"]["cantidad_servidores"]["valor"] = primary_value
        resultado_parcial["parametros_extraidos"]["cantidad_servidores"]["fragmento_texto"] = primary_fragment
        print(f"INFO: Número de servidores extraído: {primary_value} (Fuente: '{primary_fragment}')")
        unique_numeric_values = set(item["valor"] for item in found_values)
        if len(unique_numeric_values) > 1:
            advertencia = (f"Advertencia: Se encontraron múltiples valores diferentes para el número de servidores. Se utilizó el primero ({primary_value} de '{primary_fragment}'). Otros valores encontrados: ")
            otros_valores_str_list = [f"{item['valor']} (en \"{item['span_text']}\" de la oración \"{item['fragmento']}\")" for item in found_values if item["valor"] != primary_value]
            advertencia += "; ".join(otros_valores_str_list); resultado_parcial["errores"].append(advertencia); print(f"ADVERTENCIA DETALLADA: {advertencia}")
    return resultado_parcial

def extraer_capacidad_sistema(doc_spacy, resultado_parcial):
    # ... (código sin cambios, igual al proporcionado por el usuario en la v8) ...
    if NLP_SPACY is None: resultado_parcial["errores"].append("Modelo spaCy no cargado en extraer_capacidad_sistema."); return resultado_parcial
    matcher = Matcher(NLP_SPACY.vocab)
    keywords_cap = ["capacidad", "límite", "tamaño", "espacio", "máximo", "buffer"]; keywords_entidad = ["cliente", "persona", "unidad", "auto", "puesto", "elemento", "paciente", "trabajo", "item"]; keywords_lugar_espera = ["cola", "sala de espera", "almacén", "buffer"]
    pattern_cap_num1 = [{"LEMMA": {"IN": keywords_cap}}, {"LOWER": {"IN": ["del", "de la"]}, "OP": "?"}, {"LOWER": {"IN": ["sistema", "total"] + keywords_lugar_espera}, "OP": "?"}, {"LOWER": "es", "OP": "?"}, {"LOWER": "de", "OP": "?"}, {"LOWER": {"IN": ["para", "en"]}, "OP": "?"}, {"LIKE_NUM": True}, {"LEMMA": {"IN": keywords_entidad}, "OP": "?"}]
    pattern_cap_num2 = [{"LIKE_NUM": True}, {"LEMMA": {"IN": keywords_entidad}, "OP": "+"}, {"LOWER": {"IN": ["de", "en", "para"]}, "OP": "+"}, {"LEMMA": {"IN": ["capacidad"] + keywords_lugar_espera + ["sistema"]}}]
    pattern_caben = [{"LEMMA": {"IN": ["caber", "acomodar"]}}, {"LIKE_NUM": True}, {"LEMMA": {"IN": keywords_entidad}, "OP": "?"}, {"LOWER": "en", "OP": "?"}, {"LOWER": {"IN": ["la", "el"]}, "OP": "?"}, {"LEMMA": {"IN": keywords_lugar_espera}}]
    pattern_no_mas_de = [{"LOWER": "no", "OP": "?"}, {"LEMMA": {"IN": ["poder", "aceptar", "admitir", "acomodar"]}, "OP": "+"}, {"LOWER": "más"}, {"LOWER": "de"}, {"LIKE_NUM": True}, {"LEMMA": {"IN": keywords_entidad}, "OP": "?"}]
    pattern_k_igual = [{"TEXT": {"IN": ["K", "k"]}}, {"LOWER": {"IN": ["=", "es"]}, "OP": "?"}, {"LIKE_NUM": True}]
    pattern_rechazo = [{"LEMMA": {"IN": ["rechazar", "no admitir", "no aceptar", "dejar de aceptar"]}}, {"IS_PUNCT": False, "OP": "{0,5}"}, {"LOWER": "si"}, {"LOWER": "hay"}, {"LOWER": {"IN": ["más", "mas"]}}, {"LOWER": "de"}, {"LIKE_NUM": True}, {"LEMMA": {"IN": keywords_entidad}, "OP": "?"}]
    pattern_solo_permiten = [{"LOWER": {"IN": ["solo", "solamente", "únicamente"]}}, {"LOWER": {"IN": ["se", "puede", "pueden"]}, "OP": "?"}, {"LEMMA": {"IN": ["permitir", "caber", "almacenar", "haber", "tener", "aceptar"]}, "OP": "+"}, {"LIKE_NUM": True}, {"LEMMA": {"IN": keywords_entidad}, "OP": "?"}]
    pattern_infinita_directa = [{"LOWER": {"IN": ["capacidad", "límite"]}, "OP": "?"}, {"LOWER": {"IN": ["infinita", "ilimitada"]}}]; pattern_no_limite = [{"LOWER": {"IN": ["no", "sin"]}}, {"LOWER": {"IN": ["hay", "tiene", "existe"]}, "OP": "?"}, {"LEMMA": {"IN": ["límite", "restricción"]}}]
    matcher.add("CAP_NUM1", [pattern_cap_num1]); matcher.add("CAP_NUM2", [pattern_cap_num2]); matcher.add("CAP_CABEN", [pattern_caben]); matcher.add("CAP_NO_MAS_DE", [pattern_no_mas_de]); matcher.add("CAP_K_IGUAL", [pattern_k_igual]); matcher.add("CAP_RECHAZO", [pattern_rechazo]); matcher.add("CAP_SOLO_PERMITEN", [pattern_solo_permiten]); matcher.add("CAP_INFINITA", [pattern_infinita_directa]); matcher.add("CAP_NO_LIMITE", [pattern_no_limite])
    matches = matcher(doc_spacy); found_capacities = []
    for match_id, start, end in matches:
        span = doc_spacy[start:end]; match_name = NLP_SPACY.vocab.strings[match_id]; current_value = None; current_fragment = span.sent.text; match_type = "general_numeric"
        if match_name in ["CAP_INFINITA", "CAP_NO_LIMITE"]: current_value = "infinita"; match_type = "infinite"
        else:
            for token in span:
                if token.like_num:
                    if token.text.lower() in NUMEROS_EN_PALABRAS_MAP:
                        current_value = NUMEROS_EN_PALABRAS_MAP[token.text.lower()]
                    else:
                        try: current_value = int(float(token.text.replace(",", ".")))
                        except ValueError: continue
                    break
            if current_value is not None:
                sentence_text_lower = current_fragment.lower()
                if match_name in ["CAP_RECHAZO", "CAP_SOLO_PERMITEN", "CAP_K_IGUAL", "CAP_NO_MAS_DE"]: match_type = "system_direct_k";
                if any(kw_lugar in sentence_text_lower for kw_lugar in keywords_lugar_espera) and not any(kw_sys in sentence_text_lower for kw_sys in ["sistema", "total"]): match_type = "queue_explicit"
                elif match_name in ["CAP_NUM1", "CAP_NUM2", "CAP_CABEN"]:
                    if any(kw in sentence_text_lower for kw in ["sistema", "total"]): match_type = "system_explicit"
                    elif any(kw_lugar in sentence_text_lower for kw_lugar in keywords_lugar_espera) or match_name == "CAP_CABEN": match_type = "queue_explicit"
        if current_value is not None:
            if isinstance(current_value, (int, float)) and current_value <= 0: continue
            found_capacities.append({"valor": current_value, "fragmento": current_fragment, "span_text": span.text, "match_name": match_name, "match_type": match_type})
    if not found_capacities:
        print("INFO: No se encontró info de capacidad. Asumiendo infinita por defecto."); resultado_parcial["parametros_extraidos"]["capacidad_sistema"]["valor"] = "infinita"; resultado_parcial["parametros_extraidos"]["capacidad_sistema"]["fragmento_texto"] = "Asumida infinita (no encontrada explícitamente)."
    else:
        chosen_cap = None; priority_order = ["system_direct_k", "system_explicit", "general_numeric", "queue_explicit", "infinite"]
        for p_type in priority_order:
            caps_of_type = [fc for fc in found_capacities if fc["match_type"] == p_type]
            if caps_of_type: chosen_cap = caps_of_type[0]; break
        if chosen_cap:
            primary_value = chosen_cap["valor"]; primary_fragment = chosen_cap["fragmento"]
            resultado_parcial["parametros_extraidos"]["capacidad_sistema"]["valor"] = primary_value; resultado_parcial["parametros_extraidos"]["capacidad_sistema"]["fragmento_texto"] = primary_fragment
            print(f"INFO: Capacidad del sistema extraída: {primary_value} (Fuente: '{primary_fragment}', Patrón: {chosen_cap['match_name']}, Tipo: {chosen_cap['match_type']})")
            numeric_caps_all = [fc for fc in found_capacities if isinstance(fc["valor"], (int, float))]; all_found_numeric_values = set(nc["valor"] for nc in numeric_caps_all)
            if len(all_found_numeric_values) > 1 and isinstance(primary_value, (int, float)):
                adv = (f"Advertencia: Múltiples valores numéricos para capacidad. Usado: {primary_value} (Tipo: {chosen_cap['match_type']}). Otros detectados: {[f'{nc["valor"]} (Tipo: {nc["match_type"]})' for nc in found_capacities if isinstance(nc['valor'], (int,float)) and nc['valor'] != primary_value]}")
                resultado_parcial["errores"].append(adv); print(f"ADVERTENCIA: {adv}")
            if numeric_caps_all and any(fc["match_type"] == "infinite" for fc in found_capacities) and isinstance(primary_value, (int, float)):
                adv = (f"Advertencia: Se encontró capacidad numérica ({primary_value}) y mención de 'infinita'. Se priorizó el valor numérico basado en el tipo de patrón ({chosen_cap['match_type']}).")
                resultado_parcial["errores"].append(adv); print(f"ADVERTENCIA: {adv}")
        else: print("INFO: No se pudo determinar la capacidad explícita con prioridades. Asumiendo infinita por defecto."); resultado_parcial["parametros_extraidos"]["capacidad_sistema"]["valor"] = "infinita"; resultado_parcial["parametros_extraidos"]["capacidad_sistema"]["fragmento_texto"] = "Asumida infinita (lógica de selección)."
    return resultado_parcial

def extraer_disciplina_cola(doc_spacy, resultado_parcial):
    # ... (código sin cambios, igual al proporcionado por el usuario en la v8) ...
    if NLP_SPACY is None: resultado_parcial["errores"].append("Modelo spaCy no cargado en extraer_disciplina_cola."); return resultado_parcial
    matcher = Matcher(NLP_SPACY.vocab); disciplinas_encontradas = []
    pattern_fifo_sigla = [{"TEXT": "FIFO"}]; pattern_fifo_frase_es = [{"LOWER": "primero"}, {"LOWER": "en"}, {"LOWER": "llegar"}, {"IS_PUNCT": True, "OP": "?"}, {"LOWER": "primero"}, {"LOWER": "en"}, {"LOWER": "ser"}, {"LOWER": "servido"}]; pattern_fifo_frase_es_alt = [{"LOWER": "primero"}, {"LOWER": "que"}, {"LOWER": "llega"}, {"LOWER": "es"}, {"LOWER": "el"}, {"LOWER": "primero"}, {"LOWER": "que"}, {"LOWER": "se"}, {"LOWER": "atiende"}]; pattern_fifo_fcfs = [{"TEXT": "FCFS"}]; pattern_orden_llegada = [{"LOWER": "orden"}, {"LOWER": "de"}, {"LOWER": "llegada"}]; pattern_orden_ingreso = [{"LOWER": "orden"}, {"LOWER": "en"}, {"LOWER": "que"}, {"LEMMA": {"IN": ["ir", "llegar", "ingresar"]}}]
    pattern_lifo_sigla = [{"TEXT": "LIFO"}]; pattern_lifo_frase_es = [{"LOWER": "último"}, {"LOWER": "en"}, {"LOWER": "llegar"}, {"IS_PUNCT": True, "OP": "?"}, {"LOWER": "primero"}, {"LOWER": "en"}, {"LOWER": "ser"}, {"LOWER": "servido"}]; pattern_lifo_lcfs = [{"TEXT": "LCFS"}]
    pattern_siro_sigla = [{"TEXT": "SIRO"}]; pattern_siro_frase_es = [{"LOWER": "servicio"}, {"LOWER": "en"}, {"LOWER": "orden"}, {"LOWER": "aleatorio"}]; pattern_siro_random = [{"LOWER": "atención"}, {"LOWER": "aleatoria"}]
    pattern_prioridad_sigla = [{"TEXT": "PRI"}]; pattern_prioridad_frase = [{"LEMMA": {"IN": ["prioridad", "prioritario"]}}]; pattern_prioridad_clases = [{"LEMMA": {"IN": ["clase", "tipo", "nivel"]}}, {"LOWER": "de", "OP": "?"}, {"LEMMA": "prioridad"}]
    matcher.add("DISC_FIFO_SIGLA", [pattern_fifo_sigla]); matcher.add("DISC_FIFO_FRASE_ES", [pattern_fifo_frase_es]); matcher.add("DISC_FIFO_FRASE_ES_ALT", [pattern_fifo_frase_es_alt]); matcher.add("DISC_FIFO_FCFS", [pattern_fifo_fcfs]); matcher.add("DISC_ORDEN_LLEGADA", [pattern_orden_llegada]); matcher.add("DISC_ORDEN_INGRESO", [pattern_orden_ingreso])
    matcher.add("DISC_LIFO_SIGLA", [pattern_lifo_sigla]); matcher.add("DISC_LIFO_FRASE_ES", [pattern_lifo_frase_es]); matcher.add("DISC_LIFO_LCFS", [pattern_lifo_lcfs])
    matcher.add("DISC_SIRO_SIGLA", [pattern_siro_sigla]); matcher.add("DISC_SIRO_FRASE_ES", [pattern_siro_frase_es]); matcher.add("DISC_SIRO_RANDOM", [pattern_siro_random])
    matcher.add("DISC_PRIORIDAD_SIGLA", [pattern_prioridad_sigla]); matcher.add("DISC_PRIORIDAD_FRASE", [pattern_prioridad_frase]); matcher.add("DISC_PRIORIDAD_CLASES", [pattern_prioridad_clases])
    matches = matcher(doc_spacy)
    for match_id, start, end in matches:
        span = doc_spacy[start:end]; rule_id_str = NLP_SPACY.vocab.strings[match_id]; disciplina_detectada = None
        if "FIFO" in rule_id_str or "FCFS" in rule_id_str or "ORDEN_LLEGADA" in rule_id_str or "ORDEN_INGRESO" in rule_id_str: disciplina_detectada = "FIFO"
        elif "LIFO" in rule_id_str or "LCFS" in rule_id_str: disciplina_detectada = "LIFO"
        elif "SIRO" in rule_id_str or "RANDOM" in rule_id_str: disciplina_detectada = "SIRO"
        elif "PRIORIDAD" in rule_id_str: disciplina_detectada = "Prioridad"
        if disciplina_detectada: disciplinas_encontradas.append({"valor": disciplina_detectada, "fragmento": span.sent.text, "span_text": span.text})
    if not disciplinas_encontradas: print("INFO: No se encontró disciplina de cola explícita. Se establece como no especificado."); resultado_parcial["parametros_extraidos"]["disciplina_cola"]["valor"] = None; resultado_parcial["parametros_extraidos"]["disciplina_cola"]["fragmento_texto"] = None
    else:
        chosen_discipline = disciplinas_encontradas[0]; primary_value = chosen_discipline["valor"]; primary_fragment = chosen_discipline["fragmento"]
        resultado_parcial["parametros_extraidos"]["disciplina_cola"]["valor"] = primary_value; resultado_parcial["parametros_extraidos"]["disciplina_cola"]["fragmento_texto"] = primary_fragment
        print(f"INFO: Disciplina de cola extraída: {primary_value} (Fuente: '{primary_fragment}')")
        unique_disciplines = set(d["valor"] for d in disciplinas_encontradas)
        if len(unique_disciplines) > 1:
            advertencia = (f"Advertencia: Se encontraron múltiples menciones de disciplinas de cola diferentes. Se utilizó la primera detectada ('{primary_value}' de '{primary_fragment}'). Otras detectadas: {[f'{d["valor"]} (en \"{d["span_text"]}\")' for d in disciplinas_encontradas if d["valor"] != primary_value]}")
            resultado_parcial["errores"].append(advertencia); print(f"ADVERTENCIA DETALLADA: {advertencia}")
    return resultado_parcial

# --- Función Principal de Extracción ---
def extraer_parametros_colas(texto_entrada, umbral_similitud_candidatas=0.6, debug_specific_sentence_part=None):
    # ... (código sin cambios, igual al proporcionado por el usuario en la v8) ...
    if NLP_SPACY is None or MODEL_SENTENCE_TRANSFORMERS is None or \
        not EMBEDDINGS_FRASES_CLAVE or \
        EMBEDDINGS_FRASES_CLAVE.get("llegada") is None or \
        EMBEDDINGS_FRASES_CLAVE.get("servicio") is None:
        print("Intentando recargar modelos y/o embeddings...")
        cargar_modelos_y_precalcular_embeddings()
        if NLP_SPACY is None or MODEL_SENTENCE_TRANSFORMERS is None or \
            not EMBEDDINGS_FRASES_CLAVE or \
            EMBEDDINGS_FRASES_CLAVE.get("llegada") is None or \
            EMBEDDINGS_FRASES_CLAVE.get("servicio") is None:
            res_error = inicializar_estructura_salida()
            res_error["errores"].append("Fallo crítico al cargar modelos NLP o embeddings de frases clave.")
            return res_error

    doc_spacy, resultado_parcial = procesar_texto_basico(texto_entrada)
    if doc_spacy is None: return resultado_parcial

    resultado_parcial = extraer_numero_servidores(doc_spacy, resultado_parcial)
    resultado_parcial = extraer_capacidad_sistema(doc_spacy, resultado_parcial)
    resultado_parcial = extraer_disciplina_cola(doc_spacy, resultado_parcial)

    oraciones_candidatas = identificar_oraciones_candidatas(doc_spacy, umbral_similitud_candidatas, debug_specific_sentence_part=debug_specific_sentence_part)
    resultado_parcial["oraciones_candidatas_debug"] = oraciones_candidatas

    print("\n--- Oraciones Candidatas Detectadas (Debug) ---")
    if oraciones_candidatas.get("llegada"):
        print("Llegada:")
        for cand in oraciones_candidatas["llegada"]:
            print(f"  - Sim: {cand['similitud']:.4f}, Oración: \"{cand['oracion_texto']}\"")
    if oraciones_candidatas.get("servicio"):
        print("Servicio:")
        for cand in oraciones_candidatas["servicio"]:
            print(f"  - Sim: {cand['similitud']:.4f}, Oración: \"{cand['oracion_texto']}\"")
    print("--------------------------------------------")

    # --- Tarea 2.3 y 2.4 (Poblado de tasas/tiempos) ---
    # Procesar candidatas de LLEGADA
    processed_sentences_for_llegada = set()
    if resultado_parcial["oraciones_candidatas_debug"]["llegada"]:
        candidatas_llegada_sorted = sorted(resultado_parcial["oraciones_candidatas_debug"]["llegada"], key=lambda x: x["similitud"], reverse=True)

        for cand_info_llegada in candidatas_llegada_sorted:
            oracion_llegada_principal = cand_info_llegada["oracion_texto"]
            if oracion_llegada_principal in processed_sentences_for_llegada:
                continue

            servicio_cand_match = next((s_cand for s_cand in resultado_parcial["oraciones_candidatas_debug"].get("servicio", []) if s_cand["oracion_texto"] == oracion_llegada_principal), None)
            if servicio_cand_match and \
                (resultado_parcial["parametros_extraidos"]["tasa_servicio_por_servidor"]["fragmento_texto"] == oracion_llegada_principal or \
                 resultado_parcial["parametros_extraidos"]["tiempo_servicio_por_servidor"]["fragmento_texto"] == oracion_llegada_principal) and \
                servicio_cand_match["similitud"] > cand_info_llegada["similitud"] + 0.05:
                print(f"INFO: Oración '{oracion_llegada_principal}' ya asignada a servicio con mayor similitud o ya procesada. Omitiendo para llegada.")
                continue

            datos_llegada = extraer_valor_y_unidad_de_oracion(oracion_llegada_principal)
            if datos_llegada:
                print(f"DEBUG - Datos LLEGADA extraídos de '{oracion_llegada_principal}': {datos_llegada}")
                
                if resultado_parcial["parametros_extraidos"]["tasa_llegada"]["valor"] is None and \
                   resultado_parcial["parametros_extraidos"]["tiempo_entre_llegadas"]["valor"] is None:
                    
                    param_type_filled_llegada = None 
                    if datos_llegada["tipo_parametro"] == "tasa":
                        resultado_parcial["parametros_extraidos"]["tasa_llegada"]["valor"] = datos_llegada["valor"]
                        resultado_parcial["parametros_extraidos"]["tasa_llegada"]["unidades"] = datos_llegada["unidad_texto"]
                        resultado_parcial["parametros_extraidos"]["tasa_llegada"]["fragmento_texto"] = oracion_llegada_principal
                        param_type_filled_llegada = "tasa"
                    elif datos_llegada["tipo_parametro"] == "tiempo":
                        resultado_parcial["parametros_extraidos"]["tiempo_entre_llegadas"]["valor"] = datos_llegada["valor"]
                        resultado_parcial["parametros_extraidos"]["tiempo_entre_llegadas"]["unidades"] = datos_llegada["unidad_texto"]
                        resultado_parcial["parametros_extraidos"]["tiempo_entre_llegadas"]["fragmento_texto"] = oracion_llegada_principal
                        param_type_filled_llegada = "tiempo"

                    if param_type_filled_llegada: 
                        dist_tasa_llegada_detectada = None
                        dist_tiempo_llegada_detectada = None
                        oracion_lower = oracion_llegada_principal.lower()
                        mencion_poisson = "poisson" in oracion_lower
                        mencion_exponencial = "exponencial" in oracion_lower

                        if mencion_poisson and mencion_exponencial:
                            resultado_parcial["errores"].append(f"Advertencia: Menciones de Poisson Y Exponencial en la misma oración de llegada: '{oracion_llegada_principal}'. Se asume Tasa Poisson / Tiempo Exponencial.")
                            dist_tasa_llegada_detectada = "Poisson"
                            dist_tiempo_llegada_detectada = "Exponencial"
                        elif mencion_poisson:
                            dist_tasa_llegada_detectada = "Poisson"
                            dist_tiempo_llegada_detectada = "Exponencial"
                            if "tiempo entre llegadas" in oracion_lower and "poisson" in oracion_lower:
                                resultado_parcial["errores"].append(f"Advertencia: Mención de 'tiempo entre llegadas Poisson' en '{oracion_llegada_principal}'. Se mantiene inferencia estándar: Tasa Poisson / Tiempo Exponencial.")
                        elif mencion_exponencial:
                            dist_tiempo_llegada_detectada = "Exponencial"
                            dist_tasa_llegada_detectada = "Poisson"
                            if "tasa de llegada" in oracion_lower and "exponencial" in oracion_lower:
                                dist_tasa_llegada_detectada = "Exponencial"
                                dist_tiempo_llegada_detectada = None 
                                resultado_parcial["errores"].append(f"Advertencia: Mención de 'tasa de llegada exponencial' en '{oracion_llegada_principal}'. Se extrae literal para la tasa, tiempo no inferido automáticamente como Exponencial.")
                        
                        if param_type_filled_llegada == "tasa":
                            if resultado_parcial["parametros_extraidos"]["tasa_llegada"]["distribucion"] is None and dist_tasa_llegada_detectada:
                                resultado_parcial["parametros_extraidos"]["tasa_llegada"]["distribucion"] = dist_tasa_llegada_detectada
                            if resultado_parcial["parametros_extraidos"]["tiempo_entre_llegadas"]["distribucion"] is None and dist_tiempo_llegada_detectada:
                                if resultado_parcial["parametros_extraidos"]["tiempo_entre_llegadas"]["valor"] is None or \
                                   resultado_parcial["parametros_extraidos"]["tiempo_entre_llegadas"]["fragmento_texto"] == oracion_llegada_principal:
                                    resultado_parcial["parametros_extraidos"]["tiempo_entre_llegadas"]["distribucion"] = dist_tiempo_llegada_detectada
                        elif param_type_filled_llegada == "tiempo":
                            if resultado_parcial["parametros_extraidos"]["tiempo_entre_llegadas"]["distribucion"] is None and dist_tiempo_llegada_detectada:
                                resultado_parcial["parametros_extraidos"]["tiempo_entre_llegadas"]["distribucion"] = dist_tiempo_llegada_detectada
                            if resultado_parcial["parametros_extraidos"]["tasa_llegada"]["distribucion"] is None and dist_tasa_llegada_detectada:
                                if resultado_parcial["parametros_extraidos"]["tasa_llegada"]["valor"] is None or \
                                   resultado_parcial["parametros_extraidos"]["tasa_llegada"]["fragmento_texto"] == oracion_llegada_principal:
                                    resultado_parcial["parametros_extraidos"]["tasa_llegada"]["distribucion"] = dist_tasa_llegada_detectada
                        
                        if resultado_parcial["parametros_extraidos"]["tasa_llegada"]["distribucion"] == "Poisson" and \
                           resultado_parcial["parametros_extraidos"]["tiempo_entre_llegadas"]["distribucion"] is None:
                            resultado_parcial["parametros_extraidos"]["tiempo_entre_llegadas"]["distribucion"] = "Exponencial"
                        elif resultado_parcial["parametros_extraidos"]["tiempo_entre_llegadas"]["distribucion"] == "Exponencial" and \
                             resultado_parcial["parametros_extraidos"]["tasa_llegada"]["distribucion"] is None:
                            resultado_parcial["parametros_extraidos"]["tasa_llegada"]["distribucion"] = "Poisson"
                    processed_sentences_for_llegada.add(oracion_llegada_principal)

    # Procesar candidatas de SERVICIO
    processed_sentences_for_servicio = set()
    if resultado_parcial["oraciones_candidatas_debug"]["servicio"]:
        candidatas_servicio_sorted = sorted(resultado_parcial["oraciones_candidatas_debug"]["servicio"], key=lambda x: x["similitud"], reverse=True)

        for cand_info_servicio in candidatas_servicio_sorted:
            oracion_servicio_principal = cand_info_servicio["oracion_texto"]
            if oracion_servicio_principal in processed_sentences_for_servicio:
                continue

            if oracion_servicio_principal == resultado_parcial["parametros_extraidos"]["tasa_llegada"]["fragmento_texto"] or \
               oracion_servicio_principal == resultado_parcial["parametros_extraidos"]["tiempo_entre_llegadas"]["fragmento_texto"]:
                llegada_cand_match = next((l_cand for l_cand in resultado_parcial["oraciones_candidatas_debug"].get("llegada", []) if l_cand["oracion_texto"] == oracion_servicio_principal), None)
                if llegada_cand_match and llegada_cand_match["similitud"] >= cand_info_servicio["similitud"] - 0.05:
                    print(f"INFO: Oración '{oracion_servicio_principal}' ya asignada a llegada o con similitud no dominante para servicio. Omitiendo para servicio.")
                    continue

            datos_servicio = extraer_valor_y_unidad_de_oracion(oracion_servicio_principal)
            if datos_servicio:
                print(f"DEBUG - Datos SERVICIO extraídos de '{oracion_servicio_principal}': {datos_servicio}")
                if resultado_parcial["parametros_extraidos"]["tasa_servicio_por_servidor"]["valor"] is None and \
                   resultado_parcial["parametros_extraidos"]["tiempo_servicio_por_servidor"]["valor"] is None:
                    
                    if datos_servicio["tipo_parametro"] == "tasa":
                        resultado_parcial["parametros_extraidos"]["tasa_servicio_por_servidor"]["valor"] = datos_servicio["valor"]
                        resultado_parcial["parametros_extraidos"]["tasa_servicio_por_servidor"]["unidades"] = datos_servicio["unidad_texto"]
                        resultado_parcial["parametros_extraidos"]["tasa_servicio_por_servidor"]["fragmento_texto"] = oracion_servicio_principal
                    elif datos_servicio["tipo_parametro"] == "tiempo":
                        resultado_parcial["parametros_extraidos"]["tiempo_servicio_por_servidor"]["valor"] = datos_servicio["valor"]
                        resultado_parcial["parametros_extraidos"]["tiempo_servicio_por_servidor"]["unidades"] = datos_servicio["unidad_texto"]
                        resultado_parcial["parametros_extraidos"]["tiempo_servicio_por_servidor"]["fragmento_texto"] = oracion_servicio_principal

                    oracion_lower_srv = oracion_servicio_principal.lower()
                    if "poisson" in oracion_lower_srv:
                        if resultado_parcial["parametros_extraidos"]["tasa_servicio_por_servidor"]["distribucion"] is None:
                             resultado_parcial["parametros_extraidos"]["tasa_servicio_por_servidor"]["distribucion"] = "Poisson"
                        if resultado_parcial["parametros_extraidos"]["tiempo_servicio_por_servidor"]["distribucion"] is None:
                            resultado_parcial["parametros_extraidos"]["tiempo_servicio_por_servidor"]["distribucion"] = "Exponencial"
                    elif "exponencial" in oracion_lower_srv:
                        if resultado_parcial["parametros_extraidos"]["tasa_servicio_por_servidor"]["distribucion"] is None:
                             resultado_parcial["parametros_extraidos"]["tasa_servicio_por_servidor"]["distribucion"] = "Poisson" 
                        if resultado_parcial["parametros_extraidos"]["tiempo_servicio_por_servidor"]["distribucion"] is None:
                            resultado_parcial["parametros_extraidos"]["tiempo_servicio_por_servidor"]["distribucion"] = "Exponencial"
                    processed_sentences_for_servicio.add(oracion_servicio_principal)
    return resultado_parcial

if __name__ == "__main__":
    cargar_modelos_y_precalcular_embeddings()

    textos_prueba_tasas_tiempos = [
        "Los clientes llegan a una tasa de 10 por hora.",
        "El tiempo promedio de servicio es de 5 minutos por cliente.", # User test case 1
        "Se espera que arriben unidades cada 2 minutos.",
        "La capacidad de atención del sistema es de 15 trabajos por día.",
        "Un nuevo pedido entra al sistema aproximadamente cada 30 segundos.",
        "El servidor procesa los ítems con una frecuencia de 60 ítems/hora.", # User test case 2
        "La tasa esperada de llegada de pacientes es de 3 por hora.",
        "El tiempo medio entre llegadas de vehículos es de 120 segundos.",
        "La tasa media de servicio por cajero es de 25 clientes/hora.", 
        "El tiempo esperado de servicio para cada tarea es de 10 minutos.",
        "Los pacientes llegan a la clínica de un médico de acuerdo con una distribución de Poisson a razón de treinta pacientes por hora."
    ]

    for i, texto_test in enumerate(textos_prueba_tasas_tiempos):
        print(f"\n--- Procesando Texto de Prueba Tasas/Tiempos #{i+1}: '{texto_test}' ---")
        if NLP_SPACY and MODEL_SENTENCE_TRANSFORMERS and \
            EMBEDDINGS_FRASES_CLAVE.get("llegada") is not None and \
            EMBEDDINGS_FRASES_CLAVE.get("servicio") is not None:
            info_extraida = extraer_parametros_colas(texto_test, umbral_similitud_candidatas=0.55)
            print("\n--- Resultado Extracción (JSON) ---")
            print(json.dumps(info_extraida, indent=4, ensure_ascii=False))
            print("-----------------------------------\n")
        else:
            print("Modelos o embeddings de frases clave no cargados, saltando prueba.")

    print("\n--- Prueba con archivo ejemplo1.txt (si existe) ---")
    ruta_ejemplo = os.path.join(DATA_DIR, "ejemplo1.txt")

    try:
        
        if os.path.exists(ruta_ejemplo):
            with open(ruta_ejemplo, 'r', encoding='utf-8') as f:
                texto_ejemplo_regresion = f.read()
            print(f"--- Procesando Archivo de Regresión: {ruta_ejemplo} ---")
            print(texto_ejemplo_regresion)
            print("------------------------------------")
            if NLP_SPACY and MODEL_SENTENCE_TRANSFORMERS and \
                EMBEDDINGS_FRASES_CLAVE.get("llegada") is not None and \
                EMBEDDINGS_FRASES_CLAVE.get("servicio") is not None:
                info_extraida_regresion = extraer_parametros_colas(texto_ejemplo_regresion, umbral_similitud_candidatas=0.50, debug_specific_sentence_part="a razón de 20 pacientes por hora")
                print("\n--- Resultado (JSON) - Archivo Regresión ---")
                print(json.dumps(info_extraida_regresion, indent=4, ensure_ascii=False))
                print("--------------------------\n")
        else:
            print(f"Archivo de regresión {ruta_ejemplo} no encontrado o no se pudo crear.")

    except Exception as e:
        print(f"Error durante las pruebas: {e}")
        import traceback
        traceback.print_exc()
