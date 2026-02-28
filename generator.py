import re
import json
import urllib.request
from typing import Optional

# ==========================================
# CONFIGURACIÓN DEL LLM LOCAL
# ==========================================
# ./llama-server --hf-repo psychopenguin/Qwen3-4B-Thinking-2507-Q8_0-GGUF --hf-file qwen3-4b-thinking-2507-q8_0.gguf -c 2048 -t 8
LLAMA_API_URL = "http://127.0.0.1:8080/completion"

def extract_python_code(text: str) -> Optional[str]:
    """
    Usa expresiones regulares para extraer SOLO el código Python.
    """
    pattern = re.compile(r"`{3}(?:python)?(.*?)`{3}", re.DOTALL | re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    
    if "def update_step" in text:
        lines = text.split('\n')
        code_lines = []
        in_function = False
        for line in lines:
            if line.startswith("def update_step"):
                in_function = True
            if in_function:
                code_lines.append(line)
        return "\n".join(code_lines).strip()
    
    return None

# Cambia la definición de la función
def generate_candidate(best_code: str, recent_errors: list, research_notes: list = None, temperature: float = 0.8) -> Optional[str]:
    """
    Genera una mutación orientada a Edge Computing / TinyML.
    """
    
    # 1. Prompt completamente rediseñado para "Cero Memoria Vectorial"
    prompt = "<|im_start|>system\nEres un investigador experto en TinyML y Edge Computing. Responde ÚNICAMENTE con código Python.<|im_end|>\n"
    prompt += "<|im_start|>user\nModifica este optimizador para microcontroladores con RAM extremadamente limitada.\n"
    
    prompt += "`" * 3 + f"python\n{best_code}\n" + "`" * 3 + "\n"

    # "Amnesia Controlada"
    if recent_errors:
        prompt += "\nERRORES RECIENTES A EVITAR:\n"
        for error in recent_errors[-2:]:
            prompt += f"- {error}\n"
            
    if research_notes and len(research_notes) > 0:
        prompt += "\nPIZARRA DE DESCUBRIMIENTOS (Principios que DEBES mantener):\n"
        for note in research_notes:
            prompt += f"- {note}\n"

    # Reglas estrictas de castigo de memoria
    # Reglas estrictas de castigo de memoria y ROBUSTEZ AL RUIDO
    prompt += "\nOBJETIVO: Mejorar la convergencia matemática SIN usar memoria vectorial adicional, en un entorno hostil con 20% de ETIQUETAS CORRUPTAS (Ruido Extremo).\n"
    prompt += "REGLA CRÍTICA DE MEMORIA: Tienes ESTRICTAMENTE PROHIBIDO guardar arrays (vectores numpy de tamaño igual a los parámetros) en el diccionario 'state'. Si guardas historiales como el momentum clásico, el programa fallará por falta de RAM.\n"
    prompt += "PERMITIDO: Solo puedes guardar valores ESCALARES (números individuales) en 'state' (ej. contadores 't', medianas globales, normas escalares).\n"
    prompt += "REGLA DE ROBUSTEZ (NUEVA): Los gradientes contendrán valores atípicos (outliers) extremos que destruirán la red. DEBES usar estadística robusta (ej. np.clip, np.median, np.sign) para filtrar la basura antes de actualizar los pesos.\n"
    prompt += "REGLA FATAL: NUNCA uses la palabra 'import'. 'np' (numpy) ya está cargado globalmente.\n"
    prompt += "REGLA DE RENDIMIENTO: DEBES usar funciones nativas de numpy. ESTÁ PROHIBIDO usar bucles 'for' iterando sobre los gradientes.\n"
    prompt += "REGLA DE RETORNO: La variable 'params' final DEBE seguir siendo un vector numpy (np.ndarray).\n"
    prompt += "HIPERPARÁMETROS: Usa 'lr', 'c1', 'c2', 'c3' provistos en hyperparams.\n"
    prompt += "REGLA DE SINTAXIS: Tu código DEBE terminar siempre con la línea exacta 'return params, state'. No la olvides.\n"
    prompt += "Innova combinando 'grads', estadística robusta y tu memoria escalar para sobrevivir al ruido.\n"
    
    prompt += "<|im_start|>assistant\n" + "`" * 3 + "python\n"

    # 2. Configuración de límites
    data = {
        "prompt": prompt,
        "n_predict": 800,
        "temperature": temperature,
        "stop": ["`" * 3 + "\n", "<|im_end|>", "Explicación:"],
        "stream": False
    }

    req = urllib.request.Request(
        LLAMA_API_URL, 
        data=json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )

    # 3. Petición y Extracción
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            raw_text = result.get('content', '')
            
            if raw_text.startswith("def update_step"):
                 code = raw_text
            else:
                 markdown_block = "`" * 3 + "python\n" + raw_text + "`" * 3
                 code = extract_python_code(markdown_block)
                 
            return code if code else raw_text
            
    except Exception as e:
        print(f"Error conectando al LLM local: {e}")
        return None

# ==========================================
# PRUEBA DEL GENERADOR
# ==========================================
if __name__ == "__main__":
    print("Conectando con llama-server en http://127.0.0.1:8080 ...")
    
    codigo_semilla = """
def update_step(params, grads, state, hyperparams):
    lr = hyperparams.get('lr', 0.05)
    params = params - lr * grads
    return params, state
"""
    errores = []
    
    nuevo_codigo = generate_candidate(codigo_semilla, errores)
    
    if nuevo_codigo:
        print("\n--- CÓDIGO EXTRAÍDO LIMPIO ---")
        print(nuevo_codigo)
    else:
        print("\n[!] Fallo en la generación o extracción. ¿Está corriendo el servidor en la otra terminal?")