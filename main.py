import os
import json
import time
import re
from unittest import result
from generator import generate_candidate
from evaluator import evaluate_candidate

HISTORY_FILE = "evolution_history.json"

def sanitize_code(code: str) -> str:
    """Elimina residuos de Markdown y destruye cualquier intento de importación."""
    clean = code.replace("```python", "").replace("```", "")
    # Destruye cualquier línea que empiece con 'import ' o 'from '
    clean = re.sub(r'^\s*import\s+.*$', '', clean, flags=re.MULTILINE)
    clean = re.sub(r'^\s*from\s+.*$', '', clean, flags=re.MULTILINE)
    
    return clean.strip()

def load_state():
    """Carga el estado de la evolución desde el disco para poder pausar/reanudar."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return {
        "generation": 0,
        "best_fitness": float('inf'),
        "best_code": """
            def update_step(params, grads, state, hyperparams):
            lr = hyperparams.get('lr', 0.001)
            params = params - lr * grads
            return params, state
            """,
        "recent_errors": [],
        "research_notes": []
    }

def save_state(state):
    """Guarda el estado en el disco."""
    with open(HISTORY_FILE, "w") as f:
        json.dump(state, f, indent=4)

def main():
    print("=== INICIANDO BÚSQUEDA DE OPTIMIZADORES (Single-Node CPU) ===")
    state = load_state()
    
    print(f"[*] Reanudando desde Generación {state['generation']}")
    print(f"[*] Mejor Fitness Actual: {state['best_fitness']:.6f}\n")
    if state.get('research_notes'):
        print(f"[*] Pizarra activa con {len(state['research_notes'])} notas de investigación.\n")
    else:
        print("[*] Pizarra de investigación vacía.\n")

    # Bucle infinito: lo dejas corriendo, lo detienes con Ctrl+C cuando quieras
    try:
        while True:
            state['generation'] += 1
            print(f"--- GENERACIÓN {state['generation']} ---")
            
            # 1. Generar Candidato (Toma ~20 segundos)
            print(">> Generando mutación (LLM)...")
            raw_candidate = generate_candidate(
                best_code=state['best_code'], 
                recent_errors=state['recent_errors'],
                research_notes=state['research_notes']
            )
            
            if not raw_candidate:
                print("[!] Fallo de conexión con el LLM. Reintentando en 5s...")
                time.sleep(5)
                continue
                
            candidate_code = sanitize_code(raw_candidate)
            
            # 2. Evaluar Candidato (Toma < 2 segundos gracias al Sandbox multiproceso)
            print(">> Evaluando en Proxy (MLP Estocástico)...")
            result = evaluate_candidate(candidate_code)
            
            # 3. Lógica Evolutiva (Selección Natural)
            if result["error"]:
                print(f"[x] Mutación Fallida: {result['error']}")
                # Añadimos el error a la memoria a corto plazo (Amnesia controlada)
                # Reemplaza saltos de línea por espacios y toma los primeros 60 caracteres de forma segura
                codigo_seguro = candidate_code.replace('\n', ' ')[:60] if candidate_code else "Código vacío o irreconocible"
                error_msg = f"Error: {result['error']}. Snippet: {codigo_seguro}..."
                state['recent_errors'].append(error_msg)
                # Mantenemos solo los últimos 3 errores para no saturar el prompt
                state['recent_errors'] = state['recent_errors'][-3:] 
            else:
                fitness = result["fitness"]
                print(f"[v] Mutación Válida. Fitness: {fitness:.6f}")
                
                # ¿Es mejor que nuestro campeón actual? (Menor es mejor en Rosenbrock)
                if fitness < state['best_fitness']:
                    print("\n" + "="*40)
                    print(f"🏆 ¡NUEVO CAMPEÓN DESCUBIERTO! (Fitness: {fitness:.6f})")
                    print("="*40)
                    state['best_fitness'] = fitness
                    state['best_code'] = candidate_code
                    # Limpiamos errores porque encontramos un nuevo camino viable
                    state['recent_errors'] = [] 
                else:
                    print("[-] El candidato funciona, pero no mejora al campeón actual.")
                    
            # Guardamos progreso después de cada iteración
            save_state(state)
            print("")
            
    except KeyboardInterrupt:
        print("\n=== BÚSQUEDA PAUSADA POR EL USUARIO ===")
        print(f"Progreso guardado en {HISTORY_FILE}. Puedes reanudar ejecutando el script de nuevo.")

if __name__ == "__main__":
    main()