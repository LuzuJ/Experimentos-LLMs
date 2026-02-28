import ast
import numpy as np
import multiprocessing

def validate_code_ast(code_str: str) -> bool:
    """Verificación de seguridad estática básica."""
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.While)):
            return False
        if isinstance(node, ast.Attribute) and node.attr.startswith('__'):
            return False
    return True

# ==========================================
# FASE 2: Entorno Estocástico (MLP Mini-batch)
# ==========================================
def generate_non_linear_data(samples=1000, features=4):
    """Genera un problema no lineal y CORROMPE el 20% de las etiquetas"""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((samples, features))
    
    # Regla real
    y_true = ((X[:, 0] * X[:, 1]) > 0).astype(int) 
    
    # Inyectamos 20% de ruido (etiquetas falsas)
    noise_mask = rng.random(samples) < 0.2
    y_noisy = np.where(noise_mask, 1 - y_true, y_true)
    
    return X, y_noisy

def compute_loss_and_grads_flat(X_batch, y_batch, flat_params):
    """
    Desempaqueta el vector 1D, hace Forward/Backward pass de un MLP (4 -> 8 -> 1)
    y devuelve la pérdida y los gradientes aplanados en 1D.
    """
    # 1. Reconstruir tensores
    W1 = flat_params[0:32].reshape(4, 8)
    b1 = flat_params[32:40].reshape(1, 8)
    W2 = flat_params[40:48].reshape(8, 1)
    b2 = flat_params[48:49].reshape(1, 1)
    
    m = X_batch.shape[0]
    
    # 2. Forward Pass (ReLU en capa oculta, Sigmoide en salida)
    Z1 = np.dot(X_batch, W1) + b1
    A1 = np.maximum(0, Z1) # ReLU
    Z2 = np.dot(A1, W2) + b2
    
    # Sigmoide con clipping para evitar NaNs
    Z2_clipped = np.clip(Z2, -250, 250)
    A2 = 1 / (1 + np.exp(-Z2_clipped))
    
    # Pérdida (Binary Cross Entropy)
    epsilon = 1e-15
    y_batch = y_batch.reshape(-1, 1)
    loss = -np.mean(y_batch * np.log(A2 + epsilon) + (1 - y_batch) * np.log(1 - A2 + epsilon))
    
    # 3. Backward Pass (Backpropagation)
    dZ2 = A2 - y_batch
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (Z1 > 0) # Derivada de ReLU
    dW1 = np.dot(X_batch.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    # 4. Aplanar gradientes para el LLM
    flat_grads = np.concatenate([dW1.flatten(), db1.flatten(), dW2.flatten(), db2.flatten()])
    return loss, flat_grads

# ==========================================
# AISLAMIENTO: Ejecución del Optimizador
# ==========================================
def worker_evaluate(code_str: str, queue: multiprocessing.Queue):
    safe_globals = {"np": np, "math": __import__('math')}
    safe_locals = {}
    
    try:
        exec(code_str, safe_globals, safe_locals)
        if 'update_step' not in safe_locals:
            queue.put({"error": "Falta update_step", "fitness": float('inf')})
            return
        update_step = safe_locals['update_step']
    except Exception as e:
        queue.put({"error": f"Error de sintaxis: {str(e)}", "fitness": float('inf')})
        return

    # Inicialización
    X, y = generate_non_linear_data(samples=600)
    # Total de parámetros para 4->8->1 es 49. Iniciamos con distribución normal pequeña.
    rng = np.random.default_rng(99)
    params = rng.standard_normal(49) * 0.1 
    
    state = {} 
    # Proveemos un entorno rico para que el LLM no pierda tokens inventando constantes
    hyperparams = {
        "lr": 0.05, 
        "c1": 0.9,     # En lugar de beta1
        "c2": 0.999,   # En lugar de beta2
        "c3": 1e-8     # En lugar de eps
    }
    
    batch_size = 32
    num_epochs = 4 # Mantenerlo corto para evaluación rápida en CPU
    n_batches = len(X) // batch_size
    
    try:
        for epoch in range(num_epochs):
            indices = rng.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                loss, grads = compute_loss_and_grads_flat(X_batch, y_batch, params)
                
                if np.isnan(loss) or np.isinf(loss) or loss > 100:
                    queue.put({"error": "Explosión de gradientes", "fitness": float('inf')})
                    return
                
                # Ejecución de la fórmula del LLM
                params, state = update_step(params, grads, state, hyperparams)
            
        memory_penalty = 0
        for key, value in state.items():
            # Si el LLM guardó un vector del tamaño de los parámetros (ej. Momentum)
            if (isinstance(value, np.ndarray) and value.size > 1) or (isinstance(value, list) and len(value) > 1):
                memory_penalty += 2.0
        
        final_loss, _ = compute_loss_and_grads_flat(X, y, params)
        total_fitness = final_loss + memory_penalty
        
        if memory_penalty > 0:
            queue.put({"error": "INFRACCIÓN DE MEMORIA: Guardaste vectores en 'state'. Solo se permiten escalares.", "fitness": float('inf')})
            return

        queue.put({"error": None, "fitness": total_fitness, "loss": final_loss})
        
        
    except Exception as e:
         queue.put({"error": f"Fallo matemático: {str(e)}", "fitness": float('inf')})

def evaluate_candidate(code_str: str, timeout: float = 3.5):
    if not validate_code_ast(code_str):
        return {"error": "Validación AST fallida", "fitness": float('inf')}
        
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=worker_evaluate, args=(code_str, queue))
    p.start()
    p.join(timeout)
    
    if p.is_alive():
        p.terminate()
        p.join()
        return {"error": "Timeout (Proceso colgado)", "fitness": float('inf')}
        
    if not queue.empty():
        return queue.get()
    else:
        return {"error": "Fallo fatal del Worker", "fitness": float('inf')}