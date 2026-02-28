# 🧬 Symbolic-AutoML: Evolución de Optimizadores Zero-State con LLMs Locales

Este repositorio documenta un experimento de Descubrimiento Algorítmico Automatizado (Symbolic Discovery) utilizando un modelo de lenguaje de 4 Billones de parámetros ejecutado localmente en una CPU de consumidor.

El objetivo: Demostrar que técnicas usadas por Google DeepMind para descubrir algoritmos (como el optimizador Lion) pueden ser replicadas a microescala para resolver problemas de nicho, específicamente en Edge Computing (TinyML) y entornos con etiquetas ruidosas.

🧠 El Descubrimiento: "Robust Zero-State Optimizer"

A través de un ciclo evolutivo (Algoritmo Genético guiado por LLM), el modelo evaluó iterativamente funciones matemáticas en un Sandbox seguro. Le impusimos dos restricciones extremas:

Memoria O(1): Prohibido usar tensores del tamaño de la red para almacenar historial (como hacen Adam o Lion). Solo podía usar memoria escalar.

Ruido Extremo: El 20% de las etiquetas del dataset de evaluación fueron corrompidas (invertidas) artificialmente.

La Matemática Emergente (Generación 166)

El LLM redescubrió de forma autónoma el uso de estadística robusta combinada con la función de Signo (SignSGD parcial) para crear una Votación Democrática de Capa. La función descubierta fue:

## Matemática extraída del LLM

robust_grads = torch.sign(p.grad)
median_grad = torch.median(robust_grads)

## Inercia escalar aislada de la magnitud del ruido

state['m'] = c1 * state['m'] + (1 - c1) * median_grad
scale = 1.0 / (torch.abs(state['m']) + 1.0)

p.sub_(robust_grads, alpha=(lr * scale))

### 📊 Resultados Empíricos (Benchmark vs Adam)

Se evaluó la heurística extraída frente a Adam en una Red Neuronal Convolucional entrenando el dataset Fashion-MNIST con un 20% de inyección de ruido simétrico en las etiquetas.

Adam: Inicialmente aprende, pero memoriza rápidamente el ruido (overfitting), degradando su capacidad de generalización y estancándose.

Nuestro Optimizador (IA): Al ignorar la magnitud del gradiente y filtrar anomalías con la mediana de los signos, aísla la señal útil del ruido, superando el accuracy final de Adam sin utilizar la memoria RAM extra que este último requiere.

![alt text](image/results_vs_Adam.png)

🛠️ Arquitectura del Motor Evolutivo

El sistema fue diseñado para operar en recursos limitados (Ryzen 7 7730U, 16GB RAM):

Filtro de Alucinaciones: Análisis AST (Abstract Syntax Tree) en subprocesos para atrapar código inválido en <1ms.

Amnesia Controlada: El LLM (Qwen-4B) solo recibe sus últimos 3 errores y una "Pizarra de Laboratorio" con principios matemáticos fijos para evitar la dilución del contexto.

Evaluación Proxy: Redes estocásticas minúsculas para evaluar el fitness en milisegundos.

🚀 Próximos Pasos (Tesis)

El motor evolutivo será reconfigurado para relajar las restricciones de memoria y explorar paradigmas emergentes de optimización a gran escala, compitiendo contra arquitecturas de estado del arte como Lion (EvoLved Sign Momentum) y Sophia (Second-order Clipped Stochastic Optimization).

Este proyecto es parte de la investigación para tesis de grado en paradigmas emergentes de optimización neuronal.
