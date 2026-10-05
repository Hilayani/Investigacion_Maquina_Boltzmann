import numpy as np
import matplotlib.pyplot as plt

# Pesos y sesgos que nos dieron en la sección 6
W = np.array([
    [0.0,  0.8, -0.4],
    [0.8,  0.0,  0.6],
    [-0.4, 0.6,  0.0]
])
b = np.array([0.2, -0.1, 0.3])

# Funciones de la sección 9.1
def energia(estado, pesos, sesgos):
    s = np.array(estado)
    e_interaccion = -0.5 * np.dot(s, np.dot(pesos, s))
    e_sesgos = -np.dot(sesgos, s)
    return e_interaccion + e_sesgos

def probabilidad_activacion(indice, estado, pesos, sesgos, temperatura):
    s = np.array(estado)
    delta_E = np.dot(pesos[indice], s) + sesgos[indice]
    if s[indice] == 1:
        delta_E -= pesos[indice, indice]
    
    return 1.0 / (1.0 + np.exp(-delta_E / temperatura))

def actualizar_unidad(indice, estado, pesos, sesgos, temperatura, rng):
    p1 = probabilidad_activacion(indice, estado, pesos, sesgos, temperatura)
    nuevo_estado = list(estado)
    nuevo_estado[indice] = 1 if rng.random() < p1 else 0
    return nuevo_estado

def simular(estado_inicial, pesos, sesgos, temperatura, iteraciones, semilla=42):
    rng = np.random.default_rng(semilla)
    n_unidades = len(estado_inicial)
    
    historial_estados = []
    historial_energias = []
    estado_actual = list(estado_inicial)
    
    for _ in range(iteraciones):
        i = rng.integers(0, n_unidades)
        estado_actual = actualizar_unidad(i, estado_actual, pesos, sesgos, temperatura, rng)
        historial_estados.append(tuple(estado_actual))
        historial_energias.append(energia(estado_actual, pesos, sesgos))
        
    return historial_estados, historial_energias

# Parte principal para sacar los datos de la sección 10
if __name__ == "__main__":
    iteraciones = 10000
    burn_in = 1000
    
    for T in [0.2, 1.0, 5.0]:
        estados, energias = simular([0, 0, 0], W, b, T, iteraciones, semilla=42)
        
        estados_estables = estados[burn_in:]
        energias_estables = energias[burn_in:]
        
        # Gráfica de energía
        plt.figure(figsize=(7, 3.5))
        plt.plot(energias[:500], color='blue', alpha=0.7)
        plt.title(f"Energía vs Iteración (T={T})")
        plt.xlabel("Iteración")
        plt.ylabel("Energía")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"grafica_energia_T_{T}.png")
        plt.close()
        
        # Histograma de estados
        etiquetas = [f"{s[0]}{s[1]}{s[2]}" for s in estados_estables]
        conteo = {f"{i:03b}": 0 for i in range(8)}
        for e in etiquetas:
            conteo[e] += 1
            
        plt.figure(figsize=(7, 3.5))
        plt.bar(conteo.keys(), [v / len(estados_estables) for v in conteo.values()], color='gray')
        plt.title(f"Frecuencia de Estados (T={T})")
        plt.xlabel("Estado")
        plt.ylabel("Frecuencia")
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(f"histograma_estados_T_{T}.png")
        plt.close()
        
        print(f"--- Temperatura {T} ---")
        print(f"Estado más frecuente: {max(conteo, key=conteo.get)}")
        print(f"Energía media: {np.mean(energias_estables):.4f}")
        print(f"Estados visitados: {sum(1 for v in conteo.values() if v > 0)}\n")