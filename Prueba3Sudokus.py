# -*- coding: utf-8 -*-
"""
Created on Tue May 19 11:56:27 2026

@author: 34699
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
from scipy.linalg import eigh
import matplotlib.ticker as ticker
import pandas as pd
import time



#FUNCIONES NECESARIAS 

def restriccionesfijas(n, E, k, precoldic=None):
    valordistinto = -1 / (k - 1)
    Fijos = np.zeros((n, n), dtype=bool)
    ValoresFijos = np.zeros((n, n))
    for u, v in E:
        Fijos[u-1, v-1] = True
        Fijos[v-1,u-1] = True
        ValoresFijos[u-1, v-1] = valordistinto
        ValoresFijos[v-1,u-1] = valordistinto
    if precoldic:
        nodospre = list(precoldic.keys())
        for i in range(len(nodospre)):
            vertice1 = nodospre[i]
            color1 = precoldic[vertice1]
            for j in range(i + 1, len(nodospre)):
                vertice2 = nodospre[j]
                color2 = precoldic[vertice2]
                Fijos[vertice1-1, vertice2-1] = True
                Fijos[vertice2-1,vertice1-1] = True
                
                if color1 == color2:
                    ValoresFijos[vertice1-1,vertice2-1] = 1
                    ValoresFijos[vertice2-1,vertice1-1] = 1
                else:
                    ValoresFijos[vertice1-1,vertice2-1] = valordistinto
                    ValoresFijos[vertice2-1,vertice1-1] = valordistinto
    np.fill_diagonal(Fijos, True)
    np.fill_diagonal(ValoresFijos, 1)
    
    return Fijos, ValoresFijos

    
    
def proyeccionC1(X, Fijos, ValoresFijos, k):
    P = X.copy()
    P[P > (k - 2)/(2*k - 2)] = 1
    P[P <= (k - 2)/(2*k - 2)] = -1/(k-1)
    P[Fijos] = ValoresFijos[Fijos]
    return P



def proyeccionC2(X, n, k):
    valores, Q = eigh(X, subset_by_index=[n-k+1, n - 1])
    positivos = valores > 0
    valorespos = valores[positivos]
    Qpos = Q[:, positivos]
    if len(valorespos) == 0:
        return np.zeros((n, n))
    PC2Xs = (Qpos * valorespos) @ Qpos.T
    return PC2Xs


def TC1C2(n,E,k,itermax,tol,l,precoldic=None):
    X = np.random.rand(n, n)
    Xs = np.triu(X) + np.triu(X, 1).T
    iteraciones=0    
    historial_errores = [] 
    error= tol +1    
    Fijos, ValoresFijos = restriccionesfijas(n, E, k, precoldic)
    PC1 = proyeccionC1(Xs, Fijos, ValoresFijos, k)
    while iteraciones < itermax and error > tol:
        RC1 = 2*PC1 - Xs
        PC2 = proyeccionC2(RC1,n,k)
        RC2 = 2*PC2 - RC1
        Xs=(1-(l/2))*Xs+(l/2)*RC2
        iteraciones=iteraciones+1
        PC1=proyeccionC1(Xs, Fijos, ValoresFijos, k)
        A=proyeccionC2(PC1, n, k) - PC1
        error = np.linalg.norm(A) 
        historial_errores.append(error)
        if error < tol:
                return PC1 , error, historial_errores
        elif iteraciones == itermax: 
                    return PC1, error, historial_errores
                
                                

def comprobarsolucion(Xs,k,Fijos, ValoresFijos, precoldic=None): 
    n=Xs.shape[0]
    vertices = np.zeros(n,dtype='uint8')
    color_actual = 1
    for i in range(n):
        if vertices[i] == 0:
            vertices[i] = color_actual
            for j in range(i+1, n):
                if Xs[i, j] == 1:  
                    vertices[j] = color_actual
            color_actual += 1
            if color_actual > k : 
                if 0 in vertices:
                    return False, None 
    if precoldic: 
        reord=-np.ones(9,dtype='int8')
        for i in precoldic.items():
            reord[vertices[i[0]-1]-1]=i[1]
        R=set(reord)    
        T=set(range(9))
        T=np.array(list(T.difference(R)))
        if len(T) != np.sum(reord == -1):#esto lo añadimos para que no de error de longitud de colores
            return False, None
        reord[reord == -1] = T
        col=np.zeros(81,dtype='uint8')
        for i in range(81):
            col[i]=reord[vertices[i]-1]
        return True, col
    else:
        return True, vertices
    
#-----------------------esto es para generar aristas de sudokus-----------------------------
def generar_aristas_sudoku():
    E = set()
    def fila(i):
        return i // 9
    def columna(i):
        return i % 9
    def bloque(i):
        return (fila(i) // 3) * 3 + (columna(i) // 3)
    for i in range(81):
        for j in range(i + 1, 81):
            if (
                fila(i) == fila(j)
                or columna(i) == columna(j)
                or bloque(i) == bloque(j)
            ):
                # +1 para usar índices 1..81
                E.add((i + 1, j + 1))

    return list(E)
E = generar_aristas_sudoku()           


#000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#TERCERA PRUEBA :CON LA MEJOR COMBINACIÓN, RESOLVER UN CONJUNTO DE SUDOKUS MÁS GRANDE. Mejor combinación obtenida en prueba 2---> TC1C2 con lambda 1
#000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

def leer_sudokus_kaggle(fichero, n=1000000):
    df = pd.read_csv(fichero, nrows=n)
    sudokus = [tuple(int(c) for c in s) for s in df['puzzle']]
    return sudokus

sudokus_kaggle = leer_sudokus_kaggle('sudoku.csv', n=1000000)
print(f"Sudokus cargados: {len(sudokus_kaggle)}")


# =====================================================
# EXPERIMENTO
# =====================================================

lam     = 1
itermax = 50000
tol     = 1e-8
n_reps  = 1

import pickle
checkpoint_file = "experimento3_checkpoint.pkl"
resumen = []


exitos_totales = 0
total_runs     = 0

print(f"\nTC1C2 | λ=1 | {n_reps} repeticiones | itermax={itermax}")
print("-" * 70)
np.random.seed(42)
for idx, sud in enumerate(sudokus_kaggle):
    precoldic = {pos+1: val for pos, val in enumerate(sud) if val != 0}
    Fijos, ValoresFijos = restriccionesfijas(81, E, 9, precoldic)
    n_pistas  = len(precoldic)

    exitos  = 0
    iters   = []
    tiempos = []
    fallos  = []

    for rep in range(n_reps):
        inicio = time.time()
        X, error, historial = TC1C2(81, E, 9, itermax, tol, lam, precoldic)
        fin = time.time()

        iters.append(len(historial))
        tiempos.append(fin - inicio)

        if error <= tol:
            exitos += 1
        else:
            fallos.append({"rep": rep+1, "error": error, "iters": len(historial)})

    exitos_totales += exitos
    total_runs     += n_reps

    resumen.append({
        "idx":     idx+1,
        "pistas":  n_pistas,
        "exitos":  exitos,
        "iters":   iters,
        "tiempos": tiempos,
        "fallos":  fallos
    })
    if (idx + 1) % 1000 == 0:
        with open(checkpoint_file, "wb") as f:
            pickle.dump(resumen, f)
            tasa_parcial = exitos_totales / total_runs * 100
            print(f"  Sudoku {idx+1}/{len(sudokus_kaggle)} | tasa acumulada={tasa_parcial:.1f}%")

    
    
    
import pickle
with open("experimento3_checkpoint.pkl", "rb") as f:
    resumen = pickle.load(f)
print(f"Sudokus procesados: {len(resumen)}")

# =====================================================
# RESUMEN GLOBAL
# =====================================================

tasa_global       = exitos_totales / total_runs * 100
sudokus_perfectos = sum(1 for r in resumen if r["exitos"] == n_reps)
sudokus_fallidos  = sum(1 for r in resumen if r["exitos"] == 0)
sudokus_parciales = sum(1 for r in resumen if 0 < r["exitos"] < n_reps)

print(f"\n{'='*70}")
print(f"RESUMEN GLOBAL — 1.000.000 sudokus Kaggle | TC1C2 | λ=1")
print(f"{'='*70}")
print(f"  Tasa de éxito global:          {tasa_global:.1f}%  ({exitos_totales}/{total_runs})")
print(f"  Sudokus resueltos siempre:     {sudokus_perfectos}/{len(sudokus_kaggle)}")
print(f"  Sudokus resueltos a veces:     {sudokus_parciales}/{len(sudokus_kaggle)}")
print(f"  Sudokus fallidos siempre:      {sudokus_fallidos}/{len(sudokus_kaggle)}")
print(f"  Iter. medias globales:         "
      f"{np.mean([np.mean(r['iters']) for r in resumen]):.1f}")
print(f"  Tiempo medio por ejecución:    "
      f"{np.mean([np.mean(r['tiempos']) for r in resumen]):.3f}s")

# Sudokus con algún fallo
fallos_globales = [r for r in resumen if r["exitos"] < n_reps]
if fallos_globales:
    print(f"\n  Sudokus con algún fallo ({len(fallos_globales)}):")
    for r in fallos_globales:
        print(f"    Sudoku {r['idx']:>4} ({r['pistas']} pistas): "
              f"{r['exitos']}/{n_reps} éxitos")

# =====================================================
# GRÁFICAS
# =====================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("TC1C2 | λ=1 | 1.000.000 sudokus Kaggle", fontsize=13)

pistas_list     = [r["pistas"]          for r in resumen]
tasa_exito_list = [r["exitos"]/n_reps   for r in resumen]
iters_med_list  = [np.mean(r["iters"])  for r in resumen]

# --- Gráfica 1: tasa de éxito por sudoku ---
axes[0].bar(range(1, len(resumen)+1), tasa_exito_list,
            color='steelblue', alpha=0.8)
axes[0].set_xlabel("Índice del sudoku")
axes[0].set_ylabel("Tasa de éxito")
axes[0].set_title("Tasa de éxito por sudoku")
axes[0].set_ylim(0, 1.1)
axes[0].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='100%')
axes[0].legend()

# --- Gráfica 2: iteraciones medias vs número de pistas ---
axes[1].scatter(pistas_list, iters_med_list,
                color='steelblue', alpha=0.6, s=20)
axes[1].set_xlabel("Número de pistas")
axes[1].set_ylabel("Iteraciones medias")
axes[1].set_title("Iteraciones medias vs pistas")
axes[1].grid(True, linestyle='--', alpha=0.4)

# --- Gráfica 3: histograma de iteraciones medias ---
axes[2].hist(iters_med_list, bins=30, color='steelblue', alpha=0.8, edgecolor='white')
axes[2].set_xlabel("Iteraciones medias")
axes[2].set_ylabel("Frecuencia")
axes[2].set_title("Distribución de iteraciones")
axes[2].grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig("experimento3_kaggle.pdf", bbox_inches='tight', dpi=300)
plt.show()


# --- Gráfica 4: exito/fallo vs número de pistas ---
exito_list = [r["exitos"] for r in resumen]  # 0 o 1

fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # ← esto faltaba
fig.suptitle("TC1C2 | λ=1 | 1.000.000 sudokus Kaggle", fontsize=13)

axes[0].scatter(pistas_list, exito_list,
                c=['steelblue' if e == 1 else 'tomato' for e in exito_list],
                alpha=0.3, s=5)
axes[0].set_xlabel("Número de pistas")
axes[0].set_ylabel("Éxito (1) / Fallo (0)")
axes[0].set_title("Resultado por número de pistas")
axes[0].set_yticks([0, 1])
axes[0].set_yticklabels(["Fallo", "Éxito"])
axes[0].grid(True, linestyle='--', alpha=0.4)

# --- Gráfica 5: tasa de éxito por número de pistas ---
grupos = {}
for r in resumen:
    p = r["pistas"]
    if p not in grupos:
        grupos[p] = {"total": 0, "exitos": 0}
    grupos[p]["total"] += 1
    grupos[p]["exitos"] += r["exitos"]

labels_p = sorted(grupos.keys())
tasas    = [grupos[p]["exitos"] / grupos[p]["total"] * 100 for p in labels_p]
colores  = ["tomato" if t < 70 else "orange" if t < 95 else "steelblue" for t in tasas]

axes[1].bar(labels_p, tasas, color=colores, alpha=0.8)
axes[1].set_xlabel("Número de pistas")
axes[1].set_ylabel("Tasa de éxito (%)")
axes[1].set_title("Tasa de éxito por número de pistas")
axes[1].set_ylim(0, 105)
axes[1].axhline(100, color="gray", linestyle="--", alpha=0.4)
axes[1].grid(True, linestyle='--', alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("experimento3_kagglesuelto.pdf", bbox_inches='tight', dpi=300)
plt.show()

