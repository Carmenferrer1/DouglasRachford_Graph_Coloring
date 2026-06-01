# -*- coding: utf-8 -*-
"""
Created on Tue May 19 12:01:07 2026

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
    while iteraciones < itermax and error > tol:
        PC1 = proyeccionC1(Xs, Fijos, ValoresFijos, k)
        RC1 = 2*PC1 - Xs
        PC2 = proyeccionC2(RC1,n,k)
        RC2 = 2*PC2 - RC1
        Xs=(1-(l/2))*Xs+(l/2)*RC2
        iteraciones=iteraciones+1
        sombra=proyeccionC1(Xs, Fijos, ValoresFijos, k)
        A=proyeccionC2(sombra, n, k) - sombra
        error = np.linalg.norm(A) 
        historial_errores.append(error)
        if error < tol:
                return sombra , error, historial_errores
        elif iteraciones == itermax: #por si se alcanzan las iteraciones
                    #print("Se alcanzaron las iteraciones máximas")
                    return sombra, error, historial_errores
                
                

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


#00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#PRIMERA PRUEBA: CARGAR LOS SUDOKUS DEL PROYECTO EULER Y VER CÓMO TRABAJA MI ALGORITMO CON TC1C2 Y LAMBDA 0.75
#00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
    

def leer_sudokus_euler(fichero):
    sudokus = []
    with open(fichero, 'r') as f:
        lineas = f.readlines()
    
    i = 0
    while i < len(lineas):
        linea = lineas[i].strip()
        if linea.startswith('Grid'):
            sud = []
            for j in range(1, 10):
                fila = lineas[i+j].strip()
                sud.extend(int(c) for c in fila)
            sudokus.append(tuple(sud))
            i += 10
        else:
            i += 1
    
    return sudokus

sudokus = leer_sudokus_euler('fichero.txt')
print(f"Sudokus cargados: {len(sudokus)}")


# =====================================================
#PROBANDO: tomamos los sudokus del proyecto euler (50), y para cada uno hacemos 10 repeticiones. Usamos el operador TC1C2 y lambda 0.75
# =====================================================

lam = 0.75
itermax = 15000
tol = 1e-8
n_repeticiones = 10
exitos_totales = 0
total_runs = 0
resumen = []  # guardamos todo aquí

print(f"{'Sudoku':>8} {'Pistas':>7} {'Éxitos':>8} {'Iter. media':>12} {'Tiempo medio':>13}")
print("-" * 55)

for idx, sud in enumerate(sudokus):
    precoldic = {pos+1: val for pos, val in enumerate(sud) if val != 0}
    Fijos, ValoresFijos = restriccionesfijas(81, E, 9, precoldic)
    n_pistas = len(precoldic)
    exitos = 0
    iters = []
    tiempos = []

    for _ in range(n_repeticiones):
        inicio = time.time()
        X, error, historial = TC1C2(81, E, 9, itermax, tol, lam, precoldic)
        fin = time.time()
        iters.append(len(historial))
        tiempos.append(fin - inicio)
        if error <= tol:
            exitos += 1
    exitos_totales += exitos
    total_runs += n_repeticiones
    resumen.append({
        "idx": idx + 1,
        "pistas": n_pistas,
        "tasa": exitos / n_repeticiones,
        "iters_media": np.mean(iters),
        "iters_std": np.std(iters),
        "tiempo_medio": np.mean(tiempos),
        "exitos": exitos,
        "iters": iters,
    })

    print(f"{idx+1:>8} {n_pistas:>7} {exitos:>5}/{n_repeticiones:<3} "
          f"{np.mean(iters):>12.1f} {np.mean(tiempos):>12.2f}s")

tasa_global = exitos_totales / total_runs * 100
print(f"\n{'='*55}")
print(f"RESUMEN")
print(f"{'='*55}")
print(f"  Tasa de éxito global:     {tasa_global:.1f}%")
print(f"  Éxitos totales:           {exitos_totales}/{total_runs}")

# =====================================================
# GRÁFICAS
# =====================================================
indices      = [r["idx"]          for r in resumen]
pistas       = [r["pistas"]       for r in resumen]
tasas        = [r["tasa"]         for r in resumen]
iters_media  = [r["iters_media"]  for r in resumen]
iters_std    = [r["iters_std"]    for r in resumen]
tiempos      = [r["tiempo_medio"] for r in resumen]
todas_iters  = [it for r in resumen for it in r["iters"]]  # para histograma

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(f"TC1C2  |  λ=0.75  |  50 sudokus Euler  |  {n_repeticiones} reps/sudoku", fontsize=13)



# --- 1. Tasa de éxito por sudoku ---
fig, ax = plt.subplots(figsize=(10, 4))
colores = ['steelblue' if t == 1.0 else ('orange' if t > 0 else 'tomato') for t in tasas]
ax.bar(indices, tasas, color=colores, alpha=0.85, edgecolor='white')
ax.axhline(1.0, color='green', linestyle='--', linewidth=1, alpha=0.6, label='100%')
ax.set_xlabel("Índice del sudoku")
ax.set_ylabel("Tasa de éxito")
ax.set_title(f"TC1C2 | λ=0.75 | Tasa de éxito por sudoku")
ax.set_ylim(0, 1.15)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("prueba1_tasa_exito.pdf", bbox_inches='tight', dpi=300)
plt.show()

# --- 2. Pistas vs tasa de éxito ---
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(pistas, tasas, color='steelblue', alpha=0.7, edgecolors='white', s=60)
ax.set_xlabel("Número de pistas")
ax.set_ylabel("Tasa de éxito")
ax.set_title("TC1C2 | λ=0.75 | Pistas vs tasa de éxito")
ax.set_ylim(-0.05, 1.15)
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("prueba1_pistas_tasa.pdf", bbox_inches='tight', dpi=300)
plt.show()

# --- 3. Pistas vs iteraciones medias (con barra de error) ---
fig, ax = plt.subplots(figsize=(6, 5))
ax.errorbar(pistas, iters_media, yerr=iters_std,
            fmt='o', color='steelblue', ecolor='lightsteelblue',
            elinewidth=1.5, capsize=3, alpha=0.8)
ax.set_xlabel("Número de pistas")
ax.set_ylabel("Iteraciones medias")
ax.set_title("TC1C2 | λ=0.75 | Pistas vs iteraciones (± std)")
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("prueba1_pistas_iters.pdf", bbox_inches='tight', dpi=300)
plt.show()

# --- 4. Iteraciones medias por sudoku ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(indices, iters_media, color='steelblue', alpha=0.85, edgecolor='white')
ax.set_xlabel("Índice del sudoku")
ax.set_ylabel("Iteraciones medias")
ax.set_title("TC1C2 | λ=0.75 | Iteraciones medias por sudoku")
plt.tight_layout()
plt.savefig("prueba1_iters_sudoku.pdf", bbox_inches='tight', dpi=300)
plt.show()

# --- 5. Histograma de iteraciones ---
fig, ax = plt.subplots(figsize=(6, 5))
ax.hist(todas_iters, bins=30, color='steelblue', alpha=0.85, edgecolor='white')
ax.axvline(itermax, color='tomato', linestyle='--', linewidth=1.2, label=f'itermax={itermax}')
ax.set_xlabel("Iteraciones")
ax.set_ylabel("Frecuencia")
ax.set_title("TC1C2 | λ=0.75 | Distribución de iteraciones")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("prueba1_histograma_iters.pdf", bbox_inches='tight', dpi=300)
plt.show()

# --- 6. Tiempo medio por sudoku ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(indices, tiempos, color='steelblue', alpha=0.85, edgecolor='white')
ax.set_xlabel("Índice del sudoku")
ax.set_ylabel("Tiempo medio (s)")
ax.set_title("TC1C2 | λ=0.75 | Tiempo medio por sudoku")
plt.tight_layout()
plt.savefig("prueba1_tiempos.pdf", bbox_inches='tight', dpi=300)
plt.show()

