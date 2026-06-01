# -*- coding: utf-8 -*-
"""
Created on Mon May 18 19:34:06 2026

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


def TC1C2(n,E,k,itermax,tol,l,precoldic=None, X0=None):
    Xs = X0.copy()
    iteraciones=0    
    historial_errores = [] 
    error= tol +1    
    Fijos, ValoresFijos = restriccionesfijas(n, E, k, precoldic)
    PC1 = proyeccionC1(Xs, Fijos, ValoresFijos, k) #primera proyección
    while iteraciones < itermax and error > tol:
        RC1 = 2*PC1 - Xs
        PC2 = proyeccionC2(RC1,n,k)
        RC2 = 2*PC2 - RC1
        Xs=(1-(l/2))*Xs+(l/2)*RC2
        iteraciones=iteraciones+1
        PC1=proyeccionC1(Xs, Fijos, ValoresFijos, k)#es la sombra
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


#↨-------------------------leemos los 50 sudokus del proyecto euler----------------------------
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

#---------------------operadores para comparar----------------------------------------------
def TC2C1(n, E, k, itermax, tol, l, precoldic=None ,X0=None):
    Xs = X0.copy()
    iteraciones = 0
    historial_errores = []
    error = tol + 1
    Fijos, ValoresFijos = restriccionesfijas(n, E, k, precoldic)
    PC2  = proyeccionC2(Xs, n, k) #primera proyección
    while iteraciones < itermax and error > tol:
        RC2  = 2 * PC2 - Xs
        PC1  = proyeccionC1(RC2, Fijos, ValoresFijos, k)
        RC1  = 2 * PC1 - RC2
        Xs   = (1 - l/2) * Xs + (l/2) * RC1
        iteraciones += 1
        PC2 = proyeccionC2(Xs, n, k)
        error  = np.linalg.norm(proyeccionC1(PC2, Fijos, ValoresFijos, k) - PC2)
        historial_errores.append(error)
        if error < tol:
            return PC2, error, historial_errores
        elif iteraciones==itermax:
            return PC2, error, historial_errores


def TDC(n, E, k, itermax, tol, l, precoldic=None,X0=None):
    Fijos, ValoresFijos = restriccionesfijas(n, E, k, precoldic)
    X1 = X0.copy()
    X2 = X1.copy()
    iteraciones = 0
    historial_errores = []
    error = tol + 1
    while iteraciones < itermax and error > tol:
        media = (X1 + X2) / 2 #la proyección sobre D es la media
        RD1   = 2 * media - X1
        RD2   = 2 * media - X2
        PC1_  = proyeccionC1(RD1, Fijos, ValoresFijos, k)
        PC2_  = proyeccionC2(RD2, n, k)
        RC1_  = 2 * PC1_ - RD1
        RC2_  = 2 * PC2_ - RD2
        X1 = (1 - l/2) * X1 + (l/2) * RC1_
        X2 = (1 - l/2) * X2 + (l/2) * RC2_
        iteraciones += 1
        sombra = proyeccionC1((X1 + X2) / 2, Fijos, ValoresFijos, k)
        error = np.linalg.norm(proyeccionC2(sombra, n, k) - sombra)
        historial_errores.append(error)
        if error < tol:
            return sombra, error, historial_errores
        elif iteraciones==itermax:
            return sombra, error, historial_errores


def TCD(n, E, k, itermax, tol, l, precoldic=None,X0=None):
    Fijos, ValoresFijos = restriccionesfijas(n, E, k, precoldic)
    X1 = X0.copy()
    X2 = X0.copy()
    iteraciones = 0
    historial_errores = []
    error = tol + 1
    while iteraciones < itermax and error > tol:
        PC1_  = proyeccionC1(X1, Fijos, ValoresFijos, k)
        PC2_  = proyeccionC2(X2, n, k)
        RC1_  = 2 * PC1_ - X1
        RC2_  = 2 * PC2_ - X2
        media = (RC1_ + RC2_) / 2
        RD1   = 2 * media - RC1_
        RD2   = 2 * media - RC2_
        X1 = (1 - l/2) * X1 + (l/2) * RD1
        X2 = (1 - l/2) * X2 + (l/2) * RD2
        iteraciones += 1
        sombra = proyeccionC1((X1 + X2) / 2, Fijos, ValoresFijos, k)
        error = np.linalg.norm(proyeccionC2(sombra, n, k) - sombra)
        historial_errores.append(error)
        
        if error < tol:
            return sombra, error, historial_errores
        elif iteraciones==itermax:
            return sombra, error, historial_errores




#PRUEBA----------------------------------------------------------------------------------------------------------

from matplotlib.colors import LinearSegmentedColormap


lambdas    = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75]
operadores = {
    "TC1C2": TC1C2,
    "TC2C1": TC2C1,
    "TDC":   TDC,
    "TCD":   TCD,
}
itermax       = 50000
tol           = 1e-8
n_repeticiones = 10

resultados = {op: {lam: {"exitos": 0, "iters": [], "tiempos": [], "total": 0}
                   for lam in lambdas}
              for op in operadores}

total_combinaciones = len(operadores) * len(lambdas)
combinacion_actual  = 0

np.random.seed(42)  

puntos_iniciales = []
for _ in range(n_repeticiones):
    X = np.random.rand(81, 81)
    puntos_iniciales.append((X + X.T) / 2)

for op_nombre, op_func in operadores.items():
    for lam in lambdas:
        combinacion_actual += 1
        print(f"[{combinacion_actual}/{total_combinaciones}] {op_nombre} | λ={lam:.2f} ...", end=" ", flush=True)
        t0 = time.time()
        
        for idx, sud in enumerate(sudokus):
            precoldic = {pos+1: val for pos, val in enumerate(sud) if val != 0}
            Fijos, ValoresFijos = restriccionesfijas(81, E, 9, precoldic)

            for X0 in puntos_iniciales:
                X, error, historial = op_func(81, E, 9, itermax, tol, lam, precoldic, X0=X0)
                resultados[op_nombre][lam]["iters"].append(len(historial))
                resultados[op_nombre][lam]["tiempos"].append(time.time() - t0)
                resultados[op_nombre][lam]["total"] += 1

                if error <= tol:
                    resultados[op_nombre][lam]["exitos"] += 1


        elapsed = time.time() - t0
        tasa = resultados[op_nombre][lam]["exitos"] / resultados[op_nombre][lam]["total"] * 100
        print(f"tasa={tasa:.0f}%  ({elapsed:.1f}s)")
        




# =====================================================
# CONSTRUIR MATRICES PARA GRÁFICAS
# =====================================================
op_names   = list(operadores.keys())
n_ops      = len(op_names)
n_lams     = len(lambdas)

mat_tasa   = np.zeros((n_ops, n_lams))
mat_iters  = np.zeros((n_ops, n_lams))
mat_std    = np.zeros((n_ops, n_lams))
mat_tiempo = np.zeros((n_ops, n_lams))

for i, op in enumerate(op_names):
    for j, lam in enumerate(lambdas):
        r = resultados[op][lam]
        mat_tasa[i, j]   = r["exitos"] / r["total"] * 100
        mat_iters[i, j]  = np.mean(r["iters"])
        mat_std[i, j]    = np.std(r["iters"])
        mat_tiempo[i, j] = np.mean(r["tiempos"]) if r["tiempos"] else 0

lam_labels = [str(l) for l in lambdas]

# =====================================================
# GRÁFICA 1: Heatmaps tasa de éxito + iteraciones medias
# =====================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Prueba 2 — Comparativa operadores × λ | 50 sudokus Euler", fontsize=13)

cmap_verde = LinearSegmentedColormap.from_list("rv", ["#fff7bc", "#41ab5d", "#00441b"])
cmap_calor = "RdYlGn_r"

# Tasa de éxito
im1 = axes[0].imshow(mat_tasa, aspect="auto", cmap=cmap_verde, vmin=0, vmax=100)
axes[0].set_xticks(range(n_lams)); axes[0].set_xticklabels(lam_labels)
axes[0].set_yticks(range(n_ops));  axes[0].set_yticklabels(op_names)
axes[0].set_xlabel("λ"); axes[0].set_title("Tasa de éxito (%)")
for i in range(n_ops):
    for j in range(n_lams):
        axes[0].text(j, i, f"{mat_tasa[i,j]:.0f}%",
                     ha="center", va="center", fontsize=9,
                     color="white" if mat_tasa[i,j] > 60 else "black")
plt.colorbar(im1, ax=axes[0]).set_label("%")

# Iteraciones medias
im2 = axes[1].imshow(mat_iters, aspect="auto", cmap="RdYlGn_r", vmin=800, vmax=5000)
axes[1].set_xticks(range(n_lams)); axes[1].set_xticklabels(lam_labels)
axes[1].set_yticks(range(n_ops));  axes[1].set_yticklabels(op_names)
axes[1].set_xlabel("λ"); axes[1].set_title("Iteraciones medias")
for i in range(n_ops):
    for j in range(n_lams):
        axes[1].text(j, i, f"{mat_iters[i,j]:.0f}",
                     ha="center", va="center", fontsize=9,
                     color="white" if mat_iters[i,j] < 2000 else "black")
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig("prueba2_heatmaps.pdf", bbox_inches="tight", dpi=300)
plt.show()
# =====================================================
# GRÁFICA 1.a: Heatmaps tasa de éxito
# =====================================================
fig, ax = plt.subplots(figsize=(9, 5))
im1 = ax.imshow(mat_tasa, aspect="auto", cmap=cmap_verde, vmin=0, vmax=100)
ax.set_xticks(range(n_lams)); ax.set_xticklabels(lam_labels)
ax.set_yticks(range(n_ops));  ax.set_yticklabels(op_names)
ax.set_xlabel("λ")
ax.set_title("Prueba 2 — Tasa de éxito (%) | 50 sudokus Euler")
for i in range(n_ops):
    for j in range(n_lams):
        ax.text(j, i, f"{mat_tasa[i,j]:.0f}%",
                ha="center", va="center", fontsize=10,
                color="white" if mat_tasa[i,j] > 60 else "black")
plt.colorbar(im1, ax=ax).set_label("%")
plt.tight_layout()
plt.savefig("prueba2_tasa.pdf", bbox_inches="tight", dpi=300)
plt.show()

# =====================================================
# GRÁFICA 1.b: iteraciones medias
# =====================================================
fig, ax = plt.subplots(figsize=(9, 5))
im2 = ax.imshow(mat_iters, aspect="auto", cmap="RdYlGn_r", vmin=800, vmax=5000)
ax.set_xticks(range(n_lams)); ax.set_xticklabels(lam_labels)
ax.set_yticks(range(n_ops));  ax.set_yticklabels(op_names)
ax.set_xlabel("λ")
ax.set_title("Prueba 2 — Iteraciones medias | 50 sudokus Euler")
for i in range(n_ops):
    for j in range(n_lams):
        ax.text(j, i, f"{mat_iters[i,j]:.0f}",
                ha="center", va="center", fontsize=10,
                color="white" if mat_iters[i,j] < 2000 else "black")
plt.colorbar(im2, ax=ax)
plt.tight_layout()
plt.savefig("prueba2_iters.pdf", bbox_inches="tight", dpi=300)
plt.show()

# =====================================================
# GRÁFICA 2: Líneas tasa de éxito vs lambda por operador
# =====================================================
fig, ax = plt.subplots(figsize=(8, 5))
colores_op = {"TC1C2": "steelblue", "TC2C1": "tomato", "TDC": "seagreen", "TCD": "darkorange"}
markers    = {"TC1C2": "o", "TC2C1": "s", "TDC": "^", "TCD": "D"}

for i, op in enumerate(op_names):
    ax.plot(lambdas, mat_tasa[i], marker=markers[op], color=colores_op[op],
            linewidth=2, markersize=7, label=op)

ax.set_xlabel("λ"); ax.set_ylabel("Tasa de éxito (%)")
ax.set_title("Prueba 2 — Tasa de éxito vs λ por operador\n50 sudokus Euler | 10 reps")
ax.set_ylim(-5, 105)
ax.set_xticks(lambdas)
ax.axhline(100, color="gray", linestyle="--", alpha=0.4)
ax.legend(); ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("prueba2_lineas_tasa.pdf", bbox_inches="tight", dpi=300)
plt.show()

# =====================================================
# GRÁFICA 3: Líneas iteraciones medias vs lambda por operador
# =====================================================
fig, ax = plt.subplots(figsize=(8, 5))
for i, op in enumerate(op_names):
    ax.plot(lambdas, mat_iters[i], marker=markers[op], color=colores_op[op],
            linewidth=2, markersize=7, label=op)
    ax.fill_between(lambdas,
                    mat_iters[i] - mat_std[i],
                    mat_iters[i] + mat_std[i],
                    color=colores_op[op], alpha=0.12)

ax.set_xlabel("λ"); ax.set_ylabel("Iteraciones medias")
ax.set_title("Prueba 2 — Iteraciones medias vs λ por operador (± std)\n50 sudokus Euler | 10 reps")
ax.axhline(itermax, color="tomato", linestyle="--", linewidth=1, alpha=0.5, label=f"itermax={itermax}")
ax.set_xticks(lambdas)
ax.legend(); ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("prueba2_lineas_iters.pdf", bbox_inches="tight", dpi=300)
plt.show()

