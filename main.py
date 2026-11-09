import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =====================================================================
# 1. Implémentation mathématique de l'algorithme de De Boor-Cox
# =====================================================================
def de_boor(k, degree, t, x, knots):
    d = [x[j + k - degree] for j in range(0, degree + 1)]
    for r in range(1, degree + 1):
        for j in range(degree, r - 1, -1):
            numerateur = t - knots[j + k - degree]
            denominateur = knots[j + 1 + k - r] - knots[j + k - degree]
            alpha = numerateur / denominateur if denominateur != 0 else 0
            
            x_val = (1.0 - alpha) * d[j - 1][0] + alpha * d[j][0]
            y_val = (1.0 - alpha) * d[j - 1][1] + alpha * d[j][1]
            d[j] = [x_val, y_val]
    return d[degree]

# =====================================================================
# 2. Chargement des données du TP
# =====================================================================
df = pd.read_csv('points_contour_piece_bspline.txt', sep=';')
P = df[['x', 'y']].values.tolist()
n = len(P) - 1

degres = [1, 2, 3, 4]
couleurs = ['blue', 'green', 'red', 'purple']

P_x = [p[0] for p in P]
P_y = [p[1] for p in P]

# Création d'une grille 2x2 pour afficher les 4 graphiques séparément
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
axs = axs.flatten() # Transforme la matrice 2x2 en une simple liste de 4 éléments pour itérer

# =====================================================================
# 3. Construction et affichage des courbes pour chaque degré
# =====================================================================
for i, p in enumerate(degres):
    ax = axs[i]
    
    # Affichage du polygone de contrôle sur chaque sous-graphique
    ax.plot(P_x, P_y, 'o--', color='gray', alpha=0.5, label='Polygone de contrôle', markersize=4)
    
    m = n + p + 1
    noeuds = [0.0] * (m + 1)
    
    step = 1.0 / (n - p + 1)
    for j in range(p + 1, m - p):
        noeuds[j] = noeuds[j - 1] + step
    for j in range(m - p, m + 1):
        noeuds[j] = 1.0
        
    t_eval = np.linspace(0, 1, 500)
    courbe_x = []
    courbe_y = []
    
    for t in t_eval:
        if t == 1.0:
            courbe_x.append(P[-1][0])
            courbe_y.append(P[-1][1])
            continue
            
        k = p
        while k < m and t >= noeuds[k + 1]:
            k += 1
            
        pt = de_boor(k, p, t, P, noeuds)
        courbe_x.append(pt[0])
        courbe_y.append(pt[1])
        
    ax.plot(courbe_x, courbe_y, linewidth=2, color=couleurs[i], label=f'B-spline (p={p})')
    
    # Paramétrage de chaque sous-graphique
    ax.set_title(f"Degré {p}", fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True)
    ax.axis('equal')

# Titre global et ajustement des marges
plt.suptitle("Reconstruction de contour par B-spline (Algorithme de De Boor)", fontsize=18, y=1.02)
plt.tight_layout()
plt.show()