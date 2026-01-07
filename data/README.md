# PMI Project - Data Structure

## Organisation des données

```
data/
├── raw/                              # Données brutes CFD (1 dossier par simulation)
│   ├── NACA23015_AoA0/              # Profil NACA 23015, angle d'attaque 0°
│   │   ├── fields.csv               # Champs CFD exportés (X, Y, P, ρ, U, V)
│   │   ├── lift_coefficient-rfile.out
│   │   └── drag-coefficient-rfile.out
│   │
│   ├── NACA23015_AoA5/              # Même profil, angle 5°
│   │   └── ...
│   │
│   └── NACA0012_AoA0/               # Autre profil
│       └── ...
│
└── processed/                        # Données transformées pour le CNN
    ├── CFD_X.npy                     # Grilles 128x128x4 (N, H, W, C)
    ├── CFD_y.npy                     # Labels [CL, CD] (N, 2)
    ├── case_names.txt                # Liste des cas
    └── cnn_model.keras               # Modèle entraîné
```

## Convention de nommage

Format: `{PROFIL}_AoA{ANGLE}`

| Exemple | Signification |
|---------|---------------|
| `NACA23015_AoA0` | Profil NACA 23015, angle 0° |
| `NACA23015_AoA5` | Profil NACA 23015, angle 5° |
| `NACA0012_AoA-2` | Profil NACA 0012, angle -2° |

## Contenu de `fields.csv`

Export Fluent ASCII avec colonnes :
- `cellnumber` - ID de cellule
- `x-coordinate` - Position X
- `y-coordinate` - Position Y
- `pressure` - Pression statique
- `density` - Densité
- `x-velocity` - Vitesse U
- `y-velocity` - Vitesse V

## Comment ajouter une simulation

1. Créer un dossier dans `data/raw/` avec le bon nom
2. Y mettre :
   - `fields.csv` (export Fluent du domaine fluide)
   - `*lift*.out` (historique convergence CL)
   - `*drag*.out` (historique convergence CD)
3. Lancer `python main.py` → option 3
