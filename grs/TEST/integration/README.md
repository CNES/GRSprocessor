# Tests d’intégration

Ce répertoire contient les tests d’intégration de la chaîne de traitement GRS. Ils exécutent le traitement de bout en bout à partir d’une scène Sentinel‑2 et de ses données auxiliaires, puis vérifient la structure des produits générés et la cohérence de la réflectance avec une mesure in situ HYPERNETS.

## Contenu

- `conftest.py` : charge le cas de test YAML et définit les fixtures pytest. Le traitement GRS est exécuté une fois par session.
- `test_s2_l2a.py` : vérifie le produit L2A en mémoire : variables, dimensions, formes, longueurs d’onde, CRS, coordonnées spatiales, données non vides et comparaison à HYPERNETS.
- `test_output_netcdf.py` : vérifie les fichiers NetCDF écrits : présence du produit, variables et dimensions attendues, conservation de `Rrs`, des longueurs d’onde et du CRS, ainsi que le fichier auxiliaire.
- `helpers.py` : fonctions utilitaires d’échantillonnage spatial et de préparation des références HYPERNETS.
- `reference/case_S2A_20230129.yml` : cas de référence fourni par défaut.

## Prérequis

Les tests nécessitent l’environnement Python du projet et ses dépendances de traitement et de test, notamment `pytest`, `numpy`, `xarray`, `PyYAML`, ainsi que les dépendances GRS nécessaires au traitement et à la gestion des données géospatiales.

Il faut également disposer des données d’entrée référencées dans le fichier YAML :

- une scène Sentinel‑2 au format `.SAFE` ;
- un fichier CAMS ;
- un modèle numérique de terrain (DEM) ;
- un fichier HYPERNETS au format NetCDF ;
- les données GRS nécessaires, notamment les tables de lookup (LUT).

Les chemins présents dans `reference/case_S2A_20230129.yml` sont propres à l’environnement où le fichier a été créé. **Ils doivent être adaptés à votre machine** ou remplacés par un fichier de cas dédié.

## Exécuter les tests

Depuis la racine du dépôt, dans l’environnement Python du projet :

```bash
pytest -m integration
```

Cette commande utilise par défaut le cas :

```text
integration/reference/case_S2A_20230129.yml
```

Pour utiliser un autre cas de test :

```bash
GRS_TEST_CASE=/chemin/vers/mon_cas.yml pytest -m integration
```

Le chemin indiqué par `GRS_TEST_CASE` peut être absolu ou relatif au répertoire courant. Les chemins des entrées définis dans le YAML sont, eux, résolus relativement au répertoire contenant ce fichier lorsqu’ils ne sont pas absolus.

Pour lancer un fichier de tests en particulier :

```bash
pytest -m integration integration/test_s2_l2a.py
pytest -m integration integration/test_output_netcdf.py
```

## Déroulement

1. `conftest.py` charge le fichier YAML et vérifie que les entrées requises existent.
2. La fixture `process_result` lance le traitement complet avec `Process.execute()`.
3. Les tests inspectent le produit L2A en mémoire.
4. Le produit est écrit en NetCDF, puis rouvert avec `xarray` pour vérifier le résultat sérialisé.
5. Les valeurs de `Rrs` sont comparées à la référence HYPERNETS, sur les longueurs d’onde couvertes par la mesure et autour du pixel le plus proche du site.

Les résultats sont écrits dans un répertoire temporaire géré par pytest. Ils ne sont donc pas conservés comme produits de sortie permanents.

## Configurer un cas de référence

Un fichier de cas est un document YAML contenant les entrées, les options de traitement, les attentes sur le produit et les paramètres de validation. Exemple simplifié :

```yaml
inputs:
  safe: /data/scene.SAFE
  cams: /data/cams.nc
  dem: /data/dem.tif
  hypernets: /data/hypernets.nc
  grsdata: /data/grsdata

processing:
  resolution: 20
  scale_aot: 1
  allpixels: false
  snap_compliant: false

expected:
  epsg: 32631
  variables:
    - Rrs
    - BRDFg
    - aot550
    - vza
    - sza
    - raa
    - flags
    - dem
    - mask
  dimensions:
    wl: 11
    x: 5490
    y: 5490
  wavelengths: [443, 490, 560, 665, 705, 740, 783, 842, 865, 1610, 2190]

validation:
  hypernets:
    spatial_window_pixels: 3
    tolerance:
      rtol: 0.20
      atol: 0.002
```

Les dimensions et les longueurs d’onde doivent correspondre au produit attendu pour la scène choisie. Les options de `processing` sont transmises au traitement ; les valeurs de `expected` servent aux assertions. Les tolérances HYPERNETS sont utilisées par `numpy.testing.assert_allclose`.

### Champs principaux

| Champ | Rôle |
|---|---|
| `inputs.safe` | Scène Sentinel‑2 à traiter |
| `inputs.cams` | Données atmosphériques CAMS |
| `inputs.dem` | Modèle numérique de terrain |
| `inputs.hypernets` | Mesures de référence HYPERNETS |
| `inputs.grsdata` | Répertoire des données GRS/LUT |
| `processing` | Options transmises à `Process.execute()` |
| `expected.variables` | Variables attendues dans le produit |
| `expected.forbidden_variables` | Variables qui ne doivent pas être présentes |
| `expected.dimensions` | Tailles attendues des dimensions |
| `expected.shapes` | Formes attendues de variables spécifiques |
| `expected.wavelengths` | Longueurs d’onde attendues |
| `expected.epsg` | Code EPSG attendu |
| `validation.hypernets.tolerance` | Tolérances absolue (`atol`) et relative (`rtol`) |

## Validation HYPERNETS

Le test `test_rrs_against_hypernets` :

- récupère la latitude et la longitude du site depuis les attributs du fichier HYPERNETS ;
- transforme ces coordonnées dans le CRS du produit ;
- échantillonne `Rrs` au pixel le plus proche, ou sur une fenêtre carrée définie par `spatial_window_pixels` ;
- convertit la réflectance HYPERNETS `rho_w` en `Rrs` en divisant par π ;
- interpole la référence sur les longueurs d’onde du produit et ignore celles qui sont hors de la plage de mesure ;
- compare les valeurs avec les tolérances configurées.

Une erreur de comparaison affiche les longueurs d’onde ainsi que les valeurs GRS et HYPERNETS, pour faciliter le diagnostic.

## Dépannage

- **Fichier d’entrée introuvable** : vérifiez les chemins dans le YAML. Les chemins relatifs sont résolus depuis le dossier du fichier YAML.
- **Cas YAML introuvable** : vérifiez `GRS_TEST_CASE` ou utilisez le cas par défaut.
- **CRS ou dimensions inattendus** : vérifiez que le cas de référence correspond bien à la scène et à la configuration de traitement.
- **Écart HYPERNETS trop important** : vérifiez la date et le site de la mesure, le point géographique, la fenêtre spatiale, les longueurs d’onde et les tolérances.
- **Échec lié aux LUT** : vérifiez que `inputs.grsdata` pointe vers le répertoire contenant les fichiers LUT attendus par GRS.

## Remarque

Ces tests sont des tests de bout en bout : ils lancent réellement le traitement et nécessitent les données d’entrée. Ils sont donc plus coûteux que des tests unitaires et sont particulièrement adaptés à une exécution ciblée ou à une étape dédiée de la CI disposant des données nécessaires.
