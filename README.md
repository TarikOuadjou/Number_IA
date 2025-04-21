# Number_IA

**Number_IA** est un projet de reconnaissance de chiffres manuscrits, développé dans le but de reconnaître des chiffres dessinés à la main via une interface graphique. Actuellement, le projet utilise l'algorithme des k plus proches voisins (KNN) ainsi que de mon propre réseau de neuronne pour effectuer la reconnaissance d'image.

## Prérequis

Avant de pouvoir exécuter le programme, assurez-vous d'avoir les éléments suivants installés sur votre machine :

- **Python 3.x** avec les bibliothèques suivantes :
  - `pandas`
  - `tensorflow`
  - `numpy`

Vous pouvez installer ces bibliothèques via `pip` si elles ne sont pas encore installées :

```bash
pip install pandas tensorflow numpy
```

## Installation

### 1. Génération du fichier de données `mnist_train.csv`

Le fichier `mnist_train.csv` est nécessaire pour l'entraînement de l'algorithme. Vous pouvez soit :

- **Télécharger le fichier `mnist_train.csv`** depuis internet.
- **Générer le fichier** en exécutant le script Python `mnist_train_create.py`. Ce script utilise les bibliothèques `pandas`, `tensorflow` et `numpy` pour créer le fichier CSV à partir des données MNIST.

Une fois obtenu, le fichier `mnist_train.csv` doit être placé directement dans le dossier principal.
A terme le ficiher `data_modif.ipynb` réalisera un traitement sur le dataset, pour l'instant ce n'est pas finalisé.

### 2. Compilation du projet

Le projet utilise un **Makefile** pour simplifier la compilation du code source. Pour compiler le programme, exécutez la commande suivante dans le terminal :

```bash
make
```

Cela va générer un fichier exécutable `main.exe`.

### 3. Exécution du programme

Une fois la compilation terminée, vous pouvez lancer le programme avec la commande suivante :

```bash
./main.exe
```

Le programme ouvrira une interface graphique vous permettant de dessiner des chiffres manuscrits pour les reconnaître.

## Utilisation du programme

### Interaction avec l'interface graphique

- **Dessiner un chiffre** : Maintenez le clic gauche de la souris pour dessiner le chiffre.
- **Reconnaître le chiffre - KNN** : Après avoir dessiné un chiffre, appuyez sur la touche **"k"** pour que l'algorithme tente de reconnaître l'image à l'aide de l'algorithme des k - plus proches voisins
- **Reconnaître le chiffre - Neural Network** : Après avoir dessiné un chiffre, appuyez sur la touche **"n"** pour que l'algorithme tente de reconnaître l'image à l'aide d'un réseau de neuronne.
- **Réinitialiser la feuille de dessin** : Appuyez sur la touche **"e"** pour effacer la feuille et recommencer.
