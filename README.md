# pipeline — script → vidéo longue narrée

Génère une vidéo narrée de ~24 min illustrée par ~300 images fixes animées en
Ken Burns, à partir d'un fichier texte. Zéro intervention manuelle entre le
script et le MP4.

Implémente [`../SPEC.md`](../SPEC.md). Sans rapport avec `../flowapi/`.

## Installation

```bash
uv venv && uv pip install -e ".[dev]"
uv run python -m piper.download_voices fr_FR-siwis-medium --data-dir models/piper
```

`ffmpeg` et `ffprobe` doivent être dans le PATH. Sans `sudo`, une version
statique suffit :

```bash
curl -sL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
  | tar xJ --strip-components=1 -C ~/.local/bin --wildcards '*/ffmpeg' '*/ffprobe'
```

## Utilisation

```bash
python -m pipeline new snow --script mon_script.txt
python -m pipeline run snow                    # toutes les étapes manquantes
python -m pipeline run snow --sample --draft   # itération rapide
python -m pipeline run snow --step images      # une seule étape
python -m pipeline run snow --force segment    # invalide à partir de segment
```

`--sample` ne rend que 3 fenêtres de 30 s (début, milieu, fin), soit ~20 images
au lieu de 300. **On n'itère pas sur 24 minutes** : c'est là-dessus qu'on juge le
rythme et la cohérence, le rendu complet ne vient qu'après validation du script.

## Le contrat d'étape

Chaque étape lit des fichiers sous `projects/<id>/`, écrit exactement un
artefact, et ne fait rien si cet artefact existe déjà. On obtient gratuitement
la reprise après crash et la régénération partielle en secondes. C'est la seule
abstraction du système — ne pas en introduire d'autre.

| Étape | Sortie | Notes |
|---|---|---|
| `tts` | `audio.wav` | Piper local, silences insérés par ponctuation |
| `align` | `alignment.json` | forced alignment Whisper, ~1× temps réel en CPU |
| `segment` | `scenes.json` | frontières pondérées, découpe sous contrainte |
| `prompts` | `prompts.json` | API Anthropic, lots de 20 scènes |
| `images` | `images.json` + `images/*.png` | provider bouchon ou ComfyUI |
| `render` | `out.mp4` | un segment par scène en parallèle, puis concat |

## Providers d'images

```bash
PIPELINE_PROVIDER=stub    # défaut — PNG local, instantané, gratuit
PIPELINE_PROVIDER=comfy COMFY_URL=http://<hôte-windows>:8188
```

Le bouchon n'est pas décoratif : il rend le pipeline exécutable de bout en bout
sans GPU, ce qui permet de vérifier durées, synchronisation et enchaînement des
coupes indépendamment de la génération d'images.

Le GPU (RX 9060 XT, RDNA4) n'étant visible que depuis Windows, ComfyUI y tourne
en natif et le pipeline lui parle en HTTP. Aucune dépendance PyTorch ici.

## Les deux tests

```bash
uv run pytest
```

Deux invariants, et rien d'autre — ce sont les seuls endroits où une erreur est
**silencieuse** :

- **A** — `' '.join(alignment.words)` est exactement le script normalisé.
- **B** — la concaténation des textes de scènes reproduit le script normalisé.

S'y ajoute `check_plausible()`, qui couvre l'angle mort de l'invariant A : une
réconciliation Whisper peut apparier une phrase répétée à la mauvaise occurrence
et produire un alignement dont **tous les mots sont corrects et tous les temps
sont faux**. L'invariant A passe, rien ne plante, et le symptôme apparaît des
semaines plus tard sous forme d'images en retard d'une demi-phrase. Le contrôle
mesure le débit de parole sur une fenêtre glissante et échoue durement s'il
dévie.

## Journal de run

Chaque run affiche la distribution des durées de scènes et l'histogramme des
scores de frontières. C'est l'indicateur de santé de la segmentation : beaucoup
de coupes forcées signale un script mal ponctué, et une majorité de candidats à
1.0 signale que le barème ne discrimine plus rien.
