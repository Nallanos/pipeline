# Lancer une génération (configuration de production, 19/08)

## Tout démarrer d'un coup

```powershell
.\start.ps1
```

Lance ComfyUI, Chatterbox et le webui chacun dans sa fenêtre — ignore ceux
déjà en écoute sur leur port, ignore ceux dont le dossier est absent. Ouvre
le navigateur sur http://127.0.0.1:8765 une fois le webui prêt.

## La commande

```powershell
$env:PYTHONIOENCODING="utf-8"
$env:PIPELINE_TTS_PROVIDER="chatterbox"
$env:CHATTERBOX_LANGUAGES="en"
$env:COMFY_MODEL="flux2"
$env:COMFY_CHARACTER_REFS="C:\Users\nalla\Documents\pipeline\characters\refs\bldguy_head.png"
$env:COMFY_TIMEOUT="600"
.\.venv\Scripts\python.exe -m pipeline run <projet>
```

Ajouter `--sample` pour trois fenêtres de 30 s au lieu de la vidéo entière.

`COMFY_CHARACTER_REFS` ne sert plus que de repli (scènes avant le premier
palier, ou projet sans `NIVEAU:`) : sur un script à paliers, l'étape `refs`
génère automatiquement une référence par palier (costume de la section, voir
`steps/refs.py`) et la fait tourner à sa place pendant l'étape `images`. Ne
jamais y remettre `protagonist_adult_full.png` / `protagonist_adult_close.png`
— ce sont les références d'un autre personnage (costume cowboy), la cause du
bug de cohérence costume/décor corrigé le 09/09.

## Scènes en HiDream-O1 sur un GPU loué (07/10)

Les **références** (costumes, décors) restent faites par Klein sur le ComfyUI
local ; seules les **scènes** partent sur le pod. C'est automatique avec
`COMFY_MODEL=hidream` : HiDream rend des références inutilisables (crâne qui
perd son rasage, décors « vides » remplis de monde, et texture grise sur
toute scène qui réutilise ses propres images).

1. Louer un pod Runpod sur le template « ComfyUI - CUDA 13.0 », GPU ≥ 24 Go
   (RTX 4090 : ~12 s l'image, ~0,74 $/h en secure), port `8188/http`, et
   télécharger dans `models/checkpoints/` :
   `https://huggingface.co/Comfy-Org/HiDream-O1-Image/resolve/main/checkpoints/hidream_o1_image_dev_fp8_scaled.safetensors`
   (8 Go, ~3 min). Le proxy `https://<pod>-8188.proxy.runpod.net` est public
   et sans mot de passe : éteindre le pod dès la fin.
2. ComfyUI local lancé avec Klein, comme d'habitude (pour les références).
3. Lancer :

```powershell
$env:COMFY_MODEL="hidream"
$env:COMFY_URL="https://<pod>-8188.proxy.runpod.net"
$env:COMFY_CHARACTER_REFS="C:\Users\nalla\Documents\pipeline\characters\refs\bldguy_head.png"
$env:COMFY_TIMEOUT="600"
.\.venv\Scripts\python.exe -m pipeline run <projet>
```

Depuis le webui : mettre `COMFY_HIDREAM_URL=https://<pod>-8188.proxy.runpod.net`
dans `.env` et choisir « HiDream-O1 » ; `COMFY_URL` y reste celui du
ComfyUI local. Autre emplacement pour Klein : `COMFY_REFS_URL`.

Une image presque uniforme (écart-type < 12, `COMFY_FLAT_IMAGE_STD`) est
désormais comptée en erreur : 5 d'affilée arrêtent l'étape images au lieu de
remplir la vidéo de gris.

## Vitesse du montage

Le rendu final sort en **1.25x** par défaut (`PIPELINE_SPEED`) : validé à
l'oreille le 17/09, c'est le rythme du format — les silences se resserrent
sans que ça s'entende comme un accéléré. `atempo` corrige le pitch de la voix.

Appliqué dans `ffmpeg.py`, pas en post-traitement : `frame_span()` et
`title_card_command()` divisent les durées par `SPEED`, donc les segments sont
rendus directement à la bonne longueur et la vidéo n'est jamais ré-encodée une
seconde fois. Dérive mesurée sur 343 scènes : 18 ms, soit moins d'une image.

`PIPELINE_SPEED=1.0` rétablit la vitesse d'origine. Le webui expose le réglage
dans un menu **Vitesse du montage** (1.0 à 1.4), mémorisé d'un run à l'autre.

Changer la vitesse suffit à rendre le montage périmé : `steps/render.py` note
la valeur employée dans `render.json`, et le pipeline refait l'étape `render`
si elle diffère. Pas besoin de `--force render` — sans ce garde-fou on testerait
une nouvelle vitesse en regardant l'ancienne vidéo, puisque les images, elles,
n'ont pas bougé.

## Sous-titres

Produits à l'étape `render`, à partir de l'alignement mot à mot déjà calculé
par `align` — aucune reconnaissance vocale, aucun modèle appelé.

- **Incrustés** dans l'image : blancs, contour noir, en bas. Écrits pendant
  l'encodage de chaque segment, qui a lieu de toute façon : coût machine nul,
  contrairement à une incrustation sur la vidéo finale qui imposerait de tout
  ré-encoder.
- **`out.srt`** à côté de la vidéo, à envoyer à YouTube (affichage propre,
  accessibilité, transcription indexable, et le spectateur peut les couper).

Deux décalages sont appliqués, et les oublier fait dériver les sous-titres :

1. l'alignement est mesuré sur l'audio d'origine, la vidéo est montée en 1.25x
   — les timings sont divisés par `SPEED` ;
2. le carton de palier occupe les premières frames de sa scène, donc les
   sous-titres incrustés de cette scène démarrent d'autant plus tard (le
   carton est déjà du texte, il n'en porte pas). Le `.srt`, lui, n'a aucun
   décalage à appliquer : l'audio n'est jamais interrompu.

Réglages dans `subtitles.py` : 42 caractères et 7 mots par ligne au maximum,
700 ms d'affichage minimum, coupure sur la ponctuation forte.

## Contrôle du script

`pipeline new` et chaque `pipeline run` passent le script au crible
(`script_check.py`) et affichent des avertissements — jamais bloquants :

- un paragraphe qui **redit le carton** qui le précède (« Niveau un : chŏnsa »
  affiché, puis « Niveau un, le chŏnsa » lu) ;
- des **horodatages `[MM:SS]`** encore dans le texte, que le TTS lirait ;
- un script qui parle de « niveau » **sans marqueur `NIVEAU:`** — aucun carton,
  aucune référence de costume par palier ;
- un **intitulé de palier trop long** pour tenir sur un carton ;
- un **palier déséquilibré**, qui passera en un éclair.

Vu sur video5 : les dix paliers redisaient leur carton. Le contrôle l'aurait
signalé à l'import, pas après cinq heures de génération.

## Revue des images

Les scènes marquées et les prompts réécrits dans la page de revue sont
enregistrés dans `projects/<id>/review.json`, écrit à chaque modification
(différé de 400 ms). Avant, tout ne vivait qu'en mémoire du navigateur :
fermer l'onglet effaçait le marquage de plusieurs dizaines d'images, et seules
les scènes déjà régénérées étaient retrouvées au rechargement.

## Cartons de palier ("Niveau 1 : ...")

Pour un script au format « ta vie à chaque rang », insérer une ligne seule
avant le paragraphe où commence chaque palier :

```
NIVEAU: Le Commoner

Il était une fois un homme simple qui travaillait la terre...

NIVEAU: Le Chevalier

Puis un jour, tout a changé...
```

`pipeline new` extrait ces lignes à l'import (elles n'apparaissent jamais dans
`script.txt` ni ne sont lues par le TTS) et les garde dans `levels.json`. Au
rendu, un carton statique portant exactement ce texte — fond noir, aucun
mouvement de caméra — s'insère juste avant la scène où le palier commence,
silence inséré dans l'audio pour ne pas désynchroniser la suite. Seulement sur
un rendu complet, pas en `--sample`.

Le texte du marqueur est affiché tel quel : mettre "Niveau 1 : X" en toutes
lettres si c'est ce qu'on veut voir à l'écran.

Le carton est **pris sur la durée de sa scène**, jamais ajouté (depuis le
19/09) : il occupe ses ~1,5 premières secondes, l'image occupe le reste, et
l'audio n'est pas touché du tout. La voix annonce donc le niveau *pendant* que
le carton est affiché, puis l'histoire enchaîne sur l'image — au lieu d'un
silence suivi d'une répétition.

Conséquences : plus de silence inséré dans l'audio, plus de décalage à
propager dans les sous-titres, et plus d'appel TTS par palier (`_card_audio`
a disparu). La durée totale de la vidéo est exactement celle de l'audio
divisée par `SPEED` — écart mesuré sur video5 : 15 ms sur 15,5 minutes.

Les lettres ne tremblent plus non plus (`WOBBLE_AMPLITUDE_PX = 0` dans
`ffmpeg.py`, remettre 3 pour retrouver l'effet d'origine).

## Prérequis

1. **ComfyUI** sur le port 8188 : `cd C:\ComfyUI ; .\venv\Scripts\python.exe main.py`

   **Encodeur texte en fp4 — ne pas y toucher.** `COMFY_FLUX2_CLIP` vaut
   `qwen_3_4b_fp4_flux2.safetensors` (3,7 Go) et pas la version fp16 de
   7,7 Go : avec celle-ci, encodeur + UNet font 15,1 Go, ne tiennent pas
   dans les 16 Go de la carte, et ComfyUI décharge ~5 Go à chaque image.
   Mesuré le 16/09 : **60 s l'image en fp16 contre 18 s en fp4**, le
   sampling passant de 9,3 à 2,3 s/it. Le GPU ne calculait pas, il
   transférait.

   **Laisser ComfyUI tourner entre deux runs.** La première image après un
   démarrage coûte ~40 s de plus (chargement des modèles). Inutile en
   revanche de faire le ménage côté Windows : les ~3,9 Go pris par le
   bureau (DWM, navigateur) ne gênent pas une fois le fp4 en place.

   **Références de décor réduites automatiquement.** `COMFY_DECOR_REF_SCALE`
   vaut 0.34 : les références de lieu sont downscalées à ~448x256 avant
   l'upload, les références de costume restent intactes. Mesuré le 17/09,
   6 prompts alternés : décor pleine taille 59/58/22 s (instable), décor
   réduit 17/17/17 s. Une scène avec décor coûte désormais le même temps
   qu'une scène sans. Ne PAS étendre la réduction aux références de
   costume : le personnage y perd son crâne rasé dès la demi-taille.

   **Référence de décor limitée aux 3 premières scènes d'un lieu**
   (`location_bible.MAX_REF_SCENES`). Au-delà, Klein recopie son cadrage
   au lieu de s'en inspirer : sur escobart (30/09), un lieu de 15 scènes a
   donné 14 fois la même image. Les scènes suivantes gardent la
   description texte du lieu, qui suffit à la continuité.

   Plus obligatoire de le lancer soi-même : depuis le 16/09, les étapes GPU
   (`refs`, `location_refs`, `images`) le démarrent automatiquement s'il ne
   répond pas, dans sa propre fenêtre, et attendent qu'il soit prêt (~30 s).
   Réglages : `COMFY_DIR` (défaut `C:\ComfyUI`), `COMFY_BOOT_TIMEOUT`
   (défaut 180 s), `COMFY_AUTOSTART=0` pour désactiver. Un `COMFY_URL` qui
   pointe vers une autre machine n'est jamais démarré d'ici — le run
   s'arrête avec le message qui l'explique.

   **Ce qui ne peut plus arriver** : un run lancé ComfyUI éteint produisait
   343 images de repli en 2 secondes puis un `out.mp4` de bouchons (vu le
   16/09). L'étape s'arrête maintenant net, sans rien écraser. Et si ComfyUI
   tombe *en cours* de génération (vu le 15/09 : 304 replis écrits en
   silence), un coupe-circuit arrête l'étape après 5 échecs consécutifs
   (`PIPELINE_IMAGES_FAILURE_LIMIT`), garde ce qui est fait, et empêche
   `render` de partir dessus.

2. **Chatterbox** sur le port 8189 : `cd C:\ChatterboxTTS ; .\venv\Scripts\python.exe server.py`
3. Pour un run long, envelopper dans `C:\Users\nalla\lora-training\keep_awake.py` :
   la veille système a provoqué un bugcheck `0xA0` en pleine génération le 15/08.

## Pourquoi cette configuration

**FLUX.2 Klein 4B plutôt que SDXL.** SDXL perdait l'adhérence au prompt : « emerges
onto the iron gallery » devenait des rochers, « his small son » disparaissait, les
scènes à deux personnages s'effondraient sur un seul. Ce n'est ni un problème de
LoRA ni de sampler — c'est une limite du modèle. Klein tient les relations
spatiales et les scènes à plusieurs personnages.

Licence **Apache 2.0**, donc usage commercial libre. FLUX.1-dev et Klein 9B sont
non commerciaux ; FLUX.1-schnell est Apache 2.0 mais échoue sur les mêmes scènes
complexes que SDXL.

**Références plutôt que LoRA.** Deux images passées en `ReferenceLatent` tiennent
le personnage mieux que le LoRA entraîné 4 h 30 sur 104 images curées, notamment
sur les âges (enfance, vieillesse) que le LoRA ne savait pas rendre.

**Les références doivent montrer un ADULTE.** `ReferenceLatent` apparie la
référence au personnage de la scène qui lui ressemble le plus. Avec des
références d'apparence enfantine, une scène père + fils donnait le design du
protagoniste **au fils**, et le père dérivait vers un inconnu. Vérifié en A/B :
des références adultes inversent correctement l'attribution.

**4 pas, cfg 1.0, euler/simple** (dans `providers/local.py`, pas dans
`style.yaml`) : Klein est distillé. 8 pas coûtent 43 % de temps en plus sans gain
visible. Mesuré.

## Coûts mesurés

| | |
|---|---|
| Génération | **~64 s/image** |
| Vidéo de 20 min (~364 plans) | **~6 h 30** |
| Vidéo de 27 min (~490 plans) | **~8 h 45** |

L'encodeur texte quantifié fp4 **n'apporte rien** (testé) : le coût est le calcul
du Qwen3-4B, pas son chargement en VRAM.

Alternative API : ~0,014 $/image chez les hébergeurs, soit **~7 $ par vidéo** —
à considérer une fois la chaîne monétisée, en vérifiant d'abord que le
fournisseur expose bien le conditionnement par référence.

## Ce qui n'est pas encore validé

- Les références adultes n'ont été testées **que sur un plan**, pas sur une
  séquence entière. À confirmer sur un sample avant un run de 6 h.
- Le style « aplats explicites » de `style.yaml` a été validé sur 6 images.

## Personnages secondaires récurrents

`bldguy` a sa référence d'image, les lieux récurrents aussi ; les personnages
secondaires n'ont que du texte. Sans précaution, chaque lot de 20 prompts les
réinventait (vu le 30/09 sur escobart : El Mono, quatre motards différents en
quatre scènes).

L'étape `prompts` fait donc un appel de plus, sur le script entier
(`cast_bible.py`) : elle liste les personnages secondaires qui reviennent et
fige pour chacun une description visuelle de 10-15 mots, écrite dans
`cast_bible.json`. Chaque lot la reçoit avec la consigne de la recopier mot
pour mot à la place du nom — Klein ne sait pas qui est « El Mono », mais il
redessine le même « man with sun-bleached blond hair, gold chain, white
sneakers » d'une image à l'autre.

Texte seul, pas de référence d'image : deux références de personnage dans une
même scène, c'est le cas où `ReferenceLatent` se trompe d'attribution.

**Relire `cast_bible.json` avant de lancer `images`.** La détection varie
d'un appel à l'autre et peut encore donner la même tenue à deux personnages
ou glisser un nom de club sur un maillot. Le fichier se corrige à la main,
puis on supprime `prompts.json` et on relance : la bible corrigée est relue
telle quelle (seul `--force prompts` la recalcule).

## File de nuit

Sur la page d'un projet, **Ajouter à la file** enregistre le projet avec les
options du formulaire à ce moment-là (voix, modèle, vitesse, `--sample`,
forçage). Sur l'accueil, le panneau **File de nuit** les enchaîne un par un
à partir de l'heure choisie (vide = tout de suite).

- Un projet en échec est marqué comme tel, la file passe au suivant.
- La file est écrite dans `queue.json` : elle survit à un redémarrage du
  webui ou de la machine. Le projet qui tournait au moment de la coupure est
  relancé, et le pipeline reprend à l'étape où il s'était arrêté. Rien ne
  relance `start.ps1` après un crash : la file repart quand le webui
  redémarre.
- Tant qu'un run tourne (file ou run manuel), le webui empêche la mise en
  veille (`SetThreadExecutionState`) — plus besoin de `keep_awake.py`.
- La clé API Fish n'est jamais écrite dans `queue.json` : les runs de la file
  prennent celle du `.env`.

Environ 2 h à 2 h 30 par vidéo complète : 3 à 4 projets par nuit.
