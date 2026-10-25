# Modifications à appliquer au skill `dix-niveaux` (claude.ai)

Trois éditions dans `SKILL.md`. Copier-coller tel quel.

---

## 1. Remplacer le temps 1 dans « Les sept temps d'un niveau »

⚠️ **Révisé le 20/09** — le premier jet de cette note disait de ne PAS narrer
l'annonce, pour éviter qu'elle soit dite deux fois (carton parlé + narration).
Le mécanisme du carton a changé depuis : il ne parle plus jamais tout seul,
il prend juste les premières secondes de SA scène pendant que l'audio de
cette scène continue sans interruption. Dans ce nouveau mécanisme, faire dire
le niveau par la narration ne répète plus rien — c'est la seule façon de
l'entendre, synchronisée avec le carton silencieux affiché dessus. Utiliser
la version ci-dessous, pas l'ancienne.

**Chercher :**

> 1. **L'annonce** — `Niveau quatre, le tesserarius.` Le titre reste dans sa langue d'origine, jamais traduit dans l'annonce ; l'explication vient juste après, en français.

**Remplacer par :**

> 1. **L'annonce** — une ligne seule `NIVEAU: Niveau quatre : tesserarius`
> juste avant le paragraphe, ligne vide entre les deux. Le paragraphe
> COMMENCE par l'annonce parlée, dans les mêmes mots que le carton, sans
> traduire le titre : `Niveau quatre, le tesserarius. Tu as 25 ans...` Cette
> phrase se dit pendant que le carton reste affiché à l'écran — un carton
> muet sur un texte que personne ne prononce ne sert à rien. L'intitulé tient
> en 8 mots maximum, sinon il déborde du carton.

---

## 2. Ajouter une section « Compatibilité pipeline » avant « Horodatage »

> ## Compatibilité pipeline
>
> Ces scripts sont montés automatiquement (TTS + illustration par scène). Trois
> conséquences sur la rédaction :
>
> - **Les marqueurs `NIVEAU:` sont obligatoires** — une ligne seule avant chaque
>   palier, suivie d'une ligne vide. Sans eux : aucun carton, et aucune référence
>   de costume par palier n'est générée, donc le personnage change d'apparence en
>   cours de vidéo.
> - **Ne jamais décrire physiquement le narrateur.** Les âges annoncés (`Tu as
>   25 ans.`) sont la scansion du format et restent. En revanche, pas de couleur
>   de cheveux, de carrure, de visage : le personnage vient d'une image de
>   référence, et toute description écrite le fait dériver.
> - **Un paragraphe = un plan.** Les lignes d'une à trois phrases déjà demandées
>   plus bas donnent le bon découpage. Un pavé de dix lignes produit une seule
>   image tenue quinze secondes.

---

## 3. Remplacer le premier paragraphe de « Horodatage »

**Chercher :**

> Écrire d'abord le script nu, une ligne par unité de souffle (une à trois phrases), lignes vides entre les niveaux. Puis :

**Remplacer par :**

> Écrire le script nu, une ligne par unité de souffle (une à trois phrases), lignes vides entre les niveaux.
>
> **Pour un script destiné au pipeline de montage, s'arrêter ici** : les
> horodatages sont retirés à l'import, les produire ne sert à rien et un oubli
> serait lu à voix haute par le TTS. L'étape ci-dessous ne concerne qu'une
> lecture manuelle ou un montage à la main.
>
> Sinon :

---

## Optionnel — durée cible

Le montage sort en 1.25x par défaut (`PIPELINE_SPEED`). Un script de 3 500 à
4 000 mots donne 22–28 min de narration à 145 mots/minute, donc **18–22 min de
vidéo finale**. Pour viser 25 min à l'écran, monter à 4 500–5 500 mots.
