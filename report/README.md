# Système de recommandation de jeux vidéo

**Mouhamedou Yahya Cheikh Med Vall — Matricule 25 239** — Projet *Recommender
gaming*, Systèmes de recommandation, SupNum — Juin 2026.

> Hybride pondéré contenu + collaboratif, avec un front-end LLM local pour
> comprendre la requête et une pile vocale 100 % locale (faster-whisper / Piper).

Ce document est le **rendu écrit** demandé par l'énoncé : *problème, dataset,
features, méthode, captures, évaluation, limites*. Tous les chiffres viennent
d'`artifacts/eval_metrics.json` et des notes de phase dans `tasks.md` ; aucun
n'est inventé. Les figures sont produites par `notebooks/eval.ipynb` et vivent
dans `artifacts/figures/`. Une carte « claim → source » est fournie en fin de
document.

---

## 1. Problème

L'objectif est de **recommander des jeux vidéo** à partir d'une requête en
langage naturel — tapée ou dictée — et de restituer les résultats à l'écran et
à voix haute. Le cœur du système est un **hybride pondéré** au sens de Burke :
on combine deux façons indépendantes de mesurer la similarité entre deux jeux
— par leurs *métadonnées* (contenu) et par *qui les joue* (collaboratif) —
puis on retourne les plus proches voisins d'un jeu d'amorçage (*seed*).

L'idée tient en une ligne :

    score(a, b) = α · sim_contenu(a, b) + (1 − α) · sim_collab(a, b)

Le projet est volontairement **académique et pédagogique** : la clarté et la
correction priment sur l'échelle ou le vernis industriel. Chaque composant doit
rester assez simple pour être expliqué en soutenance. Conformément à l'énoncé,
le sujet « Recommender gaming » utilise un jeu de données validé par
l'enseignante, et l'application est livrée en **React** (modification validée au
préalable, à la place de Streamlit).

---

## 2. Jeu de données

Le jeu de données est **« Game Recommendations on Steam »** (Kaggle). Il a été
retenu parce qu'il porte **les deux signaux** dont l'hybride a besoin dans la
même source : métadonnées de jeux (pour le bras contenu) **et** interactions
utilisateur–jeu (pour le bras collaboratif). Beaucoup de jeux de données n'ont
que l'un ou l'autre ; celui-ci a les deux.

**Fichiers bruts :**

| Fichier                    | Volume                | Rôle                            |
|----------------------------|-----------------------|---------------------------------|
| `games.csv`                | 50 872 lignes         | titres, prix, ratio d'avis      |
| `games_metadata.json`      | 50 872 lignes         | tags + descriptions (contenu)   |
| `recommendations.csv`      | ≈ 41 M lignes         | avis = interactions (collab)    |
| `users.csv`                | table utilisateurs    | optionnel (non utilisé)         |

**Échantillonnage (choix justifié).** L'ensemble complet est trop lourd pour
calculer un cosinus item–item sur portable. On garde :

* le **catalogue des 6 000 jeux les plus interagis** (les 565 sans tags ont été
  enrichis via l'API Steam `appdetails` — 560 sur 565 récupérés, soit 99,12 %) ;
* **150 000 utilisateurs** échantillonnés (graine `42`) ayant au moins 5
  interactions dans le catalogue.

Matrice finale : `150 000 × 6 000`, **sparsité ≈ 99,81 %** (1 680 531 valeurs
non nulles). Le `game_id == app_id` est la clé de jointure unique entre toutes
les pièces du système.

---

## 3. Features

### Bras contenu — TF-IDF + cosinus
Chaque jeu devient une « soupe » de texte : ses *tags* (avec
`"Open World"` → `open_world`, le bloc étant répété 3 fois pour dominer la
prose) concaténés à sa *description nettoyée*. Une liste d'arrêt retire ~35
tokens de « plomberie » plateforme (`single_player`, `steam_achievements`, …)
qui décrivent l'infrastructure plutôt que le jeu.

La soupe est vectorisée en **TF-IDF** : matrice creuse `6 000 × 8 681`
(8 681 termes), `min_df=2`, `sublinear_tf=True`, lignes normalisées L2 — donc
le produit scalaire est directement un cosinus dans `[0, 1]`.

### Bras collaboratif — cosinus item-item pondéré IDF
Chaque jeu est un vecteur sur l'espace des utilisateurs (signal **binaire** :
présence d'un avis). La similarité est un **cosinus item-item pondéré par
IDF** : la pondération neutralise l'effondrement vers les jeux les plus
populaires (qui s'apparaîtraient mutuellement « similaires » dans toutes les
directions sans elle).

### Filtres — appliqués *après* le score, jamais avant
Les contraintes extraites de la requête (`max_price`, `tags`, `genres`) ne
contaminent pas la similarité : elles **contraignent** la liste de résultats
au moment du top-k, jamais le score lui-même. C'est ce qui permet de dire
*« quelque chose de relaxant comme Stardew sous 20 € »* et d'obtenir un seed
relaxant + un plafond de prix, sans que le plafond ne distorde le choix du
seed.

---

## 4. Méthode

### 4.1 Hybride pondéré (cœur du système)
Le score entre deux jeux `a` et `b` est une combinaison convexe des deux bras,
réglée par un seul poids `α` :

    score(a, b) = α · sim_contenu(a, b) + (1 − α) · sim_collab(a, b)

Les deux bras vivent sur des distributions de magnitude différentes (les
cosinus contenu sont ~3-4× plus grands que les cosinus CF). Si on les
additionnait directement, le mélange « basculerait » d'un bras à l'autre au
lieu d'interpoler. On **renormalise chaque vecteur de scores par son maximum
pour le seed considéré** avant le mélange — désormais `α=0.5` est un vrai
50/50. **`DEFAULT_ALPHA = 0.25`** (provisoire ; justification en §6).

### 4.2 Routage et cold-start
Tous les jeux n'ont pas les deux signaux. Le système *route* donc chaque
amorce :

| Condition                                            | Branche         | Part du catalogue |
|------------------------------------------------------|-----------------|-------------------|
| vecteur contenu nul                                  | `cf_only`       | 5 jeux (0,08 %)   |
| signal CF faible (< 50 interactions)                 | `content_only`  | 2 068 (34,5 %)    |
| les deux présents                                    | `blend` (α=0.25)| 3 927 (65,5 %)    |
| aucun                                                | `popularity`    | 0 (vide)          |

Les **angles morts des deux bras sont disjoints** : c'est la *raison d'être*
de l'hybride (les 5 MMO F2P sans métadonnées sont couverts par le CF ; les
2 068 jeux à signal CF faible sont couverts par le contenu). Aucun jeu ne
tombe en `popularity` — chaque jeu zéro-contenu a assez de signal CF, ce qui
est exactement l'asymétrie qu'exploite l'hybride.

### 4.3 Compréhension de la requête (LLM)
Un modèle local **`llama3.1:8b` (via Ollama)** transforme la requête libre en
une structure `{ seed, filters }`. Il est utilisé **uniquement** pour analyser
la requête (et, le cas échéant, proposer un jeu d'amorce représentatif) ;
**il n'écrit jamais les recommandations** — celles-ci viennent toujours de
`hybrid.recommend()`. Le titre proposé est *ancré* sur le catalogue par un
score `max(ratio_caractères, recouvrement_tokens)`, avec un **seuil
d'admission de `0.80`** : en dessous, le système répond honnêtement *« je n'ai
pas reconnu de jeu »* plutôt que de s'accrocher à un mauvais voisin. Appels
Ollama : `temperature=0`, `seed=42`, `format="json"`.

### 4.4 Amorçage par ambiance (*vibe-seeding*) — 3 tiers
Quand la requête décrit une *ambiance* sans nommer de jeu (*« quelque chose
de relaxant et atmosphérique »*), trois tiers s'enchaînent :

1. **Le LLM propose un seed.** *« relaxing farming game »* → `Stardew Valley`.
2. **Le résolveur l'ancre** (mêmes 0.80 d'admission). Anti-hallucination :
   le LLM peut s'inventer un titre, le résolveur refuse les titres
   hors-catalogue.
3. **Repli TF-IDF.** `.transform()` (et pas `.fit_transform()`) du texte de
   l'ambiance dans le même vectoriseur, plus proche voisin du catalogue,
   départage par popularité, propre seuil de confiance. Sinon → `no_seed`.

### 4.5 Front-end et voix locale
Le front-end **React (Vite)** capture la requête (texte ou micro) et affiche
les résultats : il **ne calcule rien** côté client (aucun tri, score ou
filtre). La voix est servie *localement* par le back-end :

* **STT :** [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper)
  derrière `POST /stt` — le navigateur poste l'audio capturé, reçoit le
  transcript.
* **TTS :** [`Piper`](https://github.com/rhasspy/piper) derrière `POST /tts`
  — le navigateur poste le texte, reçoit un court fichier audio à lire.

Ce choix remplace l'API vocale du navigateur (Web Speech), qui passait par
les serveurs Google (erreurs `mic: network` intermittentes, synthèse de
qualité médiocre, dépendance cloud incohérente avec le LLM local).

### 4.6 Flux d'une requête (de bout en bout)

1. **Requête** — tapée, ou dictée et POSTée à `/stt`.
2. **Analyse LLM** — la requête devient `{ seed, filters }` ; si
   `seed=null`, le pipeline vibe-seeding (§4.4) prend le relais.
3. **Score hybride** — cosinus contenu + cosinus CF, mélangés par `α` après
   re-normalisation max par seed.
4. **Filtrage + tri** — les filtres masquent les jeux non éligibles à `-∞`,
   on prend le Top-`N` aligné.
5. **Sortie** — affichage React + lecture vocale via `/tts`.

---

## 5. Captures

> Captures à insérer par le rapporteur — pour chaque emplacement, l'intention
> de la capture est décrite afin de savoir exactement quoi montrer. Les
> images vivront idéalement dans `report/figs/`.

* **C1 — Écran d'accueil :** `report/figs/c1_home.png`
  Vue d'accueil de l'app : barre de recherche, autocomplétion ouverte sur
  *« Hades »*, panneau d'aide « tape un jeu ou décris une ambiance ».
* **C2 — Résultats d'une requête nommée :** `report/figs/c2_named.png`
  Top-10 retourné pour `Hades` (la branche affichée doit être `blend` avec
  `α=0.25`).
* **C3 — Requête en langage naturel + filtre :** `report/figs/c3_filter.png`
  *« something chill like Stardew under $20 »* — le bandeau « provenance »
  doit montrer le seed extrait (`Stardew Valley`) et le filtre
  `max_price=20`.
* **C4 — Ambiance pure (vibe-seeding) :** `report/figs/c4_vibe.png`
  *« a relaxing farming game »* — l'app indique le tier déclenché (LLM →
  résolveur ancre → `Stardew Valley`) et liste les voisins.
* **C5 — Tier de confirmation « did you mean… ? » :** `report/figs/c5_didyoumean.png`
  Titre ambigu (ex. *« stardw »*) ; l'app propose une confirmation au lieu
  de deviner.
* **C6 — Réponse honnête `no_seed` :** `report/figs/c6_noseed.png`
  Requête volontairement vide / non-jeu — message guidant l'utilisateur à
  nommer un jeu.
* **C7 — Voix (round-trip) :** `report/figs/c7_voice.png`
  Bouton micro enregistre, `/stt` transcrit, `/tts` lit les résultats.
* **C8 — Figures d'évaluation :** **déjà générées**, à reproduire telles
  quelles :
  - `../artifacts/figures/phase8_ablation.png` (panneau métriques vs α)
  - `../artifacts/figures/phase8_tradeoff.png` (courbe HR@10 vs couverture)

---

## 6. Évaluation

### 6.1 Protocole (verrouillé, graine 42)

**Leave-one-out (LOO)** sur la matrice d'interactions :

* 1 000 utilisateurs échantillonnés (`np.random.default_rng(42)`,
  ≥ 2 interactions chacun) ;
* pour chaque utilisateur, on retire une interaction (la **cible**) ; le
  reste est le **profil** (toutes les autres interactions) ;
* score = `Σ` sur les seeds du profil de `α·cosinus_contenu + (1-α)·cosinus_CF`,
  chaque bras divisé par son max par seed avant mélange (même renormalisation
  qu'au runtime — c'est la même fonction de blend que `hybrid._blended_scores`) ;
* on masque les interactions connues de l'utilisateur (sauf la cible) ;
* **hit** ssi la cible est dans le Top-`k`.

**Validation préalable du protocole** (Phase 8, *avant* de croire le moindre
chiffre) :

1. **Hittabilité** : 1000 / 1000 = **100 %** des cibles retirées sont dans le
   catalogue de 6 000 jeux (par construction — la matrice est `users × catalog`).
2. **`fullprof` vs `3seed`** : le balayage de Phase 4 utilisait au plus
   3 amorces par utilisateur (raccourci de calcul). Le standard item-kNN
   somme sur **tout le profil**. À la même graine et sur le même
   échantillon, `fullprof` lève HR@10 de ~35 % en relatif à α=0
   (0.119 → 0.161). On adopte donc **`fullprof` comme protocole honnête**
   pour le headline ; on garde `3seed` uniquement pour reproduire le sweep
   de Phase 4 (toutes les déviations sont à ±10 hits/1000, dans le bruit).

### 6.2 Tableau d'ablation (`fullprof`, n=1000, graine 42)

| α    | HR@10     | HR@20     | P@10   | P@20    | couverture@10 | diversité@10 |
|------|-----------|-----------|--------|---------|---------------|--------------|
| 0.00 | 0.161     | 0.247     | 0.0161 | 0.01235 | 0.201         | 0.933        |
| 0.25 | **0.168** | **0.250** | 0.0168 | 0.01250 | **0.247**     | 0.908        |
| 0.50 | 0.148     | 0.205     | 0.0148 | 0.01025 | 0.305         | 0.865        |
| 0.75 | 0.071     | 0.103     | 0.0071 | 0.00515 | 0.368         | 0.814        |
| 1.00 | 0.029     | 0.046     | 0.0029 | 0.00230 | 0.387         | 0.793        |

> Avec une seule cible positive par essai, `HR@k = recall@k`, et
> `precision@k = HR@k / k`. **Le hit-rate est la métrique honnête** —
> precision@k est plafonné à `1/k`, ce qui fait paraître les nombres minuscules
> même quand le système se débrouille. On reporte les deux pour la
> comparabilité avec la littérature.

### 6.3 Lecture honnête (le résultat *réel* du projet)

* **Sur l'axe précision (HR@10)**, α=0.00 et α=0.25 sont **statistiquement
  indiscernables** à n=1 000 : Δ = +0.007 pour une erreur-type binomiale
  `√(0.16·0.84/1000) ≈ 0.012`. Le mélange **n'écrase pas** le CF pur — il le
  **tient**. Au-delà de α=0.5, la précision s'effondre (normal : la métrique
  prédit un co-jeu retiré, ce qui est précisément la spécialité du CF).
* **Sur l'axe couverture**, la couverture du catalogue **monte de façon
  monotone** avec α (0.201 → 0.387). À α=0.25, **~25 %** du catalogue
  apparaît dans au moins un top-10 d'utilisateur, contre **~20 %** à α=0 —
  un gain réel de **+4.6 points**, hors bruit.
* **Sur l'axe diversité**, la diversité (1 − cosinus contenu intra-liste)
  *décroît* avec α (0.933 → 0.793). **Ce métrique est circulaire** : il
  utilise le bras contenu pour juger des listes que le bras contenu a aidé
  à produire — les listes à fort α sont par construction homogènes en
  contenu. **On surface la mise en garde** plutôt que de la cacher : la
  diversité n'est *pas* un argument propre du mélange ; la couverture l'est.

**La conclusion (verrouillée, pré-enregistrée, sans retuning) :**

> α=0.25 est **un gain de Pareto** : la même HR@10 que le CF pur (à 0.007
> près, dans le bruit), **et** +4.6 points de couverture catalogue. Ce
> n'est pas un compromis précision / couverture, c'est une amélioration
> sans coût. Phase 8 **ne retouche pas** `DEFAULT_ALPHA` (retuner sur
> l'ensemble d'évaluation serait du data leakage) — la valeur 0.25 reste,
> avec la justification couverture documentée dans
> `artifacts/eval_metrics.json`.

### 6.4 Figures

![Ablation et tradeoff sur la grille d'α](../artifacts/figures/phase8_ablation.png)

*Panneau gauche : HR@10, HR@20, couverture@10, diversité@10 en fonction de α
(`n=1000`, graine 42, full-profile). La diversité descend pour la raison
expliquée en §6.3 (circularité). La ligne pointillée marque
`DEFAULT_ALPHA = 0.25`. Panneau droit : la même information vue comme
tradeoff HR@10 vs couverture — α=0.25 domine α=0 sur les deux axes.*

![Pareto HR@10 vs couverture](../artifacts/figures/phase8_tradeoff.png)

*La courbe HR@10 vs couverture, paramétrée par α. α=0.25 est strictement
au-dessus et à droite de α=0 — le pivot de Pareto. Au-delà, on troque de
la précision contre de la couverture.*

---

## 7. Limites

* **HR@10 ≈ 16,8 %, et c'est cohérent.** Pour un LOO à **une seule cible
  positive** sur un catalogue de 6 000 items, c'est l'ordre de grandeur
  attendu. Mais cela rappelle que la métrique mesure le rappel d'un *co-jeu*,
  pas la pertinence subjective.
* **Le CF gagne sur la précision** — on l'a écrit dès la lecture honnête en
  §6.3. L'hybride est défendu sur la **couverture** (gain de Pareto), pas
  sur l'exactitude maquillée.
* **La diversité est circulaire**, on l'assume ouvertement. Une diversité
  catégorique (Jaccard sur les tags) serait plus propre ; on s'en passe
  pour rester dans le périmètre pédagogique.
* **5 jeux sans contenu** (MMO F2P dont l'API Steam ne renvoie rien) — pris
  par le repli CF.
* **TF-IDF est lexical, pas sémantique** — d'où le besoin du LLM ancré pour
  l'amorçage par ambiance.
* **Échelle réduite** : 6 000 jeux × 150 000 utilisateurs, pour rester
  traçable sur portable. Le système n'est pas dimensionné pour le catalogue
  Steam complet.
* **Latence vocale** : la chaîne STT → LLM → reco → TTS est séquentielle
  (~4 s, dominés par le LLM) ; un état de chargement masque l'attente.
* **Signal d'interaction binaire** (présence d'un avis). On pourrait
  pondérer par les heures de jeu — piste non explorée par choix de
  simplicité.

---

## 8. Installation et exécution

**Prérequis :**
* Linux/macOS (testé sur Linux), Python 3.11+, Node 18+, ~2 Go de RAM libre.
* [Ollama](https://ollama.com) installé localement avec le modèle
  `llama3.1:8b` téléchargé (`ollama pull llama3.1:8b`).
* L'environnement conda `ds` (ou un venv équivalent).

**Back-end (FastAPI + artefacts pré-calculés) :**

```bash
# depuis la racine du projet
conda activate ds
pip install -r requirements.txt
uvicorn src.api:app --host 127.0.0.1 --port 8765
```

**LLM local (terminal séparé) :**

```bash
ollama serve
# modèle requis : llama3.1:8b
```

**Front-end (Vite + React) :**

```bash
cd web
npm install
npm run dev          # http://localhost:5173
```

Le front-end est codé en dur sur `http://127.0.0.1:8765` (cf.
`web/src/config.js`) et le back-end autorise CORS sur
`http://localhost:5173` — si vous changez l'un, changez les deux.

**Routes du back-end (vérifiées au boot) :**
`GET /health`, `POST /recommend`, `GET /games/search`, `POST /ask`,
`GET /resolve`, `POST /stt`, `POST /tts`.

**Reproduire l'évaluation :**

```bash
conda run -n ds jupyter nbconvert --to notebook --execute \
    notebooks/eval.ipynb --output eval.ipynb
# produit : artifacts/eval_metrics.json + artifacts/figures/*.png
```

**Déterminisme.** Toutes les étapes d'échantillonnage utilisent la graine
`42` (`np.random.default_rng(42)` pour l'éval, idem pour le sample des
utilisateurs en Phase 1, `temperature=0 seed=42 format=json` pour Ollama).
Les artefacts sont régénérables depuis `src/` et **ne doivent jamais être
édités à la main**.

---

## 9. Q&A — questions probables du jury

* **« Pourquoi un hybride et pas juste du collaboratif ? »**
  La **couverture** : ~34 % du catalogue a un signal CF trop faible pour
  recommander quoi que ce soit ; le bras contenu les sauve. Et le démarrage
  à froid : un jeu sans avis reste recommandable par son contenu. Les angles
  morts des deux bras sont **disjoints**, c'est la définition même de
  l'utilité du mélange.

* **« Pourquoi α=0.25 et pas α=0, puisque la précision préfère α=0 ? »**
  Parce que ce n'est *pas* un compromis précision/couverture, c'est un
  **gain de Pareto** : HR@10 0.168 (α=0.25) vs 0.161 (α=0) — différence
  inférieure à un écart-type binomial à n=1 000 ; **et** couverture
  passant de 20 % à 25 %. On gagne sur la couverture sans payer en
  précision. On assume que c'est provisoire et ajustable.

* **« Pourquoi le LLM n'écrit pas les recommandations directement ? »**
  Contrôle et explicabilité. Le LLM analyse et propose une amorce
  *ancrée* par le résolveur, mais l'hybride reste **déterministe** —
  même requête, même résultats, défendables ligne par ligne. Pas
  d'hallucination dans les picks.

* **« Comment évites-tu qu'il invente un jeu inexistant ? »**
  Toute proposition passe par `resolve_game_id` (seuil ≥ 0.80 sur
  `max(ratio_caractères, recouvrement_tokens)`). En dessous, réponse
  honnête `no_seed` qui guide l'utilisateur — *« je recommande des
  jeux similaires à un que tu aimes ; nomme-m'en un »*.

* **« Pourquoi TF-IDF et pas des embeddings (BERT, sentence-BERT) ? »**
  Projet pédagogique : simplicité et explicabilité. La sémantique des
  ambiances est déjà couverte par le LLM (tier 1 du vibe-seeding), donc
  pas besoin d'embeddings lourds en plus.

* **« Pourquoi 6 000 jeux et 150 000 utilisateurs ? »**
  Le cosinus item-item est coûteux. Cet échantillon (déterministe,
  graine 42) reste traçable sur portable tout en gardant les jeux les
  plus pertinents (le top 6 000 par interactions couvre la quasi-totalité
  du signal exploitable). Les 565 titres sans tags ont été enrichis via
  l'API Steam (`appdetails`) : 560 récupérés, soit 99,12 %.

* **« Pourquoi le bug des cosinus 3-4× plus grands en contenu avait un
  impact ? »**
  Sans renormalisation, à α=0.25 le contenu pesait *effectivement* ~78 %
  du score. Le mélange ne lissait pas, il *basculait*. La renormalisation
  max par seed restaure une vraie interpolation. C'est un piège
  classique des hybrides — diagnostiqué avant de coder le correctif.

* **« Et la diversité qui descend avec α ? »**
  Le métrique est **circulaire** : on utilise le bras contenu pour juger
  une liste que le bras contenu a aidé à produire — donc une liste à
  fort α paraît mécaniquement homogène. C'est la limite de cette mesure,
  pas une infériorité réelle du contenu. **L'argument propre du mélange
  passe par la couverture**, qui n'est pas circulaire.

* **« Et la latence en démo ? »**
  ~4 s à chaud, dominés par le LLM. Le modèle est préchargé au
  démarrage, un état de chargement masque l'attente. La chaîne
  STT → LLM → reco → TTS reste séquentielle.

---

## 10. Carte « claim → source » (gate méthodologique)

> Chaque chiffre quantitatif important dans ce document trace ici à sa
> source unique. `EM = artifacts/eval_metrics.json`,
> `T = tasks.md` (notes de phase), `A = artifacts/`.

| Claim                                                              | Source                                |
|--------------------------------------------------------------------|---------------------------------------|
| HR@10 ≈ 0.168 à α=0.25 (`fullprof`)                                | `EM.ablation_table[α=0.25].HR@10`     |
| HR@10 ≈ 0.161 à α=0                                                | `EM.ablation_table[α=0].HR@10`        |
| Δ HR@10 = +0.007 ; SE ≈ 0.012                                      | calcul à partir d'`EM` (binomial)     |
| Couverture catalogue : 0.201 → 0.387 sur la grille d'α             | `EM.ablation_table[*].catalog_coverage@10` |
| Couverture +4.6 pts à α=0.25 vs α=0                                | `EM` (0.247 − 0.201)                  |
| Diversité 0.933 → 0.793 (et caveat circularité)                    | `EM.ablation_table[*].intra_list_diversity@10` + `EM.tradeoff_direction` |
| Hittabilité 100 % (1000/1000)                                      | `EM.hittability`                      |
| 3-seed reproduit Phase 4 à ±10 hits/1000                           | `EM.phase4_reproducibility`           |
| Catalogue 6 000 jeux, sparsité ≈ 99,81 %                           | `T` Phase 1 Outputs                   |
| Routage : 3 927 / 2 068 / 5 / 0                                    | `T` Phase 4 Outputs                   |
| `DEFAULT_ALPHA = 0.25` (provisoire)                                | `T` Phase 4 Outputs + `src/hybrid.py` |
| Seuil résolveur 0.80                                               | `T` Phase 6 Outputs + `HANDOFF.md`    |
| Enrichissement Steam : 560 / 565 (99,12 %)                         | `T` Phase 1.5 Outputs                 |
| Figures : `phase8_ablation.png`, `phase8_tradeoff.png`             | `A/figures/`                          |

**Discrepancy à signaler.** Le document `report.tex` à la racine cite
encore la précision@10 du sweep Phase-4 (3-seed) — 0.0112, 0.0108, … —
sans le headline HR@10 `fullprof` ni le cadrage Pareto. Ces chiffres ne
sont pas *faux* (ils tracent au sweep 3-seed dans `artifacts/alpha_sweep.json`),
mais ils sont **anciens** par rapport à `eval_metrics.json`. Le présent
`report/README.md` est la version Phase-9 alignée sur la mesure honnête ;
si `report.tex` est réutilisé tel quel pour le rendu PDF, mettre à jour
sa section `Évaluation` avec le tableau de §6.2.
