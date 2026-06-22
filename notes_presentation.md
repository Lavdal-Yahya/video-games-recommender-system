# Notes de présentation — Système de recommandation de jeux vidéo

*Notes personnelles pour la soutenance. Premier ressort : raconter les **choix**
et les **problèmes résolus**, pas réciter du code. Parle des chiffres avec
assurance, assume les limites — un jury respecte l'honnêteté plus qu'un score
gonflé.*

> **⚠️ Statut au jour J — à ajuster :** la démo texte fonctionne (testée). La
> **voix locale** (faster-whisper + Piper) et l'**évaluation formelle**
> (couverture/diversité) sont en finalisation. Ne présente que ce qui tourne
> vraiment ; si la voix n'est pas prête, présente-la comme bonus en cours, le
> projet tient debout sans elle.

---

## 0. Aide-mémoire chiffres (à avoir en tête)

- Données brutes : **50 872 jeux**, ~**41 millions** d'interactions (avis Steam).
- Catalogue de travail : **6 000 jeux** (les plus interagis).
- Matrice d'interactions : **150 000 utilisateurs × 6 000 jeux**, sparsité ≈ **99,81 %**.
- Bras contenu : TF-IDF **6 000 × 8 681** termes.
- Poids du mélange : **α = 0,25**.
- LLM local : **llama3.1:8b** (via Ollama).
- Seuil d'admission du résolveur : **0,80**.
- Routage : **5** jeux sans contenu, **2 068** jeux (~34 %) à faible signal collaboratif.
- Enrichissement catalogue : **560 / 565** jeux récupérés via l'API Steam (**99,12 %**).
- Évaluation (profil complet) : **hit-rate@10 ≈ 16,8 %** à α=0,25 ; couverture
  catalogue **0,201 → 0,387** quand α monte ; déterminisme : **graine 42** partout.

---

## 1. Le problème (l'accroche, ~30 s)

- Je recommande des jeux vidéo à partir d'une requête en **langage naturel**,
  tapée ou dictée, et je restitue les résultats à l'écran et à voix haute.
- Le c\u0153ur, c'est un **hybride pondéré** : il y a deux façons de mesurer la
  similarité entre deux jeux — par leurs **métadonnées** et par **qui les joue**
  — et je mélange les deux.
- Idée en une phrase : *« donne-moi un jeu que tu aimes, je te trouve ses plus
  proches voisins » — sauf que « proche » combine le contenu et le comportement.*

## 2. Le jeu de données — et pourquoi celui-là

- J'ai pris **« Game Recommendations on Steam »** (Kaggle).
- **Pourquoi ce dataset précisément :** un hybride a besoin des **deux signaux**
  dans la même source — les métadonnées des jeux (genres, tags, description) pour
  le bras contenu, **et** les interactions utilisateur–jeu pour le bras
  collaboratif. Beaucoup de datasets n'ont que l'un ou l'autre ; celui-ci a les deux.
- Quatre fichiers : `games.csv` (titres, prix), `games_metadata.json` (tags,
  descriptions), `recommendations.csv` (~41 M d'avis = les interactions), `users.csv`.
- **Choix d'échantillonnage (à justifier si on me le demande) :** le jeu complet
  est trop lourd pour un calcul item–item sur portable. Je garde les **6 000
  jeux les plus interagis** et j'échantillonne **150 000 utilisateurs** (graine
  42). Ça garde la matrice traçable tout en restant représentatif.

## 3. La méthode / les techniques

- **Hybride pondéré (Burke).** Le score combine les deux bras avec un seul poids :
  `score = α · contenu + (1−α) · collaboratif`. Un seul bouton, α, simple à expliquer.
- **Bras contenu :** une « soupe » de texte par jeu (tags + description), nettoyée
  d'une liste d'arrêt de ~35 tokens de « plomberie » plateforme, puis **TF-IDF** +
  **cosinus**. Vocabulaire de 8 681 termes.
- **Bras collaboratif :** chaque jeu est un vecteur sur les utilisateurs (signal
  binaire : présence d'un avis), similarité **cosinus item–item pondéré par IDF**
  — la pondération empêche l'effondrement vers les jeux populaires.
- **Routage / cold-start :** tous les jeux n'ont pas les deux signaux. Je route :
  jeu sans contenu → collaboratif seul ; jeu à faible signal collaboratif → contenu
  seul ; les deux → mélange. C'est ce qui fait que **l'un couvre l'angle mort de
  l'autre**.
- **Front-end LLM :** un modèle local (`llama3.1:8b`) transforme la requête libre
  en `{ seed, filters }`. Point clé : **le LLM ne fait qu'analyser et proposer une
  amorce** — il **n'écrit jamais** les recommandations, c'est l'hybride qui les produit.
- **Amorçage par ambiance (vibe-seeding), 3 tiers :** si l'utilisateur décrit une
  ambiance sans nommer de jeu, (1) le LLM propose un jeu représentatif, (2) le
  résolveur l'**ancre** au catalogue (anti-hallucination), (3) repli TF-IDF si
  besoin, sinon réponse honnête « nomme-moi un jeu ».
- **Voix locale :** transcription par **faster-whisper**, synthèse par **Piper**,
  servies par le back-end — tout reste local, pas de dépendance cloud.

## 4. Les fonctionnalités implémentées (ce que je montre en démo)

- Recherche par **nom de jeu** avec autocomplétion → recommandations.
- Requête en **langage naturel** : *« something chill like Stardew under $20 »* →
  le LLM extrait l'amorce + le filtre prix.
- Requête par **ambiance pure** : *« a relaxing farming game »* → le système
  dérive Stardew Valley tout seul.
- **Tier de confirmation** *« did you mean… ? »* quand le titre est ambigu.
- **Provenance** affichée : pourquoi ce jeu (amorce, filtres, tier déclenché) —
  utile pour expliquer en direct.
- Réponse honnête **no_seed** guidée quand la requête n'a pas d'ancre.
- (Bonus) **Voix** : dicter la requête, entendre les résultats lus.

## 5. Les défis rencontrés et comment je les ai résolus  ⭐ (le c\u0153ur de la soutenance)

> *C'est ici que je passe le plus de temps. Chaque défi = un problème découvert,
> une cause comprise, une correction. Ça montre que j'ai compris le sujet en
> profondeur.*

**A. Le bug du catalogue — j'avais silencieusement perdu les plus gros jeux.**
- Mon premier filtre gardait les jeux ayant au moins un tag. Problème : sur Steam,
  des titres **phares** (Witcher 3, GTA V, CS:GO, Terraria…) arrivent avec des
  tags **vides** dans les métadonnées brutes.
- Résultat : mon catalogue « top jeux » avait évincé les jeux les plus joués —
  un bug invisible.
- **Solution :** j'ai retiré le filtre (catalogue = strictement le top-6 000 par
  interactions) et **enrichi** les ~565 jeux sans tags via l'**API Steam**
  (`appdetails`) — j'en ai récupéré 560, soit 99 %.

**B. Le mélange α qui ne mélangeait pas.**
- En réglant α, je voyais la qualité **basculer** brutalement du collaboratif au
  contenu — α=0,5 ne donnait pas un vrai 50/50.
- **Cause :** les cosinus contenu étaient ~3 à 4× plus grands que les cosinus
  collaboratifs. En les additionnant directement, le contenu écrasait le CF dès
  qu'α dépassait 0.
- **Solution :** je **renormalise chaque bras par son maximum** avant de mélanger.
  Là, α interpole vraiment. *(C'est un piège classique des hybrides — je l'ai
  diagnostiqué avant de coder le correctif.)*

**C. L'évaluation honnête — et un piège de protocole que j'ai corrigé.**
- Mon balayage d'α (Phase 4) montrait une précision qui décroît quand α augmente.
  Mais en validant le protocole, j'ai trouvé un **raccourci qui sous-comptait** :
  je scorais à partir de 3 amorces par utilisateur au lieu de **tout son profil**.
  Le scoring sur profil complet relève le taux de réussite d'~**35 %** en relatif
  (0,119 → 0,161 à α=0). J'ai adopté le profil complet comme protocole honnête.
- **Le vrai résultat, mesuré :** à α=0,25, le taux de réussite (0,168) est
  **statistiquement indiscernable** du collaboratif pur (0,161 ; Δ=+0,007,
  erreur-type ≈0,012) — l'exactitude est **maintenue** — pendant que la
  **couverture catalogue gagne +4,6 points**.
- Donc le mélange n'est **pas** un compromis : c'est une **amélioration de Pareto**
  (plus de couverture, aucun coût d'exactitude mesurable). *C'est mon argument
  central, et il est mesuré, pas supposé.*
- **Honnêteté assumée :** je ne maquille rien. La diversité, elle, est circulaire
  (le cosinus contenu juge une liste que le contenu a façonnée) — je le dis et je
  m'appuie sur la couverture comme signal non circulaire.

**D. Le piège du « snap » confiant du résolveur.**
- Quand le LLM proposait un titre, je le rapprochais du catalogue par similarité
  de chaînes. Deux couches « tolérantes » en série pouvaient s'accrocher à un
  **mauvais** jeu avec assurance — ex. *« stardew »* tombait sur *« StarMade »*,
  ou un jeu Nintendo absent du catalogue se faisait rattacher à un titre proche.
- **Solution :** appariement par **ratio de recouvrement de tokens** (pas juste la
  ressemblance de lettres), scan complet du catalogue, normalisation des nombres
  (« three » = « 3 » = « III »), et surtout un **seuil de confiance (0,80)** :
  en dessous, je réponds honnêtement « je n'ai pas reconnu de jeu » plutôt que de
  deviner faux. **Ancrer avec assurance ou décliner avec assurance.**

**E. L'échec lexical de l'amorçage par ambiance.**
- Ma première version dérivait l'amorce par TF-IDF pur. Mais TF-IDF est **lexical,
  pas sémantique** : *« relaxing farming game »* tombait sur **Farming Simulator**
  (des simulateurs de tracteur), *« open-world rpg with a big story »* sur
  **Warframe** (un MMO de farm). Le mot « relaxant » ne pesait rien.
- **Solution — pipeline à 3 tiers :** je laisse d'abord le **LLM proposer** le jeu
  (il *sait* que « relaxant + ferme » = Stardew), je l'**ancre** par le résolveur
  (anti-hallucination), et je garde le TF-IDF en **filet de sécurité**. Désormais
  *« relaxing farming game »* → **Stardew Valley**, *« big story RPG »* → **Witcher 3**.

**F. La voix « mic: network ».**
- La première version utilisait l'API vocale du navigateur, qui envoie l'audio aux
  serveurs Google — d'où des erreurs réseau intermittentes, une mauvaise synthèse,
  et une dépendance cloud incohérente avec mon LLM local.
- **Solution :** je suis passé à une **pile 100 % locale** — faster-whisper pour la
  transcription, Piper pour la synthèse, servies par le back-end. Ça supprime
  l'erreur réseau, améliore la voix, et fonctionne au-delà de Chrome.

## 6. L'évaluation (cadrage honnête)

- **Protocole :** leave-one-out sur les interactions, 1 000 utilisateurs
  (graine 42), scoring sur le **profil complet** de l'utilisateur. Métrique
  phare : **hit-rate@k** (= rappel@k ici) — la précision@k est plafonnée à 1/k
  avec une seule cible, donc je mène avec le hit-rate.
- **Tableau d'ablation (profil complet, n=1 000) :**

  | α | hit-rate@10 | couverture@10 | diversité@10 |
  |------|-------------|---------------|--------------|
  | 0,00 | 0,161 | 0,201 | 0,933 |
  | **0,25** | **0,168** | 0,247 | 0,908 |
  | 0,50 | 0,148 | 0,305 | 0,865 |
  | 0,75 | 0,071 | 0,368 | 0,814 |
  | 1,00 | 0,029 | 0,387 | 0,793 |

- **Lecture (le point fort) :** à α=0,25, l'exactitude est **maintenue**
  (indiscernable de α=0) et la couverture **augmente de +4,6 points** → une
  **amélioration de Pareto**, pas un compromis. Au-delà de 0,5 l'exactitude chute :
  le mélange équilibré est le bon réglage.
- **Mise en garde diversité :** la diversité décroît avec α, mais la mesure est
  **circulaire** (cosinus contenu jugeant une liste façonnée par le contenu) — je
  m'appuie sur la **couverture**, non circulaire. *Si on me pousse là-dessus, je le
  dis avant qu'on me le demande : ça montre que je connais les limites de ma métrique.*
- Couverture structurellement réelle : angles morts disjoints (5 jeux sans contenu
  couverts par le CF, 2 068 jeux faible-signal CF couverts par le contenu).

## 7. Limites et pistes

- Précision brute faible (attendu pour un LOO à une cible) → je présente le
  **hit-rate**, métrique adaptée.
- Échelle réduite (6 000 jeux, 150 000 users) pour rester traçable sur portable.
- Signal collaboratif **binaire** (présence d'avis) — on pourrait pondérer par les
  heures de jeu (piste, pas faite, par choix de simplicité).
- Latence vocale : la chaîne STT → LLM → reco → TTS est séquentielle (~4 s dominés
  par le LLM) ; un état de chargement masque l'attente.
- Pistes : plus d'utilisateurs pour densifier le CF, un signal d'interaction pondéré,
  une métrique de diversité formelle.

## 8. Questions probables du jury (prép Q&A)

- **« Pourquoi un hybride et pas juste du collaboratif ? »** → La couverture :
  ~34 % du catalogue a trop peu de signal collaboratif ; le contenu les sauve. Et
  le démarrage à froid : un jeu nouveau sans avis reste recommandable par son contenu.
- **« Pourquoi α=0,25 si la précision préfère α=0 ? »** → Parce que la précision
  mesure le co-jeu, terrain du collaboratif. 0,25 est un **vrai mélange** pour un
  coût minime (~4 sur 1 000), qui apporte de la couverture et de la diversité. Je
  l'assume comme provisoire, ajustable.
- **« Pourquoi le LLM n'écrit pas les recommandations directement ? »** →
  Contrôle et explicabilité : le LLM parse et propose une amorce **ancrée**, mais
  l'hybride reste **déterministe** et défendable. Pas d'hallucination dans les picks.
- **« Comment évites-tu qu'il invente un jeu inexistant ? »** → Toute proposition
  passe par le résolveur (seuil 0,80) ; sinon, réponse honnête no_seed.
- **« Pourquoi TF-IDF et pas des embeddings / BERT ? »** → Projet pédagogique :
  simplicité et explicabilité. Et la sémantique des ambiances est déjà apportée
  par le LLM (tier 1), donc pas besoin d'embeddings lourds.
- **« Pourquoi 6 000 jeux et 150 000 utilisateurs ? »** → Le cosinus item–item est
  coûteux ; cet échantillon (déterministe, graine 42) reste traçable sur portable
  tout en gardant les jeux les plus pertinents.
- **« Et la latence en démo ? »** → ~4 s dominés par le LLM ; je précharge le modèle
  au démarrage et j'affiche un état de chargement pour que ça ne paraisse pas figé.

---

*Rappel final : la démo texte est le socle solide. La voix est un bonus. Parle des
défis (section 5) avec confiance — c'est là que se joue la note.*
