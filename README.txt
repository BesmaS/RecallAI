#  RecallAI — « Tu crois connaître ton cours ? Vérifions. »

**Mini-projet 2 : agent conversationnel, du Transformer pré-entraîné à une application de dialogue** (CY Tech, ING3 IA)

RecallAI est un coach de révision par **rappel actif** pour les étudiants en informatique (Python et SQL). Au lieu de réciter le cours, il pose des questions de difficulté croissante, évalue les réponses et corrige les erreurs. En fin de session, il produit un diagnostic, des flashcards ciblées et une révision des erreurs.

**Modèle :** [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), adapté par LoRA.

---

## Contenu du dépôt

| Fichier / dossier | Description |
|---|---|
| `agent (1).ipynb` | **Version exécutée du notebook .** Toutes les sorties sont visibles : tokenisation, figures, courbes de perte, évaluation et interface. Exécuté sur Google Colab, GPU T4. |
| `agent.ipynb` | Même notebook, sans les sorties (version source). |
| `data/connaissances.json` | Base de connaissances : 2 domaines, 8 notions, 32 sous-notions (faits de référence, erreurs fréquentes, alias). |
| `data/banque_questions.json` | Banque de 128 questions (4 niveaux par sous-notion) servant à générer le dataset. |
| `data/recallai_{train,val,test}.jsonl` | Dataset généré par la section 5. Le test ne contient que deux notions exclues de l'entraînement (Sous-requêtes, Dictionnaires). |
| `recallai-lora/` | Adaptateur LoRA (meilleur point de contrôle selon la validation) et `entrainement.json` (hyperparamètres et pertes). |
| `resultats/` | Sorties de l'exécution : réponses de référence de la baseline, courbes de perte, grille d'évaluation. |
| `rapport.pdf` | Rapport du projet (4 pages hors annexes). |
| `README.txt` | Instructions d'exécution, en texte brut. |

### Plan du notebook

0. Installation et configuration
1. Définition de l'agent
2. Observation du Transformer (tokens, logits, softmax, température, top-k/top-p, attention)
3. Moteur de session (état, scoring, consignes)
4. Baseline sans entraînement (prompt A vs B)
5. Génération du dataset (reproductible)
6. Fine-tuning LoRA
7. Évaluation avant / après LoRA
8. Interface Gradio
9. Conclusion

---

## Résultats principaux

- **Paramètres entraînés :** 18,5 M sur 1,56 Md (1,18 %), modèle de base gelé.
- **Entraînement :** arrêt anticipé au pas 75 sur 100. Le meilleur adaptateur est au pas 60, avec une perte de validation de 0,489, contre 1,615 avant entraînement.
- **Test** (deux notions jamais vues) : la perte passe de 1,454 à 0,980 et la perplexité de 4,28 à 2,67.

**Évaluation sur 24 requêtes non vues :**

| Métrique | Baseline (prompt B) | LoRA |
|---|:---:|:---:|
| Tag présent | 31 % | 100 % |
| Verdict exact | 12 % | 81 % |
| Hors périmètre refusé | 25 % | 100 % |
| Format respecté | 33 % | 79 % |

**Limites observées :** deux refus à tort (des notions légitimes jamais vues), et des informations inventées dans certaines corrections et certains QCM. Le détail est commenté dans la section 7 du notebook et dans `rapport.pdf`.

---

##  Exécution sur Google Colab (recommandé)

1. Ouvrir `agent.ipynb` (ou `agent (1).ipynb`) dans Colab.
2. **Exécution › Modifier le type d'exécution › GPU T4.**
3. **Exécution › Tout exécuter.**
4. À la cellule de la section 0.1, téléverser `data/connaissances.json` et `data/banque_questions.json`.
5. Le notebook s'exécute ensuite de haut en bas. Durées indicatives sur T4 :
   - sections 0 à 5 : environ 10 min (téléchargement du modèle, 3 Go) ;
   - section 6 (entraînement LoRA) : environ 20 à 40 min ;
   - section 7 (évaluation, 48 générations) : quelques minutes.
6. Section 8 : la dernière cellule lance l'interface Gradio. Un lien public temporaire (`*.gradio.live`) s'affiche, et l'interface apparaît aussi dans le notebook.

> **Réexécuter sans réentraîner :** placer le dossier `recallai-lora/` à la racine (le téléverser dans Colab), puis mettre `ENTRAINER = False` au début de la section 6.

> **Sauvegarder l'adaptateur :** dans Colab, les fichiers sont effacés à la fin de la session. Décommenter les deux lignes de la dernière cellule de la section 6 pour télécharger `recallai-lora.zip`.

---

##  Exécution en local

Il faut Python 3.10 ou plus récent. Un GPU NVIDIA est conseillé : le notebook fonctionne sur CPU, mais l'entraînement y est très lent.

```bash
pip install torch transformers accelerate peft gradio pandas matplotlib
jupyter notebook agent.ipynb
```

Le code a été testé avec transformers 5.17, peft 0.21, gradio 6.28 et torch 2.x. L'interface est compatible avec Gradio 5 et 6.

---

##  Utiliser l'adaptateur sans le notebook

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", dtype="auto")
model = PeftModel.from_pretrained(base, "recallai-lora")
```

L'adaptateur a été entraîné avec le prompt construit par le moteur de la section 3 (message système, fiche et consigne du tour). Il donne de bons résultats dans ce cadre, pas avec un prompt quelconque.

---

## Reproductibilité

- Graine fixe (`SEED = 42`) pour la génération du dataset, le découpage train / validation / test et l'entraînement.
- Le dataset est entièrement régénéré par la section 5 à partir des deux fichiers JSON de `data/`.
- L'évaluation utilise le décodage glouton, qui est déterministe.

##  Données

Aucune donnée personnelle : les réponses d'étudiants sont fictives, générées à partir de la banque de questions. La banque de questions et la base de connaissances ont été rédigées pour le projet avec l'aide d'une IA générative, puis relues manuellement.
