RecallAI — « Tu crois connaître ton cours ? Vérifions. »
=======================================================

Mini-projet 2 : agent conversationnel, du Transformer pré-entraîné à une
application de dialogue.

RecallAI est un coach de révision par rappel actif pour étudiants en
informatique (Python et SQL). Au lieu de réciter le cours, il pose des
questions de difficulté croissante, évalue les réponses, corrige les erreurs,
puis produit un diagnostic, des flashcards ciblées et une révision des erreurs.

Modèle : Qwen/Qwen2.5-1.5B-Instruct (Hugging Face), adapté par LoRA.


CONTENU
-------

agent.ipynb                   Notebook complet, de l'installation à l'interface :
                                0. installation et configuration
                                1. définition de l'agent
                                2. observation du Transformer (tokens, logits,
                                   softmax, température, top-k/top-p, attention)
                                3. moteur de session (état, scoring, consignes)
                                4. baseline sans entraînement (prompt A vs B)
                                5. génération du dataset (reproductible)
                                6. fine-tuning LoRA
                                7. évaluation avant / après LoRA
                                8. interface Gradio
                                9. conclusion
data/connaissances.json       Base de connaissances : 2 domaines, 8 notions,
                              32 sous-notions (faits de référence, erreurs
                              fréquentes, alias).
data/banque_questions.json    Banque de 128 questions (4 niveaux par
                              sous-notion) servant à générer le dataset.
data/recallai_train.jsonl     Dataset généré par la section 5 (format
data/recallai_val.jsonl       « messages » + catégorie, domaine, notion).
data/recallai_test.jsonl      Le test contient uniquement deux notions exclues
                              de l'entraînement (Sous-requêtes, Dictionnaires).
recallai-lora/                Adaptateur LoRA (meilleur point de contrôle selon
                              la validation) + entrainement.json (hyperparamètres
                              et pertes). Créé par la section 6.
resultats/                    Sorties de l'exécution : réponses de référence de
                              la baseline, courbes de perte, grille d'évaluation.
rapport.pdf                   Rapport (4 pages maximum).


EXÉCUTION SUR GOOGLE COLAB (recommandé)
--------------------------------------

1. Ouvrir agent.ipynb dans Colab.
2. Exécution > Modifier le type d'exécution > GPU T4.
3. Exécution > Tout exécuter.
4. À la cellule de la section 0.1, téléverser les deux fichiers demandés :
   data/connaissances.json et data/banque_questions.json.
5. Le notebook s'exécute ensuite de haut en bas. Durées indicatives sur T4 :
   - sections 0 à 5 : environ 10 minutes (téléchargement du modèle, 3 Go) ;
   - section 6 (entraînement LoRA) : environ 20 à 40 minutes ;
   - section 7 (évaluation, 48 générations) : quelques minutes.
6. Section 8 : la dernière cellule lance l'interface Gradio. Un lien public
   temporaire (*.gradio.live) s'affiche ; l'interface apparaît aussi dans le
   notebook.

Réexécuter sans réentraîner : placer le dossier recallai-lora/ à la racine
(le téléverser dans Colab), puis mettre ENTRAINER = False au début de la
section 6. L'adaptateur est alors rechargé au lieu d'être réentraîné.

Sauvegarder l'adaptateur après l'entraînement : dans Colab, les fichiers sont
effacés à la fin de la session. Décommenter les deux lignes de la dernière
cellule de la section 6 pour télécharger recallai-lora.zip.


EXÉCUTION EN LOCAL
------------------

Python 3.10 ou plus récent, GPU NVIDIA conseillé (le notebook fonctionne sur
CPU, mais l'entraînement y est très lent).

    pip install torch transformers accelerate peft gradio pandas matplotlib
    jupyter notebook agent.ipynb

Versions avec lesquelles le code a été testé : transformers 5.17, peft 0.21,
gradio 6.28, torch 2.x. L'interface est compatible Gradio 5 et 6.


UTILISER L'ADAPTATEUR SANS LE NOTEBOOK
--------------------------------------

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", dtype="auto")
    model = PeftModel.from_pretrained(base, "recallai-lora")

Attention : l'adaptateur a été entraîné avec le prompt construit par le moteur
de la section 3 (system prompt, fiche et consigne du tour). Il donne de bons
résultats dans ce cadre, pas avec un prompt quelconque.


REPRODUCTIBILITÉ
----------------

- Graine fixe (SEED = 42) pour la génération du dataset, le découpage
  train / validation / test et l'entraînement.
- Le dataset est entièrement régénéré par la section 5 à partir des deux
  fichiers JSON de data/.
- L'évaluation utilise le décodage glouton (déterministe).


DONNÉES
-------

Aucune donnée personnelle : les réponses d'étudiants sont fictives, générées à
partir de la banque de questions. La banque de questions et la base de
connaissances ont été rédigées pour le projet avec l'aide d'une IA générative,
puis relues manuellement.
