from sentence_transformers import SentenceTransformer, util

# 1. Charger le modèle d'embedding (téléchargé au 1er lancement)
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 2. Les phrases à comparer
phrases = [
    "mildiou sur les tomates",      # phrase de référence
    "taches brunes au potager",     # sens proche
    "réparer une fuite d'eau",      # sens éloigné
]

# 3. Transformer chaque phrase en vecteur (384 nombres)
vecteurs = model.encode(phrases)
print(f"Forme des vecteurs : {vecteurs.shape}\n")  # (3, 384)

# 4. Comparer la phrase 0 aux deux autres
sim_proche  = util.cos_sim(vecteurs[0], vecteurs[1]).item()
sim_eloigne = util.cos_sim(vecteurs[0], vecteurs[2]).item()

print(f"'{phrases[0]}'  vs  '{phrases[1]}'  →  {sim_proche:.3f}")
print(f"'{phrases[0]}'  vs  '{phrases[2]}'  →  {sim_eloigne:.3f}")