import re


def extract_ids(file_path):
    """Extrait tous les IDs d'un fichier HTML/Dash en parcourant les attributs ID."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        # Utilisation d'une expression régulière pour trouver les ID
        return set(re.findall(r"id=['\"](.*?)['\"]", content))
    except FileNotFoundError:
        print(f"Erreur : le fichier {file_path} est introuvable.")
        return set()


# Chemins vers vos fichiers
file_app = r"C:\Users\hp\Desktop\PFE\my dash app\callbacks\clustering_callbacks.py"
file_layout = r"C:\Users\hp\Desktop\PFE\my dash app\layouts\analysis_layout.py"

# Extraction des IDs
ids_app = extract_ids(file_app)
ids_layout = extract_ids(file_layout)

# Comparaison des IDs
common_ids = ids_app & ids_layout
different_ids_app = ids_app - ids_layout
different_ids_layout = ids_layout - ids_app

# Affichage des résultats
print("IDs communs : ", common_ids)
print("IDs uniques à app.py : ", different_ids_app)
print("IDs uniques à analysis_layout.py : ", different_ids_layout)
