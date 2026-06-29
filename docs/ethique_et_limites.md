# Éthique, sécurité et limites
> **Author :** Badr TAJINI 
> **Solution Delivery - filière Data** 
>  **Année académique :** 2025-2026
## Ligne rouge

Ce dépôt est un support pédagogique. Il ne doit pas être utilisé pour poser un diagnostic, trier des patients, recommander un traitement ou remplacer un professionnel qualifié.

## Avertissement obligatoire

> Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise.

Cet avertissement doit apparaître dans :

- l'interface web ;
- la sortie JSON ;
- le README ;
- la soutenance ;
- le rapport final.

## Données

Utiliser uniquement :

- données synthétiques ;
- datasets publics autorisés ;
- images explicitement dé-identifiées ;
- sous-ensembles documentés par un fichier de métadonnées.

Ne jamais stocker : nom, prénom, date de naissance, identifiant patient réel, centre hospitalier, information clinique personnelle.

## Garde-fous fonctionnels

- Classe `uncertain` si qualité image faible ou signes insuffisants.
- **Escalade d'incertitude (§7.2)** : champ `uncertainty_warning` rempli dès que
  `confidence < 0.60` OU `image_quality = mauvaise`. L'avertissement non clinique
  (`warning`), lui, est **inconditionnel** (présent sur 100 % des sorties).
- Refus des conclusions définitives.
- Contrôle de validité JSON (validateur strict + bascule en `uncertain` si invalide).
- Limitation de la justification aux observations visibles (« pas d'invention »).
- Journalisation des prompts, modèles, sorties et latences (SQLite).

## Limites à documenter

- Données synthétiques ou sous-ensembles non représentatifs.
- Risque d'hallucination textuelle.
- Confiance non automatiquement calibrée.
- Sensibilité aux prompts et au modèle choisi.
- Nécessité d'une validation indépendante pour tout usage réel.
