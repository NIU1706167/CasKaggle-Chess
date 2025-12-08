# Predicció del guanyador de partides d'escacs (Lichess)  
### *Cas Kaggle – Aprenentatge Computacional (UAB)*

Aquest repositori conté el desenvolupament complet d’un projecte d’aprenentatge supervisat amb dades de partides d’escacs de **Lichess.org**, extretes del conjunt de dades de Kaggle: https://www.kaggle.com/datasnaek/chess

L’objectiu és **predir quin jugador guanyarà (blanques = 1, negres = 0)** utilitzant tant informació prèvia dels jugadors com característiques posicionalment derivades dels moviments de la partida.

---

## Estructura del repositori
**Dades**
- 'games.csv'
- 'games_EDA.csv'
- 'games_preprocessed.csv'
  
**Notebooks**
- '1_EDA.ipynb'
- '2_preprocessing.ipynb'
- '3_model_previ.ipynb'
- 3_model_partida.ipynb'
  
**Scripts**
- 'metric_selection.py'
- 'funcions.py'
  
**Documentació**
- 'README.md'

Els 4 notebooks en format Jupyter i els dos scrpits en Python s'han d'executar en el següent ordre i contenen el següent:
- 1_EDA.ipnb : Anàlisis exploratori de les dades. S'analitza el conjunt de dades, on l'objectiu és entendre les variables que tenim.
- 2_preprocessing.ipnyb : en aquest notebook fem la preparació de les dades que utilitzarem. Creem noves variables i decidim quines utilitzarem en els models.
- metric_selection.py : script de python en el qual tenim les funcions que utilitzarem amb les mètriques, juntament amb una explicació de quines mètriques utilitzarem i perquè.
- 3_model_previ.ipynb : notebook on tenim els models que entrenem per partides que no han començat, és a dir, models que entrenem sense cap de les variables relacionades amb els moviments. L'objectiu és veure fins a quin punt un model pot predir el guanyador sense informació de la partida actual i, a més, analitzar si ja es pot descartar algun dels models. Per fer això fem una cerca d'hiperparàmetres i una validació creuada en cada model per aconseguir el seu millor rendiment.
- funcions.py : script de pyhton en el qual tenim totes les funcions que es necessiten per obtenir les variables relacionades amb els moviments que hem creat al fitxer '2_preprocessing.ipynb'.
- 3_model_partida.ipynb : últim notebook en el qual entrenems models ara afegints les variables dels moviments i acabem seleccionant un model final, després de fer cerca d'hiperparàmetres i validació creuada.
---
## Resultats
El model final seleccionat ha estat el Gradient Boosting, de la mateixa manera que podríem haver seleccionat el Random Forest, ja que els resultats eren pràcticament els mateixos.

A l'hora de predir el guanyador d'una partida que encara no ha començat, aconseguim un accuracy del 0.666, de manera que s'està predint correctament 2 de cada 3 partides, la qual cosa considerem com un bon resultat.

Pel model que utilitza informació dels moviments, hem fet servir els primers 35 moviments de cada partida. Aquesta elecció ens assegurava que la gran majoria de partides no estiguessin a punt d'acabar, però alhora ens donava prou informació perquè el model pogués millorar. Dit això, el model obté un accuracy del 0.736, el que ens diu que, com ja ens podíem imaginar, afegint informació dels moviments és més fàcil predir qui guanyarà

---

## Autores
- Inés Gómez Carmona (1704361)
- Naroa Sarrià Gil (1706167)

Repositori GitHub:
https://github.com/NIU1706167/CasKaggle-Chess
