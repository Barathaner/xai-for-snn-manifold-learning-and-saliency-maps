

# 🧠 Explainable AI: Visualization of Neural Activity Dynamics of Spiking Neural Networks via Manifold Learning

Spiking Neural Networks (SNNs) sind die **dritte Generation Neuronaler Netze**. Dieses Projekt widmet sich ihrer **Erklärbarkeit (XAI)**. Was passiert während des Trainings? Welche Strukturen werden von den Daten gebildet und welche Strukturen werden von dem Spiking Neural Network gebildet? Gibt es einen Zusammenhang?

## ✨ 1. Einführung und Zielsetzung

* **SNNs** sind biologisch inspiriert und potenziell **sehr energieeffizient**.
* Sie sind komplex, was ihre Analyse erschwert.
* **Ziel:** Die **dynamische Aktivität** von SNNs verständlich machen indem man eine Pipeline schafft, bei der man verschiedene Methoden nutzen kann.
* **Hypothese:** **Manifolds** (durch Dimensionsreduktion) offenbaren die **strukturellen Einsichten** in SNNs.

---

## 🔬 2. Methodik: Manifold Learning
Nach dem Preprocessing wird zur Visualisierung folgende Dimensionalitätsreduktionsalgorithmen angewandt. 
Die neuronale Aktivität aus den SNNs wird extrahiert und mit Dimensionsreduktionstechniken visualisiert.

| Technik | Fokus/Eigenschaft |
| :--- | :--- |
| **PCA** | Lineare Dimensionsreduktion. Findet Hauptrichtungen der Varianz. |
| **t-SNE** | Fokussiert auf die **lokale Struktur** (Clustering) der Datenpunkte. |
| **UMAP** | Bewahrt oft sowohl **lokale als auch globale** Datenstruktur. |
| **Isomap** | Nutzt die **geodätische Distanz** zur Abbildung der globalen Struktur. |


---

## 📈 3. Analyse und Interpretation

* Es wird die neuronale Aktivität **jeder Schicht** und **jedes Trainingsschritts** (Epoch) erfasst.
* **Interpretation:** Die **Evolution des Manifolds** im Training wird analysiert, um zu verstehen, wie das Netzwerk lernt.
* Es erfolgt eine **topologische Analyse** der Manifolds (Form, Struktur).
* **Ergebnis:** Ein umfassender Vergleich der Visualisierungstechniken für SNNs.

---

## 🛠️ 4. Repository-Struktur

| Ordner | Inhalt und Zweck |
| :--- | :--- |
| **data/** | Datensätze und Preprocessing-Skripte für SNN-Experimente. |
| **experiments/** | Skripte zum Ausführen der Pipeline und zum Laden von Gewichten. |
| **model_export/** | Export der trainierten SNNs (z.B. als **NIR** und `.pth`). |
| **models/** | Definition der SNN-Architekturen. |
| **utils/** | Allgemeine Hilfsfunktionen für verschiedene Module. |
| **training/** | Definition des Trainingsalgorithmus, Callbacks und Evaluierungen. |

---