# Liquid State Machine mit E-Prop in SNNTorch oder Online learning für Spiking Neural Networks (Reservoir Computing Architecture)
Das Verarbeiten zeitlich kodierter Informationen (z. B. gesprochene Sprache, sensorische Daten oder Ereignisströme) stellt klassische neuronale Netzwerke vor Herausforderungen, insbesondere im Hinblick auf Energieeffizienz, Online-Lernen und zeitliche Dynamik.
Zwar liefern künstliche neuronale Netze (ANNs) bei statischen Aufgaben wie Bildklassifikation starke Ergebnisse, doch sie sind:

- schwer über Zeit zu trainieren (z. B. bei kontinuierlichen Signalen),
- nicht biologisch plausibel,
- energieintensiv.

Dieses Repository implementiert eine Liquid State Machine (LSM) – ein rekurrentes Spiking Neural Network (SNN) mit fest verdrahtetem Reservoir – und kombiniert sie mit dem E-Prop Algorithmus zur biologisch inspirierten Online-Gewichtsaktualisierung.

## Ziel:
Ein Framework für das Training eines spikenden Klassifikators auf dem Spiking Heidelberg Digits (SHD) Datensatz, das folgende Eigenschaften erfüllt:

- 🧠 Zeitliche Verarbeitung von Spikes über ein rekurrentes Reservoir

- 🔁 E-Prop als lokale, onlinefähige Lernregel ohne Backpropagation Through Time (BPTT)

- 🧱 Modularer Code, der leicht anpassbar und wiederverwendbar ist

- 🔌 Vorbereitung für spätere Integration in Anwendungen oder auf neuromorpher Hardware
## Project Structure
```bash
lsm_eprop_shd/
├── config/                     # Konfigurationsdateien
│   └── settings.yaml
├── data/                       # Rohdaten und Preprocessing-Skripte
│   ├── raw/                    # (Optional: Originale SHD-Daten)
│   └── processed/              # Preprocessed Data (npz, pt, ...)
│   └── dataloader.py
├── models/                     # Model-Architekturen
│   ├── lsm.py                  # Liquid State Machine Definition
│   ├── eprop.py                # E-Prop Algorithmus / Lernregeln
│   └── neuron_models.py        # Custom Neurons, z. B. LIF, ALIF etc.
├── training/                   # Trainings- und Evaluationslogik
│   ├── trainer.py              # Trainingsloop
│   ├── evaluator.py            # Auswertung (Accuracy, Spikes etc.)
│   └── callbacks.py            # Logging, EarlyStopping etc.
├── utils/                      # Hilfsfunktionen
│   ├── spike_tools.py          # Spike-Statistiken, Visualisierung
│   └── metrics.py              # Loss-Funktionen, Energie-Metriken
├── experiments/                # Trainingsskripte (für verschiedene Runs)
│   └── run_shd_lsm.py          # Einstiegspunkt (Trainingskonfiguration)
├── tests/                      # Unit Tests
│   └── test_models.py
├── notebooks/                  # Für Exploration, Debugging, Visualisierung
│   └── data_exploration.ipynb
├── README.md                   # Projektbeschreibung
├── requirements.txt            # Python-Abhängigkeiten
└── .gitignore

```
