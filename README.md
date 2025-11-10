# Hyperkahler Stack (Colab + Python)
This repository includes three Colab-friendly demos of the Hyperkahler AI/Quantum stack.

## Files
- `hyperkahler_physics_toy.py` – Quaternionic autoencoder on a toy PDE.
- `hyperkahler_hybrid_pennylane.py` – PennyLane hybrid quantum simulation.
- `hyperkahler_benchmark.py` – Baseline comparison.

## Run Locally
```bash
pip install torch pennylane matplotlib
python hyperkahler_physics_toy.py
python hyperkahler_hybrid_pennylane.py
python hyperkahler_benchmark.py
```

## Run in Colab
Upload all files to Colab, then execute in cells:
```python
!pip install torch pennylane matplotlib
!python hyperkahler_physics_toy.py
```
Results and figures will save to `/content/`.
