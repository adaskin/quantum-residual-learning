# Multi‑Stage Residual Quantum Learner
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18937121.svg)](https://doi.org/10.5281/zenodo.18937121)

This repository contains the code accompanying the paper:

**"Mitigating Frequency Learning Bias in Quantum Models via Multi‑Stage Residual Learning"**,   
Ammar Daskin,[https://arxiv.org/abs/2603.10083](https://arxiv.org/abs/2603.10083), March 2026

The project implements a multi‑stage residual learning framework for parameterized quantum circuits, inspired by classical Fourier neural operators (SpecB‑FNO). It systematically studies how quantum models learn multi‑frequency functions and demonstrates that residual training can significantly improve performance, especially for high‑frequency components.

## ⚙️ Requirements

- Python 3.8+
- PyTorch
- PennyLane
- NumPy
- scikit‑learn
- Matplotlib

Install all dependencies with:

```bash
pip install torch pennylane numpy scikit-learn matplotlib
```

## 🚀 How to Run

1. **Generate the dataset figure**  
   ```bash
   python plot_data.py
   ```

2. **Draw the circuit diagram**  
   ```bash
   python quantum_circuit_draw.py
   ```

3. **Run the qubit‑count experiments** (this may take a while)  
   ```bash
   python experiment_qubit_count.py
   ```

4. **Frequency‑resolved learning curves**  
   ```bash
   python experiment_frequency_resolved.py
   ```

5. **Residual spectrum analysis**  
   ```bash
   python experiment_residual_spectrum.py
   ```

6. **Barren plateau diagnostic**  
   ```bash
   python experiment_barren_plateau.py
   ```

7. **Run a single residual learner with default settings** (produces an overview plot)  
   ```bash
   python core.py
   ```

All scripts save their output figures as both PNG and PDF in the current directory.

## 📊 Output Files

| Script                         | Output Files                                                                 | Description |
|--------------------------------|------------------------------------------------------------------------------|-------------|
| `plot_data.py`                 | `generated_multifreq_data.{png,pdf}`                                        | Synthetic dataset coloured by dominant frequency region. |
| `quantum_circuit_draw.py`      | `quantum_circuit.{png,pdf}`                                                 | Circuit diagram for 2 qubits with one variational layer. |
| `experiment_qubit_count.py`    | `experiment_qubit_count1.{png,pdf}`, `experiment_qubit_count2.{png,pdf}`   | MSE vs. qubits (all stages) and comparison with baseline. |
| `experiment_frequency_resolved.py` | `experiment_frequency_resolved.{png,pdf}`                               | Bar plots of frequency amplitudes per stage. |
| `experiment_residual_spectrum.py` | `experiment_residual_spectrum-qubits*.{png,pdf}`, `experiment_residual_energy-qubits*.{png,pdf}` | Residual spectrum and energy reduction. |
| `experiment_barren_plateau.py` | `experiment_barren_plateau.{png,pdf}`                                       | Gradient variance scaling with qubits/layers. |
| `core.py` (standalone)         | `residual_learning_results-qubits*.{png,pdf}`, `training_curves-qubits*.{png,pdf}` | Overview of predictions, residuals, frequency amplitudes, and loss curves for a single run. |

## 📖 Citation

If you use this code in your research, please cite the paper:

```
@article{daskin2026mitigating,
  title={Mitigating Frequency Learning Bias in Quantum Models via Multi‑Stage Residual Learning},
  author={Daskin, Ammar},
  journal={arxiv preprint: 2603.10083},
  year={2026},
  url={https://arxiv.org/abs/2603.10083}
}
```

## Acknowledgements

The author acknowledges DeepSeek AI for assistance with proofreading and language refinement during the preparation of the manuscript and writing the simulation code.

---

**License:** 

[MIT](LICENSE)