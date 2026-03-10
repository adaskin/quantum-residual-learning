import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# ------------------------------------------------------------
# Define the quantum circuit (matching the paper's architecture)
# ------------------------------------------------------------
n_qubits = 2
input_dim = 1          # one input feature
n_layers = 1           # one variational layer for clarity

dev = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(dev, interface=None)   # interface=None for pure NumPy drawing
def circuit(x, weights):
    # Data encoding (applied to each qubit cyclically)
    for q in range(n_qubits):
        feat_idx = q % input_dim   # cycle through input features
        qml.RY(np.pi * x[feat_idx], wires=q)
        qml.RZ(np.pi * x[feat_idx]**3, wires=q)
        qml.RX(np.pi * np.sqrt(np.abs(1 - x[feat_idx]**2)), wires=q)
        print(f"Encoding qubit {q} with input feature {feat_idx} (x={x[feat_idx]:.2f})")
    # Variational layer(s)
    idx = 0
    for l in range(n_layers):
        # Single‑qubit rotations
        for q in range(n_qubits):
            qml.RY(weights[idx], wires=q); idx += 1
            qml.RZ(weights[idx], wires=q); idx += 1
            qml.RX(weights[idx], wires=q); idx += 1

        # Entangling block (all‑to‑all with controlled rotations)
        for q1 in range(n_qubits):
            for q2 in range(n_qubits):
                if q1 != q2:
                    qml.CRY(weights[idx], wires=[q1, q2]); idx += 1
                    qml.CRZ(weights[idx], wires=[q1, q2]); idx += 1
                    qml.CRX(weights[idx], wires=[q1, q2]); idx += 1
                else:
                    qml.RY(weights[idx], wires=q1); idx += 1
                    qml.RZ(weights[idx], wires=q1); idx += 1
                    qml.RX(weights[idx], wires=q1); idx += 1

    # Measurements
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# ------------------------------------------------------------
# Generate dummy parameters to fill the circuit
# ------------------------------------------------------------
# One input value
x_example = np.array([0.3])

# Total number of trainable weights (matches the circuit's parameter count)
# For n_qubits=3, n_layers=1: 3 * 3 * (3+1) = 36
weights_example = np.random.randn(36)

# ------------------------------------------------------------
# Draw the circuit using matplotlib
# ------------------------------------------------------------
fig, ax = qml.draw_mpl(circuit)(x_example, weights_example)

# Add annotations to explain the main blocks
ax.text(0.15, 0.98, 'Data encoding', transform=fig.transFigure,
        ha='center', va='top', fontsize=14, weight='bold')
ax.text(0.55, 0.98, '.'*85 + 'Variational layer' + '.'*85, transform=fig.transFigure,
        ha='center', va='top', fontsize=14, weight='bold')
ax.text(0.95, 0.98, 'Measurement', transform=fig.transFigure,
        ha='center', va='top', fontsize=14, weight='bold')

# Adjust layout to make room for the annotations
plt.tight_layout()

# Save the figure
plt.savefig('quantum_circuit.pdf', dpi=300, bbox_inches='tight')
plt.savefig('quantum_circuit.png', dpi=300, bbox_inches='tight')
plt.show()