import torch
import numpy as np
import matplotlib.pyplot as plt
from core import (
    set_seed, generate_multifreq_data, load_data,
    ResidualQuantumLearner, generate_clean_grid
)

def residual_spectrum(learner, components, x_range, n_points=1000):
    """Compute FFT of residuals after each stage on a clean grid."""
    x_grid, y_clean = generate_clean_grid(components, x_range, n_points)
    dx = x_grid[1] - x_grid[0]
    freqs = np.fft.fftfreq(n_points, d=dx)[:n_points//2]

    # Predictions at each stage (cumulative)
    preds = []
    for stage in range(1, learner.n_stages + 1):
        pred = learner.predict(torch.tensor(x_grid.reshape(-1, 1), dtype=torch.float32), stage=stage)
        preds.append(pred.numpy().flatten())

    # Compute FFT of residuals
    fft_residuals = []
    for pred in preds:
        residual = y_clean - pred
        fft_vals = np.abs(np.fft.fft(residual)[:n_points//2])
        fft_residuals.append(fft_vals)

    # True signal FFT
    fft_true = np.abs(np.fft.fft(y_clean)[:n_points//2])
    return freqs, fft_true, fft_residuals


if __name__ == "__main__":
    # -------------------- Configuration --------------------
    config = {
        'seed': 42,
        'data': {
            'n_samples': 5000,
            'x_range': (0, 2.0),
            'noise': 0.0,
            'components': [
                {'freq': 0.5, 'center': 0.3, 'width': 0.2, 'amp': 1.0},
                {'freq': 3.0, 'center': 0.8, 'width': 0.15, 'amp': 0.7,
                 'envelope': lambda x, c, w: 1 / (1 + ((x - c)/w)**2)},
                {'freq': 7.0, 'center': 1.2, 'width': 0.25, 'amp': 0.5,
                 'envelope': lambda x, c, w: np.maximum(0, 1 - np.abs(x - c)/w)},
                {'freq': 12.0, 'center': 1.7, 'width': 0.1, 'amp': 0.3},
                {'freq': 20.0, 'center': 1.9, 'width': 0.05, 'amp': 0.2},
            ],
        },
        'training': {
            'batch_size': 64,
            'val_split': 0.15,
            'test_split': 0.15,
        },
        'model': {
            'input_dim': 1,
            'n_qubits': 6,           # change as needed
            'n_layers': 2,
        },
        'optim': {
            'lr': 0.005,
            'epochs': 25,
        },
        'residual': {
            'n_stages': 4,
        },
    }

    set_seed(config['seed'])

    # -------------------- Generate data --------------------
    X, y = generate_multifreq_data(
        n_samples=config['data']['n_samples'],
        x_range=config['data']['x_range'],
        noise=config['data']['noise'],
        components=config['data']['components'],
        random_state=config['seed']
    )

    train_loader, val_loader, test_loader, _, _ = load_data(
        X, y,
        batch_size=config['training']['batch_size'],
        val_split=config['training']['val_split'],
        test_split=config['training']['test_split']
    )

    # -------------------- Train residual learner --------------------
    learner = ResidualQuantumLearner(
        input_dim=config['model']['input_dim'],
        n_qubits=config['model']['n_qubits'],
        n_layers=config['model']['n_layers'],
        lr=config['optim']['lr'],
        epochs=config['optim']['epochs'],
        n_stages=config['residual']['n_stages']
    )
    learner.fit(train_loader, val_loader)

    # -------------------- Compute spectra --------------------
    freqs, fft_true, fft_residuals = residual_spectrum(
        learner,
        config['data']['components'],
        config['data']['x_range'],
        n_points=1000
    )

    # -------------------- Plot 1: Residual spectrum --------------------
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.size'] = 14
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 14
    plt.rcParams['font.family'] = 'serif'

    stage_colors = plt.cm.viridis(np.linspace(0.2, 0.9, learner.n_stages))
    for i, fft_res in enumerate(fft_residuals):
        plt.semilogy(freqs, fft_res, color=stage_colors[i], label=f'Stage {i+1}')

    # True signal (dashed black)
    plt.semilogy(freqs, fft_true, 'k--', linewidth=2, label='True signal', alpha=0.7)

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (log scale)')
    plt.title(f'Residual Spectrum After Each Stage (qubits={learner.n_qubits})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'experiment_residual_spectrum-qubits{learner.n_qubits}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'experiment_residual_spectrum-qubits{learner.n_qubits}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # -------------------- Plot 2: Cumulative residual energy (squared) --------------------
    # Use sum of squares (Parseval) – this is proportional to MSE on the grid.
    total_energy_true = np.sum(fft_true ** 2)
    residual_energies = [np.sum(fft_res ** 2) for fft_res in fft_residuals]
    relative_energies = [e / total_energy_true for e in residual_energies]

    plt.figure(figsize=(8, 5))
    stages = range(1, learner.n_stages + 1)
    plt.plot(stages, relative_energies, 'o-', linewidth=2, markersize=8, color='steelblue')
    plt.xlabel('Stage')
    plt.ylabel('Relative Residual Energy (squared)')
    plt.title(f'Residual Energy Reduction (qubits={learner.n_qubits})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'experiment_residual_energy-qubits{learner.n_qubits}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'experiment_residual_energy-qubits{learner.n_qubits}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # -------------------- Optional: print test MSE for comparison --------------------
    X_test_t = test_loader.dataset.tensors[0]
    y_test_t = test_loader.dataset.tensors[1]
    print("\nTest MSE per stage:")
    for stage in range(1, learner.n_stages + 1):
        pred = learner.predict(X_test_t, stage=stage)
        mse = torch.mean((pred - y_test_t) ** 2).item()
        print(f"Stage {stage}: {mse:.6f}")