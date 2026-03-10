import numpy as np
import matplotlib.pyplot as plt
from core import set_seed, generate_multifreq_data, load_data, ResidualQuantumLearner, compute_frequency_amplitudes

# Use same config as before (or adjust)
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
        'n_qubits': 6,
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

# Generate data and train
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

learner = ResidualQuantumLearner(
    input_dim=config['model']['input_dim'],
    n_qubits=config['model']['n_qubits'],
    n_layers=config['model']['n_layers'],
    lr=config['optim']['lr'],
    epochs=config['optim']['epochs'],
    n_stages=config['residual']['n_stages']
)
learner.fit(train_loader, val_loader)

# Compute frequency amplitudes
target_freqs, amps_true, amps_stages = compute_frequency_amplitudes(
    components=config['data']['components'],
    x_range=config['data']['x_range'],
    n_stages=learner.n_stages,
    learner=learner,
    n_points=1000
)

# Plot: for each frequency, show amplitude across stages
n_freqs = len(target_freqs)
fig, axes = plt.subplots(1, n_freqs, figsize=(4*n_freqs, 4))
plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 8
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['font.family'] = 'serif'

if n_freqs == 1:
    axes = [axes]

for idx, freq in enumerate(target_freqs):
    ax = axes[idx]
    true_amp = amps_true[idx]
    stage_amps = [amps[idx] for amps in amps_stages]
    stages = list(range(1, learner.n_stages+1))
    ax.bar(stages, stage_amps, color='steelblue', label='Learned')
    ax.axhline(true_amp, color='red', linestyle='--', label='True')
    ax.set_xlabel('Stage')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Frequency {freq:.1f} Hz')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig('experiment_frequency_resolved.png', dpi=300)
plt.savefig('experiment_frequency_resolved.pdf', dpi=300)
plt.show()