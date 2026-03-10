import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def generate_multifreq_data(
    n_samples=5000,
    x_range=(0, 2.0),
    noise=0.0,
    random_state=42
):
    """
    Generate a 1D regression dataset with spatially localized frequency components.

    Parameters:
        n_samples : int
            Total number of samples (including train/val/test splits).
        x_range : tuple (min, max)
            Domain of the input variable x.
        noise : float
            Standard deviation of Gaussian noise added to the target.
        random_state : int
            Seed for reproducibility.

    Returns:
        X : np.ndarray, shape (n_samples, 1)
            Input values.
        y : np.ndarray, shape (n_samples, 1)
            Target values.
        components : list of dict
            The list of component definitions used.
    """
    np.random.seed(random_state)

    # ---- Define the five frequency components ----
    components = [
        # 0.5 Hz – Gaussian envelope
        {'freq': 0.5, 'center': 0.3, 'width': 0.2, 'amp': 1.0},
        # 3.0 Hz – Lorentzian envelope
        {'freq': 3.0, 'center': 0.8, 'width': 0.15, 'amp': 0.7,
         'envelope': lambda x, c, w: 1 / (1 + ((x - c)/w)**2)},
        # 7.0 Hz – Triangular envelope
        {'freq': 7.0, 'center': 1.2, 'width': 0.25, 'amp': 0.5,
         'envelope': lambda x, c, w: np.maximum(0, 1 - np.abs(x - c)/w)},
        # 12.0 Hz – Gaussian envelope
        {'freq': 12.0, 'center': 1.7, 'width': 0.1, 'amp': 0.3},
        # 20.0 Hz – Gaussian envelope (very narrow)
        {'freq': 20.0, 'center': 1.9, 'width': 0.05, 'amp': 0.2},
    ]

    # Default Gaussian envelope if none provided
    def gaussian_envelope(x, center, width):
        return np.exp(-((x - center) / width) ** 2)

    # Generate input samples uniformly
    x = np.random.uniform(*x_range, n_samples).astype(np.float32)

    # Build target signal as sum of components
    y = np.zeros(n_samples, dtype=np.float32)
    for comp in components:
        freq = comp['freq']
        center = comp['center']
        width = comp['width']
        amp = comp.get('amp', 1.0)
        # Use custom envelope if provided, else Gaussian
        env_func = comp.get('envelope', gaussian_envelope)
        weights = env_func(x, center, width).astype(np.float32)
        y += amp * weights * np.sin(2 * np.pi * freq * x)

    # Add Gaussian noise
    y += noise * np.random.randn(n_samples)

    return x.reshape(-1, 1), y.reshape(-1, 1), components

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split

# ... (the generate_multifreq_data function remains unchanged) ...

# Generate the dataset
X, y, components = generate_multifreq_data(
    n_samples=5000,
    x_range=(0, 2.0),
    noise=0.0,
    random_state=42
)

# Helper to determine dominant frequency region for each x
def get_dominant_region(x, components):
    envelopes = np.zeros((len(x), len(components)))
    for i, comp in enumerate(components):
        center = comp['center']
        width = comp['width']
        env_func = comp.get('envelope', lambda x, c, w: np.exp(-((x - c)/w)**2))
        envelopes[:, i] = env_func(x, center, width)
    return np.argmax(envelopes, axis=1)

x_flat = X.flatten()
dominant = get_dominant_region(x_flat, components)
colors = ['blue', 'green', 'red', 'cyan', 'magenta']
color_map = [colors[d] for d in dominant]

# Sort for a smooth line
sort_idx = np.argsort(x_flat)
x_sorted = x_flat[sort_idx]
y_sorted = y.flatten()[sort_idx]

# Plot
plt.figure(figsize=(10, 6))
plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 8
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['font.family'] = 'serif'

# Plot the sorted data as a line
line = plt.plot(x_sorted, y_sorted, 'k-', linewidth=2, label='Generated data (sorted)')[0]

# Scatter plot colored by dominant region
scatter = plt.scatter(x_flat, y.flatten(), c=color_map, s=10, alpha=0.7)

# Create legend handles for the frequency regions
legend_handles = []
for i, comp in enumerate(components):
    patch = Patch(color=colors[i], label=f'{comp["freq"]} Hz')
    legend_handles.append(patch)
# Add the line handle
legend_handles.append(line)

plt.xlabel('x')
plt.ylabel('y')
plt.title("Generated data (colored by dominant region)")
plt.legend(handles=legend_handles, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
plt.savefig('generated_multifreq_data.png', dpi=300)
plt.savefig('generated_multifreq_data.pdf', dpi=300)
plt.show()