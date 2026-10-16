import json, os
from pathlib import Path

h = json.load(open('checkpoints/training_history.json'))
epochs = sorted(set(e["epoch"] for e in h))
train_losses = [e["train_loss"] for e in h]
print(f"History entries: {len(h)}")
print(f"Epochs recorded: {epochs}")
print(f"Train loss range: {min(train_losses):.6f} → {max(train_losses):.6f}")
print()

# Find all .npy and .csv files, skip heavy dirs
for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in ['.venv', '.git', '__pycache__', 'runs']]
    for f in files:
        if f.endswith('.npy') or f.endswith('.csv'):
            p = Path(root) / f
            print(f"  {p}  ({p.stat().st_size:,} bytes)")

print()
# Check all checkpoint files
for p in sorted(Path('checkpoints').glob('*.pt')):
    print(f"  checkpoint: {p.name}  ({p.stat().st_size:,} bytes)")
