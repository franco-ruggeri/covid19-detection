import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

root_path = Path('/') / 'home' / 'fruggeri' / 'Downloads'

LINES = [
    (root_path / 'run-train-tag-epoch_accuracy.csv', 'training'),
    (root_path / 'run-validation-tag-epoch_accuracy.csv', 'validation'),
]
METRIC = 'accuracy'

plt.figure()
for line in LINES:
    csv_content = pd.read_csv(line[0], engine='python', sep=',')
    x = csv_content['Step']
    y = csv_content['Value']
    plt.plot(x, y, label=line[1])
plt.legend()
plt.xlabel('epoch')
plt.ylabel(METRIC)
plt.show()
