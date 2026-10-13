"""Plot completed comparisons, preserving right-censored trials."""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(sys.argv[1]).resolve()
result = json.loads((out/'results.json').read_text())
assert not result['partial'] and all(g['complete'] for g in result['groups'])
colors = ['#52667a', '#dc993b', '#167d93']
labels = ['Ordinary 1B', 'Guided sampler, scale 0', '1B + 10k guide, scale 25']
fig, axes = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={'height_ratios':[1, 1.15]}, constrained_layout=True)
for j, (group, color, label) in enumerate(zip(result['groups'], colors, labels)):
    times = np.array(group['hold_seconds'])
    censored = np.array(group['censored'])
    events = sorted(set(times[~censored]))
    xx = [0] + events + [group['observation_cap_s']]
    yy = [16] + [int(np.sum((times > t) | ((times == t) & censored))) for t in events]
    yy += [yy[-1]]
    axes[0].step(xx, yy, where='post', color=color, label=label, linewidth=2)
    if censored.any():
        axes[0].scatter(times[censored], [int(censored.sum())]*int(censored.sum()), color=color, marker='+', s=65)
    x = np.arange(16) + (j-1)*.25
    axes[1].bar(x, times, width=.23, color=color, label=label)
    axes[1].scatter(x[censored], times[censored], color='black', marker='+', s=45, zorder=4)
axes[0].set(title='Remaining within 5 cm of the initial bulb position', xlabel='Simulated time (s)', ylabel='Trials remaining (of 16)', ylim=(0,16.5))
axes[0].set_yticks([0,4,8,12,16])
axes[0].legend(loc='upper right', frameon=False)
axes[1].set(title='Matched seeds and initial states', xlabel='Diffusion seed (initial pose/placement group)', ylabel='Time to first >5 cm displacement (s)')
axes[1].set_xticks(range(16), [f'{50+k}\n({k%4+1})' for k in range(16)])
for ax in axes:
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Real hand initialization in simulation: 10k guide comparison', fontsize=15)
fig.savefig(out/'comparison.png', dpi=180)
fig.savefig(out/'comparison.pdf')
plt.close(fig)
print(out/'comparison.png')
