"""Combine completed live audits with the rechecked seed44 archive."""
import json

from analyze_h8_cross_model_comparison import audit
from run_h8_cross_model_comparison import EPISODES, METHODS, O, PHYSICS


def main():
    online = O / 'audited_seed45_46.jsonl'
    later = [json.loads(line) for line in online.read_text().splitlines()]
    assert len(later) == 144
    first = [audit(44, model, physics, episode, method)
             for model in ('1b', '10b', 'h8') for physics in PHYSICS
             for episode in EPISODES for method in METHODS]
    assert len(first) == 72
    results = first + later
    identities = {(r['seed'], r['model'], r['physics'], r['episode'], r['method']) for r in results}
    assert len(results) == len(identities) == 216
    assert all(r['validation']['initial_all_fields_exact'] and r['validation']['settle_motion_exact'] and
               r['validation']['settle_contact_force_max_delta_N'] < 1e-5 for r in results)
    (O / 'audited_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    print('ASSEMBLED 216 AUDITED ROLLOUTS', flush=True)


if __name__ == '__main__':
    main()
