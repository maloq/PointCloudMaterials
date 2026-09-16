"""Scientific pairing and whole-source uncertainty for the causal-state study."""
import numpy as np
import pytest

from src.research.mace_causal_comparison import GROUPS, source_values, summarize, verify_pair


def test_predeclared_sufficiency_subset_keeps_common_readout_comparison():
    from src.research.mace_causal_comparison import readouts_for_variant
    config = dict(variants=['C', 'D', 'repeated_anchor'], readouts=['joint', 'linear', 'nonlinear'], diagnostic_variants=['D'])
    assert readouts_for_variant(config, 'C') == config['readouts']
    assert readouts_for_variant(config, 'D') == [*config['readouts'], 'state_constant', 'state_history']
    with pytest.raises(ValueError, match='declared'):
        readouts_for_variant(dict(config, diagnostic_variants=['A']), 'D')


def test_seed_means_precede_paired_source_bootstrap():
    records = [dict(variant=variant, readout='nonlinear', population='low_order', method='encoder',
                    metric='future/9ps/block_mean', source_id=source, seed=seed,
                    value=100*source+seed+(3 if variant == 'D' else 0))
               for variant in ('C', 'D') for source in (1, 2, 3) for seed in (10, 20)]
    summary, paired = summarize(records, 500, 11, [('D', 'C')])
    assert len(summary) == 2 and paired[0]['sources'] == 3 and paired[0]['seeds'] == 2
    assert paired[0]['value'] == paired[0]['ci95_low'] == paired[0]['ci95_high'] == 3
    single, _ = summarize([r for r in records if r['source_id'] == 1], 100, 11, [])
    assert all(row['ci95_low'] is None and row['ci95_high'] is None for row in single)
    with pytest.raises(ValueError, match='Missing seed/source'):
        summarize(records[:-1], 500, 11, [('D', 'C')])


def test_pairing_rejects_reordered_or_changed_physical_targets():
    reference = {k: np.arange(4) for k in ('source_id', 'center_atom_id', 'anchor_ps', 'present_target', 'future_target')}
    verify_pair(reference, reference)
    changed = dict(reference, future_target=np.arange(4)[::-1])
    with pytest.raises(ValueError, match='future_target'):
        verify_pair(reference, changed)


def test_physical_score_balances_actual_six_producer_blocks():
    rows = [dict(population='all', method='encoder', metric=f'present/{name}', source_id='1', value=str(i))
            for i, name in enumerate(GROUPS)]
    values = source_values(rows)
    assert values[('all', 'encoder', 'present/block_mean', 1)] == 2.5
    with pytest.raises(ValueError, match='Incomplete six-block'):
        source_values(rows[:-1])
