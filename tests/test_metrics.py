import numpy as np
import pytest
from sonar.metrics import evaluate
from sonar.statistics import holm


def test_multilabel_map_checks_secondary_labels():
    truth = [[0, 1], [1], [0]]
    good = np.array([[.9, .8], [.1, .9], [.8, .1]])
    bad = np.array([[.9, .0], [.1, .9], [.8, .1]])
    a, _, _ = evaluate(good, truth, 'multilabel')
    b, _, _ = evaluate(bad, truth, 'multilabel')
    assert a['Hit@1'] == b['Hit@1'] == 100
    assert a['mAP'] > b['mAP']


def test_macro_f1_weights_classes_equally():
    scores = np.array([[.9, .1], [.8, .2], [.7, .3], [.6, .4]])
    metrics, per_class, ranks = evaluate(scores, [[0], [0], [0], [1]], 'single-label')
    assert metrics['Hit@1'] == 75
    assert metrics['Macro-F1'] == pytest.approx(100 * (6 / 7) / 2)
    assert per_class['recall'] == [100, 0]


def test_historical_ranks_only_override_exact_ties():
    scores = np.array([[.5, .5, .1]])
    metrics, _, ranks = evaluate(scores, [[1]], 'multilabel', reference_ranks=[2])
    assert ranks.tolist() == [2]
    with pytest.raises(ValueError): evaluate(scores, [[1]], 'multilabel', reference_ranks=[3])


def test_holm_is_monotonic_and_capped():
    assert holm([.01, .04, .2]).tolist() == pytest.approx([.03, .08, .2])
