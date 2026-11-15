import pytest
pytest.importorskip('torch')
from sonar.retrieval import retained_tails


def test_filtering_normalizes_and_does_not_refill_top_m():
    retrieved = [' TRAIN ', ' bell ', 'BELL']
    assert retained_tails(retrieved, ['train', 'tram']) == ['bell']
