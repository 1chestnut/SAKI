import random
import numpy as np
import pytest
torch = pytest.importorskip('torch')
from saki.runtime import Branch, PreparedInputs
from saki.inference import score
from experiments import _timing_core


def prepared(tmp_path):
    # Distinct evidence with one cross-relation tail duplicate and an empty class.
    data = dict(schema_version=1, classes=['a', 'b', 'c'], relations=['r1', 'r2', 'r3'],
        label_embeddings=torch.tensor([[1., 0.], [.8, .6], [0., 1.]]),
        embeddings={kind: torch.tensor([[.4, .9], [.7, .2], [.5, .8]]) for kind in ('aakv', 'direct', 'raw')},
        records={0:{'r1':[('tail',0)],'r2':[('tail',1)],'r3':[]},
                 1:{'r1':[('other',2)],'r2':[],'r3':[]},2:{'r1':[],'r2':[],'r3':[]}},
        fixed_embeddings=torch.tensor([[.3, .2]]),fixed_indices=[{'direct':[0],'aakv':[0]} for _ in range(3)],
        frozen_relations=['r1','r2'])
    path = tmp_path / 'inputs.pt'; torch.save(data,path)
    return PreparedInputs(path)


def test_portable_scorer_matches_accepted_timing_math(tmp_path, monkeypatch):
    p = prepared(tmp_path)
    active = p.activate(Branch(pool_size=3, nr=2, top_k=2),'cpu')
    audio = torch.tensor([[.8,.2]])
    actual, meta = active.score(audio)
    monkeypatch.setattr(_timing_core,'K',2); monkeypatch.setattr(_timing_core,'NR',2)
    expected, order = _timing_core.score_and_rank('SoNAR',audio,active.labels,None,None,
                                                  active.relations,active.records,active.embeddings)
    assert torch.equal(actual,expected[0])
    assert len(meta['selected_relations']) == 2


def test_reduced_pool_recomputes_selection_and_keeps_five_independent_slots(tmp_path):
    p = prepared(tmp_path)
    active = p.activate(Branch(pool_size=2,nr=1,top_k=2),'cpu')
    _, meta = active.score(torch.tensor([[.8,.2]]))
    assert set(active.relations) == set(p.frequency_order[:2])
    assert len(meta['selected_relations']) == 1 and set(meta['selected_relations']) <= set(active.relations)


def test_random_selection_uses_stored_identity_not_current_audio_path(tmp_path):
    p = prepared(tmp_path)
    active = p.activate(Branch(selection='random',pool_size=3,nr=2,top_k=2),'cpu')
    _, meta = active.score(torch.tensor([[.8,.2]]),{'42':123})
    assert meta['selected_relations'] == sorted(random.Random(123).sample(p.relations,2))
    with pytest.raises(ValueError): active.score(torch.tensor([[.8,.2]]))
