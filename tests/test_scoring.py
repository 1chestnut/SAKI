import pytest
torch=pytest.importorskip('torch')
from saki.scoring import hier, relation_order, apply_selected
from saki.inference import score

def test_missing_evidence_keeps_entire_base_score():
    base=torch.tensor([.8,.7,.6])
    assert float(hier(base[0],torch.empty(0)))==float(base[0])
    final,_=score(base,{},['r1','r2'],top_k=1,number_of_relations=1)
    assert torch.equal(final,base)

def test_consensus_support_precedes_larger_other_gap():
    spaces={'r1':torch.tensor([.6,.5,.1]),'r2':torch.tensor([.7,.5,.1]),
            'r3':torch.tensor([.1,.9,.1])}
    order,consensus,votes=relation_order(spaces,list(spaces))
    assert consensus==0 and votes==2 and order==['r2','r1','r3']

def test_all_classes_can_supply_relation_prediction():
    base=torch.tensor([.9,.8,.7])
    per={0:{'r': [('tail',torch.tensor(0.),torch.tensor(-1.))]}}
    final,meta=score(base,per,['r'],top_k=1,number_of_relations=1)
    assert meta['consensus_class']==1
    assert final[1]==base[1] and final[2]==base[2]

def test_duplicate_tail_retains_first_relation_text():
    base=torch.tensor([.5,.4])
    per={0:{'r1':[('tail',torch.tensor(.1),torch.tensor(.8))],
            'r2':[('tail',torch.tensor(.1),torch.tensor(.2))]}}
    result=apply_selected(base,[0],per,['r1','r2'],'aakv','hier')
    assert torch.isclose(result[0],torch.tensor(.3*.5+.7*.8))
    assert result[1]==base[1]
