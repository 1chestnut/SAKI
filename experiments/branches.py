"""Paper comparisons; all branches start from an unchanged CLAP score vector."""
from saki.runtime import Branch


def branches(suite):
    if suite == 'main':
        return [Branch('CLAP', selection='base'), Branch('iKnow_dagger', text='direct', fusion='joint', selection='fixed_texts'), Branch()]
    if suite == 'components':
        return [Branch('CLAP', selection='base'),
                Branch('iKnow_dagger', text='direct', fusion='joint', selection='fixed_texts'),
                Branch('AAKV', fusion='joint', selection='fixed_texts'),
                Branch('Hierarchical_Fusion', text='direct', selection='fixed_texts'),
                Branch('Sample_level_selection', text='direct', fusion='joint'),
                Branch('AAKV_Fusion', selection='fixed_texts'),
                Branch('AAKV_Selection', fusion='joint'), Branch('Selection_Fusion', text='direct'), Branch()]
    if suite == 'relations':
        return [Branch('FrozenRq', selection='frozen'), Branch('Frequency_Top5', selection='frequency'),
                *[Branch(f'Random_Top5_seed{s}', selection='random', seed=s) for s in (42, 43, 44)], Branch()]
    if suite == 'text':
        return [Branch('Direct_concatenation', text='direct'), Branch('Raw_triple', text='raw'), Branch()]
    if suite == 'pools':
        return [Branch(f'SoNAR_R{n}', pool_size=n) for n in (47, 30, 20)]
    if suite == 'grid':
        return [Branch(f'alpha{alpha}_Nr{nr}', alpha=alpha, nr=nr) for alpha in (.3, .5, .7) for nr in (1, 3, 5, 10)]
    raise ValueError(f'Unknown suite: {suite}')


def select_development_configuration(summaries):
    # Highest Hit@1; exact ties use MRR, then the smaller relation count.
    return max(summaries, key=lambda x: (x['metrics']['Hit@1'], x['metrics']['MRR'], -x['configuration']['nr']))
