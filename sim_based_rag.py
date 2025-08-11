import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
from rdkit import DataStructs
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import sys
from rdchiral.main import rdchiralRun
from rdchiral.initialization import rdchiralReactants, rdchiralReaction
from generate_retro_templates import process_an_example

'''
This script is implemented by Retrosim https://github.com/connorcoley/retrosim
We use it for similarity search.
'''

def get_data_df(fpath='data_processed.csv'):
    return pd.read_csv(fpath)

def split_data_df(data, val_frac=0.0, test_frac=0.0, shuffle=False, seed=None):
    # Define shuffling
    if shuffle:
        if seed is None:
            np.random.seed(int(time.time()))
        else:
            np.random.seed(seed)
        def shuffle_func(x):
            np.random.shuffle(x)
    else:
        def shuffle_func(x):
            pass

    # Go through each class
    classes = sorted(np.unique(data['class']))
    for class_ in classes:
        indeces = data.loc[data['class'] == class_].index
        N = len(indeces)
        print('{} rows with class value {}'.format(N, class_))

        shuffle_func(indeces)
        train_end = int((1.0 - val_frac - test_frac) * N)
        val_end = int((1.0 - test_frac) * N)

        for i in indeces[:train_end]:
            data.at[i, 'dataset'] = 'train'
        for i in indeces[train_end:val_end]:
            data.at[i, 'dataset'] = 'val'
        for i in indeces[val_end:]:
            data.at[i, 'dataset'] = 'test'
    print(data['dataset'].value_counts())

def do_one(product_smiles, datasub, jx_cache, max_prec=100):
    similarity_metric = DataStructs.BulkTanimotoSimilarity
    getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=False)
    ex = Chem.MolFromSmiles(product_smiles)
    rct = rdchiralReactants(product_smiles)

    fp = getfp(product_smiles)
    
    sims = similarity_metric(fp, [fp_ for fp_ in datasub['prod_fp']])
    js = np.argsort(sims)[::-1]


    
    #prec_goal = Chem.MolFromSmiles(datasub_val['rxn_smiles'][ix].split('>')[0])
    #[a.ClearProp('molAtomMapNumber') for a in prec_goal.GetAtoms()]
    #prec_goal = Chem.MolToSmiles(prec_goal, True)
    
    # Sometimes stereochem takes another canonicalization...
    #prec_goal = Chem.MolToSmiles(Chem.MolFromSmiles(prec_goal), True)

    # Get probability of precursors
    probs = {}
    
    for ji, j in enumerate(js[:max_prec]):
        jx = datasub.index[j]
        

        if jx in jx_cache:
            (rxn, template, rcts_ref_fp) = jx_cache[jx]
        else:
            template = '(' + process_an_example(datasub['rxn_smiles'][jx], super_general=True).replace('>>', ')>>')
            rcts_ref_fp = getfp(datasub['rxn_smiles'][jx].split('>')[0])
            rxn = rdchiralReaction(template)
            jx_cache[jx] = (rxn, template, rcts_ref_fp)
            
        try:
            outcomes = rdchiralRun(rxn, rct, combine_enantiomers=False)
        except Exception as e:
            print(e)
            outcomes = []
            
        for precursors in outcomes:
            precursors_fp = getfp(precursors)
            precursors_sim = similarity_metric(precursors_fp, [rcts_ref_fp])[0]

            if template in probs:
                probs[template] = max(probs[template], precursors_sim * sims[j])
            else:
                probs[template] = precursors_sim * sims[j]
    
    testlimit = 100
    template_list = []

    for r, (template, prob) in enumerate(sorted(probs.items(), key=lambda x:x[1], reverse=True)[:testlimit]):
        template_list.append((template, prob))
    print('sim search!') 
    return template_list, jx_cache