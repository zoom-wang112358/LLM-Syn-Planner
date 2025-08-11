import copy
import ast
from itertools import permutations
from rdkit import Chem
from syntheseus import Molecule
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions, DataStructs
import numpy as np
from multiprocessing import Pool, cpu_count
from rdchiral.main import rdchiralRun
from rdchiral.initialization import rdchiralReactants, rdchiralReaction

def change_to_forward_reaction(reaction:str):
    products, reactants = reaction.split(">>")
    return reactants + ">>" + products

def process_reaction_routes(route:list):
    json_list = []
    for idx,i in enumerate(route):
        products, reactants = i.split(">>")
        reaction = i
        reactants_list = reactants.split(".")
        #forward_reaction = change_to_forward_reaction(reaction)
        if idx == 0:
            step = {
                "Molecule set": str([products]),
                "Product": str([products]),
                "Reaction": str([reaction]),
                "Reactants": str(reactants_list),
                "Updated molecule set": str(reactants_list)
            }
            json_list.append(step)
        else:
            original_set = copy.deepcopy(ast.literal_eval(json_list[idx-1]["Updated molecule set"]))
            update_set = copy.deepcopy(ast.literal_eval(json_list[idx-1]["Updated molecule set"]))
            try:
                update_set.remove(products)
            except:
                print(update_set)
                print(products)
            update_set = update_set + reactants_list
            step = {
                "Molecule set": str(original_set),
                "Product": str([reaction]),
                "Reaction": str([reaction]),
                "Reactants": str(reactants_list),
                "Updated molecule set": str(update_set)
            }
            json_list.append(step)
    
    return json_list

def starting_invalid_feedback(evaluation):

    fb = '''\nIn the first step, the molecule in the molecule set should be the target molecule'''.format(step[2]['target_smi'])

    return fb

def molecule_invalid_feedback(evaluation):

    invalid_molecule_id = evaluation['invalid_updated_mol_id']
    updated_molecule_set = evaluation['updated_molecule_set']

    fb = '''\nIn the 'Updated molecule set','''

    for i in range(len(invalid_molecule_id)):
        fb = fb + '''
        the molecule {} is not a valid molecule SMILES. Please make sure all the molecules are in the SMILES format.
        '''.format(updated_molecule_set[invalid_molecule_id[i]])

    return fb

def molecule_unavailable_feedback(evaluation, inventory):
    unavailable_mol_id = evaluation['unavailable_mol_id']
    updated_molecule_set = evaluation['updated_molecule_set']
    unperchasable_molecule = check_availability(updated_molecule_set, inventory)

    fb = '''\nIn the 'Updated molecule set','''

    fb = fb + '''
    the molecule {} cannot be purchased from the market.\n
    '''.format(str(unperchasable_molecule))


    return fb

def reaction_unavailable_feedback(evaluation):
    reaction = evaluation['reaction']
    #forward_reaction = change_to_forward_reaction(reaction)
    fb = '''\nThe reaction {} does not exist in the USPTO dataset. Please make sure all the molecules in the reaction are in SMILES format.\n'''.format(reaction)

    return fb
def product_not_inside_feedback(evaluation):
    product = evaluation['product'][0]

    fb = '''\nThe product molecule {} is not in the molecule set. Please make sure the product molecule is in the molecule set.\n'''.format(product)

    return fb
def check_availability(smi_list, inventory):
    unavailable_list = []
    for smi in smi_list:
        signal = inventory.is_purchasable(Molecule(smi))
        if not signal:
            unavailable_list.append(smi)
    
    return unavailable_list
    
def retrieve_routes(target_smi, all_fps, route_list, number):
    getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=False)
    similarity_metric = DataStructs.BulkTanimotoSimilarity # BulkDiceSimilarity or BulkTanimotoSimilarity
    fp = getfp(target_smi)
    sim_score = similarity_metric(fp, [fp_ for fp_ in all_fps])

    rag_tuples = list(zip(sim_score, route_list))
    rag_tuples = sorted(rag_tuples, key=lambda x: x[0], reverse=True)[:50]

    route_list = [t[1] for t in rag_tuples]
    sims_list = [t[0] for t in rag_tuples]


    sum_scores = sum(sims_list)
    population_probs = [p / sum_scores for p in sims_list]
    sampled_index = np.random.choice(len(route_list), p=population_probs, size=3, replace=False)
    sampled_routes = [route_list[i] for i in sampled_index]

    return sampled_routes
def updated_set_mismatch_feedback(evaluation):
    reaction = evaluation['reaction']
    product = evaluation['product'][0]

    fb = '''\nThe molecule set and the updated molecule set are not aligned. In each step, you need to keep a molecule set in which are the molecules we need. After taking the backward reaction in this step, you need to remove the products from the molecule set and add the reactants to the molecule set and then store
this set as 'Updated molecule set' in this step. In the last step, all the molecules in the 'Updated molecule set' should be purchasable. Please also check whether the product of this reaction is in the molecule set.'''

    return fb

def reaction_can_not_happen(evaluation):
    reaction = evaluation['reaction']
    #forward_reaction = change_to_forward_reaction(reaction)    
    fb = '''\nThe reaction {} cannot happen with the product molecule. \n'''.format(reaction)

    return fb

def verify_reaction_step(molecule_set, updated_molecule_set, reaction, product, reactants, inventory, oracle):

    results = {
        'reaction_valid': False,
        'updated_set_valid': False
    }

    # Parse the reaction, reactants, and products
    try:
        reaction_smiles = reaction
        #reaction = AllChem.ReactionFromSmarts(reaction_smiles)

        reactants = [Chem.MolFromSmiles(smi) for smi in reactants]
        products_expected = [Chem.MolFromSmiles(smi) for smi in product]
        updated_molecule_set = [Chem.MolFromSmiles(smi) for smi in updated_molecule_set]
        original_molecule_set = [Chem.MolFromSmiles(smi) for smi in molecule_set]
    except Exception as e:
        print(f"Error parsing molecules: {e}")
    # Check if the products are generated by the reaction
    try:
        #print(products_expected)
        #print(product)
        #print(reaction_smiles)
        target_rd = rdchiralReactants(product[0])
        reaction_outputs = run_retro(target_rd, reaction_smiles)
        #rank_reactants
        if len(reaction_outputs) > 1:
            reaction_outputs = rank_reactants(reaction_outputs, inventory, oracle)
        reactants_generated = [reactant for reactant in reaction_outputs[0]]
        reactants_generated = [sanitize_smiles(smi) for smi in reactants_generated]


        if None in reactants or None in reactants_generated:
            results['reaction_valid'] = False
        elif reactants_generated == []:
            results['reaction_valid'] = False
        else:
            results['reaction_valid'] = True
    except Exception as e:
        print(f"Error running/ranking reaction: {e}")

    # Verify if the updated molecule set includes reactants and products
    # In check_route function, the reactants are proposed by LLM.
    # In check_route_extra function, the reactants are generated by the reaction.
    try:
        #print(updated_molecule_set)
        updated_smiles = {Chem.MolToSmiles(mol) for mol in updated_molecule_set if mol is not None}
        reactants_smiles = {Chem.MolToSmiles(mol) for mol in reactants if mol is not None}
        products_smiles = {Chem.MolToSmiles(mol) for mol in products_expected}
        original_smiles = {Chem.MolToSmiles(mol) for mol in original_molecule_set if mol is not None}

        expected_updated_sets = (original_smiles | reactants_smiles) - products_smiles

        if expected_updated_sets == updated_smiles and products_smiles.issubset(original_smiles):
            results['updated_set_valid'] = True
        if None in updated_molecule_set:
            print('None in updated molecule set')
            results['updated_set_valid'] = False
        elif None in original_molecule_set:
            print('None in original molecule set')
            results['updated_set_valid'] = False
        elif None in reactants:
            print('None in reactants')
            results['updated_set_valid'] = False
        elif None in products_expected:
            print('None in products')
            results['updated_set_valid'] = False

        common_elements = products_smiles & reactants_smiles
        if common_elements:
            print('product equals to reactants')
            results['updated_set_valid'] = False

    except Exception as e:
        #raise ValueError(f"Error validating molecule set: {e}")
        print(f"Error validating molecule set: {e}")

    return results['reaction_valid'], results['updated_set_valid']


def is_reaction_in_dict(reaction_smiles, preprocessed_dict):
    reaction_key = None
    try:
        products, reactants = reaction_smiles.split(">>")
        reactant_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
        product_mols = [Chem.MolFromSmiles(p) for p in products.split(".")]
    except Exception as e:
        print(f"Error parsing input reaction: {e}")
        return False, reaction_key


    if None in reactant_mols or None in product_mols:
        return False, reaction_key
    
    for key, (smarts_reactant_mols, smarts_product_mols) in preprocessed_dict.items():
        try:
            # Check if all reactants and products match
            if len(smarts_reactant_mols) != len(reactant_mols):
                continue
            if len(smarts_product_mols) != len(product_mols):
                continue
        
        #print(reactant_mols)
        #print(product_mols)
            reactant_match = is_one_to_one_match(smarts_reactant_mols, reactant_mols)
            product_match = is_one_to_one_match(smarts_product_mols, product_mols)
        #print(reactant_mols)
        #print(product_mols)
            if reactant_match and product_match:
                #print(f"Reaction found in dictionary with key: {key}")
                reaction_key = key
                return True, reaction_key
        except Exception as e:
            print(f"Error processing SMARTS {smarts_reactant_mols}: {e}")
        continue
    return False, reaction_key

def is_reaction_match(args):
    reactant_mols, product_mols, smarts_reactant_mols, smarts_product_mols = args
    if len(smarts_reactant_mols) != len(reactant_mols):
        return False
    if not is_one_to_one_match(smarts_reactant_mols, reactant_mols):
        return False
    if len(smarts_product_mols) != len(product_mols):
        return False
    if not is_one_to_one_match(smarts_product_mols, product_mols):
        return False
    return True

def is_one_to_one_match(smarts_mols, target_mols):
    for perm in permutations(target_mols, len(smarts_mols)):
        if all(target.HasSubstructMatch(smarts) for smarts, target in zip(smarts_mols, perm)):
            return True
    return False

def check_validity(mol_list: list):
    validity_signal = [False] * len(mol_list)
    for idx, smi in enumerate(mol_list):
        signal = sanitize_smiles(smi)
        if signal != None:
            validity_signal[idx] = True
    
    return validity_signal

def check_purchasable(mol_list: list, validity_signals, inventory):
    availability_signals = [False] * len(mol_list)
    for idx, smi in enumerate(mol_list):
        if validity_signals[idx] == True: 
            signal = inventory.is_purchasable(Molecule(smi))
            availability_signals[idx] = signal
    
    return availability_signals


def sanitize_smiles(smi):
    """
    Return a canonical smile representation of smi 

    Parameters
    ----------
    smi : str
        smile string to be canonicalized 

    Returns
    -------
    mol (rdkit.Chem.rdchem.Mol) : 
        RdKit mol object (None if invalid smile string smi)
    smi_canon (string)          : 
        Canonicalized smile representation of smi (None if invalid smile string smi)
    conversion_successful (bool): 
        True/False to indicate if conversion was  successful 
    """
    if smi == '':
        return None
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        smi_canon = Chem.MolToSmiles(mol, canonical=True)
        return smi_canon
    except:
        return None

def run_retro(product, template):
    """
    Run a reaction given the product and the template.
    Args:
        product (str): product
        template (str): template
    Returns:
        str: reactant SMILES string
    """
    reactants = template.split(">>")[0].split(".")
    if len(reactants) > 1:
        template = "(" + template.replace(">>", ")>>")
    template = rdchiralReaction(template)
    try:
        outputs = rdchiralRun(template, product)
    except Exception as e:
        print(f"Error {e} running retro reaction {template} on product {product}")
        return []
    result = []
    for output in outputs:
        result.append(output.split("."))
    return result

def smiles_to_reaction(smiles):
    try:
        reactants, products = smiles.split(">>")
        reactant_list = reactants.split(".")
        product_list = products.split(".")
        reactant_mols = [Chem.MolFromSmiles(r) for r in reactant_list]
        product_mols = [Chem.MolFromSmiles(p) for p in product_list]
        reaction_smarts = f"{'.'.join([Chem.MolToSmarts(mol) for mol in reactant_mols])}>>{'.'.join([Chem.MolToSmarts(mol) for mol in product_mols])}"
        return rdChemReactions.ReactionFromSmarts(reaction_smarts)
    except Exception as e:
        print(f"Failed to convert SMILES to reaction: {e}")
        return None

def rank_reactants(reactants_list, inventory, oracle):
    """
    Rank reactants based on the number of products generated
    """
    dead_molecules = dict()
    visited_molecules = dict()
    non_empty_reactant_list = [item for item in reactants_list if item != []]
    scores = [oracle.reward(inventory, reactant, visited_molecules, dead_molecules) for reactant in non_empty_reactant_list]
    sorted_list = [x for _, x in sorted(zip(scores, non_empty_reactant_list), key=lambda pair: pair[0], reverse=True)]
    return sorted_list
