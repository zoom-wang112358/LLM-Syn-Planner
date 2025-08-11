import os
import yaml
import random
import torch
import numpy as np
from rdkit import Chem
import tdc
import copy
import json
import heapq
from rdkit import Chem
from syntheseus import Molecule
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions, DataStructs
from itertools import permutations
from utils import *
from scscore.scscore.standalone_model_numpy import *
import pickle
import openai
import json
from concurrent.futures import ThreadPoolExecutor
from rdchiral.main import rdchiralRun
from rdchiral.initialization import rdchiralReactants, rdchiralReaction
from sim_based_rag import get_data_df, split_data_df, do_one
from tqdm import tqdm
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

class Route:
    def __init__(self, target):
        self.target = target
        self.raw_output = raw_output

    def get_score():
        sc_score = []




def preprocess_reaction_dict(reaction_dict):
    """Preprocess reaction_dict to compile SMARTS into RDKit Mol objects."""
    preprocessed_dict = {}
    for key, smarts in reaction_dict.items():
        try:
            products, reactants = smarts.split(">>")
            reactant_mols = [Chem.MolFromSmarts(r) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmarts(p) for p in products.split(".")]
            preprocessed_dict[key] = (reactant_mols, product_mols)
        except Exception as e:
            print(f"Error preprocessing SMARTS {smarts}: {e}")
    return preprocessed_dict

#Return SC score and store routes
class Oracle:
    def __init__(self, args=None, route_buffer={}):
        self.name = None
        self.evaluator = None
        self.task_label = None
        if args is None:
            self.max_oracle_calls = 10000
            self.freq_log = 100
        else:
            self.args = args
            self.max_oracle_calls = args.max_oracle_calls
            self.freq_log = args.freq_log
        self.route_buffer = {} if route_buffer is None else route_buffer
        self.reaction_cache = dict() #mol_smiles: [reaction]

        self.last_log = 0
        self.sa_scorer = tdc.Oracle(name = 'SA')
        self.sc_Oracle = sc_oracle()

    def store_cache(self, mol_smiles, reaction):
        if mol_smiles in self.reaction_cache:
            if reaction not in self.reaction_cache[mol_smiles]:
                self.reaction_cache[mol_smiles].append(reaction)
        else:

            self.reaction_cache[mol_smiles] = [reaction]

    def get_oracle_score(self, mol_smiles):
        smi, SC_score = self.sc_Oracle.get_score_from_smi(mol_smiles)
        sa_score = self.sa_scorer(mol_smiles)
        length = len(self.route_buffer)
        if length <= 150:
            overall_score = SC_score
        elif length > 150 and length < 220:
            alpha = (length/self.max_oracle_calls)
            overall_score = (1 - alpha) * SC_score + 0.5 * alpha * sa_score            
        else:
            overall_score = 0.5 * sa_score 
        #overall_score = SC_score + 0.5 * sa_score
        return overall_score
    @property
    def budget(self):
        return self.max_oracle_calls
    
    def reward(self, inventory, updated_molecule_set:list, visited_molecules, dead_molecules):
        score_list = []
        for smi in updated_molecule_set:
            if smi in dead_molecules:
                if dead_molecules[smi] >= 1:
                    print('dead molecules!')
                    score_list.append(100)
                    continue
            try:
                signal = inventory.is_purchasable(Molecule(smi))
                if not signal:
                    score = self.get_oracle_score(smi)
                    if smi in visited_molecules:
                        print(f"Visted times: {visited_molecules[smi]}")
                        if visited_molecules[smi] > 15:
                            score = (visited_molecules[smi]/15) * score
                            print(f"Visted times adjust score")
                    score_list.append(score)
            except Exception as e:
                print(f"Error: {e}")
                score_list.append(5)
        if len(score_list) != 0:
            score_max = np.max(score_list)
        else:
            score_max = 0
        if len(score_list) != 0:
            score_mean = sum(score_list) / len(score_list) 
        else:
            score_mean = 0
        combined_score = score_mean + sum(score_list)
        final_score = - combined_score
        return final_score
    
    def evaluate(self, inventory, route_evaluation, visited_molecules, dead_molecules):
        #print(route_evaluation)
        for idx, step in enumerate(route_evaluation):
            #print(step)
            if step[1] == False:
                score = self.reward(inventory, step[2]['molecule_set'], visited_molecules, dead_molecules)
                return score
            elif step[1] == True:
                continue
        
        #last step
        print(route_evaluation[-1])
        if route_evaluation[-1][2]['check_availability'] == True and len(route_evaluation[-1][2]['unavailable_mol_id']) == 0:
            score = 0
            return score
        else:
            score = self.reward(inventory, route_evaluation[-1][2]['updated_molecule_set'], visited_molecules, dead_molecules)
            return score

    def sort_buffer(self):
        self.route_buffer = dict(sorted(self.route_buffer.items(), key=lambda kv: kv[1][0], reverse=True))

    def save_result(self, suffix=None):
        
        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            suffix = suffix.replace("/", "")
            output_file_path = os.path.join(self.args.output_dir, 'results_' + suffix + '.yaml')

        self.sort_buffer()
        with open(output_file_path, 'w') as f:
            yaml.dump(self.route_buffer, f, sort_keys=False)

    def log_intermediate(self, finish=False):
        if finish:
            n_calls = self.max_oracle_calls
            self.save_result(self.task_label)

    def __len__(self):
        return len(self.route_buffer) 

    def score_route(self, inventory, route_evaluation, visited_molecules, dead_molecules):
        """
        Function to score one molecule

        Argguments:
            smi: One SMILES string represnets a moelcule.

        Return:
            score: a float represents the property of the molecule.
        """
        if len(self.route_buffer) > self.max_oracle_calls:
            return -15
        if route_evaluation is None:
            return -15
        dict_key = json.dumps(route_evaluation)
        if dict_key in self.route_buffer:
            pass
        else:
            self.route_buffer[dict_key] = [float(self.evaluate(inventory, route_evaluation, visited_molecules, dead_molecules)), len(self.route_buffer)+1]
        return self.route_buffer[dict_key][0]
    
    def __call__(self, inventory, route_evaluation, visited_molecules, dead_molecules):
        """
        Score
        """
        score_list = self.score_route(inventory, route_evaluation, visited_molecules, dead_molecules)
        if len(self.route_buffer) % self.freq_log == 0 and len(self.route_buffer) > self.last_log:
            self.sort_buffer()
            self.last_log = len(self.route_buffer)
            self.save_result(self.task_label)
        return score_list

    @property
    def finish(self):
        return len(self.route_buffer) >= self.max_oracle_calls


class BaseOptimizer:

    def __init__(self, args=None):
        self.model_name = "Default"
        self.args = args

        self.oracle = Oracle(args=self.args)

        args.template_path = 'dataset/idx2template_retro.json'
        args.inventory_path = 'dataset/inventory.pkl'
        args.rule_based_set_path = 'dataset/data_processed.csv'
        self.original_template_dict = self.load_template(args.template_path)
        self.template_dict =  preprocess_reaction_dict(self.original_template_dict)
        self.inventory = self.load_inventory(args.inventory_path)
        self.reaction_list, self.all_reaction_fps = self.get_reaction_fps(self.original_template_dict)
        self.datasub = self.load_rule_based_set(args.rule_based_set_path)
        #self.reaction_product_mols = [value[1][0] for key, value in self.template_dict.items()]
        #self.precomputed_db = precompute_query_fps(self.template_dict, fingerprint_func=Chem.RDKFingerprint)
        self.explored_reaction = set()
        self.visited_molecules = dict() #smiles: visit number
        self.dead_molecules = dict()
        self.jx_cache = {}
        self.template_to_key = {v: k for k, v in self.original_template_dict.items()}


    def load_template(self, template_path):
        with open(template_path, "r") as f:
            template_dict = json.load(f)
        #preprocessed_dict = preprocess_reaction_dict(template_dict)
        return template_dict

    
    def load_inventory(self, inventory_path):
        with open(inventory_path, 'rb') as file:
            inventory = pickle.load(file)
        
        return inventory
    
    def load_rule_based_set(self, rule_based_set_path):
        data = get_data_df(rule_based_set_path)
        split_data_df(data)
        similarity_metric = DataStructs.BulkTanimotoSimilarity # BulkDiceSimilarity or BulkTanimotoSimilarity

        getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=False)

        dataset = 'val'

        all_fps = []
        for smi in tqdm(data['prod_smiles']):
            all_fps.append(getfp(smi))
        data['prod_fp'] = all_fps

        datasub = data.loc[data['dataset'] == 'train']
        fps = list(datasub['prod_fp'])
        print('Size of knowledge base: {}'.format(len(fps)))
        
        return datasub

    def update_visited_molecules(self, updated_molecule_set):
        for smi in updated_molecule_set:
            if smi in self.visited_molecules:
                self.visited_molecules[smi] += 1
            else:
                self.visited_molecules[smi] = 1

    def update_dead_molecules(self, dead_molecule):
        smi = dead_molecule
        assert type(smi) == str
        print('update dead molecules!')
        print(smi)
        if smi in self.dead_molecules:
            self.dead_molecules[smi] += 1
        else:
            self.dead_molecules[smi] = 1
                
    def get_reaction_fps(self, template_dict):
        reaction_list = list(template_dict.values())
        getreactionfp = lambda smart_reaction: rdChemReactions.CreateDifferenceFingerprintForReaction(rdChemReactions.ReactionFromSmarts(smart_reaction))
        all_reaction_fps = []
        for reaction in reaction_list:
            all_reaction_fps.append(getreactionfp(reaction))
        
        return reaction_list, all_reaction_fps
    
    def convert_to_fingerprint(self, mols_list):
        fgp_list = []
        getfp = lambda mol: AllChem.GetMorganFingerprint(mol, 2, useFeatures=False)
        for mol in mols_list:
            fgp_list.append(getfp(mol))
        
        return fgp_list

    def rule_based_search(self, product_smiles, reaction_smiles):
        template_list, self.jx_cache = do_one(product_smiles, self.datasub, self.jx_cache)

        if template_list == []:
            print('sim cannot found')
            return False, None, reaction_smiles
        else:
            templates = [t[0] for t in template_list]
            scores = [t[1] for t in template_list]
            weights = np.array(scores) 
            probabilities = weights / weights.sum()  # Normalize to get probabilities
            sampled_index = np.random.choice(len(templates), p=probabilities, size=len(templates), replace=False)
            sorted_templates = [templates[i] for i in sampled_index]
            for template in sorted_templates:
                raw_template = template[1:].replace(')>>', '>>')
                if (product_smiles, raw_template) in self.explored_reaction:
                    continue
                else:
                    key = '99999999999'
                    print('sim based reaction found')
                    return True, key, raw_template
            print('sim cannot found')
            return False, None, reaction_smiles
    def sanitize_smiles(self, smiles):
        """
        Check if a SMILES string is valid and return the sanitized molecule.

        Parameters:
            smiles (str): SMILES string.

        Returns:
            str or None: Sanitized SMILES string if valid, None otherwise.
        """
        if smiles == '':
            return None
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=True)
            smi_canon = Chem.MolToSmiles(mol, canonical=True)
            return smi_canon
        except:
            return None

    def sanitize_reaction(self, reaction_smiles):
        """
        Process a reaction SMILES, removing invalid molecules in reactants and products.

        Parameters:
            reaction_smiles (str): Reaction SMILES in the format "reactants>>products".

        Returns:
            str: Sanitized reaction SMILES with only valid reactants and products.
        """
        try:
            reactants, products = reaction_smiles.split('>>')
        
            reactants_list = reactants.split('.')
            products_list = products.split('.')

            sanitized_reactants = [self.sanitize_smiles(smiles) for smiles in reactants_list]
            sanitized_products = [self.sanitize_smiles(smiles) for smiles in products_list]

            sanitized_reactants = [s for s in sanitized_reactants if s is not None]
            sanitized_products = [s for s in sanitized_products if s is not None]

            sanitized_reaction = ".".join(sanitized_reactants) + ">>" + ".".join(sanitized_products)

            return sanitized_reaction

        except Exception as e:
            print(f"Invalid reaction SMILES format: {reaction_smiles}. Error: {e}")
            return reaction_smiles

    def blurry_search(self, reaction_smiles, product_smiles, exploration_signal):
        similarity_metric = DataStructs.BulkTanimotoSimilarity
        if exploration_signal == True:
            reaction_number = 1000
        else:
            reaction_number = 100

        try:
            # remove invalid molecules in the reaction to perform similarity search
            sanitized_reaction = self.sanitize_reaction(reaction_smiles)
            #sanitized_reaction = reaction_smiles
            rxn_obj = smiles_to_reaction(sanitized_reaction)
            fp_re = rdChemReactions.CreateDifferenceFingerprintForReaction(rxn_obj)

            sims = similarity_metric(fp_re, self.all_reaction_fps)
            # heapq.nlargest to fetch top reaction_number indexes
            top_indices = heapq.nlargest(reaction_number, range(len(sims)), key=lambda i: sims[i])
            sorted_reaction_list = [self.reaction_list[i] for i in top_indices]
            
        except Exception as e:
            print(f"Error {e} getting reaction {reaction_smiles} for fingerprints!")
            #return False, None, reaction_smiles
            return self.rule_based_search(product_smiles, reaction_smiles)

        try:
            target_rd = rdchiralReactants(product_smiles)
        except Exception as e:
            print(f"Error {e} initializing rdchiralReactants for product {product_smiles}")
            return False, None, reaction_smiles

        for reaction_smarts in sorted_reaction_list:
            try:
                reaction_outputs = run_retro(target_rd, reaction_smarts)
                if len(reaction_outputs) > 1:
                    reaction_outputs = self.rank_reactants(reaction_outputs)
                if len(reaction_outputs) == 0:
                    continue
                reactants_generated = [reactant for reactant in reaction_outputs[0]]
                if reactants_generated == []:
                    continue
                elif len(reactants_generated) > 0:
                    key = self.template_to_key.get(reaction_smarts)
                    if (product_smiles, reaction_smarts) in self.explored_reaction:
                        print('redundant!')
                        continue
                    return True, key, reaction_smarts
            except Exception as e:
                print(f"Error {e} testing reaction {reaction_smarts} on product {product_smiles}")
                continue
        return self.rule_based_search(product_smiles, reaction_smiles)



    
    def sanitize(self, starting_list, route, exploration_signal):
        new_route = check_and_update_routes(route, starting_list)
        first_evaluation = self.check_route(starting_list, new_route, exploration_signal)

        new_route = map_reaction(new_route, first_evaluation)
        new_route = self.fix_reaction_error(new_route, first_evaluation)
        new_route = check_and_update_routes(new_route, starting_list)

        final_evaluation = self.check_route_extra(starting_list, new_route, first_evaluation)

        return new_route, final_evaluation

    def sort_buffer(self):
        self.oracle.sort_buffer()
    
    def log_intermediate(self, finish=False):
        self.oracle.log_intermediate(finish=finish)
    

        
    def save_result(self, suffix=None):

        print(f"Saving...")
        
        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            suffix = suffix.replace("/", "")
            output_file_path = os.path.join(self.args.output_dir, 'results_' + suffix + '.yaml')

        self.sort_buffer()
        with open(output_file_path, 'w') as f:
            yaml.dump(self.route_buffer, f, sort_keys=False)
    
    def check_route(self, target_smi, route, exploration_signal):
        """
        Check if the route is valid
        target smi is a list
        """
        results = []
        for i in range(len(route)):
            current_step_index = i
            current_step = route[current_step_index]

            step_validity = False
            molecule_set = ast.literal_eval(current_step['Molecule set'])
            updated_molecule_set = ast.literal_eval(current_step['Updated molecule set'])
            reaction = ast.literal_eval(current_step['Reaction'])[0]
            #reaction = change_to_forward_reaction(reaction)
            product = extract_molecules_from_output(current_step['Product'])
            reactants = ast.literal_eval(current_step['Reactants'])

            #step 1: check molecules' validity
            starting_signal = True
            if current_step_index == 0:
                mdd = set(molecule_set).issubset(set(target_smi))
                if not mdd:
                    starting_signal = False

            # Product in molecule set
            product_inside = False
            if product[0] in molecule_set:
                product_inside = True
            invalid_molset_mol_id = []
            invalid_updated_mol_id = []
            
            updated_set_signals = check_validity(updated_molecule_set)
            if False in updated_set_signals:
                invalid_updated_mol_id = [index for index, value in enumerate(updated_set_signals) if not value]
    
            mol_set_signals = check_validity(molecule_set)
            if False in mol_set_signals:
                invalid_molset_mol_id = [index for index, value in enumerate(mol_set_signals) if not value]

            #check purchasability
            check_availability = False
            unavailable_mol_id = []
            if i == len(route) - 1:
                availibities = check_purchasable(updated_molecule_set, updated_set_signals, self.inventory)
                check_availability = True
            if check_availability == True:
                if False in availibities:
                    unavailable_mol_id = [index for index, value in enumerate(availibities) if not value]
    
            #step 2
            reaction_valid, updated_set_valid, reaction_existance = False, False, False
            if ':' in reaction:
                keys = [key for key, value in self.original_template_dict.items() if value == reaction]
                if len(keys) == 1:
                    reaction_existance = True
                    reaction_key = keys[0]
                else:
                    reaction_existance = False
                    reaction_key = None
            else:
                if current_step_index == 0:
                    reaction_existance, reaction_key = is_reaction_in_dict(reaction, self.template_dict)
                    #reaction_existance, reaction_key, new_reaction = self.blurry_search(reaction, product[0], exploration_signal)
                #To save time, we only check the step after a valid step
                elif results[-1][1] == True:
                    reaction_existance, reaction_key = is_reaction_in_dict(reaction, self.template_dict)

                else:
                    reaction_existance = False
                    reaction_key = None
            if reaction_key == None:
                new_reaction = reaction
            else:
                new_reaction = self.original_template_dict[reaction_key]
                
            if reaction_existance == True:
                reaction_valid, updated_set_valid = verify_reaction_step(molecule_set, updated_molecule_set, new_reaction, product, reactants, self.inventory, self.oracle)
            
            if current_step_index == 0:
                if reaction_key == None:
                    reaction_existance, reaction_key, new_reaction = self.blurry_search(reaction, product[0], exploration_signal)
                elif (product[0], new_reaction) in self.explored_reaction:
                    reaction_existance, reaction_key, new_reaction = self.blurry_search(reaction, product[0], exploration_signal)
                elif reaction_existance == True and reaction_valid == False:
                    reaction_existance, reaction_key, new_reaction = self.blurry_search(reaction, product[0], exploration_signal)  
            
                if reaction_key == None:
                    new_reaction = reaction
                else:
                    #new_reaction = self.original_template_dict[reaction_key]
                    reaction_valid, updated_set_valid = verify_reaction_step(molecule_set, updated_molecule_set, new_reaction, product, reactants, self.inventory, self.oracle)
               
            if (
                len(invalid_molset_mol_id) == 0 and
                len(invalid_updated_mol_id) == 0 and
                reaction_valid and
                updated_set_valid and
                starting_signal and
                product_inside
            ):
                step_validity = True

            # Construct the dictionary
            step_info = {
                "target_smi": target_smi,
                "starting_signal": starting_signal,
                "product_inside": product_inside,
                "molecule_set": molecule_set,
                "updated_molecule_set": updated_molecule_set,
                "reaction": new_reaction,
                "reaction_key": reaction_key,
                "product": product,
                "reactants": reactants,
                "updated_set_signals": updated_set_signals,
                "invalid_updated_mol_id": invalid_updated_mol_id,
                "mol_set_signals": mol_set_signals,
                "invalid_molset_mol_id": invalid_molset_mol_id,
                "check_availability": check_availability,
                "unavailable_mol_id": unavailable_mol_id,
                "reaction_existance": reaction_existance,
                "reaction_valid": reaction_valid,
                "updated_set_valid": updated_set_valid
            }

            # Store the tuple in the results list
            results.append((current_step_index, step_validity, step_info))
        return results

    def check_route_extra(self, target_smi, route, first_evaluation):
        """
        Check if the route is valid
        """
        results = []
        for i in range(len(route)):
            current_step_index = i
            current_step = route[current_step_index]
            step_id, is_valid, current_evaluation = first_evaluation[current_step_index]

            step_validity = False
            molecule_set = ast.literal_eval(current_step['Molecule set'])
            updated_molecule_set = ast.literal_eval(current_step['Updated molecule set'])
            reaction = ast.literal_eval(current_step['Reaction'])[0]
            product = extract_molecules_from_output(current_step['Product'])
            reactants = ast.literal_eval(current_step['Reactants'])

            #step 1: check molecules' validity
            starting_signal = True
            if current_step_index == 0:
                mmd = set(molecule_set).issubset(set(target_smi))
                if not mmd:
                    starting_signal = False

            # Product in molecule set
            product_inside = False
            if product[0] in molecule_set:
                product_inside = True

            invalid_molset_mol_id = []
            invalid_updated_mol_id = []
            
            updated_set_signals = check_validity(updated_molecule_set)
            if False in updated_set_signals:
                invalid_updated_mol_id = [index for index, value in enumerate(updated_set_signals) if not value]
    
            mol_set_signals = check_validity(molecule_set)
            if False in mol_set_signals:
                invalid_molset_mol_id = [index for index, value in enumerate(mol_set_signals) if not value]


            check_availability = False
            unavailable_mol_id = []
            if i == len(route) - 1:
                availibities = check_purchasable(updated_molecule_set, updated_set_signals, self.inventory)
                check_availability = True
            if check_availability == True:
                if False in availibities:
                    unavailable_mol_id = [index for index, value in enumerate(availibities) if not value]
    
            #step 2
            reaction_valid, updated_set_valid = False, False
            reaction_existance, reaction_key = current_evaluation['reaction_existance'], current_evaluation['reaction_key']
            
            new_reaction = current_evaluation["reaction"]

            if reaction_existance == True:
                reaction_valid, updated_set_valid = verify_reaction_step(molecule_set, updated_molecule_set, new_reaction, product, reactants, self.inventory, self.oracle)
    
    
            if (
                len(invalid_molset_mol_id) == 0 and
                len(invalid_updated_mol_id) == 0 and
                reaction_valid and
                updated_set_valid and
                starting_signal and
                product_inside
            ):
                step_validity = True
            if step_validity == True:
                self.explored_reaction.add((product[0], new_reaction))
            # Construct the dictionary
            step_info = {
                "target_smi": target_smi,
                "starting_signal": starting_signal,
                "product_inside": product_inside,
                "molecule_set": molecule_set,
                "updated_molecule_set": updated_molecule_set,
                "reaction": new_reaction,
                "reaction_key": reaction_key,
                "product": product,
                "reactants": reactants,
                "updated_set_signals": updated_set_signals,
                "invalid_updated_mol_id": invalid_updated_mol_id,
                "mol_set_signals": mol_set_signals,
                "invalid_molset_mol_id": invalid_molset_mol_id,
                "check_availability": check_availability,
                "unavailable_mol_id": unavailable_mol_id,
                "reaction_existance": reaction_existance,
                "reaction_valid": reaction_valid,
                "updated_set_valid": updated_set_valid
            }

            # Store the tuple in the results list
            results.append((current_step_index, step_validity, step_info))
        return results

    def reset(self):
        del self.oracle
        self.oracle = Oracle(args=self.args)
        self.oracle.route_buffer = {}
        self.oracle.reaction_cache = dict()
        self.explored_reaction = set()
        self.visited_molecules = dict()

    @property
    def route_buffer(self):
        return self.oracle.route_buffer

    @property
    def finish(self):
        return self.oracle.finish
        
    def _optimize(self, oracle, config):
        raise NotImplementedError
            
    def rewards(self, route_evaluation):
        return self.oracle(self.inventory, route_evaluation, self.visited_molecules, self.dead_molecules)
    
    def update_cache(self, mol_smiles, reaction):
        try:
            s = rdChemReactions.CreateDifferenceFingerprintForReaction(smiles_to_reaction(reaction))
            self.oracle.store_cache(mol_smiles, reaction)
        except:
            pass


    def optimize(self, target, route_list, all_fps, config, seed=0, project="test"):
        self.reset()
        self.seed = seed 
        self.oracle.task_label = self.model_name + "_" + target + "_" + str(seed)
        self._optimize(target, route_list, all_fps, config)
        if self.args.log_results:
            self.log_result()
        self.save_result(self.model_name + "_" + target + "_" + str(seed))
        
    
    def query_LLM(self, question, model="gpt-4o", temperature=0.0):
        openai.api_type = "azure"
        openai.api_base = ''
        openai.api_version = "2024-12-01-preview"
        openai.api_key = ''
        message = [{"role": "system", "content": "You are a retrosynthesis agent who can make multi-step retrosynthesis plans based on your molecule knowledge."}]

        prompt1 = question
        message.append({"role": "user", "content": prompt1})

        params = {
            "engine": '',
            "max_tokens": 8192,
            "temperature": temperature,
            "messages": message
        }

        for retry in range(3):
            try:
                response = openai.ChatCompletion.create(**params)["choices"][0]["message"]["content"]
                message.append({"role": "assistant", "content": response})
                break
            except Exception as e:
                print(f"{type(e).__name__} {e}")


        print("=>")
        return message, response
    
    def query_deepseek(self, question):
        endpoint = ''
        model_name = "DeepSeek-V3"

        api_key = ''
        client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key),
        )

        for retry in range(3):
            try:
                response = client.complete(
                    messages=[
                        SystemMessage(content="You are a retrosynthesis agent who can make multi-step retrosynthesis plans based on your molecule knowledge."),
                        UserMessage(content=question)
                    ],
                    max_tokens=4096,
                    temperature=0.8,
                    top_p=1.0,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    model=model_name
                )
                break
            except Exception as e:
                print(f"{type(e).__name__} {e}")


        print("=>")
        answer = response.choices[0].message.content
        return response, answer    

    def fix_reaction_error(self, routes, stepwise_results):
        updated_routes = []
        for i in range(len(routes)):
            step_id, is_valid, data = stepwise_results[i]
            if data['reaction_existance'] == True and data['reaction_valid'] == True:
                reaction_smiles = data['reaction']

                #reactants = [Chem.MolFromSmiles(smi) for smi in data['reactants']]
                #products = [Chem.MolFromSmiles(smi) for smi in data['product']]
            
                target_rd = rdchiralReactants(data['product'][0])
                reaction_outputs = run_retro(target_rd, reaction_smiles)
                if len(reaction_outputs) > 1:
                    reaction_outputs = self.rank_reactants(reaction_outputs)
                reactants_generated = [reactant for reactant in reaction_outputs[0]]
                reactants_generated = [sanitize_smiles(smi) for smi in reactants_generated]
                reactants_smiles = set(reactants_generated)

                products_smiles = {smi for smi in data['product']}
                original_molecule_set = [Chem.MolFromSmiles(smi) for smi in data['molecule_set']]
                original_set = {Chem.MolToSmiles(mol) for mol in original_molecule_set if mol is not None}
                #print(reactants_smiles)
                updated_mol_set = (original_set | reactants_smiles) - products_smiles
                data['Updated molecule set'] = list(updated_mol_set)
                routes[i]['Reactants'] = str(list(reactants_smiles))
                routes[i]['Updated molecule set'] = str(list(updated_mol_set))

        return routes
    def rank_reactants(self, reactants_list):
        """
        Rank reactants based on the number of products generated
        """
        non_empty_reactant_list = [item for item in reactants_list if item != []]
        scores = [self.oracle.reward(self.inventory, reactant, self.visited_molecules, self.dead_molecules) for reactant in non_empty_reactant_list]
        sorted_list = [x for _, x in sorted(zip(scores, non_empty_reactant_list), key=lambda pair: pair[0], reverse=True)]
        return sorted_list
def check_and_update_routes(routes, target_list):
    import ast

    routes[0]['Molecule set'] = str(target_list)
    for i in range(1, len(routes)):
        current_updated_set = ast.literal_eval(routes[i]['Molecule set'])
        previous_molecule_set = ast.literal_eval(routes[i - 1]['Updated molecule set'])


        if set(current_updated_set) != set(previous_molecule_set):
            print(f"Mismatch found at step {i}:")
            print(f"  Previous Molecule set: {previous_molecule_set}")
            print(f"  Current Updated molecule set: {current_updated_set}")

            routes[i]['Molecule set'] = str(previous_molecule_set)
            print(f"  Updated step {i} to match previous Molecule set.")

    print("\nAll steps checked and updated where necessary.")
    return routes



def map_reaction(routes, stepwise_results):
    # Use found reactions to substitute the reaction proposed by the LLM
    for i in range(len(routes)):
        step_id, is_valid, data = stepwise_results[i]
        #if data['reaction_existance'] == True:
        reaction_smiles = data['reaction']

        routes[i]['Reaction'] = str([reaction_smiles])

    return routes

def extract_molecules_from_output(output):
    try:

        parsed_output = ast.literal_eval(output)

        if isinstance(parsed_output, list):
            return parsed_output
        elif isinstance(parsed_output, str):
            return [parsed_output]
        else:
            return []
    except (ValueError, SyntaxError):

        return []