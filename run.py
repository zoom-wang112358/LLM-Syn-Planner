from __future__ import print_function

import random
from typing import List

import joblib
import numpy as np
from joblib import delayed
from rdkit import Chem, rdBase
from rdkit.Chem.rdchem import Mol
from rdkit import DataStructs
import ast
from syntheseus import Molecule
import rdkit.Chem.AllChem as AllChem
rdBase.DisableLog('rdApp.error')
from utils import *
from optimizer import BaseOptimizer, check_and_update_routes, map_reaction, extract_molecules_from_output
import os
import openai
import re
import copy
from concurrent.futures import ThreadPoolExecutor, wait
from route import *
from modification_prompt import modification_hints, construct_modification_prompt
from utils import *
import time
import math

MINIMUM = 1e-10





def make_mating_pool(combined_list: List, population_scores, visited_cache, inventory, offspring_size: int):

    # scores -> probs 
    population_scores = [node_value(combined_list[i], population_scores[i], visited_cache, inventory) for i in range(len(population_scores))]
    weights = np.exp(-np.array(population_scores))  # Exponentially invert scores
    probabilities = weights / weights.sum()  # Normalize to get probabilities
    while True:
        sampled_index = np.random.choice(len(combined_list), p=probabilities, size=offspring_size, replace=True)
        if len(set(sampled_index)) > 1:
            break
    mating_pool = [combined_list[i] for i in sampled_index]
    return mating_pool

def node_value(combined_list, population_score, visited_cache, inventory, C=0.3):
    molecule_list = combined_list[1].validated_route[-1]['Updated molecule set']
    unpurchasable_list = check_availability(molecule_list, inventory)

    if len(unpurchasable_list) == 0:
        return 1
    total_visits = sum(visited_cache.values())
    node_visits = 1
    for molecule in unpurchasable_list:
        if molecule in visited_cache:
            print(f"visited times: {visited_cache[molecule]}")
            node_visits += visited_cache[molecule]
            if visited_cache[molecule] > 25:
                population_score = -10
    if total_visits == 0:
        final_score = - population_score
    else:
        final_score = - population_score - C * math.sqrt(math.log(total_visits) / node_visits)
        print(f"population_score: {- population_score}")
        print(f"visiting_score: {- C * math.sqrt(math.log(total_visits) / node_visits)}")
    return final_score

class planning_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "planning"

    def initialization(self, target_smi, rag_tuples):
        exploration_signal = True
        population = []
        route_list = [t[1] for t in rag_tuples]
        sims_list = [t[0] for t in rag_tuples]


        sum_scores = sum(sims_list)
        population_probs = [p / sum_scores for p in sims_list]
        while True:
            try:
                sampled_index = np.random.choice(len(route_list), p=population_probs, size=3, replace=False)
                sampled_routes = [route_list[i] for i in sampled_index]
                examples = ''
                for i in sampled_routes:
                    examples = examples + '<ROUTE>\n'+ str(process_reaction_routes(i)) + '\n</ROUTE>\n'
                examples = examples + '\n'
                initial_prompt = '''
                    You are a professional chemist specializing in synthesis analysis. Your task is to propose a retrosynthesis route for a target molecule provided in SMILES format.

                    Definition:
                    A retrosynthesis route is a sequence of backward reactions that starts from the target molecules and ends with commercially purchasable building blocks.

                    Key concepts:
                    - Molecule set: The working set of molecules at any given step. Initially, it contains only the target molecule.
                    - Commercially purchasable: Molecules that can be directly bought from suppliers (permitted building blocks).
                    - Non-purchasable: Molecules that must be further decomposed via retrosynthesis steps.
                    - Reaction source: All reactions must be derived from the USPTO dataset, and stereochemistry (e.g., E/Z isomers, chiral centers) must be preserved.

                    Process:
                    1. Initialization: Start with the molecule set = [target molecule].
                    2. Iteration:
                        - Select one non-purchasable molecule from the molecule set (the product).
                        - Apply a valid backward reaction from the USPTO dataset to decompose it into reactants.
                        - Remove the product molecule from the set.
                        - Add the reactants to the set.
                    3. Termination: Continue until all molecules in the set are commercially purchasable.
                '''

                task_description = '''
                My target molecule is: {}

                To assist you for the format, an example retrosynthesis route is provided.\n {}

                Please propose a retrosynthesis route for my target molecule. The provided reference routes may be helpful. You can also design a synthetic route based on your own knowledge.
                '''.format(target_smi,examples)

                requirements = '''
                You need to analyze the target molecule and make a retrosynthesis plan in the <PLAN></PLAN> before proposing the route. After making the plan, you should explain the plan in the <EXPLANATION></EXPLANATION>. The route should be a list of steps wrapped in <ROUTE></ROUTE>. Each step in the list should be a dictionary.
                At the first step, the molecule set should be the target molecules set given by the user. Here is an example:
                
                <PLAN>: Analyze the target molecule and plan for each step in the route. </PLAN>
                <EXPLANATION>: Explain the plan. </EXPLANATION>
                <ROUTE>
                [   
                    {
                        'Molecule set': "[Target Molecule]",
                        'Rational': Step analysis,
                        'Product': "[Product molecule]",
                        'Reaction': "[Reaction template]",
                        'Reactants': "[Reactant1, Reactant2]",
                        'Updated molecule set': "[Reactant1, Reactant2]"
                    },
                    {
                        'Molecule set': "[Reactant1, Reactant2]",
                        'Rational': Step analysis,
                        'Product': "[Product molecule]",
                        'Reaction': "[Reaction template]",
                        'Reactants': "[subReactant1, subReactant2]",
                        'Updated molecule set': "[Reactant1, subReactant1, subReactant2]"
                    }
                ]
                </ROUTE>
                
                \n\n
                Requirements: 1. The 'Molecule set' contains molecules we need to synthesize at this stage. In the first step, it should be the target molecule. In the following steps, it should be the 'Updated molecule set' from the previous step.\n
                2. The 'Rational' part in each step should be your analysis for syhthesis planning in this step. It should be in the string format wrapped with \'\'\n
                3. 'Product' is the molecule we plan to synthesize in this step. It should be from the 'Molecule set'. The molecule should be a molecule from the 'Molecule set' in a list. The molecule smiles should be wrapped with \'\'.\n
                4. 'Reaction' is a backward reaction which can decompose the product molecule into its reactants. The reaction should be in a list. All the molecules in the reaction template should be in SMILES format. For example, ['Product>>Reactant1.Reactant2'].\n
                5. 'Reactants' are the reactants of the reaction. It should be in a list. The molecule smiles should be wrapped with \'\'.\n
                6. The 'Updated molecule set' should be molecules we need to purchase or synthesize after taking this reaction. To get the 'Updated molecule set', you need to remove the product molecule from the 'Molecule set' and then add the reactants in this step into it. In the last step, all the molecules in the 'Updated molecule set' should be purchasable.\n
                7. In the <PLAN>, you should analyze the target molecule and plan for the whole route.\n
                8. In the <EXPLANATION>, you should analyze the plan.\n'''
                question = initial_prompt + requirements + task_description
                message, answer = self.query_LLM(question, temperature=0.7)
                #message, answer = self.query_deepseek(question)
                # Converting the extracted string to a Python list of dictionaries
                match = re.search(r'<ROUTE>(.*?)<ROUTE>', answer, re.DOTALL)
                if match == None:
                    match = re.search(r'<ROUTE>(.*?)</ROUTE>', answer, re.DOTALL)

                route_content = match.group(1)

                route = ast.literal_eval(route_content)
                comp1 = ast.literal_eval(route[-1]['Updated molecule set'])
                comp2 = ast.literal_eval(route[-2]['Updated molecule set'])
                last_step_reactants = route[-1]['Reactants']
                if set(comp1) == set(comp2) or last_step_reactants == "" or last_step_reactants == "[]" or last_step_reactants == "None" or last_step_reactants == "[None]":
                    route = route[:-1]
                    print('Route cleaned!')
                for step in route:
                    temp = ast.literal_eval(step['Molecule set'])
                    temp = ast.literal_eval(step['Reaction'])[0]
                    products, reactants = temp.split(">>")
                    temp = extract_molecules_from_output(step['Product'])[0]
                    temp = ast.literal_eval(step['Reactants'])[0]
                    temp = ast.literal_eval(step['Updated molecule set'])

                route_class_item = Route(target_smi)
                checked_route, final_evaluation = self.sanitize([target_smi], route, exploration_signal)

                if final_evaluation[0][2]['reaction_existance'] == False and final_evaluation[0][2]['product_inside'] == True:
                    self.dead_molecules.append(final_evaluation[0][2]['product'][0])
                    print('reaction not exist')
                    continue
                if final_evaluation[0][1] == False:
                    print('step invalid')
                    continue
                score = self.rewards(final_evaluation)
                route_class_item.add_route(checked_route, final_evaluation)
                route_class_item.update_reward(score)
                route_class_item.update_evaluation(final_evaluation)
                break
            except Exception as e:
                print(f"Error in generating the initial population: {e}")
                continue


        return route_class_item

    def modification(self, combined_list, population_routes, all_fps, route_list, inventory):
        exploration_signal = False
        parent_a = random.choice(combined_list)
        sampled_route = parent_a[1]
        count = 0
        final_route_item = None
        for _ in range(5):
            try:
                count = count + 1
                if count >= 5:
                    count = 0
                    parent_a = random.choice(combined_list)
                    sampled_route = parent_a[1]
                route = sampled_route.validated_route
                new_route_item = copy.deepcopy(sampled_route)
                evaluation = new_route_item.evaluation
                molecule_list = route[-1]['Updated molecule set']
                unpurchasable_list = check_availability(molecule_list, inventory)

                #self.update_visited_molecules(unpurchasable_list)
                
                retrieved_routes = modification_hints(unpurchasable_list, all_fps, route_list)

                new_q = construct_modification_prompt(unpurchasable_list, retrieved_routes, evaluation, inventory, self.oracle.reaction_cache)
                new_m, new_a = self.query_LLM(new_q, temperature=0.7)
                
                match = re.search(r'<ROUTE>(.*?)<ROUTE>', new_a, re.DOTALL)
                if match == None:
                    match = re.search(r'<ROUTE>(.*?)</ROUTE>', new_a, re.DOTALL)
                    #print(answer)
                print(new_a)
                route_content = match.group(1)

                new_route = ast.literal_eval(route_content)
                comp1 = ast.literal_eval(new_route[-1]['Updated molecule set'])
                comp2 = ast.literal_eval(new_route[-2]['Updated molecule set'])
                last_step_reactants = new_route[-1]['Reactants']
                if set(comp1) == set(comp2) or last_step_reactants == "" or last_step_reactants == "[]" or last_step_reactants == "None" or last_step_reactants == "[None]":
                    new_route = new_route[:-1]
                    print('Route cleaned!')
                for idx, step in enumerate(new_route):
                    temp = ast.literal_eval(step['Molecule set'])
                    reaction = ast.literal_eval(step['Reaction'])[0]
                    products, reactants = reaction.split(">>")
                    product = extract_molecules_from_output(step['Product'])[0]
                    temp = ast.literal_eval(step['Reactants'])[0]
                    temp = ast.literal_eval(step['Updated molecule set'])
                    if idx == 0:
                        if products == product:
                            self.update_cache(product, reaction)

                checked_route, final_evaluation = self.sanitize(molecule_list, new_route, exploration_signal)
                if final_evaluation[0][2]['product_inside'] == True:
                    self.update_visited_molecules(final_evaluation[0][2]['product'])


                if final_evaluation[0][2]['reaction_existance'] == False: 
                    if final_evaluation[0][2]['product_inside'] == True:
                        self.update_dead_molecules(final_evaluation[0][2]['product'][0])
                        print('dead molecule + 1')
                    continue
                new_route_item.update_route(checked_route, final_evaluation)
                if not check_distinct_route(population_routes, new_route_item):
                    print('Exists in the population')
                    continue
                score = self.rewards(final_evaluation)
                
                new_route_item.update_reward(score)
                new_route_item.update_evaluation(final_evaluation)
                final_route_item = new_route_item
                break
            except Exception as e:
                print(f"Error in generating the modification population {e}")

                continue

        return final_route_item

    def _optimize(self, target, route_list, all_fps, config):
        
        file_path = 'stat/'+ str(self.args.dataset_name) + '_' + str(self.args.max_oracle_calls) + '_results.txt'
        #args.dataset_name
        population_class = []
        getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=False)
        similarity_metric = DataStructs.BulkTanimotoSimilarity # BulkDiceSimilarity or BulkTanimotoSimilarity

        target = sanitize_smiles(target)
        fp = getfp(target)
    
        sims = similarity_metric(fp, [fp_ for fp_ in all_fps])

        rag_tuples = list(zip(sims, route_list))
        #rag_tuples = sorted(rag_tuples, key=lambda x: x[0], reverse=True)[:50]
        rag_tuples = sorted(rag_tuples, key=lambda x: x[0], reverse=True)
        assert len(self.oracle.route_buffer) == 0, f"route_buffer not empty: {self.oracle.route_buffer}"

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.initialization, target, rag_tuples) for _ in range(config["population_size"])]
            starting_population = [future.result() for future in futures]

        # initial population
        population_routes = starting_population

        population_scores = [route_class_item.get_reward() for route_class_item in population_routes]

        combined_list = list(zip(population_scores, population_routes))
        all_routes = copy.deepcopy(population_routes)
        while True:

            if len(self.oracle) > 5:
                self.sort_buffer()
                if 0 in population_scores:
                    self.log_intermediate(finish=True) 
                    print('find a route, abort ...... ')
                    for route in population_routes:
                        reward = route.get_reward()
                        if reward == 0:
                            route.save_result(target)
                            
                    break


            # new_population
            mating_pool = make_mating_pool(combined_list, population_scores, self.visited_molecules, self.inventory, config["population_size"])

            #modification
            with ThreadPoolExecutor() as executor:

                futures = [executor.submit(self.modification, mating_pool, all_routes, all_fps, route_list, self.inventory) for _ in range(config["offspring_size"])]
                done, not_done = wait(futures, timeout=300)
                returned_routes = [future.result() for future in done]
                offspring_routes = []
                for i in range(len(returned_routes)):
                    if returned_routes[i] is not None:
                        if check_distinct_route(offspring_routes, returned_routes[i]):
                            offspring_routes.append(returned_routes[i])

            # add new_population
            population_routes += offspring_routes
            all_routes = all_routes + offspring_routes
            # stats
            old_scores = population_scores


            all_scores = [self.oracle.reward(self.inventory, route_class_item.validated_route[-1]['Updated molecule set'],self.visited_molecules, self.dead_molecules) for route_class_item in all_routes]
            combined_list = list(zip(all_scores, all_routes))
            combined_list = sorted(combined_list, key=lambda x: x[0], reverse=True)[:config["population_size"]]
            population_routes = [t[1] for t in combined_list]
            population_scores = [t[0] for t in combined_list]



            ### stopping

            if len(self.oracle) > 5:
                print(population_scores)
                if 0 in population_scores:
                    self.log_intermediate(finish=True)  
                    print('Find route, abort ...... ')
                    for route in population_routes:
                        reward = route.get_reward()
                        if reward == 0:
                            route.save_result(target)
                            
                    break
  
            if self.finish:   
                print('finished, abort ...... ')              
                break