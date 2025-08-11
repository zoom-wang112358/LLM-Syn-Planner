import ast
import pickle
import openai
import json
import os
import yaml
import random
class Route:
    def __init__(self, target_smiles):
        """

        target_smiles (str)
        """
        self.target_smiles = target_smiles  
        self.raw_route = []  
        self.validated_route = []  
        self.feedback_list = []
        self.evaluation = ()  
        self.reward = -10  
    
    def add_raw_route(self, route_list):
        """
        route_list (list)
        """
        self.raw_route = route_list
    
    def add_feedback(self, feedback_list):
        """

        feedback_list (list)
        """
        self.feedback_list = feedback_list

    def add_route(self, route_list, evaluation_results):
        """
        reaction_id (str)
        route_details (dict)
        """
        for i in range(len(route_list)):
            step_id, is_valid, data = evaluation_results[i]
            if is_valid:
                molecule_set = ast.literal_eval(route_list[i]['Molecule set'])
                updated_molecule_set = ast.literal_eval(route_list[i]['Updated molecule set'])
                reaction = ast.literal_eval(route_list[i]['Reaction'])
                product = ast.literal_eval(route_list[i]['Product'])
                reactants = ast.literal_eval(route_list[i]['Reactants'])
                step = {
                "Molecule set": molecule_set,
                "Product": product,
                "Reaction": reaction,
                "Reactants": reactants,
                "Updated molecule set": updated_molecule_set
                }
                self.validated_route.append(step)
        
        if self.validated_route == []:
            step_id, is_valid, data = evaluation_results[0]
            step = {
                "Updated molecule set": data["molecule_set"]
            }
            self.validated_route.append(step)

    def update_route(self, route_list, evaluation_results):
        """
        """
        for i in range(len(route_list)):
            step_id, is_valid, data = evaluation_results[i]
            if is_valid:
                self.validated_route.append(self.align_route(route_list[i]))

    def align_route(self, route_step):

        previous_step = self.validated_route[-1]
        
        molecule_set = previous_step['Updated molecule set']
        reaction = ast.literal_eval(route_step['Reaction'])
        product = ast.literal_eval(route_step['Product'])
        reactants = ast.literal_eval(route_step['Reactants'])

        original_set = set(molecule_set)
        product_set = set(product)
        reactants_set = set(reactants)
        updated_set = (original_set|reactants_set) - product_set

        updated_molecule_set = list(updated_set)

        step = {
            "Molecule set": molecule_set,
            "Product": product,
            "Reaction": reaction,
            "Reactants": reactants,
            "Updated molecule set": updated_molecule_set
        }
        return step

    def update_reward(self, reward):
        self.reward = reward

    def update_evaluation(self, all_step_evaluation):
        for evaluation in all_step_evaluation:
            step_id, is_valid, data = evaluation
            if is_valid:
                continue
            else:
                self.evaluation = data
                break
        

    def get_reward(self):
        return self.reward
    
    def check_same(self, ano_route):
        if set(self.validated_route[-1]['Updated molecule set']) == set(ano_route.validated_route[-1]['Updated molecule set']):
            return True
        else:
            return False

    def save_result(self, suffix=None):

        print(f"Saving...")
        
        if suffix is None:
            output_file_path = os.path.join('found_route', 'results.yaml')
        else:
            suffix = suffix.replace("/", "")
            output_file_path = os.path.join('found_route', 'results_' + suffix + '.yaml')

        with open(output_file_path, 'w') as f:
            yaml.dump(self.validated_route, f, default_flow_style=False)

    def get_current_set_molecules(self):
        return self.validated_route[-1]['Updated molecule set']


#For simplicity, we only check the final molecule set to see if two routes are distinct. There might be some cases where two routes have the same final molecule set but different intermediate steps.
def check_distinct_route(population, current_route):
    set_list = []
    for route in population:
        set_list.append(route.validated_route[-1]['Updated molecule set'])
    
    current_updated_set = current_route.validated_route[-1]['Updated molecule set']
    for s in set_list:
        if set(current_updated_set) == set(s):
            return False
    #print(set_list)
    #print(current_updated_set)
    return True