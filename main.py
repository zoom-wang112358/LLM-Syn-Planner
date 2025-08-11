import argparse
import yaml
import os
import sys
sys.path.append(os.path.realpath(__file__))
from time import time
from rdkit import DataStructs
import ast
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import pickle 
            
def main():
    start_time = time() 
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default=None)
    parser.add_argument('--dataset_name', default='pistachio_reachable')
    parser.add_argument('--config_default', default='hparams_default.yaml')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--max_oracle_calls', type=int, default=100)
    parser.add_argument('--freq_log', type=int, default=5)
    parser.add_argument('--seed', type=int, nargs="+", default=[0])

    parser.add_argument('--log_results', action='store_true')
    args = parser.parse_args()



    args.method = args.method.lower()


    
    print(args.method)
    if args.method == 'planning':
        from run import planning_Optimizer as Optimizer 

    else:
        raise ValueError("Unrecognized method name.")

    train_data = 'dataset/routes_train.pkl'
    val_data = 'dataset/routes_val.pkl'
    test_data = 'dataset/routes_test.pkl'
    test_hard_data = 'dataset/routes_possible_test_hard.pkl'
    train_routes = pickle.load(open(train_data, 'rb'))
    val_routes = pickle.load(open(val_data, 'rb'))
    test_routes = pickle.load(open(test_data, 'rb'))
    test_hard_routes = pickle.load(open(test_hard_data, 'rb'))

    total_routes = train_routes + val_routes
    target_list = []
    route_list = []
    for route in total_routes:
        target_list.append(route[0].split('>>')[0])
        route_list.append(route)
    
    test_list = []
    test_route_list = []
    for route in test_routes:
        test_list.append(route[0].split('>>')[0])
        test_route_list.append(route)
    
    test_hard_list = []
    test_hard_route_list = []
    for route in test_hard_routes:
        test_hard_list.append(route[0].split('>>')[0])
        test_hard_route_list.append(route)

    if args.dataset_name == 'USPTO-easy':
        with open('dataset/USPTO_easy.pkl', 'rb') as file:
            test_mol = pickle.load(file)
        data = test_mol   
    elif args.dataset_name == 'USPTO-190':
        test_hard_list = []
        test_hard_route_list = []
        for route in test_hard_routes:
            test_hard_list.append(route[0].split('>>')[0])
            test_hard_route_list.append(route)
        data = test_hard_list
    elif args.dataset_name == 'pistachio_reachable':
        file_path = "dataset/pistachio_reachable_targets.txt"
        data = []

        with open(file_path, "r") as file:
            for line in file:
                # 
                line = line.strip()
                if line:  # 
                    try:
                        # 
                        molecule_tuple = eval(line)
                        # 
                        first_molecule = molecule_tuple[0]
                        # 
                        data.append(first_molecule)
                    except Exception as e:
                        print(f"{line}, error info: {e}")
    
    elif args.dataset_name == 'pistachio_hard':
        file_path = "dataset/pistachio_hard_targets.txt"
        data = []
        with open(file_path, "r") as file:
            for line in file:
                # 
                line = line.strip()
                if line:  # 
                    try:
                        molecule_tuple = eval(line)
                        first_molecule = molecule_tuple[0]
                        data.append(first_molecule)
                    except Exception as e:
                        print(f"{line}, error info: {e}")
    elif args.dataset_name == 'case_study':
        file_path = "dataset/case_study.txt"
        data = []
        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()
                if line:  # 
                    try:
                        molecule_tuple = eval(line)
                        first_molecule = molecule_tuple[0]
                        data.append(first_molecule)
                    except Exception as e:
                        print(f"{line}, error info: {e}")
    similarity_metric = DataStructs.BulkTanimotoSimilarity # BulkDiceSimilarity or BulkTanimotoSimilarity
    similarity_label = 'Tanimoto'
    getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=False)
    getfp_label = 'Morgan2noFeat'

    all_fps = []
    for smi in target_list:
        all_fps.append(getfp(smi))
    

    args.output_dir = os.path.join(args.dataset_name, 'results')
    if not os.path.exists(args.dataset_name):
        os.mkdir(args.dataset_name)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)


    config_default = yaml.safe_load(open(args.config_default))


    if args.method == "planning":
        for seed in args.seed:
            for idx in range(len(data)):
                print(f'Searching routes for: {data[idx]}')
                optimizer = Optimizer(args=args)
                optimizer.optimize(data[idx], route_list, all_fps, config=config_default, seed=seed)
    

    end_time = time()
    hours = (end_time - start_time) / 3600.0
    print('---- The whole process takes %.2f hours ----' % (hours))



if __name__ == "__main__":
    main()