import numpy as np
import pandas as pd
import os
import argparse
import joblib
from tqdm import tqdm
from collections import Counter
import torch



from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
import re
RDLogger.DisableLog("rdApp.*")


from models import Graph2Edits, BeamSearch
lg = RDLogger.logger()
lg.setLevel(4)

ROOT_DIR = './'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def molWatom(mol):  # smiles 에 smarts 처럼 위치정보를 입력하는 것이라고 보시면 되요!
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol
def SmilesToSmarts(smi):  # 해당 함수를 사용해서 확인해보시면 될것 같습니다.
    mol = Chem.MolFromSmiles(smi)
    mol = molWatom(Chem.MolFromSmiles(smi))
    sma = Chem.MolToSmarts(mol)

    smia = mol.GetAtoms()
    smial = [i.GetSymbol() for i in smia]

    t1 = re.compile('#\d+')
    result = t1.findall(sma)

    for i, j in enumerate(result):
        sma = sma.replace(j, smial[i], 1)

    return sma

def canonicalize_prod(p):
    import copy
    p = copy.deepcopy(p)
    p = canonicalize(p)
    p_mol = Chem.MolFromSmiles(p)
    for atom in p_mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    p = Chem.MolToSmiles(p_mol)
    return p


def canonicalize(smiles):
    try:
        tmp = Chem.MolFromSmiles(smiles)
    except:
        print('no mol', flush=True)
        return smiles
    if tmp is None:
        return smiles
    tmp = Chem.RemoveHs(tmp)
    [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
    return Chem.MolToSmiles(tmp)

def run_search(beam_model, prod_smi, max_steps, rxn_class ):
    return beam_model.run_search(prod_smi=prod_smi, max_steps=max_steps, rxn_class=rxn_class)


def calculate_molecular_weight(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return rdMolDescriptors.CalcExactMolWt(mol)

def run_beam_search_and_write_result(fp, beam_model, prod_smi, max_steps, use_rxn_class, depth):
    with torch.no_grad():
        top_k_results = run_search(beam_model, prod_smi=prod_smi, max_steps=max_steps, rxn_class=use_rxn_class)
    
    pred_r_layer = top_k_results[0]['final_smi']
    fp.write(f"{'    ' * depth}pred_r_layer{depth}: {canonicalize_prod(pred_r_layer)}\n")
    
    reactants = pred_r_layer.split('.')
    for i, reactant in enumerate(reactants, start=1):
        c_reactant = canonicalize_prod(reactant)
        fp.write(f"{'    ' * (depth + 1)}c_r{i}: {c_reactant}\n")  

        if c_reactant.count(':') >= 3:
            if depth < 3:  
                    run_beam_search_and_write_result(fp, beam_model, prod_smi=c_reactant, 
                                                     max_steps=max_steps, use_rxn_class=use_rxn_class, depth=depth + 1)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--beam_size', type=int,
                        default=10, help='Beam search width')
    parser.add_argument('--max_steps', type=int, default=9,
                        help='maximum number of edit steps')
    args = parser.parse_args()
    
    datadir = f'data/real/'
    # 스크리닝 쪽에서 나오는 파일
    filename = f'real_product_3.txt'
    test_data = pd.read_csv(os.path.join(datadir, filename) , names=['product_smi'])
    
    model_dir = os.path.join(
            ROOT_DIR, 'data', 'best_model')
    
    checkpoint = torch.load(os.path.join(model_dir, 'epoch_123_allNO.pt'))
    config = checkpoint['saveables']
  
    model = Graph2Edits(**config, device=DEVICE)
    model.load_state_dict(checkpoint['state'])
    model.to(DEVICE)
    model.eval()

    use_rxn_class = False
    beam_model = BeamSearch(model=model, step_beam_size=10,
                            beam_size=args.beam_size, use_rxn_class=use_rxn_class)
    p_bar = tqdm(list(range(len(test_data))))
    pred_file = os.path.join(datadir, 'pred_results.txt')
    file_num = 1
    while os.path.exists(pred_file):
        pred_file = os.path.join(datadir, f'pred_results_{file_num}.txt')
        file_num += 1

    with open(pred_file, 'a') as fp:
        for idx in p_bar:
   
            # Product smi
            p = test_data['product_smi'][idx]
            fp.write(f'({idx}) {p}\n')

            try:
                run_beam_search_and_write_result(fp, beam_model, prod_smi=p, 
                                                 max_steps=args.max_steps, use_rxn_class=use_rxn_class, depth=1)
            except:
                print(idx,p)
            fp.write('\n')
            

if __name__ == '__main__':
    main()
