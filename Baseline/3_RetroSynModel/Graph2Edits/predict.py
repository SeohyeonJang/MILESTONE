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

def canonicalize_prod(p):
    import copy
    p = copy.deepcopy(p)
    p = canonicalize(p)
    p_mol = Chem.MolFromSmiles(p)
    if p_mol is not None:
        for atom in p_mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
        p = Chem.MolToSmiles(p_mol)
    return p


def run_search(beam_model, prod_smi, max_steps, rxn_class ):
    return beam_model.run_search(prod_smi=prod_smi, max_steps=max_steps, rxn_class=rxn_class)


def calculate_molecular_weight(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return rdMolDescriptors.CalcExactMolWt(mol)

def run_beam_search_and_write_result(fp, beam_model, prod_smi, max_steps, use_rxn_class, depth, eMolecule, reaction_success):

    with torch.no_grad():
        top_k_results = run_search(beam_model, prod_smi=prod_smi, max_steps=max_steps, rxn_class=use_rxn_class)
    
    pred_r_layer =''
    # .으로 reactant가 분리 될 때까지 
    for i, result in enumerate(top_k_results):
        if '.' in top_k_results[i]['final_smi']:
            pred_r_layer = top_k_results[i]['final_smi']
            break
    
    if pred_r_layer=='':
        return
    
    this_product_re_success = reaction_success

    fp.write(f"{'    ' * depth}pred_layer{depth}: {canonicalize_prod(pred_r_layer)}\n")
    
    reactants = pred_r_layer.split('.')
    reactants = sorted(reactants, key=len)

   

    this_layer_eMol_check=True

    for i, reactant in enumerate(reactants, start=1):
        c_reactant = canonicalize_prod(reactant)
        eMol_check = check_inch_key(c_reactant, eMolecule)
        this_layer_eMol_check &=eMol_check

        fp.write(f"{'    ' * (depth + 1)}react_l{depth}_r{i}:{c_reactant}   eMol:{eMol_check} \n")  
        if this_layer_eMol_check:
            this_product_re_success = True
        else:
            this_product_re_success = False

        if eMol_check: continue
        
                 
        if c_reactant.count(':') > 3:
            if depth < 5:  
                    run_beam_search_and_write_result(fp, beam_model, prod_smi=c_reactant, 
                                                     max_steps=max_steps, use_rxn_class=use_rxn_class, depth=depth + 1, 
                                                     eMolecule=eMolecule, reaction_success=this_product_re_success)
        else:
            this_product_re_success = False
            break

    if this_product_re_success:
        fp.write(f"this product reactatnt success: {this_product_re_success}\n")
    
    return this_product_re_success

def read_eMolecule_from_file(file_path):

    try:
        # 파일 읽기
        content = pd.read_csv(file_path, sep='\t', header=None, names=['inchl_key'])
        return content

    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
        return None
    
def smiles_to_inchi(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        inchi_key = Chem.inchi.MolToInchiKey(mol)
        return inchi_key
    except:
        return None
    
def check_inch_key(smiles, content):

    this_inch = smiles_to_inchi(smiles)
    # 데이터 포함 여부 확인
    if this_inch in content['inchl_key'].values:
        return True
    else:
        return False
    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--beam_size', type=int,
                        default=10, help='Beam search width')
    parser.add_argument('--max_steps', type=int, default=9,
                        help='maximum number of edit steps')
    args = parser.parse_args()
    
    datadir = f'data/real/'
    # 스크리닝 쪽에서 나오는 파일
    filename = f'smarts_list_41.txt'
    test_data = pd.read_csv(os.path.join(datadir, filename) , names=['product_smi'])

    e_mol_file_path = './eMol/e_mol_inchi.txt'


    #eMolecule에 있는지 체크를 위해 
    eMolecule = read_eMolecule_from_file(e_mol_file_path)

    
    model_dir = os.path.join(ROOT_DIR, 'data', 'best_model')
    
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
            fp.write(f'({idx}) p:{p}\n')

            #try:
            result = run_beam_search_and_write_result(fp, beam_model, prod_smi=p, 
                                                 max_steps=args.max_steps, use_rxn_class=use_rxn_class, depth=1 , eMolecule=eMolecule, reaction_success=False)
            

            #except:
            #    print(p)
            fp.write('\n')
            

if __name__ == '__main__':
    main()
