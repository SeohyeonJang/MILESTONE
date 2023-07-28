import numpy as np
import pandas as pd
import os
import argparse
import joblib
from tqdm import tqdm
from collections import Counter
import torch
import pubchempy

from STOUT import translate_forward
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import IPythonConsole, MolsToGridImage
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

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--experiments', type=str, default='27-06-2022--10-27-22',
    #                     help='Name of edits prediction experiment')
    parser.add_argument('--beam_size', type=int,
                        default=10, help='Beam search width')
    parser.add_argument('--max_steps', type=int, default=9,
                        help='maximum number of edit steps')
    args = parser.parse_args()
    
    datadir = f'data/real/'
    # 스크리닝 쪽에서 나오는 파일
    filename = f'real_product.txt'
    test_data = pd.read_csv(os.path.join(datadir, filename) , names=['product_smi'])
    
    model_dir = os.path.join(
            ROOT_DIR, 'data', 'best_model')
    

    checkpoint = torch.load(os.path.join(model_dir, 'epoch_123_allNO.pt'))
    config = checkpoint['saveables']
  

    model = Graph2Edits(**config, device=DEVICE)
    model.load_state_dict(checkpoint['state'])
    model.to(DEVICE)
    model.eval()

    top_k = np.zeros(args.beam_size)
    edit_steps_cor = []
    counter = []
    stereo_rxn = []
    stereo_rxn_cor = []
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
   
            p = test_data['product_smi'][idx]
    
            # print('p==>', p)
            # p = SmilesToSmarts(p)
            # print('sm p==>', p)
                            
            with torch.no_grad():
                top_k_results = beam_model.run_search(
                    prod_smi=p, max_steps=args.max_steps, rxn_class=use_rxn_class)

            fp.write(f'({idx}) {p}\n')
            

            beam_matched = False
            for beam_idx, path in enumerate(top_k_results):
                pred_smi = path['final_smi']
                if pred_smi =='final_smi_unmapped':
                    continue
                prob = path['prob']
                pred_set = set(pred_smi.split('.'))
                
                fp.write(f'{beam_idx} probability:{prob:.4f} {pred_smi}')
                for id,react_smi in enumerate(pred_set):
                    cano_react_smi = canonicalize_prod(react_smi)
                    #print('react_smi===>',react_smi)
                    #iupac = pubchempy.get_compounds(react_smi, namespace="smiles")
                    #iupac = translate_forward(react_smi)
                    fp.write(f' react_smi:{cano_react_smi} ')
                fp.write('\n')

            fp.write('\n')


if __name__ == '__main__':
    main()
