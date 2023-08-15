
import pandas as pd
from joblib import Parallel, delayed

from rdkit.Chem import Crippen, QED

from .conversion import mols_from_smiles
from .sascorer.sascorer import calculateScore


def logp(mol):
    return Crippen.MolLogP(mol) if mol else None


def mr(mol):
    return Crippen.MolMR(mol) if mol else None


def qed(mol):
    return QED.qed(mol) if mol else None


def sas(mol):
    return calculateScore(mol) if mol else None


def add_property(dataset, name, n_jobs):  # 입력된 데이터셋 + name 들어온 property 에 맞는 값을 추가해서 데이터셋으로 반환.
    fn = {"qed": qed, "SAS": sas, "logP": logp, "mr": mr}[name]
    smiles = dataset.smiles.tolist()  # dataset['smiles'].tolist() 해당 데이터셋 컬럼의 values 리스트형태로 변환
    mols = mols_from_smiles(smiles)  # smiles 2 molecule
    pjob = Parallel(n_jobs=n_jobs, verbose=0)  # 병렬 연산.
    prop = pjob(delayed(fn)(mol) for mol in mols)
    new_data = pd.DataFrame(prop, columns=[name])
    return pd.concat([dataset, new_data], axis=1, sort=False)
