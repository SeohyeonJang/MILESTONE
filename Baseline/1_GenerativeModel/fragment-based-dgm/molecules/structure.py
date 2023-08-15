import copy
import pandas as pd

from collections import OrderedDict
from joblib import Parallel, delayed
from rdkit import Chem

from .conversion import mols_from_smiles


def _ordered_dict(lst):  # list 형태의 데이터를 받아와서, 리스트의 각 값에 [0] 할당해서 dict형태로 반환.
    return OrderedDict(zip(lst, [0] * len(lst)))


def count_atoms(mol, atomlist):  # molecule, atomlist 입력받고, 
    count = _ordered_dict(atomlist)
    if mol:  # molecule 이 있으면,
        for atom in mol.GetAtoms():  # mol의 Atoms 반환.
            symbol = atom.GetSymbol()  # Atoms의 atom 을 Symbol 로 반환(원소 기호).
            if symbol not in count:  # 없던 원소면, "Other"에 + 1
                count["Other"] += 1
            else:
                count[symbol] += 1  # 있으면, 해당 원소 + 1, zinc 데이터셋의 경우 C,F,N,O,Other
    return count


def count_bonds(mol, bondlist):  # 위와 같이 BondType 이 있으면 해당 타입에 +1
    count = _ordered_dict(bondlist)
    if mol:
        mol = copy.deepcopy(mol)
        Chem.Kekulize(mol, clearAromaticFlags=True)  # clearAromaticFlags = 비방향족으로 표시
        for bond in mol.GetBonds():
            count[str(bond.GetBondType())] += 1  # SINGLE, DOUBLE, TRIPLE
    return count


def count_rings(mol, ringlist):  # 분자에서 각 크기에 따른 링을 몇개씩 가지고 있는지.
    ring_sizes = {i: r for (i, r) in zip(range(3, 7), ringlist)}  # 3: Tri, 4: Quad, 5: Pent, 6: Hex
    count = _ordered_dict(ringlist)  # ring_size인 3, 4, 5, 6 와 Tri, Quad, Pent, Hex 를 각각 매칭.
    if mol:
        ring_info = Chem.GetSymmSSSR(mol)  #  symmetrized(Symm) smallest set of smallest rings (SSSR) / 링 갯수와 분자정보를 담고있음.
        for ring in ring_info:  # ring_info 에서 각 링에 대한 정보를 순서대로 가져옴.
            ring_length = len(list(ring))  # 링에 포함된 원자 갯수인 ring_size 를 반환
            if ring_length in ring_sizes:
                ring_name = ring_sizes[ring_length]  # ring_name = Tri, Quad, Pent, Hex
                count[ring_name] += 1  # 해당하는 링 이름에 +1
    return count


def _add_counts(dataset, fn, names, n_jobs):
    smiles = dataset.smiles.tolist()
    mols = mols_from_smiles(smiles)
    pjob = Parallel(n_jobs=n_jobs, verbose=0)
    counts = pjob(delayed(fn)(mol, names) for mol in mols)  # count_atoms or bonds or rings 실행
    return pd.concat([dataset, pd.DataFrame(counts)], axis=1, sort=False)  # smiles 데이터셋에 count_atoms or bonds or rings 데이터 추가.


def add_atom_counts(dataset, info, n_jobs):  # 각각 count 실행해서 몇개나 들었는지 데이터를 추가해서 반환
    return _add_counts(dataset, count_atoms, info['atoms'], n_jobs)


def add_bond_counts(dataset, info, n_jobs):
    return _add_counts(dataset, count_bonds, info['bonds'], n_jobs)


def add_ring_counts(dataset, info, n_jobs):
    return _add_counts(dataset, count_rings, info['rings'], n_jobs)
