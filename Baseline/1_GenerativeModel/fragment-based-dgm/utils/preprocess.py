import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed

from molecules.conversion import (
    mols_from_smiles, mol_to_smiles, mols_to_smiles, canonicalize)
from molecules.fragmentation import fragment_iterative, reconstruct
from molecules.properties import add_property
from molecules.structure import (
    add_atom_counts, add_bond_counts, add_ring_counts)
from utils.config import DATA_DIR, get_dataset_info


def fetch_dataset(name):  # dataset 다운로드 부분.
    info = get_dataset_info(name)  # 넘어온 dataset의 정보 확인
    filename = Path(info['filename'])  # 해당 dataset 이름.
    url = info['url']  # url 받아오기.
    unzip = info['unzip']  # 해당 dataset 파일 unzip 가능 여부.

    folder = Path("./temp").absolute()  # 현재위치의 temp폴더 절대경로.
    if not folder.exists():  # temp 폴더가 없으면 생성.
        os.makedirs(folder)

    os.system(f'wget -P {folder} {url}')  # dataset 다운로드

    raw_path = DATA_DIR / name / 'RAW'  # dataset 다운로드 경로
    if not raw_path.exists():  # 'RAW' 폴더 없으면 생성
        os.makedirs(raw_path)

    processed_path = DATA_DIR / name / 'PROCESSED'  # train/ test 등으로 나뉘어서 저장되는 폴더
    if not processed_path.exists():
        os.makedirs(processed_path)

    path = folder / filename  #  temp folder + dataset name

    if unzip is True:  # 해당 dataset이 unzip 가능한 파일이면, 압축해제.
        if ".tar.gz" in info['url']:
            os.system(f'tar xvf {path}.tar.gz -C {folder}')
        elif '.zip' in info['url']:
            os.system(f'unzip {path.with_suffix(".zip")} -d {folder}')
        elif '.gz' in info['url']:
            os.system(f'gunzip {path}.gz')

    source = folder / filename  # temp + dataset name
    dest = raw_path / filename  # raw + dataset name

    shutil.move(source, dest)  # source -> dest로 파일이나 폴더 위치 옮기기.
    shutil.rmtree(folder)  # 폴더 삭제(휴지통X 영구삭제O)


def break_into_fragments(mol, smi):  # smiles, fragments, n_fragments
    frags = fragment_iterative(mol)

    if len(frags) == 0:
        return smi, np.nan, 0

    if len(frags) == 1:
        return smi, smi, 1

    rec, frags = reconstruct(frags)
    if rec and mol_to_smiles(rec) == smi:
        fragments = mols_to_smiles(frags)
        return smi, " ".join(fragments), len(frags)

    return smi, np.nan, 0


def read_and_clean_dataset(info):
    raw_path = DATA_DIR / info['name'] / 'RAW'

    if not raw_path.exists():  # 경로에 해당 dataset의 RAW data가 없으면,
        fetch_dataset(info['name'])  # dataset download

    dataset = pd.read_csv(  # csv 파일 읽기, index_col = False
        raw_path / info['filename'],
        index_col=info['index_col'])


    if info['drop'] != []:
        dataset = dataset.drop(info['drop'], axis=1)  # zinc: drop=[], PCBA: drop=[mol_id]

    if info['name'] == 'ZINC':
        dataset = dataset.replace(r'\n', '', regex=True)  # regex=True 일 경우 regex 문법사용, r'\n' 의 'n' 을 ''로 대체.

    if info['name'] == 'GDB17':
        dataset = dataset.sample(n=info['random_sample'])  # random하게 sample 가져올 갯수
        dataset.columns = ['smiles']

    if info['name'] == 'PCBA':
        cols = dataset.columns.str.startswith('PCBA')  # colums 중에 'PCBA' 라는 글자가 들어가는지 검사, 반환 [True, False, True] 같은 형식
        dataset = dataset.loc[:, ~cols]  # PCBA dataset은 여러 0과 1로 표현된 column 들이 있는데, 각 column 이름이 PCBA-1030 이런식. 그런 columns 제외한 값들만 => mol_id 와 smiles
        dataset = dataset.drop_duplicates()  # 중복 제거
        dataset = dataset[~dataset.smiles.str.contains("\.")]  # . 있는 smiles 제외

    if info['name'] == 'QM9':
        correct_smiles = pd.read_csv(raw_path / 'qm9_ext.csv')
        dataset.smiles = correct_smiles.smiles
        dataset = dataset.sample(frac=1, random_state=42)  # frac(fraction) 일정 비율로 sample 가져오기, 0~1

    if info['name'] == 'PQC':
        dataset.rename(columns={'Can_SMILES':'smiles'}, inplace=True)
        correct_smiles = pd.read_csv(raw_path / 'PubChemQC_93683.csv')
        correct_smiles.rename(columns={'Can_SMILES':'smiles'}, inplace=True)
        dataset.smiles = correct_smiles.smiles
        dataset = dataset.sample(frac=1, random_state=42)  # frac(fraction) 일정 비율로 sample 가져오기, 0~1
    
    smiles = dataset.smiles.tolist()  # dataset의 smiles 를 list로
    dataset.smiles = [canonicalize(smi, clear_stereo=True) for smi in smiles]  # canonicalize 하면서, stereochemistry 정보 제거
    dataset = dataset[dataset.smiles.notnull()].reset_index(drop=True)  # smiles에 null 이 아닌값들만 + index 재정렬

    return dataset


def add_fragments(dataset, info, n_jobs):
    smiles = dataset.smiles.tolist()
    mols = mols_from_smiles(smiles)
    #print('1:',smiles)
    #print('2:', mols)
    pjob = Parallel(n_jobs=n_jobs, verbose=1)  # 병렬 프로그래밍 n_jobs 만큼, verbose=진행 메시지 출력
    fun = delayed(break_into_fragments)
    results = pjob(fun(m, s) for m, s in zip(mols, smiles))
    smiles, fragments, lengths = zip(*results)
    dataset["smiles"] = smiles  # smiles
    dataset["fragments"] = fragments  # fragments
    dataset["n_fragments"] = lengths  # fragments lengths

    return dataset


def save_dataset(dataset, info):  # PROCESSED .smi 저장.
    dataset = dataset[info['column_order']]
    testset = dataset[dataset.fragments.notnull()]
    trainset = testset[testset.n_fragments >= info['min_length']]  # default 2
    trainset = trainset[trainset.n_fragments <= info['max_length']]  # default 10
    processed_path = DATA_DIR / info['name'] / 'PROCESSED'
    trainset.to_csv(processed_path / 'train.smi')
    dataset.to_csv(processed_path / 'test.smi')


def preprocess_dataset(name, n_jobs):
    info = get_dataset_info(name)  # 해당 데이터셋 json 파일 정보 가져오기
    dataset = read_and_clean_dataset(info)  # 데이터셋에서 smiles 컬럼 부분만 가져오기
    dataset = add_atom_counts(dataset, info, n_jobs)  # smiles 데이터셋에 atoms, bonds, rings 각각 몇개씩 존재하는지 데이터셋으로 추가
    dataset = add_bond_counts(dataset, info, n_jobs)
    dataset = add_ring_counts(dataset, info, n_jobs)

    for prop in info['properties']:
        if prop not in dataset.columns:  # smiles, atoms, bonds, rings ?
            dataset = add_property(dataset, prop, n_jobs)  # 없으면 추가

    dataset = add_fragments(dataset, info, n_jobs)  # fragments and n_fragments 추가

    save_dataset(dataset, info)
