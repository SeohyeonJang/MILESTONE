import time
import numpy as np
import pandas as pd

from molecules.conversion import mol_from_smiles
from molecules.properties import add_property
from molecules.structure import (
    add_atom_counts, add_bond_counts, add_ring_counts)

from .config import get_dataset_info
from .filesystem import load_dataset

SCORES = ["validity", "novelty", "uniqueness"]


def dump_scores(config, scores, epoch):  # best uniquness 점수를 저장하기위함.
    filename = config.path('performance') / "scores.csv"
    df = pd.DataFrame([scores], columns=SCORES)

    if not filename.exists():  # filename csv 파일 없으면, 생성 
        df.to_csv(filename)
        is_max = True
    else:  # 있으면 csv 파일 불러와서 새로들어온 uniqueness score가 기존csv best score 이상인지 판단 True or False 반환, 스코어는 기존 csv에 합쳐서 저장.
        ref = pd.read_csv(filename, index_col=0)
        is_max = scores[2] >= ref.uniqueness.max()
        ref = pd.concat([ref, df], axis=0, sort=False, ignore_index=True)
        ref.to_csv(filename)

    return is_max


def retrieve_samples(config):  # samples path에 해당하는 csv 파일들을 합쳐서 반환.
    dfs = []
    filenames = config.path('samples').glob('*_*.csv')  # 해당 경로의 csv파일들 

    for filename in filenames:
        dfs.append(pd.read_csv(filename, index_col=0))

    samples = pd.concat(dfs, axis=0, ignore_index=True, sort=False)
    samples = samples.reset_index(drop=True)
    return samples.copy()


def mask_valid_molecules(smiles):  # validity, smiles에서 molecule로 변환, 해당 구조가 적합하면 molecule 아니면 False.
    valid_mask = []

    for smi in smiles:
        try:
            mol = mol_from_smiles(smi)
            valid_mask.append(mol is not None)
        except Exception:
            valid_mask.append(False)

    return np.array(valid_mask)


def mask_novel_molecules(smiles, data_smiles):  # novelty, 해당 smiles가 dataset에 없으면 append
    novel_mask = []

    for smi in smiles:
        novel_mask.append(smi not in data_smiles)

    return np.array(novel_mask)


def mask_unique_molecules(smiles):  # uniqueness, smiles가 중복되지 않으면 unique_mask append, uniques는 set타입이라 그냥 add
    uniques, unique_mask = set(), []

    for smi in smiles:
        unique_mask.append(smi not in uniques)
        uniques.add(smi)

    return np.array(unique_mask)


def score_samples(samples, dataset, calc=True):  # validity, novelty, uniqueness 체크, 3개 곱, 리스트
    def ratio(mask):  # validity, novelty, uniqueness 점수 반환용.
        total = mask.shape[0]
        if total == 0:
            return 0.0
        return mask.sum() / total  # validity는 전체중 유효한 값, novelty는 dataset에 없는 분자, uniqueness는 생성분자 미중복 .

    if isinstance(samples, pd.DataFrame):  # 들어온 samples 형태가 dataframe 일때,
        smiles = samples.smiles.tolist()   # samples의 smiles를 리스트로 변환.
    elif isinstance(samples, list):  # 들어온 samples 형태가 list 일때,
        smiles = [s[0] for s in samples]  # samples의 smiles를 리스트로
    data_smiles = dataset.smiles.tolist()  # dataset의 smiles 를 list로

    valid_mask = mask_valid_molecules(smiles)  # valid, smiles -> molecule 변환되는 molecule list 반환 
    novel_mask = mask_novel_molecules(smiles, data_smiles)  # dataset에 없는 smiles 반환
    unique_mask = mask_unique_molecules(smiles)  # 생성된 분자들 중복제거해서 smiles 반환

    scores = []
    if calc:  # default=True, 3개 점수 계산, train 단계에서 validate_after(기준 loss)를 이하로 내려갈 때, sampling 마지막.
        start = time.time()  # 시작 시점, 현재 시간을 초단위로 반환
        print("Start scoring...")
        validity_score = ratio(valid_mask)  # 각 스코어 표시
        novelty_score = ratio(novel_mask[valid_mask])
        uniqueness_score = ratio(unique_mask[valid_mask])

        print(f"valid: {validity_score} - "
              f"novel: {novelty_score} - "
              f"unique: {uniqueness_score}")

        scores = [validity_score, novelty_score, uniqueness_score]  # 계산된 스코어 저장
        end = time.time() - start  # 계산 끝, time.time 현재 시간을 초단위 반환.
        elapsed = time.strftime("%H:%M:%S", time.gmtime(end))  # time.strftime(가져올 값지정(str), 각 형식으로 시간 값), 입력받은 시간값 중 원하는 값을 출력.
        print(f'Done. Time elapsed: {elapsed}.')               # time.gmtime(시간) ,입력받은 시간을 각 형식으로 변환해서 반환.
                                                               # 형식: 연,달,일,시,분,초,요일,연중 경과일
    return valid_mask * novel_mask * unique_mask, scores


def postprocess_samples(config, use_train=False, n_jobs=-1): # 샘플링된 파일들을 합쳐서 atom, bond, ring 데이터와 스코어계산까지
    start = time.time()
    print("Start postprocessing...", end=" ")
    kind = 'train' if use_train else 'test'
    dataset = load_dataset(config, kind=kind)
    samples = retrieve_samples(config)

    mask, _ = score_samples(samples, dataset, calc=False) 
    samples = samples.iloc[mask, :].reset_index(drop=True)

    info = get_dataset_info(config.get('dataset'))
    samples = add_atom_counts(samples, info, n_jobs)  # C,F,N,O,Other
    samples = add_bond_counts(samples, info, n_jobs)  # SINGLE, DOUBLE, TRIPLE
    samples = add_ring_counts(samples, info, n_jobs)  # Tri, Quad, Pent, Hex

    for prop in info['properties']:  # info['properties'] = qed, logP, SAS, mr
        samples = add_property(samples, prop, n_jobs)

    samples = samples[info['column_order']]
    samples['who'] = 'OURS'
    dataset['who'] = info['name']

    data = [samples, dataset]
    aggregated = pd.concat(data, axis=0, ignore_index=True, sort=False)
    aggregated.to_csv(config.path('samples') / 'aggregated.csv')

    end = time.time() - start
    elapsed = time.strftime("%H:%M:%S", time.gmtime(end))
    print(f'Done. Time elapsed: {elapsed}.')
