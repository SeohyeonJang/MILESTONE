import torch
import pickle
import torch.utils.data
import time
import os
import os.path as osp
import csv
import dgl
from tqdm import tqdm
import sys

sys.path.append("..")
from utils.basis_transform import basis_transform
from dgl.data.utils import load_graphs, save_graphs


class MoleculeDGL(torch.utils.data.Dataset): # 그래프 데이터 및 레이블을 로드하거나 생성
    def __init__(self, data_dir, split, num_graphs, basis, epsilon, power, edgehop, degs):
        self.data_dir = data_dir # data_dir = './pickle_data/1_homo/seed100'
        self.split = split # train, val, test
        self.num_graphs = num_graphs
        self.pre_processed_file_path = osp.join(data_dir, '%s_dgl_data_processed' % self.split)
        self.labels_file_path = osp.join(data_dir, '%s_labels' % self.split)

        self.basis = basis # DEN
        self.epsilon = epsilon
        self.power = power
        self.edgehop = edgehop
        self.degs = degs

        self.graph_lists = []
        self.graph_labels = []
        self._prepare()

    def _prepare(self): # preprocessed file이 있으면 로드
        if os.path.exists(self.pre_processed_file_path):
            print("Loading the cached file for the %s set... (NOTE: delete it if you change the preprocessing settings)" % (self.split.upper()))
            print("Loading the cached file for the %s set... (NOTE: delete it if you change the preprocessing settings)" % (self.split.upper()))
            self.graph_lists = load_graphs(self.pre_processed_file_path) # 그래프 리스트 로드
            self.graph_labels = torch.load(self.labels_file_path) # 라벨 로드

            assert len(self.graph_lists[0]) == self.num_graphs, "Sample num_graphs again; available idx: train/val/test => 10k/1k/1k"
            assert len(self.graph_labels) == self.num_graphs, "Sample num_graphs again; available idx: train/val/test => 10k/1k/1k"

        else: # 없으면 생성
            print("Generating %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))

            with open(self.data_dir + "/%s.pickle" % self.split, "rb") as f:
                # with open('./pickle_data/1_homo/seed100/train_100.pkl', "rb") as f:

                data = pickle.load(f)

            # 추출할 분자 개수만큼의 인덱스를 리스트 형태로 만들어야 함.
            # 모든 데이터를 사용할거면 아래 코드 주석처리 하면 됨.
            # with open(self.data_dir + "/%s.index" % self.split, "r") as f:
            #     data_idx = [list(map(int, idx)) for idx in csv.reader(f)]
            #     data = [data[i] for i in data_idx[0]]

            assert len(data) == self.num_graphs

            """
            data is a list of Molecule dict objects with following attributes

              molecule = data[idx]
            ; molecule['num_atom'] : nb of atoms, an integer (N)
            ; molecule['atom_type'] : tensor of size N, each element is an atom type, an integer between 0 and num_atom_type
            ; molecule['bond_type'] : tensor of size N x N, each element is a bond type, an integer between 0 and num_bond_type
            ; molecule['property'] : the chemical property to regress, a float variable
            """

            # trans_start = time.time()
            for step, molecule in enumerate(tqdm(data, desc="Pre-processing")):
                node_features = molecule['atom_type'].long() # atom 타입이 노드 feature

                adj = molecule['bond_type'] # bond 타입이 adj matrix
                edge_list = (adj != 0).nonzero(as_tuple=True)  # converting adj matrix to edge_list

                edge_features = adj[edge_list].reshape(-1).long()

                # Create the DGL Graph
                # dgl.graph(num_edges, num_nodes)
                g = dgl.graph(edge_list, num_nodes=molecule['num_atom'])
                # 위에서 생성된 node, edge features를 생성된 graph 내에 key 'feat'에 value로 저장
                g.ndata['feat'] = node_features
                g.edata['feat'] = edge_features

                g = dgl.remove_self_loop(g)
                g = dgl.add_self_loop(g)
                # assert (g.edata['feat'].shape[0] == _num_edges + g.num_nodes())
                g = basis_transform(g, basis=self.basis, epsilon=self.epsilon, power=self.power, edgehop=self.edgehop, degs=self.degs)

                self.graph_lists.append(g)
                self.graph_labels.append(molecule['homo'])

            # total_time = time.time() - trans_start
            print('Saving...')
            save_graphs(self.pre_processed_file_path, self.graph_lists)
            torch.save(self.graph_labels, self.labels_file_path)

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.num_graphs

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]


class MoleculeDataset(torch.utils.data.Dataset):
    def __init__(self, name='PubChemQC', config=None):
        t0 = time.time()
        self.name = name

        self.num_atom_type = 8
        self.num_bond_type = 4

        data_dir = './pickle_data/1_homo/seed100'

        basis = config.basis
        epsilon = config.epsilon
        power = config.power
        edgehop = config.get('edgehop', None)
        degs = config.get('degs', [])
        print('Basis configurations: basis: {}, epsilon: {}, power: {}, degs: {}'.format(basis, epsilon, power, degs))

        self.train = MoleculeDGL(data_dir, 'train', num_graphs=17878, basis=basis, epsilon=epsilon, power=power,
                                 edgehop=edgehop, degs=degs)
        self.val = MoleculeDGL(data_dir, 'val', num_graphs=2235, basis=basis, epsilon=epsilon, power=power,
                               edgehop=edgehop, degs=degs)
        self.test = MoleculeDGL(data_dir, 'test', num_graphs=2235, basis=basis, epsilon=epsilon, power=power,
                                edgehop=edgehop, degs=degs)

        print('train, test, val sizes :', len(self.train), len(self.test), len(self.val))
        print("Time taken: {:.4f}s".format(time.time() - t0))

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.stack(labels)
        # labels = torch.tensor(np.array(labels)).unsqueeze(1)
        batched_graph = dgl.batch(graphs)
        return batched_graph, labels

