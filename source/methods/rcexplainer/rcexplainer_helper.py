import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import dense_to_sparse

import torch.optim as optim


class Struct_BB():
    def __init__(self, basis_=None, bias_=None):
        self.bb = None

        if (bias_ is None):
            assert (basis_ is None)
        else:
            assert (basis_ is not None)
            self.importBB(basis_, bias_)

    def importBB(self, basis_, bias_):
        assert (isinstance(basis_, np.ndarray))
        assert (isinstance(bias_, np.ndarray))
        assert (basis_.shape.__len__() == 2)
        assert (bias_.shape.__len__() == 1)

        if self.bb is None:
            self.bb = np.concatenate((basis_, bias_.reshape(1, -1)), axis=0)
        else:
            new_bb = np.concatenate((basis_, bias_.reshape(1, -1)), axis=0)
            self.bb = np.concatenate((self.bb, new_bb), axis=1)

    def extendBB(self, struct_bb_):
        assert (isinstance(struct_bb_, Struct_BB))
        if self.bb is None:
            self.bb = struct_bb_.bb
        else:
            self.bb = np.concatenate((self.bb, struct_bb_.bb), axis=1)

    def getBiasArray(self):
        return self.bb[-1, :]

    def getBasisArray(self):
        return self.bb[:-1, :]

    def getBBArray(self):
        return self.bb

    # userful for extracting bb for pooling layers and last layer
    def subPivotOverOthers(self, pivot_id1_, pivot_id2_):
        # self.bb = self.bb[:, pivot_id_:pivot_id_+1] - self.bb
        # self.bb = np.delete(self.bb, pivot_id_, axis=1)
        self.bb = self.bb[:, pivot_id1_:pivot_id1_ + 1] - self.bb[:, pivot_id2_:pivot_id2_ + 1]

    def getSizeOfBB(self):
        if self.bb is None:
            return 0
        else:
            return self.getBiasArray().shape[0]

    def computeHashVal(self, array_features_):
        assert (isinstance(array_features_, np.ndarray))

        expanded_feature = np.concatenate((array_features_, np.ones((array_features_.shape[0], 1))), axis=1)
        hashvals = np.matmul(expanded_feature, self.bb)

        return hashvals

    def computeConfigs(self, array_features_):
        assert (isinstance(array_features_, np.ndarray))

        hashvals = self.computeHashVal(array_features_)
        configs = np.zeros(hashvals.shape)
        configs[hashvals > 0] = 1

        return configs

    def computeSubHashVal(self, invariant_, array_features_):
        assert (isinstance(array_features_, np.ndarray))
        assert (np.sum(invariant_) > 0)

        expanded_feature = np.concatenate((array_features_, np.ones((array_features_.shape[0], 1))), axis=1)

        hashvals = np.matmul(expanded_feature, self.bb[:, invariant_])

        return hashvals

    def computeSubConfigs(self, invariant_, array_features_):
        assert (isinstance(array_features_, np.ndarray))
        assert (np.sum(invariant_) > 0)

        hashvals = self.computeSubHashVal(invariant_, array_features_)
        configs = np.zeros(hashvals.shape)
        configs[hashvals > 0] = 1

        return configs


class RuleMinerLargeCandiPool():
    def __init__(self, model_, train_mdata_, train_preds_, train_embs_, label_, device, layer_start_=-1):
        print("rule label: ", label_)

        self._model = model_
        self._train_mdata = train_mdata_
        self._train_labels = train_preds_
        self._train_embs = train_embs_
        self.device = device

        # The class of the seed image is considered positive
        # Rest of the classes are negative
        self._indices_list_pos = []
        self._indices_list_neg = []

        self._label = label_

        for idx in range(len(train_mdata_)):

            if self._train_labels[idx] == self._label:
                self._indices_list_pos.append(idx)
            else:
                self._indices_list_neg.append(idx)

        self._pos_sample = len(self._indices_list_pos)
        self._neg_sample = len(self._indices_list_neg)
        print("rule: pos samples", self._pos_sample)
        print("rule: neg samples", self._neg_sample)

        self._layer_start = layer_start_

    def CandidatePoolLabel(self, feature_, pool=50):

        self._bb = Struct_BB()

        boundary_list = []
        opposite_list = []
        initial_list = []

        self._model.eval()

        random_set = np.random.choice(self._neg_sample, pool, replace=False)

        for i in range(pool):

            # if i % 100 == 0:
            #     print('Extracting Candidate Pool ', i)

            neg_index = self._indices_list_neg[random_set[i]]
            # print("neg index: ", neg_index)

            pos = feature_
            #             vec = self._model._getOutputOfOneLayer(pos).cpu().detach().numpy()
            #             vec = self._model.fc(pos).cpu().detach().numpy()
            # print("pos: ", vec)

            # print("negindex: ", neg_index, self._train_labels[neg_index])
            neg = self._train_embs[neg_index].float().unsqueeze(0).to(self.device)  # self._train_mdata.getCNNFeature(neg_index).cuda()
            #             vec = self._model.fc(pos).cpu().detach().numpy()
            # vec = self._model._getOutputOfOneLayer(neg).cpu().detach().numpy()
            # print("neg: ", vec)
            initial_list.append(neg_index)
            ###
            while True:
                # Adjusted binary search
                boundary_pt = 0.9 * pos + 0.1 * neg

                # Output of the boundary point
                vec = self._model.fc(boundary_pt).cpu().detach().numpy()  # self._model._getOutputOfOneLayer(boundary_pt).cpu().detach().numpy()

                vec_order = np.argsort(vec[0])
                out1 = vec_order[-1]  # index of the largest element, which is the output
                out2 = vec_order[-2]  # index of the second largest element

                if (vec[0][out1] - vec[0][out2]) ** 2 < 0.00001 and out1 == self._label:
                    break
                # print((vec[0][out1] - vec[0][out2]) ** 2)

                if out1 == self._label:
                    pos = boundary_pt
                else:
                    neg = boundary_pt

            boundary_list.append(boundary_pt)
            opposite_list.append(out2)

            ###

            #             bb_buf = self._model._getBBOfLastLayer(boundary_pt, self._layer_start + 1)
            bb_buf = _getBBOfLastLayer(self._model, self.device, boundary_pt)
            self._bb.extendBB(bb_buf)

            array_features = self._train_embs.float().reshape(self._train_embs.size(0), -1).cpu().numpy()
            self._train_configs = self._bb.computeConfigs(array_features)

        self._boundary_list = boundary_list
        self._opposite_list = opposite_list
        self._initial_list = initial_list

    def _getCoveredIndi(self, invariant_, tgt_config_):
        num_boundaries = np.sum(invariant_)
        array_features_ = self._train_embs.float().reshape(self._train_embs.size(0), -1).cpu().numpy()  # self._train_mdata._getArrayFeatures()

        configs = self._bb.computeSubConfigs(invariant_, array_features_)
        match_mat = (configs - tgt_config_ != 0)

        if num_boundaries > 1:
            check_sum = np.sum(match_mat, axis=1)
        else:
            check_sum = match_mat.squeeze()

        cover_indi = (check_sum == 0)

        return cover_indi

    def getInvariantClassifier(self, tgt_idx_, feature, label, org_train_labels_, delta_constr_=0, peel_mask_=None):  # tgt_idx_ is probably not used
        train_configs = self._train_configs

        feature = feature.flatten()
        feature = feature[np.newaxis, :]
        tgt_config = self._bb.computeConfigs(feature)
        tgt_config = np.squeeze(tgt_config)

        train_labels = self._train_labels.cpu()
        tgt_label = label

        indi_f = (train_labels == tgt_label)
        indi_g = ~indi_f

        if peel_mask_ != None:
            indi_f[peel_mask_] = False

        match_mat_f = (train_configs[indi_f, :] - tgt_config == 0)
        match_mat_g = (train_configs[indi_g, :] - tgt_config == 0)

        from methods.rcexplainer.dnn_invariant.algorithms.mine_gcn_invariant import SubmodularMinerAG, InvariantClassifierGlb

        submodu_miner = SubmodularMinerAG(match_mat_f, match_mat_g, np.where(indi_f), np.where(indi_g), False)
        invariant, f_val, g_val = submodu_miner.mineInvariant(delta_constr_=delta_constr_)

        cover_indi = self._getCoveredIndi(invariant, tgt_config[invariant])

        # print("invariant: ", self._bb.bb.shape, invariant.shape)
        #
        # print("cover indi: ", cover_indi.shape, tgt_label, org_train_labels_[cover_indi][:10])
        # print(invariant)
        # print(self._bb.bb[:,0])
        # exit()
        org_train_subdata = self._train_embs.float().reshape(self._train_embs.size(0), -1).cpu().numpy()[cover_indi, :]
        # self._train_mdata._getArrayFeatures()[cover_indi, :]
        org_train_sublabels = org_train_labels_[cover_indi]

        array_features = self._train_embs.float().reshape(self._train_embs.size(0), -1).cpu().numpy()
        inv_classifier = InvariantClassifierGlb(self._bb, invariant, f_val, g_val, tgt_config, tgt_label,
                                                org_train_subdata, org_train_sublabels,
                                                array_features[tgt_idx_, :], self._layer_start, self._model)

        boundary_list_update = []
        opposite_list_update = []
        initial_list_update = []
        for i in range(len(self._boundary_list)):
            if invariant[i]:
                boundary_list_update.append(self._boundary_list[i])
                opposite_list_update.append(self._opposite_list[i])
                initial_list_update.append(self._initial_list[i])

        return inv_classifier, boundary_list_update, opposite_list_update, initial_list_update


def evalSingleRule(inv_classifier, data, embs, preds):
    pred_labels, cover_indi = inv_classifier.classify(embs.float().reshape(embs.size(0), -1).cpu().numpy())  # mdata._getArrayFeatures())
    label_diff = pred_labels[cover_indi] - preds.long().cpu().numpy()[cover_indi]  # mdata._getArrayLabels()[cover_indi]
    accuracy = np.sum(label_diff == 0) / label_diff.size
    return accuracy, cover_indi


def _getBBOfLastLayer(model, device, input):
    # prepare the buffer
    basis_list = []
    bias_list = []
    layer_ptr = -1

    # set true grad flag for input
    input.requires_grad_(True)

    out_of_layer = model.fc(input)

    out_of_layer = out_of_layer.reshape(out_of_layer.size(0), -1)

    model.zero_grad()
    model.eval()
    for idx in range(out_of_layer.size(1)):
        unit_mask = torch.zeros(out_of_layer.size())
        unit_mask[:, idx] = 1
        unit_mask = unit_mask.to(device)

        # compute basis of this unit
        out_of_layer.backward(unit_mask, retain_graph=True)
        basis = input.grad.clone().detach().reshape(input.size(0), -1)
        basis_list.append(basis)

        # do substraction to get bias
        basis_mul_x = torch.mul(input.clone().detach(), input.grad.clone().detach())
        # print("basis shape: ", basis_mul_x.shape)
        basis_mul_x = torch.sum(basis_mul_x, dim=1).to(device)
        bias = out_of_layer[:, idx].clone().detach() - basis_mul_x
        bias_list.append(bias)

        # clean up
        model.zero_grad()
        input.grad.data.zero_()

    # set false grad flag for input
    input.requires_grad_(False)

    # reshape basis to tensor shape
    stacked_basis = torch.stack(basis_list, dim=2)
    array_basis = stacked_basis.detach().squeeze().cpu().numpy()

    # reshape bias to tensor shape
    stacked_bias = torch.stack(bias_list, dim=1)
    array_bias = stacked_bias.detach().squeeze().cpu().numpy()

    # bb_of_logits = self._getBBOfOneLayer(input, -1, layer_start)
    bb_of_logits = Struct_BB(array_basis, array_bias)

    # identify the idx of the pivot logit
    logits = bb_of_logits.computeHashVal(input.reshape(input.size(0), -1).cpu().numpy())
    assert (logits.shape[0] == 1)
    logits = logits.squeeze()

    logits_order = np.argsort(logits)
    pivot_id1 = logits_order[-1]
    pivot_id2 = logits_order[-2]

    # pivot_id = np.argmax(logits)

    # subtract between the logits to get BB_of_last_layer
    bb_of_logits.subPivotOverOthers(pivot_id1, pivot_id2)

    return bb_of_logits


def build_optimizer(args, params, weight_decay=0.0):
    filter_fn = filter(lambda p: p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.4)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer


def check_counterfactuals(explanations, gnn_model, preds, adj, feat, num_nodes):
    all_counterfactuals = []
    real_counterfactuals = []
    empty_count = 0
    suf_count = 0
    size = 0
    ged = 0
    idx = 0
    for i, pred in enumerate(preds):
        if pred == 0:  # original graph
            explanation = explanations[idx]
            idx += 1
            num_node = num_nodes[i]
            x = feat[i][:num_node, :]
            a = adj[i][:num_node, :num_node]
            m = explanation[:num_node, :num_node]  # (torch.round(torch.FloatTensor(explanation[:num_node, :num_node]) * 100) / (100)).double()
            m = torch.from_numpy(m)

            m[m > 0.5] = 1
            m[m <= 0.5] = 0
            c = a - m

            if m.sum() == 0:
                empty_count += 1

            edge_index, edge_weight = dense_to_sparse(c)
            d = Data(edge_index=edge_index, x=x).cuda()
            loader = DataLoader([d], batch_size=1)
            n_embed, g_embed, res = gnn_model(next(iter(loader)), edge_weight=edge_weight.cuda())
            # res = output.exp()
            all_counterfactuals.append(d.cpu())
            if torch.argmax(res, 1)[0].item() == 1:
                real_counterfactuals.append(d.cpu())

            edge_index, edge_weight = dense_to_sparse(m)
            d = Data(edge_index=edge_index, x=x).cuda()
            loader = DataLoader([d], batch_size=1)
            n_embed, g_embed, res = gnn_model(next(iter(loader)), edge_weight=edge_weight.cuda())
            # res = output.exp()
            if torch.argmax(res, 1)[0].item() == 0:
                suf_count += 1

            size += m.sum() / a.sum()
            ged += m.sum() / (a.sum() + c.sum() + 4 * a.shape[0])

    return all_counterfactuals, real_counterfactuals, len(real_counterfactuals) / len(all_counterfactuals), empty_count, suf_count / len(all_counterfactuals), size / len(
        all_counterfactuals), ged / len(all_counterfactuals)


def train_explainer(explainer, model, rule_dict, adjs, feats, labels, preds, num_nodes, embs, node_embs, args, train_indices, val_indices, device):
    params_optim = []
    for name, param in explainer.named_parameters():
        params_optim.append(param)

    scheduler, optimizer = build_optimizer(args, params_optim)

    ep_count = 0
    loss_ep = 0

    epoch = -1
    best_val_loss = float('inf')
    patience = max(int(args.epochs / 5), 5)
    cur_patience = 0
    for epoch in tqdm(range(args.epochs)):
        loss_epoch = 0

        explainer.train()
        masked_adjs = []
        for b, graph_idx in enumerate(train_indices):
            optimizer.zero_grad()

            # preprocess inputs
            sub_adj = adjs[graph_idx]
            sub_nodes = num_nodes[graph_idx]
            sub_feat = feats[graph_idx]
            sub_label = labels[graph_idx]

            sub_adj = np.expand_dims(sub_adj, axis=0)
            sub_feat = np.expand_dims(sub_feat, axis=0)

            adj = torch.tensor(sub_adj, dtype=torch.float).to(device)
            x = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float).to(device)
            label = torch.tensor(sub_label, dtype=torch.long).to(device)

            # extract model embeddings from layer
            emb = node_embs[graph_idx].unsqueeze(0)
            emb = emb.clone().detach().to(device)

            gt_embedding = embs[graph_idx].to(device)

            # get boundaries for sample
            rule_ix = rule_dict['idx2rule'][graph_idx]
            rule = rule_dict['rules'][rule_ix]
            rule_label = rule['label']

            boundary_list = []
            for b_num in range(len(rule['boundary'])):
                boundary = torch.from_numpy(rule['boundary'][b_num]['basis'])
                boundary = boundary.to(device)
                boundary_label = rule['boundary'][b_num]['label']
                boundary_list.append(boundary)

            # explain prediction
            t0 = 0.5
            t1 = 4.99

            tmp = float(t0 * np.power(t1 / t0, epoch / args.epochs))
            pred, masked_adj, graph_embedding, inv_embedding, inv_pred = explainer((x[0], emb[0], adj[0], tmp, label, sub_nodes), device=device, training=True, gnn_model=model)
            loss, _ = explainer.loss(graph_embedding=graph_embedding, boundary_list=boundary_list, gt_embedding=gt_embedding, inv_embedding=inv_embedding)

            loss_ep += loss
            if ep_count < args.batch_size:
                ep_count += 1.0
                if b == len(train_indices) - 1:
                    ep_count = 0.
                    loss_ep.backward()
                    optimizer.step()
                    loss_epoch += loss_ep.detach()
                    loss_ep = 0.
            else:
                ep_count = 0.
                loss_ep.backward()
                optimizer.step()
                loss_epoch += loss_ep.detach()
                loss_ep = 0.

            # evaluate explanation
            masked_adj = masked_adj.cpu().detach().numpy() * sub_adj.squeeze()
            masked_adjs.append(masked_adj)

        if scheduler is not None:
            scheduler.step()

        # validation
        val_loss, _ = evaluator_explainer(explainer, model, rule_dict, adjs, feats, labels, preds, num_nodes, embs, node_embs, val_indices, device)
        val_loss = val_loss.item()

        if val_loss < best_val_loss:
            cur_patience = 0
            best_val_loss = val_loss
            torch.save(explainer.state_dict(), args.best_explainer_model_path)
        else:
            cur_patience += 1
            if cur_patience >= patience:
                break

    explainer.load_state_dict(torch.load(args.best_explainer_model_path, map_location=device))
    return explainer, epoch


@torch.no_grad()
def evaluator_explainer(explainer, model, rule_dict, adjs, feats, labels, preds, num_nodes, embs, node_embs, graph_indices, device):
    explainer.eval()
    masked_adjs = []
    total_loss = 0
    for graph_idx in graph_indices:

        # preprocess inputs
        sub_adj = adjs[graph_idx]
        sub_nodes = num_nodes[graph_idx]
        sub_feat = feats[graph_idx]
        sub_label = labels[graph_idx]

        sub_adj = np.expand_dims(sub_adj, axis=0)
        sub_feat = np.expand_dims(sub_feat, axis=0)

        adj = torch.tensor(sub_adj, dtype=torch.float).to(device)
        x = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float).to(device)
        label = torch.tensor(sub_label, dtype=torch.long).to(device)

        pred_label = preds[graph_idx]  # np.argmax(preds[0][graph_idx], axis=0)
        # extract model embeddings from layer
        emb = node_embs[graph_idx].unsqueeze(0)
        emb = emb.clone().detach().to(device)

        gt_embedding = embs[graph_idx].to(device)

        # get boundaries for sample
        rule_ix = rule_dict['idx2rule'][graph_idx]
        rule = rule_dict['rules'][rule_ix]
        rule_label = rule['label']

        boundary_list = []
        for b_num in range(len(rule['boundary'])):
            boundary = torch.from_numpy(rule['boundary'][b_num]['basis'])
            boundary = boundary.to(device)
            boundary_label = rule['boundary'][b_num]['label']
            boundary_list.append(boundary)

        # explain prediction
        t0 = 0.5
        t1 = 4.99

        tmp = float(t0 * np.power(t1 / t0, 1.0))
        pred, masked_adj, graph_embedding, inv_embedding, inv_pred = explainer((x[0], emb[0], adj[0], tmp, label, sub_nodes), device=device, training=False, gnn_model=model)
        loss, _ = explainer.loss(graph_embedding=graph_embedding, boundary_list=boundary_list, gt_embedding=gt_embedding, inv_embedding=inv_embedding)
        total_loss += loss

        # evaluate explanation
        masked_adj = masked_adj.cpu().detach().numpy() * sub_adj.squeeze()
        masked_adjs.append(masked_adj)

    explanations = masked_adjs
    return total_loss, explanations


class ExplainModule(nn.Module):
    def __init__(
            self,
            # model,
            num_nodes,
            emb_dims,
            device,
            args
    ):
        super(ExplainModule, self).__init__()
        self.device = device

        # self.model = model.to(self.device)
        self.num_nodes = num_nodes

        input_dim = np.sum(emb_dims)

        self.elayers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)

        rc = torch.unsqueeze(torch.arange(0, self.num_nodes), 0).repeat([self.num_nodes, 1]).to(torch.float32)
        # rc = torch.repeat(rc,[nodesize,1])
        self.row = torch.reshape(rc.T, [-1]).to(self.device)
        self.col = torch.reshape(rc, [-1]).to(self.device)

        self.softmax = nn.Softmax(dim=-1)

        self.mask_act = 'sigmoid'
        self.args = args

        # self.mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        # std = nn.init.calculate_gain("relu") * math.sqrt(
        #     2.0 / (num_nodes + num_nodes)
        # )
        # with torch.no_grad():
        #     self.mask.normal_(1.0, std)
        #     # mask.clamp_(0.0, 1.0)

        self.coeffs = {
            "size": 0.01,  # syn1, 0.009
            # "size": 0.012,#mutag
            # "size": 0.009,#synthetic graph
            # "size": 0.007, #0.04 for synthetic,#0.007=>40%,#0.06,#0.005,
            "feat_size": 0.0,
            "ent": 0.0,
            "feat_ent": 0.1,
            "grad": 0,
            "lap": 1.0,
            "weight_decay": 0,
            "sample_bias": 0
        }

    def _masked_adj(self, mask, adj):

        mask = mask.to(self.device)
        sym_mask = mask
        sym_mask = (sym_mask.clone() + sym_mask.clone().T) / 2

        # Create sparse tensor TODO: test and "maybe" a transpose is needed somewhere
        sparseadj = torch.sparse_coo_tensor(
            indices=torch.transpose(torch.cat([torch.unsqueeze(torch.Tensor(adj.row), -1), torch.unsqueeze(torch.Tensor(adj.col), -1)], dim=-1), 0, 1).to(torch.int64),
            values=adj.data,
            size=adj.shape
        )

        adj = sparseadj.coalesce().to_dense().to(torch.float32).to(self.device)  # FIXME: tf.sparse.reorder was also applied, but probably not necessary. Maybe it needs a .coalesce() too tho?
        self.adj = adj

        masked_adj = torch.mul(adj, sym_mask)

        num_nodes = adj.shape[0]
        ones = torch.ones((num_nodes, num_nodes))
        diag_mask = ones.to(torch.float32) - torch.eye(num_nodes)
        diag_mask = diag_mask.to(self.device)
        return torch.mul(masked_adj, diag_mask)

    def mask_density(self, adj):
        mask_sum = torch.sum(self.masked_adj).cpu()
        adj_sum = torch.sum(adj)
        return mask_sum / adj_sum

    def concrete_sample(self, log_alpha, beta=1.0, training=True):
        """Uniform random numbers for the concrete distribution"""

        if training:
            bias = self.coeffs['sample_bias']
            random_noise = bias + torch.FloatTensor(log_alpha.shape).uniform_(bias, 1.0 - bias)
            random_noise = random_noise.to(self.device)
            gate_inputs = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (gate_inputs.clone() + log_alpha) / beta
            gate_inputs = torch.sigmoid(gate_inputs)
        else:
            gate_inputs = torch.sigmoid(log_alpha)

        return gate_inputs

    def forward(self, inputs, device=None, training=None, gnn_model=None):
        x, embed, adj, tmp, label, sub_nodes = inputs
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.to(self.device)
        if not isinstance(adj, torch.Tensor):
            adj = torch.tensor(adj)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)
        adj = adj.to(self.device)
        # embed = embed.to('cpu')
        self.label = label
        self.tmp = tmp

        row = self.row.type(torch.LongTensor).to(self.device)  # ('cpu')
        col = self.col.type(torch.LongTensor).to(self.device)
        if not isinstance(embed[row], torch.Tensor):
            f1 = torch.Tensor(embed[row]).to(self.device)  # .to(self.device)  # <-- torch way to do tf.gather(embed, self.row)
            f2 = torch.Tensor(embed[col]).to(self.device)
        else:
            f1 = embed[row]  # .to(self.device)  # <-- torch way to do tf.gather(embed, self.row)
            f2 = embed[col]

        h = torch.cat([f1, f2], dim=-1)

        h = h.to(self.device)
        for elayer in self.elayers:
            h = elayer(h)

        self.values = torch.reshape(h, [-1])

        values = self.concrete_sample(self.values, beta=tmp, training=training)

        sparsemask = torch.sparse.FloatTensor(
            indices=torch.transpose(torch.cat([torch.unsqueeze(self.row, -1), torch.unsqueeze(self.col, -1)], dim=-1), 0, 1).to(torch.int64),
            values=values,
            size=[self.num_nodes, self.num_nodes]
        ).to(self.device)
        sym_mask = sparsemask.coalesce().to_dense().to(torch.float32)  # FIXME: again a reorder() is omitted, maybe coalesce

        self.mask = sym_mask
        sym_mask = (sym_mask + sym_mask.T) / 2

        masked_adj = torch.mul(adj, sym_mask)
        self.masked_adj = masked_adj

        inverse_mask = (1.0 - sym_mask)
        orig_adj = adj + 0.
        inverse_masked_adj = torch.mul(adj, inverse_mask)
        self.inverse_masked_adj = inverse_masked_adj

        x = torch.unsqueeze(x.detach().requires_grad_(True), 0).to(torch.float32)  # Maybe needs a .clone()
        adj = torch.unsqueeze(self.masked_adj, 0).to(torch.float32)
        x.to(self.device)
        if sub_nodes is not None:
            sub_num_nodes_l = [sub_nodes.cpu().numpy()]
        else:
            sub_num_nodes_l = None

        # graph mode
        from torch_geometric.data import Data, DataLoader
        from torch_geometric.utils import dense_to_sparse

        adj_i = adj[0][:sub_num_nodes_l[0]][:, :sub_num_nodes_l[0]]
        x_i = x[0][:sub_num_nodes_l[0], :]
        edge_index, edge_weight = dense_to_sparse(adj_i)
        d = Data(edge_index=edge_index, x=x_i).to(device)  #
        loader = DataLoader([d], batch_size=1)
        n_embed, g_embed, res = gnn_model(next(iter(loader)), edge_weight=edge_weight.to(device))

        inv_adj = torch.unsqueeze(self.inverse_masked_adj, 0).to(torch.float32)
        adj_i = inv_adj[0][:sub_num_nodes_l[0]][:, :sub_num_nodes_l[0]]
        x_i = x[0][:sub_num_nodes_l[0], :]
        edge_index, edge_weight = dense_to_sparse(adj_i)
        d = Data(edge_index=edge_index, x=x_i).to(device)
        loader = DataLoader([d], batch_size=1)
        n_inv_embed, inv_embed, inv_res = gnn_model(next(iter(loader)), edge_weight=edge_weight.to(device))

        return res, masked_adj, g_embed, inv_embed, inv_res

    def loss(self, graph_embedding, boundary_list, gt_embedding, inv_embedding):
        boundary_loss = 0.
        if self.args.lambda_ > 0:
            sigma = 1.0
            for boundary in boundary_list:
                gt_proj = torch.sum(gt_embedding * boundary[:20]) + boundary[20]
                ft_proj = torch.sum(graph_embedding * boundary[:20]) + boundary[20]
                boundary_loss += torch.sigmoid(-1.0 * sigma * (gt_proj * ft_proj))
            boundary_loss = self.args.lambda_ * (boundary_loss / len(boundary_list))

        inverse_boundary_loss = 0.
        if self.args.lambda_ < 1.0:
            sigma = 1.0
            inv_losses = []
            for boundary in boundary_list:
                gt_proj = torch.sum(gt_embedding * boundary[:20]) + boundary[20]
                inv_proj = torch.sum(inv_embedding * boundary[:20]) + boundary[20]
                inv_loss = torch.sigmoid(sigma * (gt_proj * inv_proj))
                inv_losses.append(inv_loss)
            inv_losses_t = torch.stack(inv_losses)
            inverse_boundary_loss = (1 - self.args.lambda_) * torch.min(inv_losses_t)

        net_boundary_loss = boundary_loss + inverse_boundary_loss

        # size
        mask = self.mask
        size_loss = self.args.beta_ * torch.sum(mask)  # len(mask[mask > 0]) #torch.sum(mask)

        # entropy
        mask = mask * 0.99 + 0.005  # maybe a .clone()
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = self.args.mu_ * torch.mean(mask_ent)

        loss = net_boundary_loss + size_loss + mask_ent_loss

        return loss, boundary_loss.item()
