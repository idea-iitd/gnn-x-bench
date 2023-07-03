import cv2
import numpy as np

H = 224
W = 224


class InvariantClassifierGlb():
    def __init__(self, bb_, invariant_, f_val, g_val, tgt_config_, tgt_label_, org_train_subdata, org_train_sublabels, tgt_feature, layer_start_, model_):
        assert np.sum(invariant_) > 0

        # self._tgt_feature = tgt_feature
        # self._min_dist = min_dist

        self._layer_start = layer_start_
        self._model = model_

        # assign self._bb that stores all decision boundaries
        self._bb = bb_
        self._invariant = invariant_

        self._gt_config = tgt_config_[invariant_]

        # log the f_val and g_val for tihs rule
        self._f_val = f_val
        self._g_val = g_val

        # assign label for this convex polytope
        self._label = tgt_label_

        if np.unique(org_train_sublabels).__len__() <= 1:
            self._is_pure = True
        else:
            self._is_pure = False

        '''
        # train a logistic regression
        if self._is_pure == False:
            #self.classifier = LogisticRegression()
            self.classifier = SVC(kernel='rbf')
            self.classifier.fit(org_train_subdata, org_train_sublabels)
        '''

        # print('subtrain labels: ', [(lb, np.sum(org_train_sublabels == lb)) for lb in set(org_train_sublabels)], '\n')

    def updateSupportScore(self, spt_score_):
        self._spt_score = spt_score_

    def getNumBoundaries(self):
        return np.sum(self._invariant)

    def classify_one_boundary(self, array_features_, labels_):
        for j in range(len(self._invariant)):

            if self._invariant[j] == 0:
                continue

            else:
                invariant = np.zeros(len(self._invariant), dtype=bool)
                invariant[j] = 1

                num_boundaries = self.getNumBoundaries()
                configs = self._bb.computeSubConfigs(invariant, array_features_)
                match_mat = (configs - self._gt_config != 0)
                if num_boundaries > 1:
                    check_sum = np.sum(match_mat, axis=1)
                else:
                    check_sum = match_mat.squeeze()
                cover_indi = (check_sum == 0)

            print(j, np.count_nonzero(labels_[cover_indi] == 0), np.count_nonzero(labels_[cover_indi] == 1), np.count_nonzero(labels_[cover_indi] == 2))

    def classify_one_boundary_specific(self, array_features_, db):

        count = 0
        for j in range(len(self._invariant)):

            if self._invariant[j] == 0:
                continue

            else:
                if count != db:
                    count += 1
                    continue
                else:
                    invariant = np.zeros(len(self._invariant), dtype=bool)
                    invariant[j] = 1

                    pixels = []
                    for pixel in range(196):
                        array = array_features_.copy()
                        for variable in range(512 * 14 * 14):
                            if variable % 196 != pixel:
                                array[0, variable] = 0
                        val = self._bb.computeSubHashVal(invariant, array)
                        pixels.append(val)
                    count += 1
        pixels = np.array(pixels)
        pixels = np.reshape(pixels, (14, 14))
        # pixels = np.maximum(pixels, 0)
        pixels = cv2.resize(pixels, (H, W))
        pixels = pixels - np.min(pixels)
        if np.max(pixels) != 0:
            pixels = pixels / np.max(pixels)

        return pixels

    def classify(self, array_features_):
        num_boundaries = self.getNumBoundaries()

        # dist = np.sqrt(np.sum(np.power((array_features_ - self._tgt_feature), 2), axis=1))

        configs = self._bb.computeSubConfigs(self._invariant, array_features_)
        match_mat = (configs - self._gt_config != 0)

        if num_boundaries > 1:
            check_sum = np.sum(match_mat, axis=1)
        else:
            check_sum = match_mat.squeeze()

        cover_indi = (check_sum == 0)  # & (dist <= self._min_dist)

        pred_labels = np.zeros(array_features_.shape[0]) - 1

        '''
        if np.sum(cover_indi) > 0:
            if self._is_pure == True:
                pred_labels[cover_indi] = self._label
            else:
                pred_labels[cover_indi] = self.classifier.predict(array_features_[cover_indi, :])

            pred_labels[cover_indi] = self._label
        '''

        pred_labels[cover_indi] = self._label

        return pred_labels, cover_indi


# Let OracleSP handle all the complicated computations
# SP stands for sparse
# such as maintaining precomputed statistics and computing marginal gains
class OracleSP():
    def __init__(self, match_mat_f_, match_mat_g_):
        self._match_mat_f = match_mat_f_
        self._match_mat_g = match_mat_g_

        assert match_mat_f_.shape[1] == match_mat_g_.shape[1]

        self._D = match_mat_f_.shape[1]
        self._f_N = match_mat_f_.shape[0]  # number of samples in all positive data
        self._g_N = match_mat_g_.shape[0]  # number of samples in all negative data
        self._N = self._f_N + self._g_N  # number of samples in all training data

        # Weighted
        self._f_N_weights = (np.sum(self._match_mat_f, axis=1) / self._D) ** 4
        self._g_N_weights = (np.sum(self._match_mat_g, axis=1) / self._D) ** 4
        self._f_N_weighted = np.sum(self._f_N_weights)
        self._g_N_weighted = np.sum(self._g_N_weights)

    # init the precomputed statistics
    def _init_precomp_stat(self):
        # init the buffer of merged cols for new_inv_Y=all zeros
        self._buf_ids_f = np.array(range(self._f_N))
        self._buf_ids_g = np.array(range(self._g_N))

    def _compute_nom_j(self, j_):
        if isinstance(self._match_mat_f, np.ndarray):
            matchmat_f_colj = self._match_mat_f[:, j_]
        else:
            matchmat_f_colj = np.asarray(self._match_mat_f[:, j_].todense()).squeeze()

        nom_j = np.sum(matchmat_f_colj[self._buf_ids_f] == False)

        assert nom_j >= 0

        return nom_j

    def _compute_denom_j(self, j_):
        if isinstance(self._match_mat_g, np.ndarray):
            match_mat_g_colj = self._match_mat_g[:, j_]
        else:
            match_mat_g_colj = np.asarray(self._match_mat_g[:, j_].todense()).squeeze()

        denom_j = np.sum(match_mat_g_colj[self._buf_ids_g] == False)

        assert denom_j >= 0

        return denom_j

    def _compute_ratio_vec(self):
        assert isinstance(self._match_mat_f, np.ndarray)
        assert isinstance(self._match_mat_g, np.ndarray)

        nom = np.sum(self._match_mat_f[self._buf_ids_f, :] == False, axis=0) + 1e-5
        denom = np.sum(self._match_mat_g[self._buf_ids_g, :] == False, axis=0) + 1e-10

        ratio_vec = nom / denom

        return ratio_vec

    def _update_by_j(self, sel_j_):
        # update precomputed statistics self._buf_ids_f
        if isinstance(self._match_mat_f, np.ndarray):
            match_mat_f_colj = self._match_mat_f[:, sel_j_]
        else:
            match_mat_f_colj = np.asarray(self._match_mat_f[:, sel_j_].todense()).squeeze()

        self._buf_ids_f = self._buf_ids_f[np.where(match_mat_f_colj[self._buf_ids_f])[0]]

        # update precomputed statistics self._buf_ids_g
        if isinstance(self._match_mat_g, np.ndarray):
            match_mat_g_colj = self._match_mat_g[:, sel_j_]
        else:
            match_mat_g_colj = np.asarray(self._match_mat_g[:, sel_j_].todense()).squeeze()

        self._buf_ids_g = self._buf_ids_g[np.where(match_mat_g_colj[self._buf_ids_g])[0]]

    def compute_gval(self):
        gval_Y = self._g_N - self._buf_ids_g.__len__()
        return gval_Y

    def compute_fval(self):
        fval_Y = self._f_N - self._buf_ids_f.__len__()
        return fval_Y

    def compute_fval_idx(self):
        return self._buf_ids_f

    # init the precomputed statistics
    def _init_precomp_stat_rce(self, old_inv_X_):
        # initializes all kinds of precomputed statistics
        # such as self._buf_colsum_f, self._buf_colnum_f, self._fval_X and self._buf_vec_g

        # init the buffer of all matching cols of old_inv_X_
        if np.sum(old_inv_X_) > 0:
            self._buf_colsum_f = np.sum(self._match_mat_f[:, old_inv_X_], axis=1)
            self._buf_colnum_f = np.sum(old_inv_X_)
            self._fval_X = self._f_N_weighted - np.dot(self._buf_colsum_f == self._buf_colnum_f, self._f_N_weights)
        else:
            self._fval_X = self._f_N_weighted

        entire_set = np.ones(self._match_mat_f.shape[1], dtype=bool)
        self._entire_colsum_f = np.sum(self._match_mat_f[:, entire_set], axis=1)
        self._entire_colnum_f = np.sum(entire_set)
        self._fval_V = self._f_N_weighted - np.dot(self._entire_colnum_f == self._entire_colsum_f, self._f_N_weights)

        # init the buffer of merged cols for new_inv_Y=all zeros
        self._buf_vec_g = np.ones(self._g_N, dtype=bool)  # all True

    def _compute_nom_j_rce(self, j_, old_inv_X_):
        nom_j = -1

        if old_inv_X_[j_] == True:
            # j in old_inv_X_
            buf_colsum_f_woj = self._buf_colsum_f - self._match_mat_f[:, j_]
            buf_colnum_f_woj = self._buf_colnum_f - 1
            fval_V_wo_j = self._f_N_weighted - np.dot(buf_colsum_f_woj == buf_colnum_f_woj, self._f_N_weights)
            nom_j = self._fval_X - fval_V_wo_j
        else:
            # j not in old_inv_X_
            if np.sum(old_inv_X_) > 0:
                buf_colsum_f_wj = self._buf_colsum_f + self._match_mat_f[:, j_]
                buf_colnum_f_wj = self._buf_colnum_f + 1
                fval_X_wj = self._f_N_weighted - np.dot(buf_colsum_f_wj == buf_colnum_f_wj, self._f_N_weights)
                nom_j = fval_X_wj - self._fval_X
            else:
                nom_j = self._f_N_weighted - np.dot(self._match_mat_f[:, j_], self._f_N_weights)  # Use it when X is empty

        assert nom_j >= -0.00001

        return nom_j

    def _compute_denom_j_rce(self, j_):
        gval_Y = self._g_N_weighted - np.sum(np.dot(self._buf_vec_g, self._g_N_weights))
        # gval_Y = self._g_N - np.sum(self._buf_vec_g)
        if isinstance(self._match_mat_g, np.ndarray):
            match_mat_g_colj = self._match_mat_g[:, j_]
        else:
            match_mat_g_colj = np.asarray(self._match_mat_g[:, j_].todense()).squeeze()

        gval_Y_w_j = self._g_N_weighted - np.sum(np.dot(self._buf_vec_g & match_mat_g_colj, self._g_N_weights))
        # gval_Y_w_j = self._g_N - np.sum(self._buf_vec_g & match_mat_g_colj)

        denom_j = (gval_Y_w_j - gval_Y) ** 1

        assert denom_j >= 0

        return denom_j

    def _compute_gval_mval_rce(self, old_inv_X_, new_inv_Y_):
        # compute g_val
        g_val = self._g_N - np.sum(self._buf_vec_g)  # the g_val of new_inv_Y (with sel_j)

        # compute m_val
        m_val = self._fval_X

        for _, j in np.ndenumerate(np.where(old_inv_X_ & ~new_inv_Y_)):
            if j.size > 0:
                m_val -= self._compute_nom_j_rce(j, old_inv_X_)

        for _, j in np.ndenumerate(np.where(new_inv_Y_ & ~old_inv_X_)):
            if j.size > 0:
                m_val += self._compute_nom_j_rce(j, old_inv_X_)

        return np.asscalar(g_val), np.asscalar(m_val)

    def _update_by_j_rce(self, sel_j_):
        # update precomputed statistics self._buf_vec_g
        if isinstance(self._match_mat_g, np.ndarray):
            match_mat_g_colj = self._match_mat_g[:, sel_j_]
        else:
            match_mat_g_colj = np.asarray(self._match_mat_g[:, sel_j_].todense()).squeeze()

        self._buf_vec_g = self._buf_vec_g & match_mat_g_colj

    def compute_fval_rce(self, invariant_):
        colsum_f = np.sum(self._match_mat_f[:, invariant_], axis=1)
        colnum_f = np.sum(invariant_)
        f_val = self._f_N - np.sum(colsum_f == colnum_f)

        f_val_idx = np.where(colsum_f == colnum_f)

        return f_val, f_val_idx

    def compute_gval_rce(self, invariant_):
        colsum_g = np.sum(self._match_mat_g[:, invariant_], axis=1)
        colnum_g = np.sum(invariant_)
        g_val = self._g_N - np.sum(colsum_g == colnum_g)

        g_val_idx = np.where(colsum_g == colnum_g)

        return g_val, g_val_idx


# This is the core of the invariant mining algorithm.
# This solves the submodular minimization with submodular constraint problem.
# AG means Accelerated Greedy version using priority queue.
class SubmodularMinerAG():
    def __init__(self, match_mat_f_, match_mat_g_, glb_idx_f_, glb_idx_g_, verbal_=False):
        # match_mat_f_: the true false match matrix for f (i.e., configurations of positive samples)
        # match_mat_g_: the true false match matrix for g (i.e., configurations of negative samples)
        self._oracle = OracleSP(match_mat_f_, match_mat_g_)
        self._glb_idx_f = np.array(glb_idx_f_).squeeze()
        self._glb_idx_g = np.array(glb_idx_g_).squeeze()
        self.mat_1 = match_mat_f_
        self.mat_2 = match_mat_g_
        self._verbal = verbal_

    def mineInvariant(self, delta_constr_=0):
        # this uses greedy submodular minimization with constraints to mine
        # the invariant w.r.t. the sample with idx_
        # the real constraint is self._g_N - delta_constr_, where self._g_N is the total number of training data for g.

        self._constr = self._oracle._g_N - delta_constr_
        if self._verbal: print('constraint: %d' % (self._constr))

        invariant, f_val, g_val = self._mineInvariantCore()

        # invariant = self._tightenInvariant(invariant)
        invariant = self._lossenInvariant(invariant)

        f_val_idx = self._oracle.compute_fval_idx()

        if self._verbal: print('FINAL ===> f_val: %d\tg_val: %d\tf_N: %d\tg_N: %d\n' % (
            f_val, g_val, self._oracle._f_N, self._oracle._g_N))
        if self._verbal: print('global:\n', self._glb_idx_f[f_val_idx])
        if self._verbal: print('local:\n', f_val_idx)

        return invariant, f_val, g_val

    def _tightenInvariant(self, invariant_):
        new_bits = 0
        for j in range(self._oracle._D):
            if invariant_[j] == False:
                nom_j = self._oracle._compute_nom_j(j)
                if nom_j < 2:  # nom_j == 0
                    invariant_[j] = True
                    new_bits += 1

        # print('new_bits: ', new_bits)

        return invariant_

    def _lossenInvariant(self, invariant_):
        new_bits = 0
        record_f = self._oracle.compute_fval_rce(invariant_)[0]
        record_g = self._oracle.compute_gval_rce(invariant_)[0]
        for j in range(self._oracle._D):
            if invariant_[j] == True:

                temp = invariant_.copy()
                temp[j] = 0
                if self._oracle.compute_fval_rce(temp)[0] == record_f and self._oracle.compute_gval_rce(temp)[0] == record_g:
                    invariant_[j] = False
                    new_bits += 1

        # print('new_bits: ', new_bits)

        return invariant_

    def _mineInvariantCore(self):
        # init the precomputed statistics
        self._oracle._init_precomp_stat()

        # start iteration
        steps = 0
        new_inv_Y = np.zeros(self._oracle._D, dtype=bool)

        while True:
            steps += 1

            '''
            Plot2D().drawMiningStep(new_inv_Y, self._oracle._match_mat_f, self._oracle._match_mat_g,
                                    self._glb_idx_f, self._glb_idx_g, ('step_' + str(steps) + '.jpg'))
            '''

            sel_j = self._select_j_and_update(new_inv_Y)
            g_val = self._oracle.compute_gval()

            if self._verbal: print('steps: %d\tsel_j: %d\tg_val: %d\tf_val: %d' % (steps, sel_j, g_val, self._oracle.compute_fval()))

            if sel_j < 0 or g_val >= self._constr or steps > 110:  # max no. of boundaries
                break

        return new_inv_Y, self._oracle.compute_fval(), self._oracle.compute_gval()

    def _select_j_and_update(self, new_inv_Y_):
        ratio_vec = self._oracle._compute_ratio_vec()
        ratio_vec[new_inv_Y_] = 1e10

        sel_j = np.argmin(ratio_vec)

        new_inv_Y_[sel_j] = True
        self._oracle._update_by_j(sel_j)

        return sel_j

    # applies mined invariant to do classification
    @staticmethod
    def classify(invariant_, configs_, labels_, gt_config_, gt_label_):
        match_mat = (configs_[:, invariant_] - gt_config_[invariant_] == 0)
        colsum = np.sum(match_mat, axis=1)
        colnum = np.sum(invariant_)
        match_row_indi = (colsum == colnum)

        num_total_samples = np.sum(match_row_indi)
        num_correct_samples = np.sum(labels_[match_row_indi] == gt_label_)
        accuracy = num_correct_samples / num_total_samples

        return accuracy, num_total_samples, match_row_indi
