import numpy as np
from sklearn.model_selection import KFold


class GroupKFold:

    # ref:https://zenn.dev/mst8823/articles/cd40cb971f702e masatoさんのベースラインから

    def __init__(self, n_splits=5, shuffle=True, random_state=0):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, group=None):
        return self.n_splits

    def split(self, X=None, y=None, group=None):
        kfold = KFold(n_splits=self.n_splits, shuffle=self.shuffle,
                      random_state=self.random_state)
        unique_ids = group.unique()
        for tr_group_idx, va_group_idx in kfold.split(unique_ids):
            tr_group, va_group = unique_ids[tr_group_idx], unique_ids[va_group_idx]
            train_idx = np.where(group.isin(tr_group))[0]
            val_idx = np.where(group.isin(va_group))[0]
            yield train_idx, val_idx


def make_gkf(X, y, n_splits=5, random_state=0):
    #  groupkfoldでcvを渡すための関数
    #  予めgroup=data[target]としておく必要がある。

    gkf = GroupKFold(n_splits=n_splits, random_state=random_state)
    return list(gkf.split(X, y, group))


def make_folds(X, y, n_splits=5, random_state=0):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(kfold.split(X, y))
