from sklearn.model_selection import train_test_split


def fsplit(X, y, cut_size=0.30, seed=34):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=cut_size, stratify=y, random_state=seed)
    return Xtr, Xte, ytr, yte
