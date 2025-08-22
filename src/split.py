"""
Utility function for splitting datasets into training and testing subsets.
"""

from sklearn.model_selection import train_test_split


def fsplit(X, y, cut_size=0.30, seed=34):
    """
        Split features and labels into train and test subsets using stratification.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Target labels.
        cut_size : float, default=0.30
            Proportion of the dataset to include in the test split.
            Example: 0.30 means 30% test and 70% train.
        seed : int, default=34
            Random seed for reproducibility.

        Returns
        -------
        Xtr : ndarray
            Training features.
        Xte : ndarray
            Testing features.
        ytr : ndarray
            Training labels.
        yte : ndarray
            Testing labels.
        """

    # train_test_split automatically divides the data into training and testing sets
    # stratify=y ensures class distribution is preserved in both subsets
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=cut_size, stratify=y, random_state=seed)
    return Xtr, Xte, ytr, yte
