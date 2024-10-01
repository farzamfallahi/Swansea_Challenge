import numpy as np
from typing import Union, List

def confusion_matrix(y_true: Union[List, np.ndarray], 
                     y_pred: Union[List, np.ndarray], 
                     labels: Union[List, np.ndarray] = None) -> np.ndarray:
    """
    Compute the confusion matrix to evaluate the accuracy of a classification.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.
    labels : array-like of shape (n_classes,), optional (default=None)
        List of labels to index the matrix. This may be used to reorder or
        select a subset of labels. If None, those that appear at least once in
        y_true or y_pred are used in sorted order.

    Returns:
    --------
    confusion_matrix : numpy.ndarray of shape (n_classes, n_classes)
        Confusion matrix whose i-th row and j-th column entry indicates the
        number of samples with true label being i-th class and predicted label
        being j-th class.

    Raises:
    -------
    ValueError
        If the length of y_true doesn't match the length of y_pred.
    TypeError
        If y_true or y_pred are not lists or numpy arrays.

    Examples:
    ---------
    >>> y_true = [2, 0, 2, 2, 0, 1]
    >>> y_pred = [0, 0, 2, 2, 0, 2]
    >>> confusion_matrix(y_true, y_pred)
    array([[2, 0, 0],
           [0, 0, 1],
           [1, 0, 2]])

    >>> labels = [0, 1, 2]
    >>> confusion_matrix(y_true, y_pred, labels=labels)
    array([[2, 0, 0],
           [0, 0, 1],
           [1, 0, 2]])

    >>> labels = [0, 1]
    >>> confusion_matrix(y_true, y_pred, labels=labels)
    array([[2, 0],
           [0, 0]])
    """
    if not isinstance(y_true, (list, np.ndarray)) or not isinstance(y_pred, (list, np.ndarray)):
        raise TypeError("y_true and y_pred must be lists or numpy arrays")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    else:
        labels = np.asarray(labels)
        if np.all([l not in y_true and l not in y_pred for l in labels]):
            raise ValueError("At least one label must be in y_true or y_pred")

    n_labels = labels.size
    label_to_ind = {y: x for x, y in enumerate(labels)}
    
    cm = np.zeros((n_labels, n_labels), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in label_to_ind and p in label_to_ind:
            cm[label_to_ind[t], label_to_ind[p]] += 1

    return cm

# Example usage and test
if __name__ == "__main__":
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]
    
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:")
    print(cm)
    
    # Test with specific labels
    labels = [0, 1, 2]
    cm_with_labels = confusion_matrix(y_true, y_pred, labels=labels)
    print("\nConfusion matrix with specific labels:")
    print(cm_with_labels)
    
    # Test with subset of labels
    subset_labels = [0, 1]
    cm_subset = confusion_matrix(y_true, y_pred, labels=subset_labels)
    print("\nConfusion matrix with subset of labels:")
    print(cm_subset)
    
    # Test error handling
    try:
        confusion_matrix([1, 2, 3], [1, 2])  # Different lengths
    except ValueError as e:
        print(f"\nCaught expected ValueError: {e}")
    
    try:
        confusion_matrix("invalid", [1, 2, 3])  # Invalid input type
    except TypeError as e:
        print(f"Caught expected TypeError: {e}")