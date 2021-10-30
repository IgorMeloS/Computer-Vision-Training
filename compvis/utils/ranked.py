# =============================================================================
# Ranked Accuracies rk1 and rk5
# =============================================================================

# Importing libraries

import numpy as np

def rank5_accuracy(preds, labels):
    """ Rank 5 function calculates the rank 5 and 1.
    Args:
        preds: the predictions
        lables: true targets
    return rank1 and rank5
    """
    # Initilizing the variables for the rank 1 and 5
    rank1 = 0
    rank5 = 0
    # loop over the predictions and ground-truth
    for (p, gt) in zip(preds, labels):
        # sorting the probabilities by their index in descendinf order so that
        # the more confident guesses are at the front of the list
        p = np.argsort(p)[::-1]
        # Checkig if the ground-truth is in the top 5
        if gt in p[:5]:
            rank5 += 1
        # checking if the predictions it the top 1
        if gt == p[0]:
            rank1 += 1
    rank1 /= float(len(labels))
    rank5 /= float(len(labels))

    # Returning the tuple of accuracies
    return (rank1, rank5)
