def mcm(tn, fp, fn, tp):
    """Let be a confusion matrix like this:
    
    
      N    P
    +----+----+
    |    |    |
    | TN | FP |
    |    |    |
    +----+----+
    |    |    |
    | FN | TP |
    |    |    |
    +----+----+
    
    The observed values by columns and the expected values by rows and the positive class in right column. With these definitions, the TN, FP, FN and TP values are that order.
    
    
    Parameters
    ----------
    TN : integer
        True Negative
    FP: integer
        False Positive
    FN: integer
        False Negative
    TP: integer
        True Positive
   
    Returns
    -------
    sum : float
        Sum of values
    
    Notes
    -----
    https://en.wikipedia.org/wiki/Confusion_matrix
    https://developer.lsst.io/python/numpydoc.html
    
    Examples
    --------
    data = pd.DataFrame({
    'y_true': ['Positive']*47 + ['Negative']*18,
    'y_pred': ['Positive']*37 + ['Negative']*10 + ['Positive']*5 + ['Negative']*13})
    
    tn, fp, fn, tp = confusion_matrix(y_true = data.y_true, 
                                  y_pred = data.y_pred, 
                                  labels = ['Negative', 'Positive']).ravel()
    
    """
    mcm = []
    
    mcm.append(['Sensitivity', tp / (tp + fn)])
    mcm.append(['Recall', tp / (tp + fn)])
    mcm.append(['True Positive rate (TPR)', tp / (tp + fn)])
    mcm.append(['Specificity', tn / (tn + fp)])
    mcm.append(['True Negative Rate (TNR)', tn / (tn + fp)])
    
    mcm.append(['Precision', tp / (tp + fp)])
    mcm.append(['Positive Predictive Value (PPV)', tp / (tp + fp)])
    mcm.append(['Negative Predictive Value (NPV)', tn / (tn + fn)])
        
    mcm.append(['False Negative Rate (FNR)', fn / (fn + tp)])
    mcm.append(['False Positive Rate (FPR)', fp / (fp + tn)])
    mcm.append(['False Discovery Rate (FDR)', fp / (fp + tp)])
    
    mcm.append(['Accuracy', (tp + tn) / (tp + tn + fp + fn)])
    mcm.append(['F1 Score', 2*tp / (2*tp + fp + fn)])
    
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    
    fnr = fn / (fn + tp)
    tnr = tn / (tn + fp)
    
    mcm.append(['Positive Likelihood Ratio (LR+)', tpr / fpr])
    mcm.append(['Negative Likelihood Ratio (LR-)', fnr / tnr])
    
    return pd.DataFrame(mcm, columns = ['Metric', 'Value'])