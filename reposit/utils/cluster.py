







def gmm():


    """


    # of data points required.
    As a lower bound, consider a Gaussian mixture model with d dimensions and
    p mixture components, each with a separate covariance matrix.
    For each component, you'll have to estimate a mean vector (d elements) and
    covariance matrix (d*(d+1)/2 independent elements, because it's a symmetric
    matrix). You'll also have to estimate the p mixture weights. So, your total
    number of parameters is p*(d**2/2+3*d/2+1).
    You will certainly need more data points than parameters.
    Several times more.

    """




    pass