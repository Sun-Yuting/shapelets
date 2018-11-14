import numpy as np
from sklearn.cluster import KMeans


def _gen_segments(dataset, len_shapelet):
    """_gen_segments
    Generate segments of length `len_shapelet` from `dataset`.

    Parameters
    ----------
    dataset: numpy.ndarray
        shape of dataset I * Q, then J = Q - L + 1
    
    len_shapelet: int
        length of segments(shapelets)

    Returns
    -------
    segments: numpy.ndarray
        shape of segments I * J * L

    """
    I, Q = dataset.shape
    L = len_shapelet
    J = Q - L + 1
    segments = np.zeros((I, J, L))

    for i in I:
        for j in J:
            segment = dataset[i, j:j+L]
            segments[i, j, :] = segment
    return segments


def _init(segments, n_shapelets):
    """_init
    Initialize shapelets using K-Means and weights by random.

    Parameters
    ----------
    segments: numpy.ndarray
        segments with shape I * J * L

    n_shapelets: int
        number of clusters(shapelets)

    Returns
    -------
    S: numpy.ndarray
        shapelets with shape K * L

    W: numpy.ndarray
        weight of length K + 1
    """
    kmeans = KMeans(n_clusters=n_shapelets)
    kmeans.fit(segments.reshape(-1, segments.shape[2]))
    S = kmeans.cluster_centers_
    W = np.random.randn(n_shapelets + 1)

    return S, W


def _min_distance(segments, shapelet):
    """_min_distance (Eq 1)
    Calculate the distances between segments from a series and a shapelet.
    Then return the min one.

    Parameters
    ----------
    segments: numpy.ndarray
        segments from a series. shape J * L

    shapelet: numpy.ndarray
        shapelet. shape L

    Returns
    -------
    M: float
        the min distance.
    """
    J = len(segments)
    distances = np.zeros(J)

    for j in range(J):
        distances[j] = sum((segments[j]-shapelet) ** 2)
    
    return min(distances)


def _linear_predict(min_distances, weights):
    """_linear_predict (Eq 2)
    Predicts the classification result for a series.
    The series is presented by min_distances.

    Parameters
    ----------
    distances: numpy.ndarray
        min distances from K shapelets. shape K 

    weights: numpy.ndarray
        weights. shape K+1

    Returns
    -------
    Y_hat: float
        predicted Y for a series
    """
    M = min_distances
    W0 = weights[0]
    Wk = weights[1:]

    return W0 + sum(M*Wk)


def _sigmoid(y):
    return 1 / (1+np.exp(y))


def _loss_func(y_true, y_pred):
    """_loss_func (Eq 3)
    Calculate loss function ~L
    
    Parameters
    ----------
    y_true: float
        true result

    y_pred: float
        predicted result

    Returns
    -------
    Result of the loss function.
    """
    # -Yln(sig(V)) - (1-Y)ln(1-sig(V))
    return -1 * ((y_true*np.log(_sigmoid(y_pred)) + (1-y_true)*np.log(1-_sigmoid(y_pred))))


def _calc_distance(segment, shapelet):
    """_calc_distance (Eq 5)
    Calculate the distance between a segment and a shapelet.

    Parameters
    ----------
    segment: numpy.ndarray
        the segment. shape L
    
    shapelet: numpy.ndarray
        the shapelet. shape L

    Returns
    -------
    D: float
        the distance
    """
    D = sum((segment-shapelet) ** 2)
    return D


def _softmin_m(segments, shapelet, alpha):
    """_softmin_m (Eq 6)
    Calculate M_hat.

    Parameters
    ----------
    segments: numpy.ndarray
        shape J * L

    shaplet: numpy.ndarray
        shape L

    alpha: float

    Returns
    -------
    the result of softmin min distance
    """
    numerater = sum([ _calc_distance(segments[j], shapelet) * np.exp(alpha*_calc_distance(segments[j], shapelet)) for j in range(len(segments))])
    denominater = sum([np.exp(alpha * _calc_distance(segments[j], shapelet)) for j in range(len(segments))])
    return numerater / denominater


def _dFdWk(y_true, y_pred, m_hat, lambda_w, I, Wk):
    """_dFdWk (Eq 13)
    calculate dF / dW0

    Parameters
    ----------
    y_true: float
        true class value.

    y_pred: float
        predicted class value.

    m_hat: float
        m_hat value.

    lambda_w: float

    I: int

    Wk: float

    Returns
    -------
    """
    return -1*(y_true - _sigmoid(y_pred))*m_hat + 2*lambda_w*Wk/I


def _dFdW0(y_true, y_pred):
    """_dFdW0 (Eq 14)
    calculate dF / dW0.

    Parameters
    ----------
    y_true: float
        true class value.

    y_pred: float
        predicted class value.

    Returns
    -------
    dFdw0: float
    """
    dFdW0 = -1 * (y_true-_sigmoid(y_pred))
    return dFdW0


def _dFdSk(y_true, y_pred, Wk, alpha, segments, shapelet, m_hat, l):
    """_dFdSk (Eq 9, 10, 11, 12)
    dF / dS = dL/dY * dY/dM * sum(dM/dD * dD/dS)
    """
    dLdY = -1 * (y_true - _sigmoid(y_pred))
    dYdM = Wk
    dMdD_mul_dDdS = 0
    for i in range(len(segments)):
        dMdD = np.exp(alpha*_calc_distance(segments[i], shapelet)*(1+alpha*(_calc_distance(segments[i], shapelet)-m_hat))) / sum([np.exp(alpha*_calc_distance(segments[j], shapelet)) for j in range(len(segments))])
        dDdS = 2/len(shapelet) * (shapelet[l] - segments[i,l])
        dMdD_mul_dDdS += dMdD * dDdS
    
    return dLdY * dYdM * dMdD_mul_dDdS


def learning_shapelets(T, Y, K, L, lambda_w, learning_rate, maxIter, alpha=-100):
    """learning_shapelet
    Implementation of time-series shapelets learning algorithm 
    with fixed length binary classification.
    From article Learnig Time-Series Shaplets, 2014, Josif Grabocka et al.

    Parameters
    ----------
    T: numpy.ndarray
        dataset. shape I * Q
    
    Y: numpy.ndarray
        class info. targets. shape I. 1 or 0 values.

    K: int
        number of shapelets to learning.
        Usually K << Q

    L: int
        length of shapelet

    lambda_w: float
        hyper parameter `lambda_w`.

    learning_rate: float
        hyper parameter `learning_rate`.

    maxIter: int
        max iteration for update.

    Returns
    -------
    S: numpy.ndarray
        shapelets. shape K * L

    W: numpy.ndarray
        weights. shape K+1
    """
    # init
    print('Initalizing...')
    I, _ = T.shape
    segments = _gen_segments(T, L)
    S, W = _init(segments, K)

    # learning
    print('Processing...')
    for iteration in range(maxIter):
        for i in range(I):
            # pre compute
            M = np.zeros(K)
            M_hat = np.zeros(K)
            for k in range(k):
                M[k] = _min_distance(segments[i], S[k])
                M_hat = _softmin_m(segments[i], S[k], alpha)
            y_hat = _linear_predict(M, W)
            
            # update
            for k in range(K):
                # update Wk
                W[k+1] -= learning_rate * _dFdWk(Y[i], y_hat, M_hat[k], lambda_w, I, W[K+1])
                for l in range(L):
                    # update shapelets
                    S[k,l] -= learning_rate * _dFdSk(Y[i], y_hat, W[k+1], alpha, segments[i], S[k], M_hat[k], l)
            # update W0
            W[0] -= learning_rate * _dFdW0(Y[i], y_hat)
        
        # visualizaion
        print(f'Progress: {iteration} of {maxIter}. ')
    
    return S, W


if __name__ == "__main__":
    pass
