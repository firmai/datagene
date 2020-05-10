## A list of utilities that would eventually get packaged

############### CURVES #######################

from __future__ import division
from numba import jit
from scipy.spatial import distance, minkowski_distance
import matplotlib.pyplot as plt
import numpy as np
from numpy import cov, trace, iscomplexobj, asarray, expand_dims, log, mean, exp, pi
from numpy import array, shape, where, in1d
from numpy.random import random
import numpy.linalg as la
from scipy.linalg import sqrtm, det
import warnings
from scipy import ndimage
import math
import time
from scipy.special import gamma,psi, digamma
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import BallTree, KDTree
from scipy.stats import kde
import os
from sklearn.feature_extraction.text import CountVectorizer
from scipy import stats
from scipy import special


### to github 

__all__ = ['inception', 'braycurtis']




def area_u_c(matrix_org_s,matrix_gen_s,):

  x = matrix_org_s[:, 0]
  z = matrix_org_s[:, 1]-matrix_gen_s[:, 1]
  dx = x[1:] - x[:-1]
  cross_test = np.sign(z[:-1] * z[1:])

  x_intersect = x[:-1] - dx / (z[1:] - z[:-1]) * z[:-1]
  dx_intersect = - dx / (z[1:] - z[:-1]) * z[:-1]

  areas_pos = abs(z[:-1] + z[1:]) * 0.5 * dx # signs of both z are same
  areas_neg = 0.5 * dx_intersect * abs(z[:-1]) + 0.5 * (dx - dx_intersect) * abs(z[1:])

  areas = np.where(cross_test < 0, areas_neg, areas_pos)
  total_area = np.sum(areas)
  return total_area



def poly_area(x, y):
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


def is_simple_quad(ab, bc, cd, da):
    #   Compute all four cross products
    temp0 = np.cross(ab, bc)
    temp1 = np.cross(bc, cd)
    temp2 = np.cross(cd, da)
    temp3 = np.cross(da, ab)
    cross = np.array([temp0, temp1, temp2, temp3])
    #   See that cross products are greater than or equal to zero
    crossTF = cross >= 0
    #   if the cross products are majority false, re compute the cross products
    #   Because they don't necessarily need to lie in the same 'Z' direction
    if sum(crossTF) <= 1:
        crossTF = cross <= 0
    if sum(crossTF) > 2:
        return True
    else:
        return False


def makeQuad(x, y):
    AB = [x[1]-x[0], y[1]-y[0]]
    BC = [x[2]-x[1], y[2]-y[1]]
    CD = [x[3]-x[2], y[3]-y[2]]
    DA = [x[0]-x[3], y[0]-y[3]]

    isQuad = is_simple_quad(AB, BC, CD, DA)

    if isQuad is False:
        # attempt to rearrange the first two points
        x[1], x[0] = x[0], x[1]
        y[1], y[0] = y[0], y[1]
        AB = [x[1]-x[0], y[1]-y[0]]
        BC = [x[2]-x[1], y[2]-y[1]]
        CD = [x[3]-x[2], y[3]-y[2]]
        DA = [x[0]-x[3], y[0]-y[3]]

        isQuad = is_simple_quad(AB, BC, CD, DA)

        if isQuad is False:
            # place the second and first points back where they were, and
            # swap the second and third points
            x[2], x[0], x[1] = x[0], x[1], x[2]
            y[2], y[0], y[1] = y[0], y[1], y[2]
            AB = [x[1]-x[0], y[1]-y[0]]
            BC = [x[2]-x[1], y[2]-y[1]]
            CD = [x[3]-x[2], y[3]-y[2]]
            DA = [x[0]-x[3], y[0]-y[3]]

            isQuad = is_simple_quad(AB, BC, CD, DA)

    # calculate the area via shoelace formula
    area = poly_area(x, y)
    return area


def get_arc_length(dataset):
    #   split the dataset into two discrete datasets, each of length m-1
    m = len(dataset)
    a = dataset[0:m-1, :]
    b = dataset[1:m, :]
    #   use scipy.spatial to compute the euclidean distance
    dataDistance = distance.cdist(a, b, 'euclidean')
    #   this returns a matrix of the euclidean distance between all points
    #   the arc length is simply the sum of the diagonal of this matrix
    arcLengths = np.diagonal(dataDistance)
    arcLength = sum(arcLengths)
    return arcLength, arcLengths


def area_between_two_curves(exp_data, num_data):
    n_exp = len(exp_data)
    n_num = len(num_data)

    # the length of exp_data must be larger than the length of num_data
    if n_exp < n_num:
        temp = num_data.copy()
        num_data = exp_data.copy()
        exp_data = temp.copy()
        n_exp = len(exp_data)
        n_num = len(num_data)

    # get the arc length data of the curves
    # arcexp_data, _ = get_arc_length(exp_data)
    _, arcsnum_data = get_arc_length(num_data)

    # let's find the largest gap between point the num_data, and then
    # linearally interpolate between these points such that the num_data
    # becomes the same length as the exp_data
    for i in range(0, n_exp-n_num):
        a = num_data[0:n_num-1, 0]
        b = num_data[1:n_num, 0]
        nIndex = np.argmax(arcsnum_data)
        newX = (b[nIndex] + a[nIndex])/2.0
        #   the interpolation model messes up if x2 < x1 so we do a quick check
        if a[nIndex] < b[nIndex]:
            newY = np.interp(newX, [a[nIndex], b[nIndex]],
                             [num_data[nIndex, 1], num_data[nIndex+1, 1]])
        else:
            newY = np.interp(newX, [b[nIndex], a[nIndex]],
                             [num_data[nIndex+1, 1], num_data[nIndex, 1]])
        num_data = np.insert(num_data, nIndex+1, newX, axis=0)
        num_data[nIndex+1, 1] = newY

        _, arcsnum_data = get_arc_length(num_data)
        n_num = len(num_data)

    # Calculate the quadrilateral area, by looping through all of the quads
    area = []
    for i in range(1, n_exp):
        tempX = [exp_data[i-1, 0], exp_data[i, 0], num_data[i, 0],
                 num_data[i-1, 0]]
        tempY = [exp_data[i-1, 1], exp_data[i, 1], num_data[i, 1],
                 num_data[i-1, 1]]
        area.append(makeQuad(tempX, tempY))
    return np.sum(area)


def get_length(x, y):
    n = len(x)
    xmax = np.max(np.abs(x))
    ymax = np.max(np.abs(y))

    # if your max x or y value is zero... you'll get np.inf
    # as your curve length based measure
    if xmax == 0:
        xmax = 1e-15
    if ymax == 0:
        ymax = 1e-15

    le = np.zeros(n)
    le[0] = 0.0
    l_sum = np.zeros(n)
    l_sum[0] = 0.0
    for i in range(0, n-1):
        le[i+1] = np.sqrt((((x[i+1]-x[i])/xmax)**2)+(((y[i+1]-y[i])/ymax)**2))
        l_sum[i+1] = l_sum[i]+le[i+1]
    return le, np.sum(le), l_sum


def curve_length_measure(exp_data, num_data):
    x_e = exp_data[:, 0]
    y_e = exp_data[:, 1]
    x_c = num_data[:, 0]
    y_c = num_data[:, 1]

    _, le_nj, le_sum = get_length(x_e, y_e)
    _, lc_nj, lc_sum = get_length(x_c, y_c)

    xmean = np.mean(x_e)
    ymean = np.mean(y_e)

    n = len(x_e)

    r_sq = np.zeros(n)
    for i in range(0, n):
        lieq = le_sum[i]*(lc_nj/le_nj)
        xtemp = np.interp(lieq, lc_sum, x_c)
        ytemp = np.interp(lieq, lc_sum, y_c)

        r_sq[i] = np.log(1.0 + (np.abs(xtemp-x_e[i])/xmean))**2 + \
            np.log(1.0 + (np.abs(ytemp-y_e[i])/ymean))**2
    return np.sqrt(np.sum(r_sq))


def frechet_dist(exp_data, num_data, p=2):
    n = len(exp_data)
    m = len(num_data)
    ca = np.ones((n, m))
    ca = np.multiply(ca, -1)
    ca[0, 0] = minkowski_distance(exp_data[0], num_data[0], p=p)
    for i in range(1, n):
        ca[i, 0] = max(ca[i-1, 0], minkowski_distance(exp_data[i], num_data[0],
                                                      p=p))
    for j in range(1, m):
        ca[0, j] = max(ca[0, j-1], minkowski_distance(exp_data[0], num_data[j],
                                                      p=p))
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(min(ca[i-1, j], ca[i, j-1], ca[i-1, j-1]),
                           minkowski_distance(exp_data[i], num_data[j], p=p))
    return ca[n-1, m-1]


def normalizeTwoCurves(x, y, w, z):

    minX = np.min(x)
    maxX = np.max(x)
    minY = np.min(y)
    maxY = np.max(y)

    xi = (x - minX) / (maxX - minX)
    eta = (y - minY) / (maxY - minY)
    xiP = (w - minX) / (maxX - minX)
    etaP = (z - minY) / (maxY - minY)
    return xi, eta, xiP, etaP


def pcm(exp_data, num_data):
    # normalize the curves to the experimental data
    xi1, eta1, xi2, eta2 = normalizeTwoCurves(exp_data[:, 0], exp_data[:, 1],
                                              num_data[:, 0], num_data[:, 1])
    # compute the arc lengths of each curve
    le, le_nj, le_sum = get_length(xi1, eta1)
    lc, lc_nj, lc_sum = get_length(xi2, eta2)
    # scale each segment to the total polygon length
    le = le / le_nj
    le_sum = le_sum / le_nj
    lc = lc / lc_nj
    lc_sum = lc_sum / lc_nj
    # right now exp_data is curve a, and num_data is curve b
    # make sure a is shorter than a', if not swap the defintion
    if lc_nj > le_nj:
        # compute the arc lengths of each curve
        le, le_nj, le_sum = get_length(xi2, eta2)
        lc, lc_nj, lc_sum = get_length(xi1, eta1)
        # scale each segment to the total polygon length
        le = le / le_nj
        le_sum = le_sum / le_nj
        lc = lc / lc_nj
        lc_sum = lc_sum / lc_nj
        # swap xi1, eta1 with xi2, eta2
        xi1OLD = xi1.copy()
        eta1OLD = eta1.copy()
        xi1 = xi2.copy()
        eta1 = eta2.copy()
        xi2 = xi1OLD.copy()
        eta2 = eta1OLD.copy()

    n_sum = len(le_sum)

    min_offset = 0.0
    max_offset = le_nj - lc_nj

    # make sure the curves aren't the same length
    # if they are the same length, don't loop 200 times
    if min_offset == max_offset:
        offsets = [min_offset]
        pcm_dists = np.zeros(1)
    else:
        offsets = np.linspace(min_offset, max_offset, 200)
        pcm_dists = np.zeros(200)

    for i, offset in enumerate(offsets):
        # create linear interpolation model for num_data based on arc length
        # evaluate linear interpolation model based on xi and eta of exp data
        xitemp = np.interp(le_sum+offset, lc_sum, xi2)
        etatemp = np.interp(le_sum+offset, lc_sum, eta2)

        d = np.sqrt((eta1-etatemp)**2 + (xi1-xitemp)**2)
        d1 = d[:-1]

        d2 = d[1:n_sum]

        v = 0.5*(d1+d2)*le_sum[1:n_sum]
        pcm_dists[i] = np.sum(v)
    return np.min(pcm_dists)


def dtw(exp_data, num_data, metric='euclidean', **kwargs):
    c = distance.cdist(exp_data, num_data, metric=metric, **kwargs)

    d = np.zeros(c.shape)
    d[0, 0] = c[0, 0]
    n, m = c.shape
    for i in range(1, n):
        d[i, 0] = d[i-1, 0] + c[i, 0]
    for j in range(1, m):
        d[0, j] = d[0, j-1] + c[0, j]
    for i in range(1, n):
        for j in range(1, m):
            d[i, j] = c[i, j] + min((d[i-1, j], d[i, j-1], d[i-1, j-1]))
    return d[-1, -1], d


def dtw_path(d):
    path = []
    i, j = d.shape
    i = i - 1
    j = j - 1
    # back propagation starts from the last point,
    # and ends at d[0, 0]
    path.append((i, j))
    while i > 0 or j > 0:
        if i == 0:
            j = j - 1
        elif j == 0:
            i = i - 1
        else:
            temp_step = min([d[i-1, j], d[i, j-1], d[i-1, j-1]])
            if d[i-1, j] == temp_step:
                i = i - 1
            elif d[i, j-1] == temp_step:
                j = j - 1
            else:
                i = i - 1
                j = j - 1
        path.append((i, j))
    path = np.array(path)
    # reverse the order of path, such that it starts with [0, 0]
    return path[::-1]


#################### INFORMATION MEASURES  ############################


__all__=['entropy', 'mutual_information', 'entropy_gaussian']


def nearest_distances(X, k=1):
    '''
    X = array(N,M)
    N = number of points
    M = number of dimensions
    returns the distance to the kth nearest neighbor for every point in X
    '''
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X)
    d, _ = knn.kneighbors(X) # the first nearest neighbor is itself
    return d[:, -1] # returns the distance to the kth nearest neighbor


def entropy_gaussian(C):
    '''
    Entropy of a gaussian variable with covariance matrix C
    '''
    if np.isscalar(C): # C is the variance
        return .5*(1 + np.log(2*pi)) + .5*np.log(C)
    else:
        n = C.shape[0] # dimension
        return .5*n*(1 + np.log(2*pi)) + .5*np.log(abs(det(C)))


def entropy_nearest(X, k=1):
    ''' Returns the entropy of the X.
    Parameters
    ===========
    X : array-like, shape (n_samples, n_features)
        The data the entropy of which is computed
    k : int, optional
        number of nearest neighbors for density estimation
    Notes
    ======
    Kozachenko, L. F. & Leonenko, N. N. 1987 Sample estimate of entropy
    of a random vector. Probl. Inf. Transm. 23, 95-101.
    See also: Evans, D. 2008 A computationally efficient estimator for
    mutual information, Proc. R. Soc. A 464 (2093), 1203-1215.
    and:
    Kraskov A, Stogbauer H, Grassberger P. (2004). Estimating mutual
    information. Phys Rev E 69(6 Pt 2):066138.
    '''

    # Distance to kth nearest neighbor
    r = nearest_distances(X, k) # squared distances
    n, d = X.shape
    volume_unit_ball = (pi**(.5*d)) / gamma(.5*d + 1)
    '''
    F. Perez-Cruz, (2008). Estimation of Information Theoretic Measures
    for Continuous Random Variables. Advances in Neural Information
    Processing Systems 21 (NIPS). Vancouver (Canada), December.
    return d*mean(log(r))+log(volume_unit_ball)+log(n-1)-log(k)
    '''
    return (d*np.mean(np.log(r + np.finfo(X.dtype).eps))
            + np.log(volume_unit_ball) + psi(n) - psi(k))


def mutual_information(variables, k=1):
    '''
    Returns the mutual information between any number of variables.
    Each variable is a matrix X = array(n_samples, n_features)
    where
      n = number of samples
      dx,dy = number of dimensions
    Optionally, the following keyword argument can be specified:
      k = number of nearest neighbors for density estimation
    Example: mutual_information((X, Y)), mutual_information((X, Y, Z), k=5)
    '''
    if len(variables) < 2:
        raise AttributeError(
                "Mutual information must involve at least 2 variables")
    all_vars = np.hstack(variables)
    return (sum([entropy(X, k=k) for X in variables])
            - entropy(all_vars, k=k))


EPS = np.finfo(float).eps

def mutual_information_2d(x, y, sigma=1, normalized=False):
    """
    Computes (normalized) mutual information between two 1D variate from a
    joint histogram.
    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    sigma: float
        sigma for Gaussian smoothing of the joint histogram
    Returns
    -------
    nmi: float
        the computed similariy measure
    """
    bins = (256, 256)

    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    ndimage.gaussian_filter(jh, sigma=sigma, mode='constant', output=jh)

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2))) /
              np.sum(jh * np.log(jh))) - 1
    else:
        mi = (np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1)) -
              np.sum(s2 * np.log(s2)))

    return mi
    
 
def fid_1(act1, act2):
  # calculate mean and covariance statistics
  mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
  mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
  # calculate sum squared difference between means
  ssdiff = np.sum((mu1 - mu2)**2.0)
  # calculate sqrt of product between cov
  covmean = sqrtm(sigma1.dot(sigma2))
  # check and correct imaginary numbers from sqrt
  if iscomplexobj(covmean):
    covmean = covmean.real
  # calculate score
  fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
  return fid

def fid_2(X, Y):
    m = X.mean(0)
    m_w = Y.mean(0)
    X_np = X
    Y_np = Y

    C = np.cov(X_np.transpose())
    C_w = np.cov(Y_np.transpose())
    C_C_w_sqrt = sqrtm(C.dot(C_w), True).real

    score = m.dot(m) + m_w.dot(m_w) - 2 * m_w.dot(m) + \
        np.trace(C + C_w - 2 * C_C_w_sqrt)
    return np.sqrt(score)


# calculate the inception score for p(y|x)
def inception(p_yx, eps=1E-16):
	# calculate p(y)
	p_y = expand_dims(p_yx.mean(axis=0), 0)
	# kl divergence for each image
	kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
	# sum over classes
	sum_kl_d = kl_d.sum(axis=1)
	# average over images
	avg_kl_d = mean(sum_kl_d)
	# undo the logs
	is_score = exp(avg_kl_d)
	return is_score


############ Non-Parametric Entropy Estimation

# CONTINUOUS ESTIMATORS


def entropy(x, k=3, base=2):
    """ The classic K-L k-nearest neighbor continuous entropy estimator
        x should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
        if x is a one-dimensional scalar and we have four samples
    """
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x = np.asarray(x)
    n_elements, n_features = x.shape
    x = add_noise(x)
    tree = build_tree(x)
    nn = query_neighbors(tree, x, k)
    const = digamma(n_elements) - digamma(k) + n_features * log(2)
    return (const + n_features * np.log(nn).mean()) / log(base)


def centropy(x, y, k=3, base=2):
    """ The classic K-L k-nearest neighbor continuous entropy estimator for the
        entropy of X conditioned on Y.
    """
    xy = np.c_[x, y]
    entropy_union_xy = entropy(xy, k=k, base=base)
    entropy_y = entropy(y, k=k, base=base)
    return entropy_union_xy - entropy_y


def tc(xs, k=3, base=2):
    xs_columns = np.expand_dims(xs, axis=0).T
    entropy_features = [entropy(col, k=k, base=base) for col in xs_columns]
    return np.sum(entropy_features) - entropy(xs, k, base)


def ctc(xs, y, k=3, base=2):
    xs_columns = np.expand_dims(xs, axis=0).T
    centropy_features = [centropy(col, y, k=k, base=base)
                         for col in xs_columns]
    return np.sum(centropy_features) - centropy(xs, y, k, base)


def corex(xs, ys, k=3, base=2):
    xs_columns = np.expand_dims(xs, axis=0).T
    cmi_features = [mi(col, ys, k=k, base=base) for col in xs_columns]
    return np.sum(cmi_features) - mi(xs, ys, k=k, base=base)


def mi(x, y, z=None, k=3, base=2, alpha=0):
    """ Mutual information of x and y (conditioned on z if z is not None)
        x, y should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
        if x is a one-dimensional scalar and we have four samples
    """
    assert len(x) == len(y), "Arrays should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x, y = np.asarray(x), np.asarray(y)
    x, y = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)
    x = add_noise(x)
    y = add_noise(y)
    points = [x, y]
    if z is not None:
        z = np.asarray(z)
        z = z.reshape(z.shape[0], -1)
        points.append(z)
    points = np.hstack(points)
    # Find nearest neighbors in joint space, p=inf means max-norm
    tree = build_tree(points)
    dvec = query_neighbors(tree, points, k)
    if z is None:
        a, b, c, d = avgdigamma(x, dvec), avgdigamma(
            y, dvec), digamma(k), digamma(len(x))
        if alpha > 0:
            d += lnc_correction(tree, points, k, alpha)
    else:
        xz = np.c_[x, z]
        yz = np.c_[y, z]
        a, b, c, d = avgdigamma(xz, dvec), avgdigamma(
            yz, dvec), avgdigamma(z, dvec), digamma(k)
    return (-a - b + c + d) / log(base)


def cmi(x, y, z, k=3, base=2):
    """ Mutual information of x and y, conditioned on z
        Legacy function. Use mi(x, y, z) directly.
    """
    return mi(x, y, z=z, k=k, base=base)


def kldiv(x, xp, k=3, base=2):
    """ KL Divergence between p and q for x~p(x), xp~q(x)
        x, xp should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
        if x is a one-dimensional scalar and we have four samples
    """
    assert k < min(len(x), len(xp)), "Set k smaller than num. samples - 1"
    assert len(x[0]) == len(xp[0]), "Two distributions must have same dim."
    x, xp = np.asarray(x), np.asarray(xp)
    x, xp = x.reshape(x.shape[0], -1), xp.reshape(xp.shape[0], -1)
    d = len(x[0])
    n = len(x)
    m = len(xp)
    const = log(m) - log(n - 1)
    tree = build_tree(x)
    treep = build_tree(xp)
    nn = query_neighbors(tree, x, k)
    nnp = query_neighbors(treep, x, k - 1)
    return (const + d * (np.log(nnp).mean() - np.log(nn).mean())) / log(base)


def lnc_correction(tree, points, k, alpha):
    e = 0
    n_sample = points.shape[0]
    for point in points:
        # Find k-nearest neighbors in joint space, p=inf means max norm
        knn = tree.query(point[None, :], k=k+1, return_distance=False)[0]
        knn_points = points[knn]
        # Substract mean of k-nearest neighbor points
        knn_points = knn_points - knn_points[0]
        # Calculate covariance matrix of k-nearest neighbor points, obtain eigen vectors
        covr = knn_points.T @ knn_points / k
        _, v = la.eig(covr)
        # Calculate PCA-bounding box using eigen vectors
        V_rect = np.log(np.abs(knn_points @ v).max(axis=0)).sum()
        # Calculate the volume of original box
        log_knn_dist = np.log(np.abs(knn_points).max(axis=0)).sum()

        # Perform local non-uniformity checking and update correction term
        if V_rect < log_knn_dist + np.log(alpha):
            e += (log_knn_dist - V_rect) / n_sample
    return e

# DISTANCES


def braycurtis(a, b):
    return np.sum(np.fabs(a - b)) / np.sum(np.fabs(a + b))

def canberra(a, b):
    return np.sum(np.fabs(a - b) / (np.fabs(a) + np.fabs(b)))

def chebyshev(a, b):
    return np.amax(a - b)

def correlation(a, b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    return 1.0 - np.mean(a * b) / np.sqrt(np.mean(np.square(a)) * np.mean(np.square(b)))

def cosine(a, b):
    return 1 - np.dot(a, b) / (np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b)))

def dice(a, b):
    nft = ((1 - a) * b).sum()
    ntf = (a * (1 - b)).sum()
    ntt = (a * b).sum()
    return float((ntf + nft) / np.array(2.0 * ntt + ntf + nft))

def euclidean(a, b):
    return np.sqrt(np.sum(np.dot((a - b), (a - b))))

def hamming(a, b, w = None):
    if w is None:
        w = np.ones(a.shape[0])
    return np.average(a != b, weights = w)

def jaccard(a, b):
    return np.double(np.bitwise_and((a != b), np.bitwise_or(a != 0, b != 0)).sum()) / np.double(np.bitwise_or(a != 0, b != 0).sum())

def kulsinski(a, b):
    nft = ((1 - a) * b).sum()
    ntf = (a * (1 - b)).sum()
    ntt = (a * b).sum()
    return (ntf + nft - ntt + len(a)) / (ntf + nft + len(a))

def manhattan(a, b):
    return np.sum(np.fabs(a - b))


def rogerstanimoto(a, b):
    nff = ((1 - a) * (1 - b)).sum()
    nft = ((1 - a) * b).sum()
    ntf = (a * (1 - b)).sum()
    ntt = (a * b).sum()
    return float(2.0 * (ntf + nft)) / float(ntt + nff + (2.0 * (ntf + nft)))

def russellrao(a, b):
    return float(len(a) - (a * b).sum()) / len(a)

def sokalmichener(a, b):
    nff = ((1 - a) * (1 - b)).sum()
    nft = ((1 - a) * b).sum()
    ntf = (a * (1 - b)).sum()
    ntt = (a * b).sum()
    return float(2.0 * (ntf + nft)) / float(ntt + nff + 2.0 * (ntf + nft))

def sokalsneath(a, b):
    nft = ((1 - a) * b).sum()
    ntf = (a * (1 - b)).sum()
    ntt = (a * b).sum()
    return float(2.0 * (ntf + nft)) / np.array(ntt + 2.0 * (ntf + nft))

def sqeuclidean(a, b):
    return np.sum(np.dot((a - b), (a - b)))

def yule(a, b):
    nff = ((1 - a) * (1 - b)).sum()
    nft = ((1 - a) * b).sum()
    ntf = (a * (1 - b)).sum()
    ntt = (a * b).sum()
    return float(2.0 * ntf * nft / np.array(ntt * nff + ntf * nft))


#+++++++++++++


# DISCRETE ESTIMATORS
def entropyd(sx, base=2):
    """ Discrete entropy estimator
        sx is a list of samples
    """
    unique, count = np.unique(sx, return_counts=True, axis=0)
    # Convert to float as otherwise integer division results in all 0 for proba.
    proba = count.astype(float) / len(sx)
    # Avoid 0 division; remove probabilities == 0.0 (removing them does not change the entropy estimate as 0 * log(1/0) = 0.
    proba = proba[proba > 0.0]
    return np.sum(proba * np.log(1. / proba)) / log(base)


def midd(x, y, base=2):
    """ Discrete mutual information estimator
        Given a list of samples which can be any hashable object
    """
    assert len(x) == len(y), "Arrays should have same length"
    return entropyd(x, base) - centropyd(x, y, base)


def cmidd(x, y, z, base=2):
    """ Discrete mutual information estimator
        Given a list of samples which can be any hashable object
    """
    assert len(x) == len(y) == len(z), "Arrays should have same length"
    xz = np.c_[x, z]
    yz = np.c_[y, z]
    xyz = np.c_[x, y, z]
    return entropyd(xz, base) + entropyd(yz, base) - entropyd(xyz, base) - entropyd(z, base)


def centropyd(x, y, base=2):
    """ The classic K-L k-nearest neighbor continuous entropy estimator for the
        entropy of X conditioned on Y.
    """
    xy = np.c_[x, y]
    return entropyd(xy, base) - entropyd(y, base)


def tcd(xs, base=2):
    xs_columns = np.expand_dims(xs, axis=0).T
    entropy_features = [entropyd(col, base=base) for col in xs_columns]
    return np.sum(entropy_features) - entropyd(xs, base)


def ctcd(xs, y, base=2):
    xs_columns = np.expand_dims(xs, axis=0).T
    centropy_features = [centropyd(col, y, base=base) for col in xs_columns]
    return np.sum(centropy_features) - centropyd(xs, y, base)


def corexd(xs, ys, base=2):
    xs_columns = np.expand_dims(xs, axis=0).T
    cmi_features = [midd(col, ys, base=base) for col in xs_columns]
    return np.sum(cmi_features) - midd(xs, ys, base)


# MIXED ESTIMATORS
def micd(x, y, k=3, base=2, warning=True):
    """ If x is continuous and y is discrete, compute mutual information
    """
    assert len(x) == len(y), "Arrays should have same length"
    entropy_x = entropy(x, k, base)

    y_unique, y_count = np.unique(y, return_counts=True, axis=0)
    y_proba = y_count / len(y)

    entropy_x_given_y = 0.
    for yval, py in zip(y_unique, y_proba):
        x_given_y = x[(y == yval).all(axis=1)]
        if k <= len(x_given_y) - 1:
            entropy_x_given_y += py * entropy(x_given_y, k, base)
        else:
            if warning:
                warnings.warn("Warning, after conditioning, on y={yval} insufficient data. "
                              "Assuming maximal entropy in this case.".format(yval=yval))
            entropy_x_given_y += py * entropy_x
    return abs(entropy_x - entropy_x_given_y)  # units already applied


def midc(x, y, k=3, base=2, warning=True):
    return micd(y, x, k, base, warning)


def centropycd(x, y, k=3, base=2, warning=True):
    return entropy(x, base) - micd(x, y, k, base, warning)


def centropydc(x, y, k=3, base=2, warning=True):
    return centropycd(y, x, k=k, base=base, warning=warning)


def ctcdc(xs, y, k=3, base=2, warning=True):
    xs_columns = np.expand_dims(xs, axis=0).T
    centropy_features = [centropydc(
        col, y, k=k, base=base, warning=warning) for col in xs_columns]
    return np.sum(centropy_features) - centropydc(xs, y, k, base, warning)


def ctccd(xs, y, k=3, base=2, warning=True):
    return ctcdc(y, xs, k=k, base=base, warning=warning)


def corexcd(xs, ys, k=3, base=2, warning=True):
    return corexdc(ys, xs, k=k, base=base, warning=warning)


def corexdc(xs, ys, k=3, base=2, warning=True):
    return tcd(xs, base) - ctcdc(xs, ys, k, base, warning)


# UTILITY FUNCTIONS

def add_noise(x, intens=1e-10):
    # small noise to break degeneracy, see doc.
    return x + intens * np.random.random_sample(x.shape)


def query_neighbors(tree, x, k):
    return tree.query(x, k=k + 1)[0][:, k]


def count_neighbors(tree, x, r):
    return tree.query_radius(x, r, count_only=True)


def avgdigamma(points, dvec):
    # This part finds number of neighbors in some radius in the marginal space
    # returns expectation value of <psi(nx)>
    tree = build_tree(points)
    dvec = dvec - 1e-15
    num_points = count_neighbors(tree, points, dvec)
    return np.mean(digamma(num_points))


def build_tree(points):
    if points.shape[1] >= 20:
        return BallTree(points, metric='chebyshev')
    return KDTree(points, metric='chebyshev')

# TESTS


def shuffle_test(measure, x, y, z=False, ns=200, ci=0.95, **kwargs):
    """ Shuffle test
        Repeatedly shuffle the x-values and then estimate measure(x, y, [z]).
        Returns the mean and conf. interval ('ci=0.95' default) over 'ns' runs.
        'measure' could me mi, cmi, e.g. Keyword arguments can be passed.
        Mutual information and CMI should have a mean near zero.
    """
    x_clone = np.copy(x)  # A copy that we can shuffle
    outputs = []
    for i in range(ns):
        np.random.shuffle(x_clone)
        if z:
            outputs.append(measure(x_clone, y, z, **kwargs))
        else:
            outputs.append(measure(x_clone, y, **kwargs))
    outputs.sort()
    return np.mean(outputs), (outputs[int((1. - ci) / 2 * ns)], outputs[int((1. + ci) / 2 * ns)])


def return_intersection(hist_1, hist_2):
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection

# METHOD #3: ROLL YOUR OWN
def chi2_distance(histA, histB, eps = 1e-10):
	# compute the chi-squared distance
	d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
		for (a, b) in zip(histA, histB)])
	# return the chi-squared distance
	return d


def jensen_shannon_divergence(a, b):
    """Compute Jensen-Shannon Divergence
    Parameters
    ----------
    a : array-like
        possibly unnormalized distribution.
    b : array-like
        possibly unnormalized distribution. Must be of same shape as ``a``.
    Returns
    -------
    j : float
    See Also
    --------
    jsd_matrix : function
        Computes all pair-wise distances for a set of measurements
    entropy : function
        Computes entropy and K-L divergence
    """
    a = np.asanyarray(a, dtype=float)
    b = np.asanyarray(b, dtype=float)
    a = a/a.sum(axis=0)
    b = b/b.sum(axis=0)
    m = (a + b)
    m /= 2.
    m = np.where(m, m, 1.)
    return 0.5*np.sum(special.xlogy(a, a/m) + special.xlogy(b, b/m), axis=0)

def ks_statistic(data1, data2):
    """Calculate the Kolmogorov-Smirnov statistic to compare two sets of data.
    The empirical cumulative distribution function for each set of
    set of 1-dimensional data points is calculated. The K-S statistic
    the maximum absolute distance between the two cumulative distribution
    functions.
    Parameters:
        data1, data2: 1-dimensional lists or arrays of data points to compare.
    """
    x1 = np.sort(data1)
    x2 = np.sort(data2)
    x = np.sort(np.concatenate([x1, x2]))
    y1 = np.linspace(0, 1, len(x1)+1)[1:] # empirical CDF for data1: curve going up by 1/len(data1) at each observed data-point
    y2 = np.linspace(0, 1, len(x2)+1)[1:] # as above but for data2
    cdf1 = np.interp(np.mean(x,axis=0), np.mean(x1,axis=0), y1, left=0) # linearly interpolate both CDFs onto a common set of x-values.
    cdf2 = np.interp(np.mean(x,axis=0), np.mean(x2,axis=0), y2, left=0)
    return abs(cdf1-cdf2).max()

def ks_statistic_vec(data1, data2):
    """Calculate the Kolmogorov-Smirnov statistic to compare two sets of data.
    The empirical cumulative distribution function for each set of
    set of 1-dimensional data points is calculated. The K-S statistic
    the maximum absolute distance between the two cumulative distribution
    functions.
    Parameters:
        data1, data2: 1-dimensional lists or arrays of data points to compare.
    """
    x1 = np.sort(data1)
    x2 = np.sort(data2)
    x = np.sort(np.concatenate([x1, x2]))
    y1 = np.linspace(0, 1, len(x1)+1)[1:] # empirical CDF for data1: curve going up by 1/len(data1) at each observed data-point
    y2 = np.linspace(0, 1, len(x2)+1)[1:] # as above but for data2
    cdf1 = np.interp(x, x1, y1, left=0) # linearly interpolate both CDFs onto a common set of x-values.
    cdf2 = np.interp(x, x2, y2, left=0)
    return abs(cdf1-cdf2).max()

def ks_statistic_kde(data1, data2, num_points=None,vector=False):
    """Calculate the Kolmogorov-Smirnov statistic to compare two sets of data.
    Kernel Density Estimation is used to estimate the distribution of each set
    set of 1-dimensional data points. From this, the K-S statistic is
    calculated: the maximum absolute distance between the two cumulative
    distribution functions.
    Parameters:
        data1, data2: 1-dimensional lists or arrays of data points to compare.
        num_points: number of points to evaluate the density along.
    """
    xs, kd1, kd2 = _get_estimators_and_xs(data1, data2, num_points,vector)
    with np.errstate(under='ignore'):
        cdf1 = np.array([kd1.integrate_box_1d(-np.inf, x) for x in xs])
        cdf2 = np.array([kd2.integrate_box_1d(-np.inf, x) for x in xs])
    return abs(cdf1 - cdf2).max()

def js_metric(data1, data2, num_points=None,vector=False):
    """Calculate the Jensen-Shannon metric to compare two sets of data.
    Kernel Density Estimation is used to estimate the distribution of each set
    set of 1-dimensional data points. From this, the J-S metric (square root of
    J-S divergence) is calculated.
    Note: KDE will often underestimate the probability at the far tails of the
    distribution (outside of where supported by the data), which can lead to
    overestimates of K-L divergence (and hence J-S divergence) for highly
    non-overlapping datasets.
    Parameters:
        data1, data2: 1-dimensional lists or arrays of data points to compare.
        num_points: number of points to evaluate the density along.
    """
    xs, p1, p2 = _get_point_estimates(data1, data2, num_points,vector)
    m = (p1 + p2)/2
    return ((_kl_divergence(xs, p1, m) + _kl_divergence(xs, p2, m))/2)**0.5

def kl_divergence(data1, data2, num_points=None,vector=False):
    """Calculate the Kullback-Leibler divergence between two sets of data.
    Kernel Density Estimation is used to estimate the distribution of each set
    set of 1-dimensional data points. From this, the K-L divergence is
    calculated.
    Note: KDE will often underestimate the probability at the far tails of the
    distribution (outside of where supported by the data), which can lead to
    overestimates of K-L divergence for highly non-overlapping datasets.
    Parameters:
        data1, data2: 1-dimensional lists or arrays of data points to compare.
        num_points: number of points to evaluate the density along.
    """
    xs, p1, p2 = _get_point_estimates(data1, data2, num_points,vector)
    return _kl_divergence(xs, p1, p2)

def _get_kd_estimator_and_xs(data, num_points):
    """Get KDE estimator for a given dataset, and generate a good set of
    points to sample the density at."""
    data = np.asarray(data, dtype=np.float)
    kd_estimator = kde.gaussian_kde(np.mean(data,axis=0))
    data_samples = kd_estimator.resample(num_points//2)[0]
    xs = np.sort(data_samples)
    return kd_estimator, xs

def _get_kd_estimator_and_xs_vec(data, num_points):
    """Get KDE estimator for a given dataset, and generate a good set of
    points to sample the density at."""
    data = np.asarray(data, dtype=np.float)
    kd_estimator = kde.gaussian_kde(np.mean(data,axis=0))
    data_samples = kd_estimator.resample(num_points//2)[0]
    xs = np.sort(data_samples)
    return kd_estimator, xs

def _get_estimators_and_xs(data1, data2, num_points,vector=False):
    """Get KDE estimators for two different datasets and a set of points
    to evaluate both distributions on."""
    if num_points is None:
        num_points = min(5000, (len(data1) + len(data2))//2)
    if vector:
        kd1, xs1 = _get_kd_estimator_and_xs_vec(data1, num_points//2)
        kd2, xs2 = _get_kd_estimator_and_xs_vec(data2, num_points//2)
    else:
        kd1, xs1 = _get_kd_estimator_and_xs(data1, num_points//2)
        kd2, xs2 = _get_kd_estimator_and_xs(data2, num_points//2)
    xs = np.sort(np.concatenate([xs1, xs2]))
    return xs, kd1, kd2

def _get_point_estimates(data1, data2, num_points,vector=False):
    """Get point estimates for KDE distributions for two different datasets.
    """
    xs, kd1, kd2 = _get_estimators_and_xs(data1, data2, num_points)
    with np.errstate(under='ignore'):
        p1 = kd1(xs)
        p2 = kd2(xs)
    return xs, p1, p2

def _kl_divergence(xs, p1, p2):
    """Calculate Kullback-Leibler divergence of p1 and p2, which are assumed to
    values of two different density functions at the given positions xs.
    Return divergence in nats."""
    with np.errstate(divide='ignore', invalid='ignore'):
        kl = p1 * (np.log(p1) - np.log(p2))
    kl[~np.isfinite(kl)] = 0 # small numbers in p1 or p2 can cause NaN/-inf, etc.
    return np.trapz(kl, x=xs) #integrate curve


def jensen_shannon_divergence_2(repr1, repr2):
    """Calculates Jensen-Shannon divergence (https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)."""
    avg_repr = 0.5 * (repr1 + repr2)
    sim = 1 - 0.5 * (scipy.stats.entropy(repr1, avg_repr) + scipy.stats.entropy(repr2, avg_repr))
    return sim




def renyi_divergence(repr1, repr2, alpha=0.99):
    """Calculates Renyi divergence (https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy#R.C3.A9nyi_divergence)."""
    log_sum = np.sum([np.power(p, alpha) / np.power(q, alpha-1) for (p, q) in zip(repr1, repr2)])
    sim = 1 / (alpha - 1) * np.log(log_sum)
    if np.isinf(sim):
        # the similarity is -inf if no term in the document is in the vocabulary
        return 0
    return sim


def cosine_similarity(repr1, repr2):
    """Calculates cosine similarity (https://en.wikipedia.org/wiki/Cosine_similarity)."""
    if repr1 is None or repr2 is None:
        return 0
    assert not (np.isnan(repr2).any() or np.isinf(repr2).any())
    assert not (np.isnan(repr1).any() or np.isinf(repr1).any())
    sim = 1 - scipy.spatial.distance.cosine(repr1, repr2)
    if np.isnan(sim):
        # the similarity is nan if no term in the document is in the vocabulary
        return 0
    return sim


def euclidean_distance(repr1, repr2):
    """Calculates Euclidean distance (https://en.wikipedia.org/wiki/Euclidean_distance)."""
    sim = np.sqrt(np.sum([np.power(p-q, 2) for (p, q) in zip(repr1, repr2)]))
    return sim


def variational_distance(repr1, repr2):
    """Also known as L1 or Manhattan distance (https://en.wikipedia.org/wiki/Taxicab_geometry)."""
    sim = np.sum([np.abs(p-q) for (p, q) in zip(repr1, repr2)])
    return sim


def kl_divergence(repr1, repr2):
    """Calculates Kullback-Leibler divergence (https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)."""
    sim = scipy.stats.entropy(repr1, repr2)
    return sim


def bhattacharyya_distance(repr1, repr2):
    """Calculates Bhattacharyya distance (https://en.wikipedia.org/wiki/Bhattacharyya_distance)."""
    sim = - np.log(np.sum([np.sqrt(p*q) for (p, q) in zip(repr1, repr2)]))
    assert not np.isnan(sim), 'Error: Similarity is nan.'
    if np.isinf(sim):
        # the similarity is -inf if no term in the review is in the vocabulary
        return 0
    return sim


def _fancy(m):
    a = sum(m)
    b = 0
    for i in m:
        b += i * i
    return len(m) * b - (a ** 2)

def pearson_rho(m, n):
    """ 
    return the Pearson rho coefficient; based off stats.py 
    >>> x = [2, 8, 5, 4, 2, 6, 1, 4, 5, 7, 4]
    >>> y = [3, 9, 4, 3, 1, 7, 2, 5, 6, 8, 3]
    >>> pearson_rho(x, y)
    0.9245404356092288
    """
    if len(m) != len(n):
        raise ValueError('Iterables (m, n) must be the same length')
    num = len(m) * (sum([i * j for i, j in zip(m, n)])) - sum(m) * sum(n)
    return num / np.sqrt(_fancy(m) * _fancy(n))


## a few more distance measures


def eucFunc(d1, d2, t="sqrt"):
	d = d1-d2
	eucD = np.sqrt(np.sum(np.square(d)))
	if t == "1":
		return eucD
	if t == "sqrt":
		return np.sqrt(eucD)

def matchMat(d1, d2):
	A = float(np.sum((d1 == 1) & (d2 == 1)))
	B = float(np.sum((d1 == 1) & (d2 == 0)))
	C = float(np.sum((d1 == 0) & (d2 == 1)))
	D = float(np.sum((d1 == 0) & (d2 == 0)))
	return A, B, C, D

def kulSim(d1, d2, t="sqrt"):
	A = np.sum(d1)
	B = np.sum(d2)
	W = np.sum(np.minimum(d1, d2))
	S = 0.5*(W/A + W/B)
	if t == "1":
		return 1 - S
	if t == "sqrt":
		return np.sqrt(1 - S)
		
def braySim(d1, d2, t="sqrt"):
	A = np.sum(d1)
	B = np.sum(d2)
	W = np.sum(np.minimum(d1, d2))
	S = (2*W)/(A + B)
	if t == "1":
		return 1 - S
	if t == "sqrt":
		return np.sqrt(1 - S)

def gowerSim(d1, d2, t="sqrt"):
	diffs = 1 - (np.abs(d1-d2))
	isabs = d1+d2 ==0
	S = np.sum(~isabs*diffs)/np.sum(~isabs)
	if t == "1":
		return 1 - S
	if t == "sqrt":
		return np.sqrt(1 - S)

def chordDis(d1, d2, t="sqrt"):
	norm1 = 1/(np.sqrt(np.sum(d1**2))) * d1
	norm2 = 1/(np.sqrt(np.sum(d2**2))) * d2
	d = norm1-norm2
	chordD = np.sqrt(np.sum(np.square(d)))
	if t == "1":
		return chordD
	if t == "sqrt":
		return np.sqrt(chordD)

def manDist(d1, d2, t="sqrt"):
	manD = np.sum(np.abs(d1 - d2))
	if t == "1":
		return manD
	if t == "sqrt":
		return np.sqrt(manD)

def charDist(d1, d2, t="sqrt"):
	charD = 1./len(d1) * np.sum(np.abs(d1 - d2))
	if t == "1":
		return charD
	if t == "sqrt":
		return np.sqrt(charD)


def whitDist(d1, d2, t="sqrt"):
	d1 = d1/np.sum(d1)
	d2 = d2/np.sum(d2)
	whitD = 0.5*sum(np.abs(d1-d2))
	if t == "1":
		return whitD
	if t == "sqrt":
		return np.sqrt(whitD)

def canDist(d1, d2, t="sqrt"):
	isabs = d1+d2==0
	d1 = d1[~isabs]
	d2 = d2[~isabs]
	canD = np.sum(np.abs(d1-d2)/(d1+d2)) * 1./np.sum(~isabs)
	if t == "1":
		return canD
	if t == "sqrt":
		return np.sqrt(canD)

def m_gowDist(d1, d2, t="sqrt"):
	isabs = d1+d2==0
	charD = 1./np.sum(~isabs) * np.sum(np.abs(d1 - d2))
	if t == "1":
		return charD
	if t == "sqrt":
		return np.sqrt(charD)
		

@jit(nopython=True)
def earth_movers_distance(a, b):
    n = len(a)
    ac = 0
    bc = 0
    diff = 0
    for i in range(n):
        ac += a[i]
        bc += b[i]
        diff += abs(ac - bc)
    return diff


## Additional
#========================================

# Running Calculation Distance Calcultion for Sanity
def simple_diff_ratio(real, fake, mult=1):
  val = abs(((abs(real+1)-abs(fake+1))/(real+1))*100).mean(); print(val)
  
