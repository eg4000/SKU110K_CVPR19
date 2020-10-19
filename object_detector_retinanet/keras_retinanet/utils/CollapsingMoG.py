import math

import scipy
import numpy

import signal

class Timeout():
    """Timeout class using ALARM signal."""

    class Timeout(Exception):
        pass

    def __init__(self, sec):
        self.sec = sec

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, self.sec)

    def __exit__(self, *args):
        signal.alarm(0)  # disable alarm

    def raise_timeout(self, *args):
        raise Timeout.Timeout()


def agglomerative_init(alpha, mu, covariance, n, k):

    mu_stack = numpy.zeros(shape=[n - k, mu.shape[1]], dtype=mu.dtype)
    mu_stack.fill(numpy.inf)
    mu_temp = numpy.vstack([mu.copy(), mu_stack])
    covariance_temp = numpy.vstack(
        [covariance, numpy.zeros(shape=[n - k, covariance.shape[1], covariance.shape[2]], dtype=covariance.dtype)])
    alpha_temp = numpy.hstack([alpha, numpy.zeros(shape=(n - k), dtype=alpha.dtype)])
    distances = scipy.spatial.distance.cdist(mu_temp, mu_temp)
    distances = numpy.triu(distances)
    distances = numpy.nan_to_num(distances)
    distances[distances == 0] = numpy.inf
    deleted = []
    for l in range(n, 2 * n - k):
        i, j = numpy.unravel_index(numpy.argmin(distances), distances.shape)

        alpha_ij = alpha_temp[i] + alpha_temp[j]
        mu_ij = (alpha_temp[i] * mu_temp[i] + alpha_temp[j] * mu_temp[j]) / alpha_ij
        harmonic_mean = (alpha_temp[i] * alpha_temp[j]) / alpha_ij
        delta_mu = (mu_temp[i] - mu_temp[j])
        delta_mu = numpy.expand_dims(delta_mu, axis=1)
        covariance_ij = (alpha_temp[i] * covariance_temp[i] + alpha_temp[j] * covariance_temp[
            j] + harmonic_mean * numpy.dot(delta_mu, delta_mu.transpose())) / alpha_ij

        mu_temp[l] = mu_ij
        covariance_temp[l] = covariance_ij
        alpha_temp[l] = alpha_ij

        distances[:, i] = numpy.inf
        distances[:, j] = numpy.inf
        distances[i, :] = numpy.inf
        distances[j, :] = numpy.inf
        mu_temp[i] = numpy.inf
        mu_temp[j] = numpy.inf
        deleted.append(i)
        deleted.append(j)

        d = scipy.spatial.distance.cdist(mu_temp, numpy.expand_dims(mu_ij, axis=0))[:, 0]
        d[d == 0] = numpy.inf
        distances[:, l] = d
    deleted_indexes = numpy.array(deleted)
    mask = numpy.ones(alpha_temp.shape[0], dtype=bool)
    if deleted_indexes.shape[0] > 0:
        mask[deleted_indexes] = False
    return alpha_temp[mask], mu_temp[mask], covariance_temp[mask]


def gaussian_kl(mu1, cov1, mu2, cov2):
    cov2inv = numpy.linalg.inv(cov2)
    log_det_ratio = numpy.log(numpy.linalg.det(cov2) / numpy.linalg.det(cov1))
    delta_mu = (mu1 - mu2)
    delta_mu = numpy.expand_dims(delta_mu, axis=1)
    return 0.5 * (log_det_ratio
                  + numpy.trace(numpy.dot(cov2inv, cov1))
                  + numpy.dot(numpy.dot(delta_mu.transpose(), cov2inv), delta_mu))[0][0]


def gaussian_kl_diag(mu1, cov1, mu2, cov2):
    cov2sqrt = numpy.sqrt(cov2)
    cov1sqrt = numpy.sqrt(cov1)
    log_ratio = math.log(cov2sqrt[0, 0] / cov1sqrt[0, 0]) + math.log(cov2sqrt[1, 1] / cov1sqrt[1, 1])
    delta_mu = (mu1 - mu2)
    div = (cov1[0, 0] + delta_mu[0] * delta_mu[0]) / (2 * cov2[0, 0]) + (cov1[1, 1] + delta_mu[1] * delta_mu[1]) / (
            2 * cov2[1, 1])
    return div + log_ratio


def collapse(original_detection_centers, k, offset, max_iter=100, epsilon=1e-100):
    try:
        with Timeout(3):
            n = original_detection_centers.shape[0]
            mu_x = original_detection_centers.x - offset[0]
            mu_y = original_detection_centers.y - offset[1]
            sigma_xx = original_detection_centers.sigma_x * original_detection_centers.sigma_x
            sigma_yy = original_detection_centers.sigma_y * original_detection_centers.sigma_y

            alpha = numpy.array(original_detection_centers.confidence / original_detection_centers.confidence.sum())
            mu = numpy.array([mu_x.values, mu_y.values]).transpose()
            covariance = numpy.array([[sigma_xx.values, sigma_xx.values * 0], [0 * sigma_yy.values, sigma_yy.values]]).transpose()

            beta, mu_prime, covariance_prime = agglomerative_init(alpha.copy(), mu.copy(), covariance.copy(), n, k)
    except Timeout.Timeout:
        print ("agglomerative_init Timeout - using fallback")
        return None, None, None

    try:
        with Timeout(10):

            beta_init = beta.copy()
            mu_prime_init = mu_prime.copy()
            covariance_prime_init = covariance_prime.copy()
            iteration = 0
            d_val = float('inf')
            delta = float('inf')
            min_kl_cache = {}
            while delta > epsilon and iteration < max_iter:
                iteration += 1
                clusters, clusters_inv = e_step(alpha, beta, covariance, covariance_prime, mu, mu_prime, min_kl_cache)
                m_step(alpha, beta, clusters, covariance, covariance_prime, mu, mu_prime)

                prev_d_val = d_val
                d_val = 0
                for t, (alpha_, mu_, cov_) in enumerate(zip(alpha, mu, covariance)):
                    min_dist, selected_cluster = min_kl(beta, cov_, covariance_prime, mu_, mu_prime)
                    min_kl_cache[t] = (min_dist, selected_cluster)
                    d_val += alpha_ * min_dist
                delta = prev_d_val - d_val
                if delta < 0:
                    print('EM bug - not monotonic- using fallback')
                    return beta_init, mu_prime_init, covariance_prime_init
                #Log.debug('Iteration {}, d_val={}, delta={}, k={}, n={}'.format(iteration, d_val, delta, k, n))

            if delta > epsilon:
                print('EM did not converge- using fallback')
                return beta_init, mu_prime_init, covariance_prime_init
    except Timeout.Timeout:
        print ("EM Timeout - using fallback")
    return beta, mu_prime, covariance_prime


def e_step(alpha, beta, covariance, covariance_prime, mu, mu_prime, min_kl_cache):
    clusters = {}
    clusters_inv = {}
    for t, (alpha_, mu_, cov_) in enumerate(zip(alpha, mu, covariance)):
        if t in min_kl_cache:
            min_dist, selected_cluster = min_kl_cache[t]
        else:
            min_dist, selected_cluster = min_kl(beta, cov_, covariance_prime, mu_, mu_prime)

        if selected_cluster not in clusters:
            clusters[selected_cluster] = []
        clusters[selected_cluster].append(t)
        clusters_inv[t] = selected_cluster
    return clusters, clusters_inv


def min_kl(beta, cov_, covariance_prime, mu_, mu_prime):
    cov_g = numpy.zeros_like(mu_prime)
    cov_g[:, 0] = covariance_prime[:, 0, 0]
    cov_g[:, 1] = covariance_prime[:, 1, 1]

    cov_f = numpy.zeros_like(mu_prime)
    cov_f[:, 0] = cov_[0, 0]
    cov_f[:, 1] = cov_[1, 1]

    mu_f = numpy.zeros_like(mu_prime)
    mu_f[:, 0] = mu_[0]
    mu_f[:, 1] = mu_[1]
    mu_g = mu_prime

    cov_g_sqrt = numpy.sqrt(cov_g)
    cov_f_sqrt = numpy.sqrt(cov_f)
    log_ratio = numpy.log(cov_g_sqrt[:, 0] / cov_f_sqrt[:, 0]) + numpy.log(cov_g_sqrt[:, 1] / cov_f_sqrt[:, 1])
    delta_mu = mu_f - mu_g
    delta_mu_square = delta_mu * delta_mu
    div = (cov_f[:, 0] + delta_mu_square[:, 0]) / (2 * cov_g[:, 0]) + (cov_f[:, 1] + delta_mu_square[:, 1]) / (
            2 * cov_g[:, 1])
    kl = div + log_ratio
    return kl.min(), kl.argmin()


def m_step(alpha, beta, clusters, covariance, covariance_prime, mu, mu_prime):
    for j, t_vals in clusters.items():
        beta_update = 0
        for t in t_vals:
            beta_update += alpha[t]
        beta[j] = beta_update

        mu_update = numpy.array([0, 0])
        for t in t_vals:
            mu_update = numpy.add(mu_update, alpha[t] * mu[t])
        mu_update /= beta[j]
        mu_prime[j] = mu_update

        cov_update = numpy.array([[0, 0], [0, 0]])
        for t in t_vals:
            delta_mu = (mu[t] - mu_prime[j])
            delta_mu = numpy.expand_dims(delta_mu, axis=1)
            cov_update = numpy.add(cov_update, alpha[t] * (covariance[t] + numpy.dot(delta_mu, delta_mu.transpose())))
        cov_update /= beta[j]
        covariance_prime[j] = cov_update
