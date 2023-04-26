import numpy as np


class Metrics:
    def __init__(self, init=0):
        self.init = init

    def mpjpe(self, p_ref, p, scale=False, mean_align=False):
        import numpy as np

        # reshape pose if necessary
        if p.shape[0] == 1:
            p = p.reshape(3, int(p.shape[1] / 3))
        if p_ref.shape[0] == 1:
            p_ref = p_ref.reshape(3, int(p_ref.shape[1] / 3))

        if mean_align:
            p = p - p.mean(axis=1, keepdims=True)
            p_ref = p_ref - p_ref.mean(axis=1, keepdims=True)
        if scale:
            scale_p = np.linalg.norm(p.reshape(-1, 1), ord=2)
            scale_p_ref = np.linalg.norm(p_ref.reshape(-1, 1), ord=2)
            scale = scale_p_ref/scale_p
            p = p * scale

        sum_dist = 0

        for i in range(p.shape[1]):
            sum_dist += np.linalg.norm(p[:, i] - p_ref[:, i], 2)

        err = np.sum(sum_dist) / p.shape[1]

        return err

    def pmpjpe(self, p_ref, p, reflection=False):
        # reshape pose if necessary
        if p.shape[0] == 1:
            p = p.reshape(3, int(p.shape[1] / 3))

        if p_ref.shape[0] == 1:
            p_ref = p_ref.reshape(3, int(p_ref.shape[1] / 3))

        d, Z, tform = self.procrustes(p_ref.T, p.T, reflection=reflection)
        err = self.mpjpe(p_ref, Z.T)

        return err

    #def PCK(self, p_ref, p, reflection=False):
    #    # reshape pose if necessary
    #    if p.shape[0] == 1:
    #        p = p.reshape(3, int(p.shape[1] / 3))
    #
    #    if p_ref.shape[0] == 1:
    #        p_ref = p_ref.reshape(3, int(p_ref.shape[1] / 3))
    #
    #    d, Z, tform = self.procrustes(p_ref.T, p.T, reflection=reflection)

    #    err = self.mpjpe(p_ref, Z.T)

    #    return err

    def procrustes(self, X, Y, scaling=True, reflection='best'):
        """
        A port of MATLAB's `procrustes` function to Numpy.

        Procrustes analysis determines a linear transformation (translation,
        reflection, orthogonal rotation and scaling) of the points in Y to best
        conform them to the points in matrix X, using the sum of squared errors
        as the goodness of fit criterion.

            d, Z, [tform] = procrustes(X, Y)

        Inputs:
        ------------
        X, Y
            matrices of target and input coordinates. they must have equal
            numbers of  points (rows), but Y may have fewer dimensions
            (columns) than X.

        scaling
            if False, the scaling component of the transformation is forced
            to 1

        reflection
            if 'best' (default), the transformation solution may or may not
            include a reflection component, depending on which fits the data
            best. setting reflection to True or False forces a solution with
            reflection or no reflection respectively.

        Outputs
        ------------
        d
            the residual sum of squared errors, normalized according to a
            measure of the scale of X, ((X - X.mean(0))**2).sum()

        Z
            the matrix of transformed Y-values

        tform
            a dict specifying the rotation, translation and scaling that
            maps X --> Y

        """

        n, m = X.shape
        ny, my = Y.shape

        muX = X.mean(0)
        muY = Y.mean(0)

        X0 = X - muX
        Y0 = Y - muY

        ssX = (X0 ** 2.).sum()
        ssY = (Y0 ** 2.).sum()

        # centred Frobenius norm
        normX = np.sqrt(ssX)
        normY = np.sqrt(ssY)

        # scale to equal (unit) norm
        X0 /= normX
        Y0 /= normY

        if my < m:
            Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

        # optimum rotation matrix of Y
        A = np.dot(X0.T, Y0)
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        V = Vt.T
        T = np.dot(V, U.T)

        if reflection is not 'best':

            # does the current solution use a reflection?
            have_reflection = np.linalg.det(T) < 0

            # if that's not what was specified, force another reflection
            if reflection != have_reflection:
                V[:, -1] *= -1
                s[-1] *= -1
                T = np.dot(V, U.T)

        traceTA = s.sum()

        if scaling:

            # optimum scaling of Y
            b = traceTA * normX / normY

            # standarised distance between X and b*Y*T + c
            d = 1 - traceTA ** 2

            # transformed coords
            Z = normX * traceTA * np.dot(Y0, T) + muX

        else:
            b = 1
            d = 1 + ssY / ssX - 2 * traceTA * normY / normX
            Z = normY * np.dot(Y0, T) + muX

        # transformation matrix
        if my < m:
            T = T[:my, :]
        c = muX - b * np.dot(muY, T)

        # transformation values
        tform = {'rotation': T, 'scale': b, 'translation': c}

        return d, Z, tform
