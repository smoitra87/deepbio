# ----------------------------------------------------
# Training a Centered Deep Boltzmann Machine
# ----------------------------------------------------
#
# Copyright: Gregoire Montavon
#
# This code is released under the MIT licence:
# http://www.opensource.org/licenses/mit-license.html
#
# ----------------------------------------------------
#
# This code is based on the paper:
#
#   G. Montavon, K.-R. Mueller
#   Deep Boltzmann Machines and the Centering Trick
#   in Neural Networks: Tricks of the Trade, 2nd Edn
#   Springer LNCS, 2012
#
# ----------------------------------------------------
#
# This code is a basic implementation of the centered
# deep Boltzmann machines (without model averaging,
# minibatching and other optimization hacks). The code
# also requires the MNIST dataset that can be
# downloaded at http://yann.lecun.com/exdb/mnist/.
#
# ----------------------------------------------------

import numpy
from PIL import Image
#msaf = "allprotease.msa"
trainf = "sim1.train.msa"
validf = "sim1.valid.msa"
testf = "sim1.test.msa"
import util
import numpy as np

# ====================================================
# Global parameters
# ====================================================
lr = 0.005     # learning rate
rr = 0.005     # reparameterization rate
mb = 25        # minibatch size
hlayers = [400, 100]  # size of hidden layers
biases = [-1, -1]   # initial biases on hidden layers
dimvis = 20

# ====================================================
# Helper functions
# ====================================================


def arcsigm(x):
    return numpy.arctanh(2*x-1)*2


def sigm(x):
    return (numpy.tanh(x/2)+1)/2

def softmax(x, dim=0):
    """
    x - activations to run softmax over
    dim: Dimension along which to calculate softmax
    """
    y = np.exp(x)
    return y / np.expand_dims(y.sum(axis=dim)+1, axis=dim)

def realize(x):
    return (x > numpy.random.uniform(0, 1, x.shape))*1.0

def realize_softmax(x, dim=0):
    #TODO  Might be buggy
    shape = list(x.shape)
    shape[dim]=1
    samples = np.asarray(np.cumsum(x, axis=dim) > np.random.rand(*shape),dtype=np.float)
    indices = tuple(slice(d) if idx!=dim else slice(0,d-1) for idx,d in enumerate(x.shape))
    indices2 = tuple(slice(d) if idx!=dim else slice(1,d) for idx,d in enumerate(x.shape))
    samples[indices2] -= samples[indices]
    return samples

def render(x, name):
    x = x - x.min() + 1e-9
    x = x / (x.max() + 1e-9)
    Image.fromarray((x*255).astype('byte'), 'L').save(name)

# ====================================================
# Centered deep Boltzmann machine
# ----------------------------------------------------
# - self.W: list of weight matrices between layers
# - self.B: list of bias associated to each unit
# - self.O: list of offsets associated to each unit
# - self.X: free particles tracking model statistics
# ====================================================


class DBM:
    # --------------------------------------------
    # Initialize model parameters and particles
    # --------------------------------------------

    def __init__(self, M, B):
        self.M = M
        self.W = [np.zeros((dimvis-1, M[0],M[1]))] + [numpy.zeros([m, n]).astype('float32')
                  for m, n in zip(M[1:-1], M[2:])]
        self.B = [np.zeros([dimvis-1,M[0]]) + B[0]] + [numpy.zeros([m]).astype('float32')+b for m, b in zip(M[1:], B[1:])]
        self.O = [softmax(B[0])] + [sigm(b) for b in self.B[1:]]

        self.X = [np.zeros([mb, dimvis-1, M[0]]) + self.O[0]] + \
                [numpy.zeros([mb, m]).astype(
            'float32')+o for m, o in zip(M[1:], self.O[1:])]

    # --------------------------------------------
    # Gibbs activation of a layer
    # --------------------------------------------
    def gibbs(self, X, l):
        if l==0 :
            td = np.tensordot(X[1]-self.O[1], self.W[0], axes=([1],[2]))
            X[0] = realize_softmax(softmax(td+self.B[0], dim=1), dim=1)
        else:
            if l==1:
                bu = np.tensordot(X[0]-self.O[0], self.W[0], axes=([1,2],[0,1]))
            else :
                bu = numpy.dot(X[l-1]-self.O[l-1], self.W[l-1])
            td = numpy.dot(X[l+1]-self.O[l+1], self.W[l].T) if l+1 < len(X) else 0
            X[l] = realize(sigm(bu+td+self.B[l]))

    # --------------------------------------------
    # Reparameterization
    # --------------------------------------------
    def reparamB(self, X, l):
        if l==0 :
            td = np.tensordot(X[1]-self.O[1], self.W[0], \
                              axes=([1],[2])).mean(axis=0)
            bu = 0.
        else:
            if l==1:
                bu = np.tensordot(X[0]-self.O[0], self.W[0], \
                                  axes=([1,2],[0,1])).mean(axis=0)
            else :
                bu = numpy.dot(X[l-1]-self.O[l-1], self.W[l-1]).mean(axis=0)
            td = numpy.dot(X[l+1]-self.O[l+1], self.W[l].T).mean(axis=0) if l+1 < len(X) else 0

        self.B[l] = (1-rr)*self.B[l] + rr*(self.B[l] + bu + td)

    def reparamO(self, X, i):
        self.O[i] = (1-rr)*self.O[i] + rr*X[i].mean(axis=0)

    # --------------------------------------------
    # Learning step
    # --------------------------------------------
    def learn(self, Xd):

        # Initialize a data particle
        X = [Xd]+[self.X[l]*0+self.O[l]
                           for l in range(1, len(self.X))]

        # Alternate gibbs sampler on data and free particles
        for l in (range(1, len(self.X), 2)+range(2, len(self.X), 2))*5:
            self.gibbs(X, l)
        for l in (range(1, len(self.X), 2)+range(0, len(self.X), 2))*1:
            self.gibbs(self.X, l)

        # Parameter update
        self.W[0] += lr*(np.tensordot(X[0]-self.O[0], X[1]-self.O[1], axes=([0],[0])) -
                        np.tensordot(self.X[0]-self.O[0], self.X[1]-self.O[1], axes=([0],[0])))/len(Xd)
        for i in range(1, len(self.W)):
            self.W[i] += lr*(numpy.dot((X[i]-self.O[i]).T,     X[i+1]-self.O[i+1]) -
                             numpy.dot((self.X[i]-self.O[i]).T, self.X[i+1]-self.O[i+1]))/len(Xd)
        for i in range(0, len(self.B)):
            self.B[i] += lr*(X[i]-self.X[i]).mean(axis=0)

        # Reparameterization
        for l in range(0, len(self.B)):
            self.reparamB(X, l)
        for l in range(0, len(self.O)):
            self.reparamO(X, l)

    def samples(self, Xd, nSamples=1, burnin=100, skip=10):
        # Initialize a data particle
        numInput = len(Xd)
        X = [Xd]+[np.zeros([numInput,self.M[l]]) +self.O[l]
                           for l in range(1, len(self.M))]
        # Alternate gibbs sampler on data and free particles
        for l in (range(1, len(self.X), 2)+range(2, len(self.X), 2))*burnin:
            self.gibbs(X, l)

        # Sample
        samples = []
        for idx in range(nSamples):
            X[0] = Xd
            for _ in range(skip):
                for l in (range(1, len(self.X), 2)+range(2, len(self.X), 2))*skip:
                    self.gibbs(X, l)
            self.gibbs(X, 0)
            samples.append(X[0])
        return samples

    def reconstuction_err(self, Xd, nSamples=100, burnin=100, skip=1):
        # Initialize a data particle
        numInput = len(Xd)
        X = [Xd]+[np.zeros([numInput,self.M[l]]) +self.O[l]
                           for l in range(1, len(self.M))]
        # Alternate gibbs sampler on data and free particles
        for l in (range(1, len(self.X), 2)+range(2, len(self.X), 2))*burnin:
            self.gibbs(X, l)

        # Sample
        samples_l1 = np.zeros_like(X[1])
        for idx in range(nSamples):
            for _ in range(skip):
                for l in (range(1, len(self.X), 2)+range(2, len(self.X), 2))*skip:
                    self.gibbs(X, l)
            samples_l1 += X[1]

        samples_l1 /= (nSamples+0.)
        td = np.tensordot(samples_l1-self.O[1], self.W[0], axes=([1],[2]))
        p_v_given_h1 = softmax(td+self.B[0], dim=1)
        logProbs = np.sum(Xd*np.log(p_v_given_h1),axis=1) +\
            (1-Xd.sum(axis=1))*np.log(1-p_v_given_h1.sum(axis=1))
        ch = -logProbs.mean()
        return ch

# ====================================================
# Example of execution
# ====================================================

def tensor_to_msa(Xd):
    """ Converts the data 3D visual tensor to 2D msa"""
    return dimvis-1-np.cumsum(Xd,axis=1).sum(axis=1)

def msa_to_tensor(X):
    XX = np.zeros([X.shape[0], dimvis-1, X.shape[1]])
    for idx1, seq in enumerate(X):
        for idx3, idx2 in enumerate(seq):
            if idx2 == dimvis-1 : continue
            XX[idx1,idx2,idx3] = 1.
    assert XX.sum() == (X!=dimvis-1).sum()
    return XX

def get_counts(X):
    ninst, nvis = X.shape
    counts = []
    for i in range(dimvis):
        counts.append((X==i).sum(axis=0))
    counts = np.asarray(counts, dtype=np.float)
    counts = np.clip(counts/X.shape[0],0.01,0.99)
    counts /= counts.sum(axis=0).reshape(1,nvis)
    return counts


# Load sample msa matrices
Xtrain = util.convert_mat_from_msa(trainf)
Xvalid = util.convert_mat_from_msa(validf)

ninst, nvis = Xtrain.shape
counts = get_counts(Xtrain)

# Get biases of softmax function
assert(counts.shape[0] - 1 == dimvis-1)
Bvis = np.zeros((dimvis-1, nvis))
for i in range(nvis):
    a = np.ones([dimvis-1,dimvis-1]) - np.diag(1./counts[:-1,i])
    b = np.zeros(dimvis-1)-1.
    Bvis[:,i] = np.log(np.linalg.solve(a,b))

err = ((softmax(Bvis) - counts[:-1])**2).sum()
assert err < 1e-8

nn = DBM([nvis]+hlayers, [Bvis]+biases)

# Convert X matrix to 3D
Xtrain = msa_to_tensor(Xtrain)
Xvalid = msa_to_tensor(Xvalid)


import os
with open("results_"+os.path.basename(__file__),'w') as fout:


    for it in range(1000):
        # Perform some learning steps
        for _ in range(100):
            nn.learn(Xtrain[numpy.random.permutation(len(Xtrain))[:mb]])

        # Output some debugging information
        if it%5 == 0:

            train_err = nn.reconstuction_err(Xtrain[:len(Xtrain)/4])
            valid_err = nn.reconstuction_err(Xvalid)
            print(("%03d |" + " %.3f "*len(nn.W) + " | %.3f %.3f") %
                tuple([it]+[W.std() for W in nn.W] + [train_err, valid_err]))
            print >>fout, "%d %.3f %.3f"%(it, train_err, valid_err)
            fout.flush()

        else :
            print(("%03d |" + " %.3f "*len(nn.W)) %
                tuple([it]+[W.std() for W in nn.W]))

