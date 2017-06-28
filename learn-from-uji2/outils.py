
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf


def render2dmodel(model, X, Y=None):
    # create a mesh to plot in
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.25)
    plt.scatter(X[:,0], X[:,1], c=Y, marker='x')
    plt.xlabel("Caractéristique 1")
    plt.ylabel("Caractéristique 2")
    if Y is not None:
        errs = np.sum(model.predict(X) != Y)
        plt.title("Nombre d'erreurs : "+str(errs))
    plt.show()

def render2dmodel_tf(session, model, x, X, Y=None):
    # create a mesh to plot in
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = session.run(model, feed_dict={x: np.c_[xx.ravel(), yy.ravel()]})
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.25)
    plt.scatter(X[:,0], X[:,1], c=Y, marker='x')
    plt.xlabel("Caractéristique 1")
    plt.ylabel("Caractéristique 2")
    if Y is not None:
        errs = np.sum(session.run(model, feed_dict={x: X}) != Y)
        plt.title("Nombre d'erreurs : "+str(errs))
    plt.show()


def one_hot(v, N):
    def one_hot_int(i):
        res = np.zeros(N)
        res[i] = 1.0
        return res
    return np.array([one_hot_int(i) for i in v])
