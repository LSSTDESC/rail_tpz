"""
.. module:: utils_mlz
.. moduleauthor:: Matias Carrasco Kind
"""
__author__ = 'Matias Carrasco Kind'
import numpy as np
import time
import sys, os
from scipy.interpolate import interp1d as spl
from scipy.integrate import romberg as rom

try:
    from mpi4py import MPI

    PLL = 'MPI'
except:
    PLL = 'SERIAL'


def zconf_dist(conf, nbins):
    """
    Computes the distribution of Zconf for different bins between 0 and 1

    :param float conf: zConf values
    :param int nbins: number of bins
    :return: zConf dist, bins
    :rtype: float,float
    """
    bins = np.linspace(0., 1, nbins)
    s_conf = np.sort(conf)
    z_conf = np.zeros(len(bins))
    for i in range(len(bins)):
        z_conf[i] = percentile(s_conf, bins[i])
    return z_conf, bins


def get_probs(z, pdf, z1, z2):
    pdf = pdf / np.sum(pdf)
    PP = spl(z, pdf, bounds_error=False, fill_value=0.0)
    dzo = z[1] - z[0]
    dz = 0.001
    Ndz = int((z2 - z1) / dz)
    A = 0
    for i in range(Ndz):
        A += dz * PP((z1) + dz / 2. + dz * i)
    return A / dzo


def get_prob_Nz(z, pdf, zbins):
    pdf = pdf / np.sum(pdf)
    PP = spl(z, pdf, bounds_error=False, fill_value=0.0)
    dzo = z[1] - z[0]
    dz = 0.001
    Ndz = int((zbins[1] - zbins[0]) / dz)
    Nzt = np.zeros(len(zbins) - 1)
    for j in range(len(Nzt)):
        for i in range(Ndz):
            Nzt[j] += dz * PP((zbins[j]) + dz / 2. + dz * i)
    return Nzt / dzo


def compute_error(z, pdf, zv):
    """
    Computes the error in the PDF calculation using a reference values from PDF
    it computes the 68% percentile limit around this value

    :param float z: redshift
    :param float pdf: photo-z PDF
    :param float zv: Reference value from PDF (can be mean, mode, median, etc.)
    :return: error associated to reference value
    :rtype: float
    """
    res = 0.001
    PP = spl(z, pdf, bounds_error=False, fill_value=0.0)
    dz = z[1] - z[0]
    j = 0
    area = 0
    while area <= 0.68:
        j += 1
        za = zv - res * j
        zb = zv + res * j
        area = rom(PP, za, zb, tol=1.0e-04, rtol=1.0e-04) / dz
    return j * res


def compute_error2(z, pdf, zv):
    L1 = 0.0001
    L2 = (max(z) - min(z)) / 2.
    PP = spl(z, pdf, bounds_error=False, fill_value=0.0)
    dz = z[1] - z[0]
    eps = 0.05
    za1 = zv - L1
    zb1 = zv + L1
    area = 0
    LM = L2
    while abs(area - 0.68) > eps:
        za2 = zv - LM
        zb2 = zv + LM
        area = rom(PP, za2, zb2, tol=1.0e-04, rtol=1.0e-04) / dz
        Lreturn = LM
        if area > 0.68:
            L2 = LM
            LM = (L1 + L2) / 2.
        else:
            L1 = LM
            LM = (L1 + L2) / 2.
    return Lreturn


def compute_error3(z, pdf, zv):
    dz = z[1] - z[0]
    ib = np.argmin(abs(zv - z))
    area = pdf[ib]
    nm = len(z) - 1
    j = 0
    i2 = ib + 1
    i1 = ib
    if sum(pdf) < 0.00001 : return 9.99
    while area <= 0.68:
        area1 = sum(pdf[i1:i2])
        e681 = dz * (i2 - i1)
        j += 1
        i1 = max(0, ib - j)
        i2 = min(nm, ib + j) + 1
        area = sum(pdf[i1:i2])
        e68 = dz * (i2 - i1)
        ef = ((e68 - e681) / (area - area1)) * (0.68 - area1) + e681
    return ef / 2.


def compute_zConf(z, pdf, zv, sigma):
    """
    Computes the confidence level of the pdf with respect a reference value
    as the area between zv-sigma(1+zv) and zv+sigma(1+zv)

    :param float z: redshift
    :param float pdf: photo-z PDF
    :param float zv: reference value
    :param float sigma: extent of confidence
    :return: zConf
    :rtype: float
    """
    z1a = zv - sigma * (1. + zv)
    z1b = zv + sigma * (1. + zv)
    zC1 = get_area(z, pdf, z1a, z1b)
    return zC1


def compute_zConf2(z, pdf, zv, sigma):
    z1a = zv - sigma * (1. + zv)
    z1b = zv + sigma * (1. + zv)
    ib1 = np.argmin(abs(z1a - z))
    ib2 = np.argmin(abs(z1b - z)) + 1
    return sum(pdf[ib1:ib2])


def get_limits(ntot, Nproc, rank):
    """
    Get limits for farming an array to multiple processors

    :param int ntot: Number of objects in array
    :param int Nproc: number of processor
    :param int rank: current processor id
    :return: L1,L2 the limits of the array for given processor
    :rtype: int, int
    """
    jpproc = np.zeros(Nproc) + int(ntot / Nproc)
    for i in range(Nproc):
        if (i < ntot % Nproc): jpproc[i] += 1
    jpproc = [int(x) for x in jpproc]
    st = rank
    st = np.sum(jpproc[:rank]) - 1
    s0 = int(st + 1)
    s1 = int(st + jpproc[rank]) + 1
    return s0, s1
