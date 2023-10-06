"""
.. module:: analysis
.. moduleauthor:: Matias Carrasco Kind
"""
__author__ = 'Matias Carrasco Kind'
#from numpy import *
import numpy as np
import os
import datetime

from scipy.stats import mode
from astropy.io import fits as pf

from . import utils_mlz


def get_zbins(Pars):
    Nbins = int(Pars.nzbins)
    zfine = np.linspace(Pars.zmin, Pars.zmax, Nbins + 1)
    resz = zfine[1] - zfine[0]
    resz2 = resz / 1.
    zfine2 = np.arange(Pars.zmin - resz2 * 20.-resz2/2., Pars.zmax + resz2 * 20., resz2)
    wzin = np.where((zfine2 >= Pars.zmin) & (zfine2 <= Pars.zmax))[0]
    return zfine, zfine2, resz, resz2, wzin


class GetPz():
    """
    Computes PDF and general results given a set of predicted values for each object in a dictionary format

    :param dict dict_zp: Dictionary with predicted values, keys are object index
    :param int nobj: Number of objetcs
    :param class Pars: class Pars with information taken from the :ref:`input-file`
    """

    def __init__(self, dict_zp, nobj, Pars):
        zfine, zfine2, resz, resz2, wzin = get_zbins(Pars)
        self.nD = nobj
        self.Pars = Pars
        self.Nbins = int(self.Pars.nzbins)
        self.zfine = zfine
        self.resz = resz
        self.zbins = (self.zfine[1:] + self.zfine[:-1]) * 0.5
        self.resolution = resz2
        self.zfine2 = zfine2
        self.wzin = wzin
        self.sigma_g = self.Pars.sigmafactor * self.resolution
        self.dz = self.resolution
        self.x = np.arange(-3*self.sigma_g - self.dz/10, 
                           3*self.sigma_g + self.dz/10, self.dz)
        self.gaus2 = np.exp(-0.5*(self.x/self.sigma_g)**2)
        self.zp = np.zeros(self.nD) - 1
        self.zs = np.zeros(self.nD)
        self.dict_zp = dict_zp
        self.fac = np.zeros(self.nD)
        self.bigpdf = np.zeros(len(self.zbins))
        self.bigpdf2 = np.zeros((self.nD, len(self.zfine2[self.wzin])))
        self.zs0 = np.zeros(self.nD)
        self.zs1 = np.zeros(self.nD)
        self.err0 = np.zeros(self.nD)
        self.err1 = np.zeros(self.nD)
        self.zConf0 = np.zeros(self.nD)
        self.zConf1 = np.zeros(self.nD)

    def compute(self, do_pdf='yes'):
        for i in range(self.nD):
            self.bigpdf = np.zeros(len(self.zbins))
            if i in self.dict_zp:
                out = np.array(self.dict_zp[i]['zp'])
                #wout=array(self.dict_zp[i]['wp'])
                if 'zs' in self.dict_zp[i]: 
                    self.zs[i] = self.dict_zp[i]['zs']

                if self.Pars.predictionclass == 'Reg':
                    for zpi in range(len(out)):
                        mybin = int(np.floor(out[zpi] / self.resz))

                        if mybin > self.Nbins - 1: continue
                        self.bigpdf[mybin] += 1.

                    pdf = self.bigpdf 
                    pdf2 = np.interp(self.zfine2, self.zbins, pdf)
                    pdf2 = np.where(np.greater(pdf2, np.max(pdf2) * 0.01), pdf2, 0.)
                    pdf2 = np.convolve(pdf2, self.gaus2, 1)
                    pdf2 = np.where(np.greater(pdf2, np.max(pdf2) * 0.005), pdf2, 0.)
                    if sum(pdf2) > 0.: pdf2 /= np.sum(pdf2)
                    self.zs0[i] = self.zfine2[np.argmax(pdf2)]
                    self.zs0[i] = np.min(self.zs0[i], self.Pars.zmax)
                    self.zs0[i] = np.max(self.zs0[i], self.Pars.zmin)
                    self.zs1[i] = np.sum(self.zfine2 * pdf2)
                    self.zs1[i] = np.min(self.zs1[i], self.Pars.zmax)
                    self.zs1[i] = np.max(self.zs1[i], self.Pars.zmin)
                    if do_pdf == 'yes':
                        self.err0[i] = utils_mlz.compute_error(self.zfine2, pdf2, self.zs0[i])
                        self.err1[i] = utils_mlz.compute_error(self.zfine2, pdf2, self.zs1[i])
                        self.zConf0[i] = utils_mlz.compute_zConf(self.zfine2, pdf2, self.zs0[i], self.Pars.rmsfactor)
                        self.zConf1[i] = utils_mlz.compute_zConf(self.zfine2, pdf2, self.zs1[i], self.Pars.rmsfactor)
                        pdf2 = pdf2[self.wzin]
                        if sum(pdf2) > 0.: pdf2 /= sum(pdf2)
                        self.bigpdf2[i, :] = pdf2
                if self.Pars.predictionclass == 'Class':
                    if len(out) > 0:
                        self.zs0[i] = mode(out * 1.)[0][0]
                        self.zs1[i] = np.mean(out * 1.)
                    if len(out) > 0.: self.err0[i] = mode(out * 1.)[1][0]/len(out)
                    self.err1[i] = np.std(out * 1.)

        bigZ = np.zeros((self.nD, 7))
        bigZ[:, 0] = self.zs
        bigZ[:, 1] = self.zs0
        bigZ[:, 2] = self.zs1
        bigZ[:, 3] = self.zConf0
        bigZ[:, 4] = self.zConf1
        bigZ[:, 5] = self.err0
        bigZ[:, 6] = self.err1
        if do_pdf == 'no':
            return bigZ
        else:
            return bigZ, self.bigpdf2


class GetPz_short():
    """
    Computes PDF and general results given a set of predicted values for each object in a dictionary format

    :param dict dict_zp: Dictionary with predicted values, keys are object index
    :param int nobj: Number of objetcs
    :param class Pars: class Pars with information taken from the :ref:`input-file`
    """

    def __init__(self, Pars):
        zfine, zfine2, resz, resz2, wzin = get_zbins(Pars)
        self.Pars = Pars
        self.Nbins = int(self.Pars.nzbins)
        self.zfine = zfine
        self.resz = resz
        self.zbins = (self.zfine[1:] + self.zfine[:-1]) * 0.5
        self.resolution = resz2
        self.zfine2 = zfine2
        self.wzin = wzin
        self.sigma_g = self.Pars.sigmafactor * self.resolution
        self.dz = self.resolution
        self.x = np.arange(-3*self.sigma_g - self.dz/10, 
                            3*self.sigma_g + self.dz/10, self.dz)
        self.gaus2 = np.exp(-0.5*(self.x/self.sigma_g)**2)
        self.bigpdf = np.zeros(len(self.zbins))
        self.bigpdf2 = np.zeros(len(self.zfine2[self.wzin]))

    def get_hist(self, vals):
        self.bigpdf = np.zeros(len(self.zbins))
        out = np.array(vals)
        #if self.Pars.predictionclass == 'Reg':
        for zpi in range(len(out)):
            mybin = int(np.floor(out[zpi] / self.resz))
            if mybin > self.Nbins - 1: continue
            self.bigpdf[mybin] += 1
        return self.bigpdf

    def get_pdf(self, rawpdf, zs=0, zbins=''):
        pdf = rawpdf
        self.zs = zs
        if zbins != '': self.zbins = zbins
        pdf2 = np.interp(self.zfine2, self.zbins, pdf)
        pdf2 = np.where(np.greater(pdf2, np.max(pdf2) * 0.20), pdf2, 0.)
        pdf2 = np.convolve(pdf2, self.gaus2, 1)
        pdf2 = np.where(np.greater(pdf2, np.max(pdf2) * 0.005), pdf2, 0.)
        if np.sum(pdf2) > 0.: pdf2 /= np.sum(pdf2)
        self.zs0 = self.zfine2[np.argmax(pdf2)]
        self.zs0 = min(self.zs0, self.Pars.zmax)
        self.zs0 = max(self.zs0, self.Pars.zmin)
        self.zs1 = sum(self.zfine2 * pdf2)
        self.zs1 = min(self.zs1, self.Pars.zmax)
        self.zs1 = max(self.zs1, self.Pars.zmin)
        self.err0 = utils_mlz.compute_error3(self.zfine2, pdf2, self.zs0)
        self.err1 = utils_mlz.compute_error3(self.zfine2, pdf2, self.zs1)
        self.zConf0 = utils_mlz.compute_zConf2(self.zfine2, pdf2, self.zs0, self.Pars.rmsfactor)
        self.zConf1 = utils_mlz.compute_zConf2(self.zfine2, pdf2, self.zs1, self.Pars.rmsfactor)
        pdf2 = pdf2[self.wzin]
        if sum(pdf2) > 0.: pdf2 /= sum(pdf2)
        self.bigpdf2 = pdf2
        bigZ = np.zeros(7)
        bigZ[0] = self.zs
        bigZ[1] = self.zs0
        bigZ[2] = self.zs1
        bigZ[3] = self.zConf0
        bigZ[4] = self.zConf1
        bigZ[5] = self.err0
        bigZ[6] = self.err1
        return bigZ, self.bigpdf2


def class_stat(s1, s2, nv, Pars):
    z1 = s1 // nv
    midp = 0.5 * (Pars.zmin + Pars.zmax)
    z0 = np.where(z1 >= midp, Pars.zmax, Pars.minz)
    s0 = np.sqrt((s2 - nv*z1*z1)/nv)
    return z0, z1, s0


def save_single(Zall, Pars, path='', fileout='', oob='no', var=''):
    """
    Save single estimation results
    """
    ztrue = Zall[:, 0]
    zs0 = Zall[:, 1]
    zs1 = Zall[:, 2]
    zC0 = Zall[:, 3]
    zC1 = Zall[:, 4]
    e0 = Zall[:, 5]
    e1 = Zall[:, 6]
    if path == '': path = Pars.path_results
    if not os.path.exists(path): os.system('mkdir -p ' + path)
    if fileout == '':
        filebase = Pars.finalfilename
        for j in range(100):
            if os.path.exists(path + filebase + '.' + str(j) + '.mlz'):
                continue
            else:
                if oob == 'no': fileout = path + filebase + '.' + str(j) + '.mlz'
                if oob == 'yes': fileout = path + filebase + '_oob' + var + '.' + str(j) + '.mlz'
                fileinputs = path + filebase + '.' + str(j) + '.inputs'
                break
    else:
        filebase = fileout
        if oob == 'no': fileout = path + filebase + '.mlz'
        if oob == 'yes': fileout = path + filebase + '_oob' + '.mlz'
        fileinputs = path + filebase + '.inputs'

    if oob == 'no': utils_mlz.print_dtpars(Pars, fileinputs)
    fout = open(fileout, 'w')
    header = '# ztrue     zmode0    zmean1     zConf0    zConf1    err0    err1\n'
    fout.write(header)
    for i in range(len(zs0)):
        Line = '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'
        Line = Line.format(ztrue[i], zs0[i], zs1[i], zC0[i], zC1[i], e0[i], e1[i])
        fout.write(Line + '\n')
    fout.close()


def get_path_new(Pars):
    path = Pars.path_results
    if not os.path.exists(path): os.system('mkdir -p ' + path)
    filebase = Pars.finalfilename
    for j in range(100):
        if os.path.exists(path + filebase + '.' + str(j) + '.mlz'):
            continue
        else:
            break
    return path, filebase, j


def save_PDF(zfine, pdfs, Pars, path='', filebase='', num=-1, oob='no', 
             var='', multiple='no', rank=0):
    """
    Saves photo-z PDFs
    """

    if path == '':
        path = Pars.path_results
    if not os.path.exists(path): os.system('mkdir -p ' + path)
    if filebase == '':
        filebase = Pars.finalfilename
    if num == -1:
        for j in range(100):
            if (os.path.exists(path + filebase + '.' + str(j) + '.mlz') and 
                os.path.exists(path + filebase + '.' + str(j) + '.P.npy')):
                continue
            else:
                fileoutPDF = path + filebase + '.' + str(j) + '.P'
                if oob == 'yes': fileoutPDF = path + filebase + '_oob' + var + '.' + str(j) + '.P'
                break
    else:
        fileoutPDF = path + filebase + '.' + str(num) + '.P'
        if oob == 'yes': fileoutPDF = path + filebase + '_oob' + var + '.' + str(num) + '.P'

    if multiple == 'yes': fileoutPDF = fileoutPDF + '_' + str(rank)
    pdfs = np.concatenate((pdfs, [zfine]))
    if Pars.writefits == 'no':
        np.save(fileoutPDF, pdfs)
    else:
        head = pf.Header()
        head['N_TOT'] = len(pdfs) - 1
        head['DZ'] = zfine[1]-zfine[0]
        head['NPOINTS'] = len(zfine)
        head['COMMENT'] = 'The last row of the table are the redshift positions'
        head['COMMENT'] = 'This file was created using MLZ'
        head['HISTORY'] = 'Created on ' + datetime.datetime.now().strftime("%Y-%m-%d  %H:%M")
        fmt = '%dE' % len(zfine)
        col0 = pf.Column(name='PDF values', format=fmt, array=pdfs)
        #table0 = pf.new_table(pf.ColDefs([col0]))
        table0 = pf.BinTableHDU.from_columns(pf.ColDefs([col0]))
        prihdu = pf.PrimaryHDU(header=head)
        hdulist = pf.HDUList([prihdu, table0])
        hdulist.writeto(fileoutPDF + '.fits', overwrite=True)


def save_PDF_sparse(zfine, pdfs, head, Pars, path='', filebase='', num=-1, oob='no', var='', multiple='no', rank=0):
    """
    Saves photo-z PDFs in sparse representation
    """

    if path == '':
        path = Pars.path_results
    if not os.path.exists(path): os.system('mkdir -p ' + path)
    if filebase == '':
        filebase = Pars.finalfilename
    if num == -1:
        for j in range(100):
            if (os.path.exists(path + filebase + '.' + str(j) + '.mlz') and 
                os.path.exists(path + filebase + '.' + str(j) + '.P.npy')):
                continue

            else:
                fileoutPDF = path + filebase + '.' + str(j) + '.Psparse'
                if oob == 'yes': fileoutPDF = path + filebase + '_oob' + var + '.' + str(j) + '.P'
                break
    else:
        fileoutPDF = path + filebase + '.' + str(num) + '.Psparse'
        if oob == 'yes': fileoutPDF = path + filebase + '_oob' + var + '.' + str(num) + '.P'

    if multiple == 'yes': fileoutPDF = fileoutPDF + '_' + str(rank)

    col1 = pf.Column(name='redshift', format='E', array=zfine)
    fmt = '%dJ' % head['N_SPARSE']
    col2 = pf.Column(name='Sparse_indices', format=fmt, array=pdfs)
    table1 = pf.BinTableHDU.from_columns(pf.ColDefs([col1]))
    table2 = pf.BinTableHDU.from_columns(pf.ColDefs([col2]))
    prihdu = pf.PrimaryHDU(header=head)
    hdulist = pf.HDUList([prihdu, table1, table2])
    hdulist.writeto(fileoutPDF + '.fits', overwrite=True)


def save_single_t(Zall, Pars, path='', fileout='', oob='no', var=''):
    """
    Save single estimation results for types
    """
    type_phot = Zall[:, 0]
    type_ml = Zall[:, 1]
    if path == '': path = Pars.path_results
    if not os.path.exists(path): os.system('mkdir -p ' + path)
    if fileout == '': filebase = Pars.finalfilename
    for j in range(100):
        if os.path.exists(path + filebase + '.' + str(j) + '.mlz'):
            continue
        else:
            if oob == 'no': fileout = path + filebase + '.' + str(j) + '.type'
            break
    fout = open(fileout, 'w')
    header = '# type_phot    type_ml\n'
    fout.write(header)
    for i in range(len(type_phot)):
        Line = '{:.4f} {:.4f}'.format(type_phot[i], type_ml[i])
        fout.write(Line + '\n')
    fout.close()
