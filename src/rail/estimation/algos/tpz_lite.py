"""
Port of *some* parts of MLZ/TPZ, not the entire codebase.
Much of the code is directly ported from TPZ, written
by Matias Carrasco-Kind and Robert Brunner, though then
ported to python 3 compatibility in a fork of a fork.

Missing from full MLZ:
-SOM method

Missing from full TPZ:
-no classification method, (only regression tree)
-no out of bag uncertainties
-no var importance sampling

"""

import numpy as np
import qp
from sklearn.tree import DecisionTreeRegressor
from ceci.config import StageParameter as Param
from rail.estimation.estimator import CatEstimator, CatInformer
from rail.core.common_params import SHARED_PARAMS

from .mlz_utils import data
from .mlz_utils import utils_mlz
from .mlz_utils import analysis
from .ml_codes import TPZ

# value copied from ceci/ceci/stage.py, it is set there but not carried around
MPI_PARALLEL = "mpi"


# this is handled internally by ceci's stage.py in PipelineStage
# replace rank with self._rank, size with self._size, comm with self._comm
# and PLL == 'MPI'  with self._parallel == 'mpi'
#try:
#    from mpi4py import MPI
#
#    PLL = 'MPI'
#except ImportError:  # pragma: no cover
#    PLL = 'SERIAL'
#
#if PLL == 'MPI':
#    comm = MPI.COMM_WORLD
#    size = comm.Get_size()
#    rank = comm.Get_rank()
#else:  # pragma: no cover
#    size = 1
#    rank = 0
#Nproc = size


bands = ['u', 'g', 'r', 'i', 'z', 'y']
def_train_atts = []
for band in bands:
    def_train_atts.append(f"mag_{band}_lsst")

def_err_dict = {}
for band in bands:
    def_err_dict[f"mag_{band}_lsst"] = f"mag_err_{band}_lsst"
def_err_dict["redshift"] = None


def make_index_dict(inputdict, datacols):
    """
    Function to constract the dictionary with column indices for each parameter
    and its associated error.  If a column does not have an error, e.g. the
    spectroscopic redshift, then it should have `None` in the input dict, and
    will assign -1 as the column index.  If a column is not in the input data
    ind and eind of -1 will both be assigned a value of -1
    Parameters
    ----------
    inputdict: dict
      dictionary consisting of keys with names of input column and value that is
      the value of the associated error name
    datacols: list
      the list of column names in the input data
    """
    colmapdict = {}
    for key, val in inputdict.items():
        if key not in datacols:  # pragma: no cover
            keyind = -1
            errind = -1
            colmapdict[key] = dict(type="real", ind=keyind, eind=errind)
            continue
        keyind = datacols.index(key)
        if val not in datacols:
            errind = -1
        else:
            errind = datacols.index(val)

        colmapdict[key] = dict(type="real", ind=keyind, eind=errind)
    return colmapdict


class TPZliteInformer(CatInformer):
    """Inform stage for TPZliteEstimator, this stage uses training
    data to train up a set of decision trees that are then stored
    as a pickled model file for use by the Estimator stage.

    ntrees controls how many bootstrap realizations are created from a
    single catalog realization to train one tree.
    nransom controls how many catalog realizations are created. Each
    random catalog consists of adding Gaussian scatter to each attribute
    based on its associated error column.  If the error column `eind` is
    -1 then a small error of 0.00005 is hardcoded into TPZ. The key
    attribute is not included in this random catalog creation.

    So, a total of nrandom*ntrees trees are trained and stored in the
    final model i.e. if nrandom=3 and ntrees=5 then 15 total trees
    are trained and stored.
    """
    name = "TPZliteInformer"
    config_options = CatInformer.config_options.copy()
    config_options.update(zmin=SHARED_PARAMS,
                          zmax=SHARED_PARAMS,
                          nzbins=SHARED_PARAMS,
                          nondetect_val=SHARED_PARAMS,
                          mag_limits=SHARED_PARAMS,
                          bands=SHARED_PARAMS,
                          err_bands=SHARED_PARAMS,
                          redshift_col=SHARED_PARAMS,
                          seed=Param(int, 8758, msg="random seed"),
                          use_atts=Param(list, def_train_atts,
                                         msg="attributes to use in training trees"),
                          err_dict=Param(dict, def_err_dict, msg="dictionary that contains the columns that will be used to \
                                         predict as the keys and the errors associated with that column as the values. \
                                         If a column does not havea an associated error its value shoule be `None`"),
                          nrandom=Param(int, 8, msg="number of random bootstrap samples of training data to create"),
                          ntrees=Param(int, 5, msg="number of trees to create"),
                          minleaf=Param(int, 5, msg="minimum number in terminal leaf"),
                          natt=Param(int, 3, msg="number of attributes to split for TPZ"),
                          sigmafactor=Param(float, 3.0, msg="Gaussian smoothing with kernel Sigma1*Resolution"),
                          rmsfactor=Param(float, 0.02, msg="RMS for zconf calculation"),
                          tree_strategy=Param(str, "native", msg="which decision tree function to use when constructing the forest, \
                                              valid choices are 'native' or 'sklearn'.  If 'native', use the trees written for TPZ,\
                                              if 'sklearn' then use sklearn's DecisionTreeRegressor")
                          )

    def __init__(self, args, comm=None):
        """Init function, init config stuff
        """
        CatInformer.__init__(self, args, comm=comm)
        self.szs = None
        self.treedata = None

    def run(self):
        """compute the best fit prior parameters
        """
        rng = np.random.default_rng(seed=self.config.seed)
        if self._parallel == MPI_PARALLEL:
            self._comm.Barrier()
        if self._rank == 0:
            print(f"self._parallel is {self._parallel}, number of processors we will use is {self._size}")

        if self.config.hdf5_groupname:
            training_data = self.get_data("input")[self.config.hdf5_groupname]
        else:  # pragma: no cover
            training_data = self.get_data("input")

        # replace non-detects with limiting mag and mag_err with 1.0
        for bandname, errname in self.config.err_dict.items():
            if bandname == self.config.redshift_col:
                continue
            if np.isnan(self.config.nondetect_val):  # pragma: no cover
                magmask = np.isnan(training_data[bandname])
                errmask = np.isnan(training_data[errname])
            else:
                magmask = np.isclose(training_data[bandname], self.config.nondetect_val)
                errmask = np.isclose(training_data[errname], self.config.nondetect_val)

            detmask = np.logical_or(magmask, errmask)
            training_data[bandname][detmask] = self.config.mag_limits[bandname]
            training_data[errname][detmask] = 1.0

        valid_strategies = ["sklearn", "native"]
        if self.config.tree_strategy not in valid_strategies:  # pragma: no cover
            raise ValueError(f"value of {self.config.tree_strategy} not valid! Valid values for tree_strategy are 'native' or 'sklearn'")
        if self.config.tree_strategy == "sklearn" and self._rank == 0:
            print("using sklearn decision trees")
        if self.config.tree_strategy == "native" and self._rank == 0:
            print("using native TPZ decision trees")

        # TPZ expects a param called `keyatt` that is just the redshift column, copy redshift_col
        self.config.keyatt = self.config.redshift_col
        # same for atts, now use_atts
        self.config.atts = self.config.use_atts

        trainkeys = list(training_data.keys())
        npdata = np.array(list(training_data.values()))

        for key in self.config.use_atts:  # pragma: no cover
            if key not in trainkeys:
                raise KeyError(f"attribute {key} not found in input data!")

        # ngal = len(training_data[self.config.ref_band])

        if self.config.redshift_col not in training_data.keys():  # pragma: no cover
            raise KeyError(f"redshift column {self.config.redshift_col} not found in input data!")

        # construct the attribute dictionary
        train_att_dict = make_index_dict(self.config.err_dict, trainkeys)
        traindata = data.catalog(self.config, npdata.T, trainkeys, self.config.use_atts, train_att_dict)
        #####
        # make random data
        # So make_random takes the error columns and just adds Gaussian scatter to the input (or 0.00005 if no error supplied)
        # it saves `nrandom` copies of this in a dictionary for each attribute for each galaxy
        # not how I would have done things, but we're keeping it to try to duplicate MLZ's code exactly.
        if self.config.nrandom > 1:
            if self._rank == 0:
                print(f"creating {self.config.nrandom} random realizations...")
                traindata.make_random(ntimes=int(self.config.nrandom))
                temprandos = traindata.BigRan
            else:  # pragma: no cover
                temprandos = None
        if self._parallel == MPI_PARALLEL:
            self._comm.Barrier()

        # Matias writes out randoms from make_random for rank=0, then reads them all back in from file so that all ranks have access,
        # that seems slow so, instead, let's just assign them here (after broadcasting to all):
        if self._parallel == MPI_PARALLEL:
            if self.config.nrandom > 1:
                temprandos = self._comm.bcast(temprandos, root=0)
        if self.config.nrandom > 1:
            traindata.BigRan = temprandos
        if self._parallel == MPI_PARALLEL:
            self._comm.Barrier()

        ntot = int(self.config.nrandom * self.config.ntrees)
        if self._rank == 0:
            print(f"making a total of {ntot} trees for {self.config.nrandom} random realizations * {self.config.ntrees} bootstraps")

        zfine, zfine2, resz, resz2, wzin = analysis.get_zbins(self.config)
        zfine2 = zfine2[wzin]

        # Add this assignment of Nproc to grab ceci's size param
        Nproc = self._size
        s0, s1 = utils_mlz.get_limits(ntot, Nproc, self._rank)
        if self._rank == 0:
            for i in range(Nproc):
                Xs_0, Xs_1 = utils_mlz.get_limits(ntot, Nproc, i)
                if Xs_0 == Xs_1:  # pragma: no cover
                    print(f"idle...  -------------> to core  {i}")
                else:
                    print(f"{Xs_0} - {Xs_1} -------------> to core {i}")

        treedict = {}
        if self._parallel == MPI_PARALLEL:
            self._comm.Barrier()
        # copy some stuff from the runMLZ script:
        for kss in range(s0, s1):
            print(f"making {kss+1} of {ntot}...")
            if self.config.nrandom > 1:
                ir = kss // int(self.config.ntrees)
                if ir != 0:
                    traindata.newcat(ir)
            DD = 'all'

            traindata.get_XY(bootstrap='yes', curr_at=DD)
            if self.config.tree_strategy == "native":
                T = TPZ.Rtree(traindata.X, traindata.Y, forest='yes',
                              minleaf=int(self.config.minleaf), mstar=int(self.config.natt),
                              dict_dim=DD)
            elif self.config.tree_strategy == "sklearn":
                randx = rng.integers(low=0, high=25000, size=1)[0]
                T = DecisionTreeRegressor(random_state=randx,
                                          min_samples_leaf=self.config.minleaf,
                                          max_features=int(self.config.natt))
                T.fit(traindata.X, traindata.Y)
            else:  # pragma: no cover  already tested above
                raise ValueError("invalid value for tree_strategy")

            treedict[f"tree_{kss}"] = T

        if self._parallel == MPI_PARALLEL:
            if self._rank == 0:
                for i in range(1, self._size, 1):
                    print(f"receiving data from rank {i}")
                    xdata = self._comm.recv(source=i, tag=11)
                    for key in xdata:
                        treedict[key] = xdata[key]
            else:
                xdata = treedict.copy()
                self._comm.send(xdata, dest=0, tag=11)

        if self.parallel == MPI_PARALLEL:
            self._comm.Barrier()
        if self._rank == 0:
            self.model = dict(trainkeys=trainkeys,
                              treedict=treedict,
                              use_atts=self.config.use_atts,
                              zmin=self.config.zmin,
                              zmax=self.config.zmax,
                              nzbins=self.config.nzbins,
                              redshift_col=self.config.redshift_col,
                              att_dict=train_att_dict,
                              keyatt=self.config.keyatt,
                              nrandom=self.config.nrandom,
                              ntrees=self.config.ntrees,
                              minleaf=self.config.minleaf,
                              natt=self.config.natt,
                              sigmafactor=self.config.sigmafactor,
                              bands=self.config.bands,
                              rmsfactor=self.config.rmsfactor,
                              tree_strategy=self.config.tree_strategy
                              )
            self.add_data("model", self.model)


class objfromdict(object):
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


class TPZliteEstimator(CatEstimator):
    """CatEstimator subclass for regression mode of TPZ
    Requires the trained model with decision trees that are computed by
    TPZliteInformer, and data that has all of the same columns and
    column names as used by that stage!
    """
    name = "TPZliteEstimator"
    config_options = CatEstimator.config_options.copy()
    config_options.update(nondetect_val=SHARED_PARAMS,
                          mag_limits=SHARED_PARAMS,
                          test_err_dict=Param(dict, def_err_dict, msg="dictionary that contains the columns that will be used to \
                                         predict as the keys and the errors associated with that column as the values. \
                                         If a column does not havea an associated error its value shoule be `None`"))

    def __init__(self, args, comm=None):
        """Constructor, build the CatEstimator, then do BPZ specific setup
        """
        CatEstimator.__init__(self, args, comm=comm)

    def open_model(self, **kwargs):
        CatEstimator.open_model(self, **kwargs)
        self.Pars = self.model
        self.attPars = objfromdict(self.model)

    def _process_chunk(self, start, end, inputdata, first):
        """
        Run TPZ on a chunk of data
        """

        testkeys = list(inputdata.keys())

        # replace non-detects with limiting mag and mag_err with 1.0
        for bandname, errname in self.config.test_err_dict.items():
            if bandname == self.attPars.redshift_col:
                continue
            if np.isnan(self.config.nondetect_val):  # pragma: no cover
                magmask = np.isnan(inputdata[bandname])
                errmask = np.isnan(inputdata[errname])
            else:
                magmask = np.isclose(inputdata[bandname], self.config.nondetect_val)
                errmask = np.isclose(inputdata[errname], self.config.nondetect_val)

            detmask = np.logical_or(magmask, errmask)
            inputdata[bandname][detmask] = self.config.mag_limits[bandname]
            inputdata[errname][detmask] = 1.0

        # make dictionary of attributes and error columns
        test_att_dict = make_index_dict(self.config.test_err_dict, testkeys)
        zfine, zfine2, resz, resz2, wzin = analysis.get_zbins(self.attPars)
        zfine2 = zfine2[wzin]
        ntot = int(self.attPars.nrandom * self.attPars.ntrees)

        Ng_temp = np.array(list(inputdata.values()))
        # Ng = np.array(Ng_temp, 'i')

        Test = data.catalog(self.attPars, Ng_temp.T, testkeys, self.attPars.use_atts, test_att_dict)
        Test.get_XY()

        if Test.has_Y():
            yvals = Test.Y
        else:  # pragma: no cover
            yvals = np.zeros(Test.nobj)

        Z0 = np.zeros((Test.nobj, 7))
        BP0 = np.zeros((Test.nobj, len(zfine2)))
        BP0raw = np.zeros((Test.nobj, len(zfine) - 1))
        Test_S = analysis.GetPz_short(self.attPars)

        # Load trees
        alltreedict = self.model["treedict"]
        print(f"loading {ntot} total trees from model")
        for k in range(ntot):

            S = alltreedict[f"tree_{k}"]
            # DD = S.dict_dim

            # Loop over all objects
            for i in range(Test.nobj):
                if self.attPars.tree_strategy == "native":
                    temp = S.get_vals(Test.X[i])
                else:
                    temp = S.predict(Test.X[i].reshape(1, -1))
                if temp[0] != -1.:
                    BP0raw[i, :] += Test_S.get_hist(temp)

        for k in range(Test.nobj):
            z_phot, pdf_phot = Test_S.get_pdf(BP0raw[k], yvals[k])
            Z0[k, :] = z_phot
            BP0[k, :] = pdf_phot
        del BP0raw, yvals

        zgrid = np.linspace(self.attPars.zmin,
                            self.attPars.zmax,
                            self.attPars.nzbins)

        qp_dstn = qp.Ensemble(qp.interp, data=dict(xvals=zfine2, yvals=BP0))
        zmode = qp_dstn.mode(grid=zgrid)

        qp_dstn.set_ancil(dict(zmode=zmode))
        self._do_chunk_output(qp_dstn, start, end, first)
