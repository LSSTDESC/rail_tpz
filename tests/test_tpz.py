import numpy as np
import os
from rail.core.stage import RailStage
from rail.core.algo_utils import one_algo
from rail.core.utils import RAILDIR
from rail.estimation.algos.tpz_lite import TPZliteInformer, TPZliteEstimator


traindata = os.path.join(RAILDIR, "rail/examples_data/testdata/training_100gal.hdf5")
validdata = os.path.join(RAILDIR, "rail/examples_data/testdata/validation_10gal.hdf5")
DS = RailStage.data_store
DS.__class__.allow_overwrite = True


def test_tpz():
    train_config_dict = {"hdf5_groupname": "photometry", "nrandom": 2, "ntrees": 2,
                         "model": "tpz_tests.pkl"}
    estim_config_dict = {"hdf5_groupname": "photometry", "model": "tpz_tests.pkl"}
    train_algo = TPZliteInformer
    pz_algo = TPZliteEstimator
    zb_expected = np.array([0.13, 0.14, 0.14, 0.12, 0.12, 0.13, 0.14, 0.15,
                            0.12, 0.12])
    results, rerun_results, _ = one_algo("TPz_lite", train_algo, pz_algo,
                                         train_config_dict, estim_config_dict)
    flatres = results.ancil["zmode"].flatten()
    assert np.isclose(flatres, zb_expected, atol=2.e-02).all()
    assert np.isclose(results.ancil["zmode"], rerun_results.ancil["zmode"]).all()


def test_tpz_with_extra_keys():
    bands = ["u", "g", "r", "i", "z", "y"]
    usekeys = {}
    for band in bands:
        usekeys[f"mag_{band}_lsst"] = f"mag_err_{band}_lsst"
    usekeys["redshift"] = None
    usekeys["not_real"] = "also_not_real"
    train_config_dict = {"hdf5_groupname": "photometry", "nrandom": 2, "ntrees": 2,
                         "model": "tpz_extra_tests.pkl", "err_dict": usekeys}
    estim_config_dict = {"hdf5_groupname": "photometry", "model": "tpz_extra_tests.pkl"}
    train_algo = TPZliteInformer
    pz_algo = TPZliteEstimator
    zb_expected = np.array([0.13, 0.14, 0.14, 0.12, 0.12, 0.13, 0.14, 0.15,
                            0.12, 0.12])
    results, rerun_results, _ = one_algo("TPz_lite", train_algo, pz_algo,
                                         train_config_dict, estim_config_dict)
    flatres = results.ancil["zmode"].flatten()
    assert np.isclose(flatres, zb_expected, atol=2.e-02).all()
    assert np.isclose(results.ancil["zmode"], rerun_results.ancil["zmode"]).all()
