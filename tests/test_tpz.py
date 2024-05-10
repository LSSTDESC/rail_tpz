import numpy as np
import os
import pytest
from rail.core.stage import RailStage
from rail.utils.testing_utils import one_algo
from rail.core.data import TableHandle
from rail.utils.path_utils import RAILDIR
from rail.utils.path_utils import find_rail_file
from rail.estimation.algos.tpz_lite import TPZliteInformer, TPZliteEstimator

@pytest.mark.parametrize(
    "treestrat, nrand",
    [("native", 2),
     ("sklearn", 1)]
)
def test_tpz_larger_training(treestrat, nrand):
    train_config_dict = {"hdf5_groupname": "photometry", "nrandom": nrand, "ntrees": 5,
                         "model": "tpz_tests.pkl", "tree_strategy": treestrat}
    estim_config_dict = {"hdf5_groupname": "photometry", "model": "tpz_tests.pkl"}
    train_algo = TPZliteInformer
    pz_algo = TPZliteEstimator
    zb_expected = np.array([0.16, 1.24, 0.35, 0.16, 0.23, 0.17, 0.22, 0.27, 0.29, 0.13])

    #traindata = "./tests/tpz_testflie_1000.hdf5"
    #traindata = os.path.join(RAILDIR, 'rail/examples_data/estimation_data/tpz_testfile_1000.hdf5')
    traindata = find_rail_file('examples_data/estimation_data/tpz_testfile_1000.hdf5')
    validdata = os.path.join(RAILDIR, 'rail/examples_data/testdata/validation_10gal.hdf5')
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True
    DS.clear()
    training_data = DS.read_file('training_data', TableHandle, traindata)
    validation_data = DS.read_file('validation_data', TableHandle, validdata)


    train_pz = TPZliteInformer.make_stage(**train_config_dict)
    train_pz.inform(training_data)

    pz = TPZliteEstimator.make_stage(name="TPZ_lite", **estim_config_dict)
    estim = pz.estimate(validation_data)
    pz_2 = None
    estim_2 = estim
    pz_3 = None
    estim_3 = estim

    copy_estim_config_dict = estim_config_dict.copy()
    model_file = copy_estim_config_dict.pop('model', 'None')

    if model_file != 'None':
        copy_estim_config_dict['model'] = model_file
        pz_2 = TPZliteEstimator.make_stage(name=f"{pz.name}_copy", **copy_estim_config_dict)
        estim_2 = pz_2.estimate(validation_data)

    copy3_estim_config_dict = estim_config_dict.copy()
    copy3_estim_config_dict['model'] = train_pz.get_handle('model')
    pz_3 = TPZliteEstimator.make_stage(name=f"{pz.name}_copy3", **copy3_estim_config_dict)

    estim_3 = pz_3.estimate(validation_data)

    os.remove(pz.get_output(pz.get_aliased_tag('output'), final_name=True))
    if pz_2 is not None:
        os.remove(pz_2.get_output(pz_2.get_aliased_tag('output'), final_name=True))

    if pz_3 is not None:
        os.remove(pz_3.get_output(pz_3.get_aliased_tag('output'), final_name=True))
    model_file = estim_config_dict.get('model', 'None')
    if model_file != 'None':
        try:
            os.remove(model_file)
        except FileNotFoundError:  #pragma: no cover
            pass
        #    return estim.data, estim_2.data, estim_3.data
        #results, rerun_results, _
    flatres = estim.data.ancil["zmode"].flatten()
    #assert np.isclose(flatres, zb_expected, atol=2.e-01).all()
    #assert np.isclose(estim.data.ancil["zmode"], estim_2.data.ancil["zmode"]).all()
