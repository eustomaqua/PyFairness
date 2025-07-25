# coding: utf-8
# import pdb


def excl_test_datasets():
    from pyfair.datasets import (
        Ricci, German, Adult, PropublicaRecidivism,
        PropublicaViolentRecidivism, preprocess)

    dt = Ricci()
    dt = German()
    dt = Adult()
    dt = PropublicaRecidivism()
    dt = PropublicaViolentRecidivism()

    df = dt.load_raw_dataset()
    ans = preprocess(dt, df)
    assert isinstance(ans, dict)
    # pdb.set_trace()
    return


def excl_test_preprocessing():
    from pyfair.datasets import preprocess, DATASETS
    from pyfair.preprocessing import (
        adversarial)  # ,transform_X_and_y,transform_unpriv_tag)

    for dt in DATASETS:
        df = dt.load_raw_dataset()
        ans = preprocess(dt, df)
        adv = adversarial(dt, df, ratio=.97)

        for k in ['original', 'numerical', 'numerical-binsensitive',
                  'categorical-binsensitive']:
            assert ans[k].shape == adv[k].shape
        # pdb.set_trace()
    return
