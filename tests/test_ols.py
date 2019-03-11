# flake8: noqa

# This is a kludgy way about things--it uses hardcoded values pulled
#     from Excel's Data Analysis > Regression tool as target values.
# Cover cases of:
# - 1d x, 1d y
# - 2d x, 1d y
# - 2d x, 2dy (empty dimension, squeezed later)

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression

from pyfinance import ols
from pyfinance.ols import OLS, RollingOLS, PandasRollingOLS
from pyfinance import utils

RTOL = 1e-03
WINDOW = 25

# Static regression - NumPy data structures
# ---------------------------------------------------------------------

# Get away from 1.0 rsq; add some drift
x2, y1 = make_regression(n_features=2, n_samples=75, random_state=123)

# Forgot to set a random state initially...
# This adds a drift term to second col vector
x2[:, 1] = \
np.array([  1.42503884,  -2.15692001,   3.10637452,   6.37145368,
            3.25639561,   5.9022381 ,   0.08851186,   1.47479427,
           -1.52062869,   1.81508594,  -1.7388635 ,   3.98862242,
            0.53964544,   6.02197246,   2.1420056 ,  -7.15621066,
            7.47808515,  -1.74697063,   2.93951758,   5.8779689 ,
            6.88293886,   4.18950492,   1.96456695,   1.5662138 ,
            4.47014934,   3.02311781,   5.58521039,   5.1545682 ,
            6.41127486,   0.93574269,   2.14285132,  -0.30560296,
            8.95355862,   5.13532356,   0.48455874,   0.02499129,
            2.12700812,   6.98748339,   3.28662808,  -1.4976681 ,
            3.97447394,   7.33831657,   4.44938017,   2.9010048 ,
            2.29413521,   3.3960674 ,   2.67819579,   2.86040553,
           -1.15955521,  -2.02075441,   1.59271255,   5.12249457,
            2.94093683,  12.08288461,   3.60086385,  -1.74555322,
           -0.07509666,   2.87191934,   4.09538527,   3.56113037,
            1.81521781,   5.33232982,   9.62294018,   0.69376186,
            8.32701046,   7.77646993,   4.45523781,   0.54358695,
            0.56845768,   5.55399514,   2.84556713,   1.56294916,
            4.61861153,  -0.57093837,   4.47752259])

x1 = x2[:, 0].copy()
y2 = np.expand_dims(y1, axis=1)

# Predicted values for each case
pred = {
    '1d':
        np.array([ 117.814,  173.196,  -23.225,  -35.09 ,   42.744,  206.665,
                    48.413, -137.37 , -126.625, -163.506,  108.452, -102.092,
                   117.992,  -92.095,   27.067,  105.095,  -34.052,  159.614,
                    25.916,   42.901,  -81.96 ,  -41.579,  -16.641,   17.473,
                   -47.039,  105.852,   -6.014,   38.97 ,  -41.68 ,  -82.209,
                  -159.362, -105.362, -111.093,  242.833,    6.148, -149.174,
                    85.425,  108.459,  150.22 ,   28.924,   -4.167,   58.526,
                   -23.703,   84.331,   39.399,  123.706,  102.127,  145.49 ,
                   -76.826,   13.432,  -96.914,   -0.789,  -46.072,   43.718,
                     3.395,  -58.577,  -14.749,  -55.669,   58.393,   74.959,
                   -80.966,   32.008, -117.914, -126.674,   85.419,   64.903,
                   -61.705,  -18.853, -231.492,  211.179,  168.706,   -9.278,
                    50.762,  -61.266,   80.231]),
    '2d':
        np.array([ 107.349,  140.639,  -22.173,  -14.287,   44.04 ,  222.237,
                    30.603, -144.992, -152.367, -168.822,   79.055,  -94.949,
                   102.201,  -72.825,   21.818,   43.155,   -6.604,  129.658,
                    25.474,   59.96 ,  -57.613,  -33.832,  -22.521,    8.857,
                   -37.55 ,  105.116,    9.772,   51.718,  -20.572,  -93.623,
                  -162.747, -124.009,  -74.004,  253.433,   -8.859, -165.397,
                    79.504,  131.536,  150.626,    1.77 ,    1.914,   84.21 ,
                   -14.571,   83.075,   34.942,  125.034,   99.354,  143.38 ,
                  -100.893,  -16.713, -104.23 ,   12.163,  -45.788,   98.081,
                     7.155,  -86.35 ,  -32.913,  -55.704,   64.578,   77.766,
                   -87.104,   45.894,  -76.733, -139.099,  116.78 ,   93.158,
                   -52.159,  -33.257, -243.626,  224.612,  166.276,  -17.647,
                    60.169,  -81.949,   88.497])
    }

# Residuals for each case
resid = {
    '1d':
        np.array([-101.643,  -72.011, -220.557,   29.529,  -81.325,  162.555,
                    69.599,  -73.771,   31.363,  165.538,  -18.057,   20.146,
                    -5.673,  -70.461,  -35.891,  -49.179,  -35.005,  -73.672,
                   -85.984,   49.067,   91.702, -118.57 ,  118.089,  -16.784,
                  -162.7  , -202.895,   -9.225,   70.962,  117.363,  -93.869,
                    52.055,  -99.855,   71.682,   31.83 ,  -72.271,   11.642,
                   123.636,  107.926,   15.877,  163.949,   34.574,  -80.532,
                   -50.135,  -85.439,  -91.686,  -58.358, -113.493,   40.343,
                  -119.724,   14.694,   91.648,   66.23 ,   97.355,   92.529,
                   -48.414,   30.554, -130.382,  139.96 ,   52.472,  -18.176,
                   -35.291, -118.597,   18.49 ,  -77.232,  102.697,  -51.358,
                    99.295,   20.309,  -20.381,   56.797,   52.005,  125.328,
                   188.756,  -21.564,   -8.388]),
    '2d':
        np.array([ -91.178,  -39.455, -221.608,    8.725,  -82.621,  146.983,
                    87.41 ,  -66.149,   57.104,  170.853,   11.341,   13.004,
                    10.119,  -89.732,  -30.642,   12.761,  -62.452,  -43.716,
                   -85.542,   32.009,   67.356, -126.317,  123.969,   -8.168,
                  -172.189, -202.159,  -25.011,   58.215,   96.254,  -82.455,
                    55.44 ,  -81.207,   34.593,   21.23 ,  -57.264,   27.865,
                   129.557,   84.849,   15.471,  191.103,   28.493, -106.216,
                   -59.267,  -84.183,  -87.228,  -59.686, -110.72 ,   42.453,
                   -95.657,   44.839,   98.964,   53.278,   97.07 ,   38.166,
                   -52.173,   58.327, -112.217,  139.995,   46.287,  -20.983,
                   -29.154, -132.484,  -22.691,  -64.807,   71.336,  -79.614,
                    89.749,   34.712,   -8.247,   43.364,   54.435,  133.697,
                   179.349,   -0.882,  -16.653])
    }

targets = {
    '1d': {
        'alpha': 14.47190296,
        'beta': 87.88845298,
        'df_tot': 74,
        'df_reg': 1,
        'df_err': 73,
        'fstat': 86.95141546,
        'fstat_sig': 4.59336E-14,
        'ms_err': 8333.666705,
        'ms_reg': 724624.116,
        'predicted': pred['1d'],
        'pvalue_alpha': 0.174403135,
        'pvalue_beta': 4.59336E-14,
        'resids': resid['1d'],
        'rsq': 0.543611416,
        'rsq_adj': 0.537359518,
        'se_alpha': 10.55145785,
        'se_beta': 9.425263314,
        'ss_tot': 1332981.785,
        'ss_reg': 724624.116,
        'ss_err': 608357.6694,
        'std_err': 91.28891885,
        'tstat_alpha': 1.371554828,
        'tstat_beta': 9.324774285,
        'ybar': 10.12105207
        },
    '2d': {
        'alpha': -3.532326073,
        'beta': np.array([87.01235606, 6.013341957]),
        'df_tot': 74,
        'df_reg': 2,
        'df_err': 72,
        'fstat': 46.61849853,
        'fstat_sig': 1.02827E-13,
        'ms_err': 8067.090356,
        'ms_reg': 376075.6399,
        'predicted': pred['2d'],
        'pvalue_alpha': 0.804792078,
        'pvalue_beta': np.array([4.27585E-14, 0.068821542]),
        'resids': resid['2d'],
        'rsq': 0.564262234,
        'rsq_adj': 0.552158407,
        'se_alpha': 14.23965488,
        'se_beta': np.array([9.285411279, 3.255319566]),
        'ss_tot': 1332981.785,
        'ss_reg': 752151.2797,
        'ss_err': 580830.5056,
        'std_err': 89.81698256,
        'tstat_alpha': -0.248062618,
        'tstat_beta': np.array([9.370867207, 1.847235528]),
        'ybar': 10.12105207
        }
    }

models = {
    'model_1dy_1dx': OLS(y=y1, x=x1),
    'model_1dy_2dx': OLS(y=y1, x=x2),
    'model_2dy_2dx': OLS(y=y2, x=x2)
    }


def test_ols():
    for name, model in models.items():
        if name.endswith('1dx'):
            for k, v in targets['1d'].items():
                assert np.allclose(v, getattr(model, k), rtol=RTOL)
        elif name.endswith('2dx'):
            for k, v in targets['2d'].items():
                assert np.allclose(v, getattr(model, k), rtol=RTOL)


# Rolling regression - NumPy data structures
# Rather than checking each rolling period we check [0] and [-1] pos.
# ---------------------------------------------------------------------

rpred = {
    '1d': {
        'start':
            np.array([  92.316,  144.489,  -40.55 ,  -51.728,   21.597,  176.019,
                        26.937, -148.081, -137.958, -172.702,   83.497, -114.846,
                        92.484, -105.429,    6.828,   80.334,  -50.75 ,  131.694,
                         5.744,   21.744,  -95.881,  -57.84 ,  -34.347,   -2.211,
                       -62.984]),
        'end':
            np.array([ -80.303,   27.834,  -23.107,   77.903,   32.541,  -37.175,
                        12.13 ,  -33.903,   94.412,  113.047,  -62.363,   64.729,
                      -103.927, -113.782,  124.815,  101.735,  -40.694,    7.512,
                      -231.698,  266.29 ,  218.51 ,   18.284,   85.827,  -40.201,
                       118.979])
        },
    '2d': {
        'start':
            np.array([  93.699,  136.9  ,  -39.347,  -40.873,   25.782,  194.65 ,
                        21.534, -156.285, -155.023, -180.851,   74.731, -113.924,
                        91.134,  -97.832,    6.966,   54.674,  -36.43 ,  124.856,
                         8.306,   34.049,  -85.232,  -53.986,  -36.427,   -4.221,
                        -58.47 ]),
        'end':
            np.array([ -84.502,   34.901,  -23.95 ,  116.896,   31.696,  -60.684,
                        -5.936,  -34.45 ,   92.364,  107.211,  -66.514,   70.652,
                       -66.448, -120.497,  142.179,  117.712,  -32.895,   -7.176,
                      -232.023,  261.326,  202.807,    8.067,   86.914,  -57.639,
                       117.385])
        }
    }

rresid = {
    '1d': {
        'start':
            np.array([ -76.146,  -43.304, -203.232,   46.166,  -60.177,  193.202,
                        91.076,  -63.06 ,   42.696,  174.734,    6.899,   32.901,
                        19.836,  -57.127,  -15.652,  -24.418,  -18.307,  -45.752,
                       -65.811,   70.224,  105.624, -102.308,  135.795,    2.899,
                      -146.755]),
        'end':
            np.array([  75.037,   37.606,   74.39 ,   58.344,  -77.56 ,    9.153,
                      -157.261,  118.195,   16.454,  -56.265,  -53.895, -151.319,
                         4.503,  -90.124,   63.301,  -88.19 ,   78.284,   -6.057,
                       -20.175,    1.686,    2.201,   97.766,  153.691,  -42.63 ,
                       -47.136])
        },
    '2d': {
        'start':
            np.array([ -77.529,  -35.715, -204.435,   35.311,  -64.363,  174.571,
                        96.478,  -54.856,   59.76 ,  182.882,   15.665,   31.979,
                        21.186,  -64.724,  -15.79 ,    1.242,  -32.626,  -38.914,
                       -68.373,   57.919,   94.975, -106.162,  137.875,    4.91 ,
                      -151.269]),
        'end':
            np.array([  79.236,   30.54 ,   75.233,   19.351,  -76.714,   32.662,
                      -139.195,  118.741,   18.502,  -50.428,  -49.743, -157.241,
                       -32.976,  -83.409,   45.937, -104.167,   70.485,    8.632,
                       -19.85 ,    6.65 ,   17.904,  107.983,  152.604,  -25.192,
                       -45.542])
        }
    }

rtargets = {
    '1d': {
        'start': {
            'alpha': -5.037450748,
            'beta': 82.79557323,
            'df_tot': 24,
            'df_reg': 1,
            'df_err': 23,
            'fstat': 23.43162196,
            'fstat_sig': 6.92852E-05,
            'ms_err': 9374.380222,
            'ms_reg': 219656.9334,
            'predicted': rpred['1d']['start'],
            'pvalue_alpha': 0.7971472,
            'pvalue_beta': 6.92852E-05,
            'resids': rresid['1d']['start'],
            'rsq': 0.504647931,
            'rsq_adj': 0.483110885,
            'se_alpha': 19.37188279,
            'se_beta': 17.10432505,
            'ss_tot': 435267.6785,
            'ss_reg': 219656.9334,
            'ss_err': 215610.7451,
            'std_err': 96.82138308,
            'tstat_alpha': -0.260039295,
            'tstat_beta': 4.840622063,
            'ybar': -7.664949499
            },
        'end': {
            'alpha': 45.00203012,
            'beta': 98.8712744,
            'df_tot': 24,
            'df_reg': 1,
            'df_err': 23,
            'fstat': 41.92639276,
            'fstat_sig': 1.32E-06,
            'ms_err': 6684.632025,
            'ms_reg': 280262.5077,
            'predicted': rpred['1d']['end'],
            'pvalue_alpha': 0.012808213,
            'pvalue_beta': 1.316E-06,
            'resids': rresid['1d']['end'],
            'rsq': 0.64575269,
            'rsq_adj': 0.630350633,
            'se_alpha': 16.67364217,
            'se_beta': 15.26955508,
            'ss_tot': 434009.0443,
            'ss_reg': 280262.5077,
            'ss_err': 153746.5366,
            'std_err': 81.75959898,
            'tstat_alpha': 2.698992198,
            'tstat_beta': 6.475059286,
            'ybar': 23.89585737
            }
        },
    '2d': {
        'start': {
            'alpha': -12.00980735,
            'beta': np.array([86.15055245, 3.094894]),
            'df_tot': 24,
            'df_reg': 2,
            'df_err': 22,
            'fstat': 11.452182,
            'fstat_sig': 0.000390369,
            'ms_err': 9693.215531,
            'ms_reg': 111008.4684,
            'predicted': rpred['2d']['start'],
            'pvalue_alpha': 0.625229391,
            'pvalue_beta': np.array([0.000134913, 0.626601084]),
            'resids': rresid['2d']['start'],
            'rsq': 0.51006989,
            'rsq_adj': 0.465530789,
            'se_alpha': 24.24260736,
            'se_beta': np.array([18.67456553, 6.272250535]),
            'ss_tot': 435267.6785,
            'ss_reg': 222016.9368,
            'ss_err': 213250.7417,
            'std_err': 98.45412907,
            'tstat_alpha': -0.495400811,
            'tstat_beta': np.array([4.613256052, 0.4934264]),
            'ybar': -7.664949499
            },
        'end': {
            'alpha': 25.35591104,
            'beta': np.array([92.98600769, 5.015191519]),
            'df_tot': 24,
            'df_reg': 2,
            'df_err': 22,
            'fstat': 21.20945903,
            'fstat_sig': 7.37058E-06,
            'ms_err': 6737.291736,
            'ms_reg': 142894.3131,
            'predicted': rpred['2d']['end'],
            'pvalue_alpha': 0.364800613,
            'pvalue_beta': np.array([1.29506E-05, 0.374925765]),
            'resids': rresid['2d']['end'],
            'rsq': 0.658485416,
            'rsq_adj': 0.627438636,
            'se_alpha': 27.40008118,
            'se_beta': np.array([16.65003849, 5.537581194]),
            'ss_tot': 434009.0443,
            'ss_reg': 285788.6261,
            'ss_err': 148220.4182,
            'std_err': 82.08100716,
            'tstat_alpha': 0.925395471,
            'tstat_beta': np.array([5.584732298, 0.905664647]),
            'ybar':23.89585737
            }
        }
    }

rmodels = {
    'model_1dy_1dx': RollingOLS(y=y1, x=x1, window=WINDOW),
    'model_1dy_2dx': RollingOLS(y=y1, x=x2, window=WINDOW),
    'model_2dy_2dx': RollingOLS(y=y2, x=x2, window=WINDOW)
    }


def test_rolling_ols():
    for name, model in rmodels.items():
        if name.endswith('1dx'):
            for k, v in rtargets['1d']['start'].items():
                try:
                    attr = getattr(model, k)[0]
                except:
                    attr = getattr(model, k)
                assert np.allclose(v, attr, rtol=RTOL)
            for k, v in rtargets['1d']['end'].items():
                try:
                    attr = getattr(model, k)[-1]
                except:
                    attr = getattr(model, k)
                assert np.allclose(v, attr, rtol=RTOL)
        elif name.endswith('2dx'):
            for k, v in rtargets['2d']['start'].items():
                try:
                    attr = getattr(model, k)[0]
                except:
                    attr = getattr(model, k)
                assert np.allclose(v, attr, rtol=RTOL)
            for k, v in rtargets['2d']['end'].items():
                try:
                    attr = getattr(model, k)[-1]
                except:
                    attr = getattr(model, k)
                assert np.allclose(v, attr, rtol=RTOL)



window = 4
y = np.array([1, 3, 5, 7, 9, 11, 13], dtype=np.float64)
x1 = np.arange(7, dtype=np.float64)
x2 = np.array(
    [
        0.976745506,
        0.13081688,
        0.415920635,
        0.154081322,
        0.359122273,
        0.917051723,
        0.705650721
    ],
    dtype=np.float64
)

x = np.stack((x1, x2)).T  # shape (7, 2) - x1 and x2 as columns
x_with = np.stack((x1, x2, np.repeat(1., len(x1)))).T  # with artificial constant
x1_with = np.stack((x1, np.repeat(1., len(x1)))).T  # with artificial constant

outs = (
    np.array([[ 2.32241042,  1.04305696],
              [ 2.18848609,  1.51863796],
              [ 2.29824933, -0.2877955 ],
              [ 2.29513731, -0.6887367 ]]),
    np.array([[2., 0., 1.],
              [2., 0., 1.],
              [2., 0., 1.],
              [2., 0., 1.]]),
    np.array([[2., 1.],
              [2., 1.],
              [2., 1.],
              [2., 1.]]),
    np.array([2.42857143, 2.33333333, 2.25925926, 2.20930233])
)


@pytest.mark.parametrize('x,y,window,out', [
    (x, y, window, outs[0]),
    (x_with, y, window, outs[1]),
    (x1_with, y, window, outs[2]),
    (x1, y, window, outs[3]),
])
def test__rolling_lstsq(x, y, window, out):
    xwins = utils.rolling_windows(x, window)
    ywins = utils.rolling_windows(y, window)
    assert np.allclose(
        ols._rolling_lstsq(xwins, ywins),
        out
    )


def test_const_false():
    # Case where use_const=False and has_const=False
    # See Issue # 6
    X = pd.DataFrame(np.arange(5), columns=['X'])
    Y = pd.DataFrame(np.arange(0, 10, 2) + 1, columns=['Y'])
    window = 2
    reg_df = pd.concat([Y, X], axis=1)
    rr = PandasRollingOLS(y=reg_df.iloc[:, 0],  # Series
                              x=reg_df.iloc[:, 1:],  # DataFrame
                              window=window,
                              has_const=False,
                              use_const=False)
    assert np.allclose(
        rr.beta.values,
        np.array([[3.        ],
                  [2.6       ],
                  [2.38461538],
                  [2.28      ]])
    )


def test_add_const():
    X = pd.DataFrame(np.arange(5), columns=['X'])
    Y = pd.Series(np.arange(0, 10, 2) + 1)
    rr = ols.OLS(y=Y,
                 x=X,
                 has_const=False,
                 use_const=True)
    assert rr.x.ndim == 2
    assert rr.y.ndim == 1
    assert ols._confirm_constant(rr.x)


def test_confirm_constant():
    has = np.array([[0, 1], [1, 1], [2, 1], [3, 1], [4, 1]])
    missing = np.array([[0, 1], [1, 1], [2, 1], [3, 1], [4, 2]])
    assert ols._confirm_constant(has)
    assert not ols._confirm_constant(missing)


def test_datareader_frame():
    import os.path
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pdr.csv')
    data = pd.read_csv(p)
    y = data['usd']
    x = data.drop('usd', axis=1)
    window = 12  # months
    model = PandasRollingOLS(y=y, x=x, window=window)
    assert isinstance(model.beta, pd.DataFrame)
    assert model.beta.shape == (219, 2)
    tgt = np.array([[ 3.28409826e-05, -5.42606172e-02],
                    [ 2.77474638e-04, -1.88556396e-01],
                    [ 2.43179753e-03, -2.94865331e-01],
                    [ 2.79584924e-03, -3.34879522e-01],
                    [ 2.44759386e-03, -2.41902450e-01]])
    assert np.allclose(model.beta.head().values, tgt)


def test_1d_x():
    # See issue #12
    data = {'A': [2, 3, 4, 5, 6], 'B': [10, 11, 12, 13, 14]}
    df = pd.DataFrame(data)
    rolling = ols.RollingOLS(y=df['B'], x=df['A'], window=3, has_const=False, use_const=False)
    assert np.allclose(rolling.x, np.array([2, 3, 4, 5, 6])[:, None])
    assert np.allclose(rolling.y, np.array([10, 11, 12, 13, 14]))
