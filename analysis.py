import numpy as np
import matplotlib.pyplot as plt
from rasterstats import zonal_stats
from collections import defaultdict


def get_abs_error(dasymetric, estimated):
    true = np.array(dasymetric.trg[dasymetric.y_col])
    estimated = np.array(estimated)
    return np.abs(true - estimated)


def get_standardized_abs_error(dasymetric, estimated):
    true = np.array(dasymetric.trg[dasymetric.y_col])
    estimated = np.array(estimated)
    standardized = np.abs((true - estimated) / true)
    return np.nan_to_num(standardized)


def get_rmse(true, estimated):
    true = np.array(true)
    estimated = np.array(estimated)
    return np.sqrt(np.mean((true - estimated) ** 2))


def get_corr(vars):
    return np.corrcoef(*vars)


def get_rmse_by_src(dasymetric, estimated):
    trg = dasymetric.trg.assign(estimated=estimated)
    trg = trg.assign(squared_error=lambda x: (x[dasymetric.y_col] - x.estimated) ** 2)
    rmse_by_src = trg.loc[:, [dasymetric.src_id_col, 'squared_error']].groupby(dasymetric.src_id_col).mean()
    rmse_by_src = rmse_by_src.assign(rmse=lambda x: np.sqrt(x.squared_error))
    rmse_by_src = rmse_by_src.drop(columns=['squared_error'])
    return rmse_by_src


def get_cv_by_src(dasymetric, estimated):
    rmse_by_src = get_rmse_by_src(dasymetric, estimated)
    cv_by_src = dasymetric.src.join(rmse_by_src)
    cv_by_src = cv_by_src.assign(cv=lambda x: x.rmse / x[dasymetric.y_col])
    cv_by_src = cv_by_src.loc[:, ['cv']]
    return cv_by_src


def get_top_n_abs_error_zone(dasymetric, estimated, n):
    trg = dasymetric.trg.assign(estimated=estimated)
    trg = trg.assign(abs_error=lambda x: np.abs(x[dasymetric.y_col] - x.estimated))
    return trg.nlargest(n, 'abs_error')

 
def get_count_of_trg_zones(dasymetric, zones):
    counts = defaultdict(dict)
    for zone in zones:
        raw_counts = zonal_stats(dasymetric.trg.loc[zone, 'geometry'], dasymetric.aux_path, categorical=True)[0]
        for k, v in raw_counts.items():
            _class = dasymetric.class_mapper.get(k, -1)
            counts[zone][_class] = counts.get(zone, {}).get(_class, 0) + v
    return counts


def plot_abs_error_hist(true, estimated, bins=None, xscale='linear', yscale='linear'):
    true = np.array(true)
    estimated = np.array(estimated)
    abs_error = np.abs(true - estimated)
    plt.hist(abs_error, bins=bins)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.show()


def plot_abs_error_choropleth(dasymetric, estimated, cmap=None, scheme='equal_interval'):
    trg = dasymetric.trg.assign(estimated=estimated)
    trg = trg.assign(abs_error=lambda x: np.abs(x[dasymetric.y_col] - x.estimated))
    trg.plot(column='abs_error', cmap=cmap, scheme=scheme)