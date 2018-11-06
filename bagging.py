import random
from multiprocessing import Pool
from collections import defaultdict
import numpy as np
from statistics import median
from .dasymetric import EM, GWEM


class Bagging:
    def __init__(self, dasymetric, n_duplicates):
        assert type(dasymetric) in [EM, GWEM], "Bagging only works with EM or GWEM."
        self.dasymetric = dasymetric
        self.method = dasymetric.__class__
        self.duplicates = self._generate_duplicates(n_duplicates)

    def _generate_duplicates(self, n_duplicates):
        n_samples = self.dasymetric.src.shape[0]
        duplicates = []
        for i in range(n_duplicates):
            sample = random.choices(self.dasymetric.src.index, k=n_samples)
            sample_src = self.dasymetric.src.copy().loc[sample]
            sample_trg = self.dasymetric.trg.copy()
            sample_src, sample_trg = self._handle_duplicates(sample_src, sample_trg)
            if self.method is EM:
                duplicate = EM(src=sample_src, trg=sample_trg, y_col=self.dasymetric.y_col,
                               src_id_col=self.dasymetric.src_id_col, trg_id_col=self.dasymetric.trg_id_col,
                               aux_path=self.dasymetric.aux_path, n_iter=self.dasymetric.n_iter)
            elif self.method is GWEM:
                duplicate = GWEM(src=sample_src, trg=sample_trg, y_col=self.dasymetric.y_col,
                                 src_id_col=self.dasymetric.src_id_col, trg_id_col=self.dasymetric.trg_id_col,
                                 aux_path=self.dasymetric.aux_path, n_iter=self.dasymetric.n_iter,
                                 N=self.dasymetric.N)
                if self.dasymetric.density_cap is not None:
                    duplicate.set_density_cap(self.dasymetric.density_cap, self.dasymetric.benchmark_class)
            duplicate.set_class_mapper(self.dasymetric.class_mapper)
            duplicates.append(duplicate)
        return duplicates

    def _handle_duplicates(self, sample_src, sample_trg):
        src_id_col = self.dasymetric.src_id_col
        duplicate_appear = {}
        duplicated_src_ids = sample_src.loc[sample_src.index.duplicated()]
        for src_id in duplicated_src_ids:
            n = duplicate_appear.get(src_id, 1) + 1
            trg_dup_part = sample_trg.loc[sample_trg[src_id_col] == src_id].copy()
            trg_dup_part.loc[src_id_col] = src_id + '_' + str(n)
            sample_trg = sample_trg.append(trg_dup_part)
            duplicate_appear[src_id] = n
        sample_src = self._rename_duplicate_index(sample_src)
        sample_trg = self._rename_duplicate_index(sample_trg)
        return sample_src, sample_trg

    @staticmethod
    def _rename_duplicate_index(df):
        duplicate_appear = {}
        idx_list = df.index.tolist()
        for i, duplicated in enumerate(df.index.duplicated()):
            if duplicated:
                n = duplicate_appear.get(df.iloc[i, :].name, 1) + 1
                idx_list[i] = idx_list[i] + '_' + str(n)
        df.index = idx_list
        return df

    def set_density_mapper(self, method='mean'):
        with Pool() as pool:
            duplicate_density_mappers = pool.map(self._set_duplicate_density_mapper, self.duplicates)
        density_by_class = self._get_density_by_class(duplicate_density_mappers)
        agg_func = self._get_agg_func(method)
        density_mapper = self._get_density_mapper(density_by_class, agg_func)
        self.dasymetric.density_mapper = density_mapper

    @staticmethod
    def _set_duplicate_density_mapper(duplicate):
        duplicate.set_density_mapper(verbose=False)
        return duplicate.density_mapper

    def _get_density_by_class(self, duplicate_density_mappers):
        if self.method is EM:
            density_by_class = defaultdict(list)
            for density_mapper in duplicate_density_mappers:
                for _class, density in density_mapper.items():
                    density_by_class[_class].append(density)
        elif self.method is GWEM:
            density_by_class = defaultdict(dict)
            for density_mapper in duplicate_density_mappers:
                for src_id, dm in density_mapper.items():
                    for _class, density in dm.items():
                        if _class not in density_by_class[src_id].keys():
                            density_by_class[src_id][_class] = []
                        density_by_class[src_id][_class].append(density)
        return density_by_class

    @staticmethod
    def _get_agg_func(method):
        if method == 'mean':
            return np.mean
        elif method == 'median':
            return median
        else:
            raise ValueError("Method must be either mean or median.")


    def _get_density_mapper(self, density_by_class, agg_func):
        if self.method is EM:
            density_mapper = {}
            for _class, densities in density_by_class.items():
                density_mapper[_class] = agg_func(densities)
        elif self.method is GWEM:
            density_mapper = defaultdict(dict)
            for src_id, dbc in density_by_class.items():
                for _class, densities in dbc.items():
                    density_mapper[src_id][_class] = agg_func(densities)
        return density_mapper

    def estimate(self):
        if self.method is EM:
            return self.dasymetric.__class__.__bases__[0].estimate(self.dasymetric)
        elif self.method is GWEM:
            return self.dasymetric.estimate()
