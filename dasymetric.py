from abc import ABC, abstractmethod
import rasterio
from rasterstats import zonal_stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pysal.cg.kdtree import KDTree
from pysal.weights.Distance import Kernel
from collections import defaultdict


class Dasymetric(ABC):
    def __init__(self, src, trg, y_col, src_id_col, trg_id_col):
        self.src = src.set_index(src_id_col)
        self.trg = trg.set_index(trg_id_col)
        self.y_col = y_col
        self.src_id_col = src_id_col
        self.trg_id_col = trg_id_col
        self.trg = self._filter_trg_by_src()
        self.counts = {
            'src': None,
            'trg': None
        }
        self.aux_path = None
        self.class_mapper = None
        self.class_mapper_changed = False
        self.density_mapper = None

    def _filter_trg_by_src(self):
        return self.trg.loc[self.trg[self.src_id_col].isin(self.src.index)]

    def set_class_mapper(self, class_mapper):
        self.class_mapper = class_mapper
        self.class_mapper_changed = True

    def _check_class_mapper(self):
        if self.class_mapper is None:
            raise ValueError('Class mapper is not set.')

    def _count_cell_by_zone(self, zone_type):
        self._check_class_mapper()
        if zone_type == 'src':
            gdf = self.src
        elif zone_type == 'trg':
            gdf = self.trg
        else:
            raise ValueError("Zone type should be one of ('src', 'trg').")
        counts = defaultdict(dict)
        for _, row in gdf.iterrows():
            name = row.name
            raw_counts = zonal_stats(row['geometry'], self.aux_path, categorical=True)[0]
            for k, v in raw_counts.items():
                _class = self.class_mapper.get(k, -1)
                counts[name][_class] = counts.get(name, {}).get(_class, 0) + v
        return counts

    def count_cell_by_zone(self):
        self.counts['src'] = self._count_cell_by_zone('src')
        self.counts['trg'] = self._count_cell_by_zone('trg')

    def _initialize_cell_counts(self):
        if self.counts['src'] is None or self.counts['trg'] is None:
            print('Initializing raster cell counts...', end='')
            if self.counts['src'] is None:
                self.counts['src'] = self._count_cell_by_zone('src')
            if self.counts['trg'] is None:
                self.counts['trg'] = self._count_cell_by_zone('trg')
            print('Done.')

    @abstractmethod
    def estimate(self):
        pass


class Areal_Weighting(Dasymetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def estimate(self):
        estimates = []
        src_sea = self.src.to_crs({'proj': 'cea'})
        trg_sea = self.trg.to_crs({'proj': 'cea'})
        for _, row in trg_sea.iterrows():
            src_id = row[self.src_id_col]
            src_area = src_sea.loc[src_id, 'geometry'].area
            if src_area != 0:
                trg_area = row.geometry.area
                y_s = self.src.loc[src_id, self.y_col]
                y_t = y_s * trg_area / src_area
            else:
                y_t = 0
            estimates.append(y_t)
        return estimates


class IDM_Super(Dasymetric):
    def __init__(self, aux_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_path = aux_path

    @abstractmethod
    def set_density_mapper(self):
        pass

    def _initialize_density_mapper(self):
        if self.density_mapper is None:
            print('Initializing density mapper...', end='')
            self.set_density_mapper()
            print('Done')

    def estimate(self):
        if self.class_mapper_changed:
            self._initialize_cell_counts()
            self._initialize_density_mapper()
            self.class_mapper_changed = False
        estimates = []
        for _, row in self.trg.iterrows():
            src_id = row[self.src_id_col]
            y_s = self.src.loc[src_id, self.y_col]
            trg_count = self.counts['trg'][row.name]
            trg_classes = self.density_mapper.keys() & trg_count.keys()
            numerator = sum(self.density_mapper[c] * trg_count[c] for c in trg_classes)
            src_count = self.counts['src'][src_id]
            src_classes = self.density_mapper.keys() & src_count.keys()
            denom = sum(self.density_mapper[c] * src_count[c] for c in src_classes)
            if denom != 0:
                y_t = y_s * numerator / denom
            else:
                y_t = 0
            estimates.append(y_t)
        return estimates


class Binary_Dasymetric(IDM_Super):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_density_mapper(self):
        self.density_mapper = {1: 1, -1: 0}


class IDM(IDM_Super):
    def __init__(self, method, threshold=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method
        self.threshold = threshold
        if self.method == 'centroid':
            self.aux = rasterio.open(self.aux_path)
            self.aux_values = self.aux.read(1)

    def set_density_mapper(self):
        sampled = self._sample()
        self.density_mapper = {}
        unsampled_classes = []
        for _class, sampled_zones in sampled.items():
            if sampled_zones:
                y_sum = self.src.loc[sampled_zones, self.y_col].sum()
                area_sum = np.array([row.area for row in self.src.loc[sampled_zones, 'geometry']]).sum()
                if area_sum != 0:
                    self.density_mapper[_class] = y_sum / area_sum
                else:
                    self.density_mapper[_class] = 0
            else:
                unsampled_classes.append(_class)
        temp_density_mapper = self.density_mapper.copy()
        self._set_unsampled_class_density(temp_density_mapper)
        self.density_mapper.update({-1: 0})

    def _sample(self):
        sampled = {}
        for _class in set(self.class_mapper.values()):
            sampled[_class] = []
        if self.method == 'containment':
            sampled = self._sample_by_containment(sampled)
        elif self.method == 'centroid':
            sampled = self._sample_by_centroid(sampled)
        elif self.method == 'percent':
            sampled = self._sample_by_percent(sampled)
        else:
            raise ValueError('{} is not a proper method.'.format(self.method))
        return sampled

    def _sample_by_containment(self, sampled):
        for _, row in self.src.iterrows():
            count = self.counts['src'][row.name]
            nonzero_class = [_class for _class, value in count.items() if value != 0]
            if len(nonzero_class) == 1:
                _class = nonzero_class[0]
                sampled[_class].append(row.name)
        return sampled

    def _sample_by_centroid(self, sampled):
        for _, row in self.src.iterrows():
            x = row.geometry.centroid.x
            y = row.geometry.centroid.y
            idx = self.aux.index(x, y)
            category = self.aux_values[idx]
            _class = self.class_mapper.get(category, -1)
            if _class != -1:
                sampled[_class].append(row.name)
        return sampled

    def _sample_by_percent(self, sampled):
        for _, row in self.src.iterrows():
            count = self.counts['src'][row.name]
            num_total_cells = sum(count.values())
            if num_total_cells != 0:
                dominant_class = max(count, key=lambda _class: count[_class])
                if count[dominant_class] / num_total_cells >= self.threshold:
                    sampled[dominant_class].append(row.name)
        return sampled

    def _set_unsampled_class_density(self, temp_density_mapper):
        total_unsampled_classes_estimated = {}
        total_unsampled_classes_count = {}
        for _, row in self.src.iterrows():
            y_s = row[self.y_col]
            src_count = self.counts['src'][row.name]
            src_sampled_classes = temp_density_mapper.keys() & src_count.keys()
            src_estimated = sum(temp_density_mapper[c] * src_count[c] for c in src_sampled_classes)
            src_unestimated = y_s - src_estimated
            src_unsampled_classes = src_count.keys() - temp_density_mapper.keys()
            src_unsampled_classes_count = {_class: src_count[_class] for _class in src_unsampled_classes}
            src_unsampled_count_total = sum(src_unsampled_classes_count.values())
            for _class, count in src_unsampled_classes_count.items():
                total_unsampled_classes_estimated[_class] = total_unsampled_classes_estimated.get(_class, 0) + \
                                                            src_unestimated * count / src_unsampled_count_total
                total_unsampled_classes_count[_class] = total_unsampled_classes_count.get(_class, 0) + count
        for _class, count in total_unsampled_classes_count.items():
            class_total_estimate = total_unsampled_classes_estimated[_class]
            class_total_count = total_unsampled_classes_count[_class]
            if class_total_count != 0:
                self.density_mapper[_class] = class_total_estimate / class_total_count
            else:
                self.density_mapper[_class] = 0


class EM(IDM_Super):
    def __init__(self, n_iter, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_iter = n_iter
        self.density_history = pd.DataFrame()

    def set_density_mapper(self):
        self._initialize_cell_counts()
        self.density_mapper = {_class: 1 for _class in set(self.class_mapper.values())}
        self.density_mapper[-1] = 0
        for i in range(self.n_iter):
            self.e_step()
            self.m_step()
            self.density_history = self.density_history.append(self.density_mapper, ignore_index=True)
            print("Iteration: {}, Density mapper: {}".format(i + 1, self.density_mapper))

    def e_step(self):
        class_y_estimates = {}
        for _, row in self.src.iterrows():
            class_y_estimates[row.name] = {}
            y_s = row[self.y_col]
            src_count = self.counts['src'][row.name]
            src_classes = self.density_mapper.keys() & src_count.keys()
            numerators = np.array([self.density_mapper[c] * src_count[c] for c in src_classes])
            denom = numerators.sum()
            if denom != 0:
                y_sc_array = numerators * y_s / denom
            else:
                y_sc_array = [0 for i in len(numerators)]
            for i, c in enumerate(src_classes):
                class_y_estimates[row.name][c] = y_sc_array[i]
        self.class_y_estimates = class_y_estimates

    def m_step(self):
        for _class in self.density_mapper.keys():
            numerator = np.array([estm.get(_class, 0) for estm in self.class_y_estimates.values()]).sum()
            denom = np.array([counts.get(_class, 0) for counts in self.counts['src'].values()]).sum()
            if denom != 0:
                self.density_mapper[_class] = numerator / denom
            else:
                self.density_mapper[_class] = 0

    def _initialize_density_mapper(self):
        if self.density_mapper is None:
            print('Initializing density mapper...')
            self.set_density_mapper()
            print('Done')

    def estimate(self):
        if self.class_mapper_changed:
            self._initialize_cell_counts()
            self._initialize_density_mapper()
            self.class_mapper_changed = False
        estimates = []
        for _, row in self.trg.iterrows():
            src_id = row[self.src_id_col]
            trg_count = self.counts['trg'][row.name]
            src_count = self.counts['src'][src_id]
            vec1 = np.array([trg_count[c] / src_count[c] if src_count[c] != 0 else 0
                             for c in trg_count.keys()])
            y_sc = self.class_y_estimates[src_id]
            vec2 = np.array([y_sc[c] for c in trg_count.keys()])
            y_t = np.dot(vec1, vec2)
            estimates.append(y_t)
        return estimates

    def plot_density_history(self):
        for _class in self.density_history.columns:
            plt.plot(range(1, self.n_iter + 1), self.density_history[_class])
        plt.legend()
        plt.show()


class GWEM(IDM_Super):
    def __init__(self, n_iter, N, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_iter = n_iter
        self.N = N
        self.weight = self._get_weight()
        self.weight_idx_mapper = {src_id: i for i, src_id in enumerate(self.src.index)}
        self.density_cap = None
        self.benchmark_class = None
        self.benchmark = None
        self.density_history = pd.DataFrame()

    def _get_weight(self):
        n_src = self.src.shape[0]
        weight = np.zeros((n_src, n_src))
        src_longlat = self.src.to_crs({'init': 'epsg:4326'})
        centroids = np.array(list(zip(src_longlat.geometry.centroid.x, src_longlat.geometry.centroid.y)))
        kdtree = KDTree(centroids, leafsize=(src_longlat.shape[0] + 1), distance_metric='Arc')
        kw = Kernel(kdtree, fixed=False, k=self.N, ids=range(src_longlat.shape[0]), function='quartic')
        for i in range(kw.neigh.shape[0]):
            for j, n in enumerate(kw.neigh[i]):
                weight[i, n] = kw.weights[i][j]
        return weight

    def set_density_cap(self, density_cap, benchmark_class):
        self.density_cap = density_cap
        self.benchmark_class = benchmark_class

    def set_density_mapper(self):
        self._initialize_cell_counts()
        self.density_mapper = {subsrc_id: {_class: 1 for _class in set(self.class_mapper.values())} for subsrc_id in
                               self.src.index}
        self.benchmark = {subsrc_id: np.iinfo('int32').max for subsrc_id in self.src.index}
        for key in self.density_mapper.keys():
            self.density_mapper[key][-1] = 0
        for i in range(self.n_iter):
            self.e_step()
            self.m_step()
            if self.density_cap is not None:
                self.benchmark = {subsrc_id: self.density_mapper[subsrc_id][self.benchmark_class] for subsrc_id in
                                  self.src.index}
            mean_density = self._get_mean_density()
            self.density_history = self.density_history.append(mean_density, ignore_index=True)
            print("Iteration: {}, Mean density: {}".format(i + 1, mean_density))

    def e_step(self):
        class_y_estimates = {subsrc_id: defaultdict(dict) for subsrc_id in self.src.index}
        for _, row in self.src.iterrows():
            y_s = row[self.y_col]
            for subsrc_id in self.src.index:
                src_count = self.counts['src'][row.name]
                src_classes = self.density_mapper[subsrc_id].keys() & src_count.keys()
                numerators = np.array([self.density_mapper[subsrc_id][c] * src_count[c] for c in src_classes])
                denom = numerators.sum()
                if denom != 0:
                    y_sc_array = numerators * y_s / denom
                else:
                    y_sc_array = [0 for i in len(numerators)]
                for i, c in enumerate(src_classes):
                    class_y_estimates[subsrc_id][row.name][c] = y_sc_array[i]
        self.class_y_estimates = class_y_estimates

    def m_step(self):
        for subsrc_id, dm in self.density_mapper.items():
            for _class in dm.keys():
                w_vec = self.weight[self.weight_idx_mapper[subsrc_id]]
                y_vec = np.array([self.class_y_estimates.get(subsrc_id, {}).get(src_id, {}).get(_class, 0) for src_id in
                                  self.src.index])
                n_vec = np.array([self.counts['src'].get(src_id, {}).get(_class, 0) for src_id in self.src.index])
                numerator = np.dot(w_vec, y_vec)
                denom = np.dot(w_vec, n_vec)
                if denom != 0:
                    density = numerator / denom
                    density = self._cap(density, subsrc_id, _class)
                    self.density_mapper[subsrc_id][_class] = density
                else:
                    self.density_mapper[subsrc_id][_class] = 0

    def _cap(self, density, subsrc_id, _class):
        if self.density_cap is not None and _class in self.density_cap.keys():
            return min(density, self.density_cap[_class] * self.benchmark[subsrc_id])
        else:
            return density

    def _get_mean_density(self):
        mean_density = {_class: 0 for _class in set(self.class_mapper.values())}
        mean_density[-1] = 0
        for _class in mean_density.keys():
            mean = np.nanmean(
                np.array([dm[_class] if _class else np.nan in dm.keys() for dm in self.density_mapper.values()]))
            mean_density[_class] = mean
        return mean_density

    def _initialize_density_mapper(self):
        if self.density_mapper is None:
            print('Initializing density mapper...')
            self.set_density_mapper()
            print('Done')

    def estimate(self):
        if self.class_mapper_changed:
            self._initialize_cell_counts()
            self._initialize_density_mapper()
            self.class_mapper_changed = False
        estimates = []
        for _, row in self.trg.iterrows():
            src_id = row[self.src_id_col]
            y_s = self.src.loc[src_id, self.y_col]
            trg_count = self.counts['trg'][row.name]
            trg_classes = self.density_mapper[src_id].keys() & trg_count.keys()
            numerator = sum(self.density_mapper[src_id][c] * trg_count[c] for c in trg_classes)
            src_count = self.counts['src'][src_id]
            src_classes = self.density_mapper[src_id].keys() & src_count.keys()
            denom = sum(self.density_mapper[src_id][c] * src_count[c] for c in src_classes)
            if denom != 0:
                y_t = y_s * numerator / denom
            else:
                y_t = 0
            estimates.append(y_t)
        return estimates

    def plot_density_history(self):
        for _class in self.density_history.columns:
            plt.plot(range(1, self.n_iter + 1), self.density_history[_class])
        plt.legend()
        plt.show()

    def plot_density_histogram(self, bins=None):
        density_by_class = {}
        for src_id, densities in self.density_mapper.items():
            for _class, density in densities.items():
                if _class not in density_by_class.keys():
                    density_by_class[_class] = []
                density_by_class[_class].append(density)
        n_class = len(density_by_class)
        f, axes = plt.subplots(nrows=-(-n_class // 3), ncols=3, figsize=(15, 5 * -(-n_class // 3)))
        for i, _class in enumerate(density_by_class.keys()):
            if bins is None:
                axes[i // 3, i % 3].hist(density_by_class[_class])
            elif isinstance(bins, int):
                axes[i // 3, i % 3].hist(density_by_class[_class], bins=bins)
            elif isinstance(bins, list):
                assert len(bins) == n_class, "Number of bins does not match with number of classes."
                axes[i // 3, i % 3].hist(density_by_class[_class], bins=bins[i])
            else:
                raise ValueError("Bins must be either int or list.")
            axes[i // 3, i % 3].set_title(_class)
        plt.tight_layout()
        plt.show()
