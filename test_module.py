import unittest
import geopandas as gpd
import dasymetric


SRC_PATH = 'test_data/tract_part.shp'
TRG_PATH = 'test_data/block_part.shp'
AUX_PATH = 'test_data/lulc_part.tif'
SRC = gpd.read_file(SRC_PATH)
TRG = gpd.read_file(TRG_PATH)
SRC = SRC.rename(columns={'GEOID10': 'tract_id'})
TRG = TRG.rename(columns={'GEOID10': 'block_id'})
TRG = TRG.assign(tract_id=TRG.STATEFP10 + TRG.COUNTYFP10 + TRG.TRACTCE10)
Y_COL = 'nhgis00038'
SRC_ID_COL = 'tract_id'
TRG_ID_COL = 'block_id'
TOTAL_POP = SRC[Y_COL].sum()
CLASS_MAPPER = {11: 'water', 12: 'water',
                21: 'dev1', 22: 'dev2', 23: 'dev3', 24: 'dev4',
                31: 'barren',
                41: 'forest', 42: 'forest', 43: 'forest',
                51: 'shrub', 52: 'shrub',
                71: 'herbaceous', 72: 'herbaceous', 73: 'herbaceous', 74: 'herbaceous',
                81: 'planted', 82: 'planted',
                90: 'wetland', 95: 'wetland'}


class DasymetricTestCase(unittest.TestCase):

    def test_areal_weighting(self):
        areal = dasymetric.Areal_Weighting(src=SRC, trg=TRG, y_col=Y_COL, src_id_col=SRC_ID_COL, trg_id_col=TRG_ID_COL)
        estm = areal.estimate()
        self.assertAlmostEqual(TOTAL_POP, sum(estm), delta=1)

    def test_binary(self):
        binary = dasymetric.Binary_Dasymetric(src=SRC, trg=TRG, y_col=Y_COL, src_id_col=SRC_ID_COL,
                                              trg_id_col=TRG_ID_COL, aux_path=AUX_PATH)
        binary.set_class_mapper({22: 1, 23: 1, 24: 1})
        estm = binary.estimate()
        self.assertAlmostEqual(TOTAL_POP, sum(estm), delta=1)

    def test_idm_containment(self):
        idm_contain = dasymetric.IDM(src=SRC, trg=TRG, y_col=Y_COL, src_id_col=SRC_ID_COL, trg_id_col=TRG_ID_COL,
                                     aux_path=AUX_PATH, method='containment')
        idm_contain.set_class_mapper(CLASS_MAPPER)
        estm = idm_contain.estimate()
        self.assertAlmostEqual(TOTAL_POP, sum(estm), delta=1)

    def test_idm_centroid(self):
        idm_centroid = dasymetric.IDM(src=SRC, trg=TRG, y_col=Y_COL, src_id_col=SRC_ID_COL, trg_id_col=TRG_ID_COL,
                                     aux_path=AUX_PATH, method='centroid')
        idm_centroid.set_class_mapper(CLASS_MAPPER)
        estm = idm_centroid.estimate()
        self.assertAlmostEqual(TOTAL_POP, sum(estm), delta=1)

    def test_idm_percent(self):
        idm_percent = dasymetric.IDM(src=SRC, trg=TRG, y_col=Y_COL, src_id_col=SRC_ID_COL,
                                     trg_id_col=TRG_ID_COL, aux_path=AUX_PATH, method='percent', threshold=0.7)
        idm_percent.set_class_mapper(CLASS_MAPPER)
        estm = idm_percent.estimate()
        self.assertAlmostEqual(TOTAL_POP, sum(estm), delta=1)

    def test_em(self):
        em = dasymetric.EM(src=SRC, trg=TRG, y_col=Y_COL, src_id_col=SRC_ID_COL, trg_id_col=TRG_ID_COL,
                           aux_path=AUX_PATH, n_iter=50)
        em.set_class_mapper(CLASS_MAPPER)
        estm = em.estimate()
        self.assertNotEqual(em.density_mapper, {})
        self.assertAlmostEqual(TOTAL_POP, sum(estm), delta=1)

    def test_gwem(self):
        gwem = dasymetric.GWEM(src=SRC, trg=TRG, y_col=Y_COL, src_id_col=SRC_ID_COL, trg_id_col=TRG_ID_COL,
                               aux_path=AUX_PATH, n_iter=50, N=1)
        gwem.set_class_mapper(CLASS_MAPPER)
        estm = gwem.estimate()
        self.assertNotEqual(gwem.density_mapper, {})
        self.assertAlmostEqual(TOTAL_POP, sum(estm), delta=1)
