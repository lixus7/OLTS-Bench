"""Dataset registry and the two split protocols.

The boundaries below reproduce the main study exactly: they were checked against
the number of TEST forecast origins of all 60 delivered (dataset, split, H) scenarios.

O ("online" split, 20/5/75):  TRAIN = int(0.2 N), TEST = int(0.75 N), VALIDATION = the rest.
T ("traditional" split):      TRAIN = int(r N) with r = 0.6 (ETT, PEMS08) or 0.7 (Weather,
                              Electricity, Solar, PEMS-BAY), TEST = int(0.2 N), VALIDATION = the rest.
N is the number of rows used: the first 14,400 rows of ETTh* and 57,600 rows of ETTm*
(12/4/4 months, the usual ETT convention), the full file for the others.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Spec:
    name: str
    file: str          # file name inside the data directory
    channels: int      # number of variables
    rows_used: int     # N above
    t_train: float     # TRAIN fraction of the T split
    reader: str        # csv (with a date column) | txt | npz | h5
    missing: str = ''  # which raw values count as missing (not scored, not used as labels)


REGISTRY = {
    'ETTh1': Spec('ETTh1', 'ETTh1.csv', 7, 14400, 0.6, 'csv'),
    'ETTh2': Spec('ETTh2', 'ETTh2.csv', 7, 14400, 0.6, 'csv'),
    'ETTm1': Spec('ETTm1', 'ETTm1.csv', 7, 57600, 0.6, 'csv'),
    'ETTm2': Spec('ETTm2', 'ETTm2.csv', 7, 57600, 0.6, 'csv'),
    'Weather': Spec('Weather', 'weather.csv', 21, 52696, 0.7, 'csv', missing='-9999 sentinel'),
    'Electricity': Spec('Electricity', 'electricity.csv', 321, 26304, 0.7, 'csv'),
    'Solar': Spec('Solar', 'solar_AL.txt', 137, 52560, 0.7, 'txt'),
    'PEMS-BAY': Spec('PEMS-BAY', 'pems-bay.h5', 325, 52116, 0.7, 'h5', missing='zero speed'),
    'PEMS08': Spec('PEMS08', 'PEMS08.npz', 170, 17856, 0.6, 'npz'),
}
# The tenth dataset of the main study (Beijing air quality, rebuilt from raw Quotsoft files
# with the original missing-value pattern) is not distributed here.


def boundaries(name, split):
    """[0, end of TRAIN, end of VALIDATION, end of TEST] in rows."""
    s = REGISTRY[name]
    n = s.rows_used
    if split == 'O':
        nt, nte = int(n * 0.2), int(n * 0.75)
    elif split == 'T':
        nt, nte = int(n * s.t_train), int(n * 0.2)
    else:
        raise ValueError('split must be O or T, not ' + repr(split))
    return [0, nt, n - nte, n]
