import os
import numpy as np
from moabb.datasets import BNCI2014_001, PhysionetMI, Cho2017, Lee2019_MI
from moabb.paradigms import MotorImagery

import mne

mne.set_log_level("ERROR")
mne.set_config("MNE_DATA", "/data/istiaqfuad/mne_data", set_env=True)

# ✅ Proper path setup (important for all datasets)
DATA_PATH = "/data/istiaqfuad/mne_data"

os.environ["MNE_DATA"] = DATA_PATH
os.environ["MOABB_DATA_PATH"] = DATA_PATH
os.environ["MNE_DATASETS_BNCI_PATH"] = DATA_PATH
os.environ["MNE_DATASETS_EEGBCI_PATH"] = DATA_PATH
os.environ["MNE_DATASETS_GIGADB_PATH"] = DATA_PATH
os.environ["MNE_DATASETS_LEE2019_MI_PATH"] = DATA_PATH

# ✅ Common channels (for cross-dataset compatibility)
COMMON_CHANNELS = [
    "FC3",
    "FC1",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
]

# ✅ Use paradigm (correct way)
paradigm = MotorImagery(
    events=["left_hand", "right_hand"],
    n_classes=2,
    fmin=4,
    fmax=40,
    tmin=0,
    tmax=4,
    resample=128,
    channels=COMMON_CHANNELS,
)

# ✅ Datasets
datasets = [
    BNCI2014_001(),
    PhysionetMI(imagined=True, executed=False),
    Cho2017(),
    Lee2019_MI(),
]


# ✅ Safe loader
def load_one_subject(ds):
    subj = ds.subject_list[0]

    try:
        X, y, meta = paradigm.get_data(dataset=ds, subjects=[subj])
        print(f"Loaded subject {subj}")
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        return X, y
    except Exception as e:
        print(f"❌ Failed for subject {subj}: {e}")
        return None, None


# ✅ Run
for ds in datasets:
    print(f"\n===== Dataset: {ds.code} =====")
    print("subjects:", ds.subject_list)

    X, y = load_one_subject(ds)

    if X is not None:
        print("Classes:", np.unique(y))
