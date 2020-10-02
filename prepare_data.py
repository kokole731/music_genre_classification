from pre_processing import dataset_split, feature_extractor
from hparam import hps
import warnings
warnings.filterwarnings("ignore")

# split origin data to [train, val, test]
# dataset_split.split()

# extract data to numpy file
feature_extractor.save_feature(hps) 