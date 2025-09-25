"""Check mini vs extended"""

#imports
import pandas as pd
import sys
sys.path.append('../../src')                                    #needed for calibrator
sys.path.append('finetuning_text_classifiers')
#sys.path.append(os.join(os.getcwd(), '/finetuning_text_classifiers'))    #for metadataset
#from my code base
from calibrator import PrecomputedCalibrator
from llm_helper_fn import create_new_split, retrieve_data
# for the data
from metadataset.ftc.metadataset import FTCMetadataset

def main():
    #get the data
    #data_version = "mini"
    #data_version = "extended"                              # NOTE not yet downloaded
    data_versions = ["mini", "extended"]
    data_dir = "../data"

    for data_version in data_versions:
        print("Data version:", data_version, flush=True)
        metadataset = FTCMetadataset(data_dir=str(data_dir), 
                                    metric_name="error",
                                    data_version=data_version)
        dataset_names = metadataset.get_dataset_names()
        splits = ["valid", "test"]      # based of github
        for dataset in dataset_names:
            print("Processing dataset:", dataset, data_version, flush=True)
            results = retrieve_data(metadataset, dataset, splits)
            val_len = results['valid'][2].shape[1]
            #default_val_labels = results['valid'][3]
            test_len = results['test'][2].shape[1]
            #default_test_labels = results['test'][3]
            print("Data version:",data_version , "Dataset:" ,dataset ,"Valid length:", val_len, "Test length:", test_len, flush=True)
            print()


if __name__ == "__main__":
    main()
    print("Main done")