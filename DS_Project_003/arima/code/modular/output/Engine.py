from pathlib import Path
import os
import sys

module_path = Path(__file__).parents[1]
sys.path.append(str(module_path))

from MLPipeline import DataPipeline


def run_pipeline():

    data_path = os.path.join(module_path, "Input/CallCenterData.xlsx")
    dp_object = DataPipeline(data_path)

    dp_object.preprocess_data()
    dp_object.perform_EDA()
    dp_object.train_models()
    dp_object.evaluate_models()


if __name__ == '__main__':
    run_pipeline()