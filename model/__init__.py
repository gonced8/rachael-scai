from .pegasus import Pegasus
from .coqa import CoQA
from .ubuntu import Ubuntu
from .qrecc import QReCC


def get_model(model_name):
    if "pegasus" in model_name.lower():
        return Pegasus
    else:
        print(f"Unrecognized model name: {model_name}")


def get_data(data_name):
    if "coqa" in data_name.lower():
        return CoQA
    elif "ubuntu" in data_name.lower():
        return Ubuntu
    elif "qrecc" in data_name.lower():
        return QReCC
    else:
        print(f"Unrecognized model name: {model_name}")
