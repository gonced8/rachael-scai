from .pegasus import Pegasus
from .coqa import  CoQA
from .ubuntu import Ubuntu


def get_model(model_name):
    if model_name.lower() == "pegasus":
        return Pegasus
    else
        print(f"Unrecognized model name: {model_name}")

def get_data(data_name):
    if data_name.lower() == "coqa":
        return CoQA
    elif data_name.lower() == "ubuntu":
        return Ubuntu
    else
        print(f"Unrecognized model name: {model_name}")
