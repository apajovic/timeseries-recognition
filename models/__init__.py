
from models.fcn_transformer import init_default_fcnt

model_init_functions = {
            "fcnt":init_default_fcnt
        }

class ModelFactory:
    @staticmethod
    def get(model_name, **kwargs):
        if model_name not in model_init_functions:
            raise ValueError(f"Invalid model name! Available:{model_init_functions.keys()}")

        return model_init_functions.get(model_name)(**kwargs)
