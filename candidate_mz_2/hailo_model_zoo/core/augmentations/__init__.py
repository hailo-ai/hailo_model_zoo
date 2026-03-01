from .left_to_right import AugmentedLRModel


def make_model_callback(network_info):
    if network_info.evaluation.infer_type == "facenet_infer":
        return lambda m: AugmentedLRModel(m)

    return lambda m: m
