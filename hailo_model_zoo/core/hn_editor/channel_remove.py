import numpy as np


def _channel_remove(db, layer, mask, orig_shape, npz):
    for i, sh in enumerate(db["layers"][layer]["output_shapes"]):
        if sh[3] == orig_shape:
            db["layers"][layer]["output_shapes"][i] = [sh[0], sh[1], sh[2], int(sum(mask))]
    for i, sh in enumerate(db["layers"][layer]["input_shapes"]):
        if sh[3] == orig_shape:
            db["layers"][layer]["input_shapes"][i] = [sh[0], sh[1], sh[2], int(sum(mask))]
            pred_layer = db["layers"][layer]["input"][i]
            _channel_remove(db, pred_layer, mask, orig_shape, npz)
    if "params" in db["layers"][layer].keys() and "kernel_shape" in db["layers"][layer]["params"]:
        for k, v in npz.items():
            if layer in k:
                if len(npz[k].shape) == 1:
                    # bias
                    npz[k] = v[np.array(mask, bool)]
                elif len(npz[k].shape) == 4:
                    # kernel
                    sh = db["layers"][layer]["params"]["kernel_shape"]
                    if sh[2] == orig_shape:
                        tmp = v[:, :, :, np.array(mask, bool)]
                        npz[k] = tmp[:, :, np.array(mask, bool), :]
                        db["layers"][layer]["params"]["kernel_shape"] = [sh[0], sh[1], int(sum(mask)), int(sum(mask))]
                    else:
                        npz[k] = v[:, :, :, np.array(mask, bool)]
                        db["layers"][layer]["params"]["kernel_shape"] = [sh[0], sh[1], sh[2], int(sum(mask))]


def channel_remove(runner, remove_info):
    """Function gets runner and updates its HN and NPZ

    Args:
        runner: ~hailo_sdk_client.runner.client_runner.ClientRunner
        remove_info: dict containing two fields.
            layer_name: list of strings to remove channels from (output layers)
            mask: list of one hot vectors indicating which channels to remove
    """
    db = runner.get_hn()
    tmp_params = runner.get_params()
    num_of_anchors = remove_info["num_of_anchors"] if "num_of_anchors" in remove_info else None
    for idx, (layer, mask) in enumerate(zip(remove_info["layer_name"], remove_info["mask"])):
        assert layer in db["layers"], "Layer does not exist in the HN"
        assert db["layers"][layer]["type"] == "output_layer", "Chosen layer is not an output"
        sh = db["layers"][layer]["output_shapes"][0]
        mask_tile = list(mask) if num_of_anchors is None else list(np.tile(mask, [num_of_anchors[idx]]))
        if not sh[3] == 1:
            # normal output
            assert len(mask_tile) == sh[3], "One hot vector is not in the right length"
            _channel_remove(db, layer, mask_tile, sh[3], tmp_params)
        else:
            # output layer after argmax
            pred_layer = db["layers"][layer]["input"][0]
            sh = db["layers"][pred_layer]["input_shapes"][0]
            assert len(mask) == sh[3], "One hot vector is not in the right length"
            _channel_remove(db, pred_layer, mask, sh[3], tmp_params)
    runner.set_hn(db)
    runner.load_params(tmp_params)
