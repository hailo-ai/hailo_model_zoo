def bgr2rgb(runner):
    hn = runner.get_hn()
    network_name = hn["name"]
    params = runner.get_params()

    layer_name = "conv1"
    parameter_name = "/".join([network_name, layer_name, "kernel:0"])
    shape = params[parameter_name].shape
    channels = shape[2]
    if channels not in [3, 12]:
        raise ValueError(
            f"Expected 3 channels in the first convolution (or 12 after space_to_depth) but got shape: {shape}"
        )
    if channels == 3:
        params[parameter_name] = params[parameter_name][:, :, ::-1, :]
    if channels == 12:
        # if the first convulution has 12 channels, we assume the network does space_to_depth as in yolov5 variants
        _verify_after_space_to_depth(runner, layer_name)

        k = params[parameter_name]
        h, w, c_in, c_out = k.shape
        k_expanded = k.reshape(h, w, 4, 3, c_out)  # h x w x 4 x 3 x c_out
        k_swapped = k_expanded[:, :, :, ::-1, :]  # h x w x 4 x 3 x c_out
        params[parameter_name] = k_swapped.reshape(h, w, c_in, c_out)

    runner.load_params(params)


def _verify_after_space_to_depth(runner, layer_name):
    net = runner.get_hn_model()
    layer = net.get_layer_by_name(layer_name)
    predecessors = list(net.predecessors(layer))
    if len(predecessors) != 1:
        predecessor_names = [p.name for p in predecessors]
        raise ValueError(
            f"For a conv with 12 channels expected a single single_to_depth predecessors, but got {predecessor_names}"
        )

    predecessor = predecessors[0]
    op = predecessor.op.name
    if op != "space_to_depth":
        name = predecessor.name
        raise ValueError(
            f"For a conv with 12 channels expected predecessor of type space_to_depth, but got {op} ({name})"
        )
