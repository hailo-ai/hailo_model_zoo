def bgr2rgb(runner):
    hn = runner.get_hn()
    name = hn['name']
    params = runner.get_params()

    kernel_name = '/'.join([name, 'conv1', 'kernel:0'])
    if params[kernel_name].shape[2] != 3:
        raise ValueError("Expected 3 channels in the first convolution, but got shape: {}".format(
            params[kernel_name].shape))
    params[kernel_name] = params[kernel_name][:, :, ::-1, :]

    runner.load_params(params)
