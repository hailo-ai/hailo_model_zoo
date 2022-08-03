import numpy as np


def fold_normalization(runner, mean_list, std_list):
    hn = runner.get_hn()
    name = hn['name']
    params = runner.get_params()

    layer_name = 'conv1'
    kernel_name = '/'.join([name, layer_name, 'kernel:0'])
    kernel = params[kernel_name]
    if kernel.shape[2] != 3:
        raise ValueError(f"Expected 3 channels in the first convolution, but got shape: {kernel.shape}")
    new_kernel = np.copy(kernel[:, :, :, :])
    for c in range(kernel.shape[2]):
        new_kernel[:, :, c, :] /= std_list[c]
    assert new_kernel.shape == kernel.shape
    params[kernel_name] = new_kernel

    bias_name = '/'.join([name, layer_name, 'bias:0'])
    bias = params[bias_name]
    new_bias_term = np.zeros_like(new_kernel)
    for c in range(kernel.shape[2]):
        new_bias_term[:, :, c, :] = mean_list[c] * new_kernel[:, :, c, :]
    new_bias_term = new_bias_term.sum((0, 1, 2))
    new_bias = bias - new_bias_term
    assert new_bias.shape == bias.shape
    params[bias_name] = new_bias

    runner.load_params(params)
