import numpy as np
import copy


class LayerSplitter(object):
    def __init__(self, runner, network_info, split_fc=False):
        supported_networks = ['yolov3',
                              'yolov3_gluon',
                              'yolov3_416',
                              'yolov3_gluon_416',
                              'yolov4_leaky',
                              'tiny_yolov4',
                              'tiny_yolov4_license_plates',
                              'polylanenet_resnet_v1_34',
                              'smoke_regnetx_800mf']
        self._runner = runner
        self._hn = self._runner.get_hn()
        self._npz = self._runner.get_params()
        self._layer_names = network_info.hn_editor.output_scheme.outputs_to_split
        self._split_type = self._get_network_name()
        assert self._split_type in supported_networks, \
            'LayerSplitter does not yet support {} network architecture.'.format(self._split_type)
        if network_info.postprocessing.anchors.sizes is not None:
            self._num_anchors = len(network_info.postprocessing.anchors.sizes[0]) // 2
        self._num_classes = network_info.evaluation.classes
        self._hn_modified = None
        self._npz_modified = None
        self._num_lane_coeffs = 4
        self._max_layers = 1000

    def modify_network(self):
        self._modify_hn()
        self._runner.set_hn(self._hn_modified)
        self._modify_npz()
        self._runner.load_params(self._npz_modified)
        return self._runner

    def _modify_hn(self):
        self._hn_modified = copy.deepcopy(self._hn)
        output_index = 0
        self._remove_output_layers()
        self._clear_names_and_shapes_of_prev_layer_outputs()
        self._clear_net_params_output_layer_order()
        for layer in self._layer_names:
            if 'polylanenet' in self._split_type:
                output_layer_ordinals = list(range(0, self._max_layers))
                split_names, split_kernel_shapes = self.split_polylanenet_hn_layer(layer)
            elif 'yolo' in self._split_type:
                output_layer_ordinals = list(range(0, self._max_layers))
                split_names, split_kernel_shapes = self.split_yolov3_hn_layer(layer)
            elif 'tiny_yolov4' in self._split_type:
                output_layer_ordinals = list(range(0, self._max_layers))
                split_names, split_kernel_shapes = self.split_yolov3_hn_layer(layer)
            elif 'smoke' in self._split_type:
                output_layer_ordinals = [0, 1, 3, 4, 5, 6]
                split_names, split_kernel_shapes = self.split_smoke_hn_layer(layer)
            for split_name, split_kernel_shape in zip(split_names, split_kernel_shapes):
                output_index += 1
                self._add_layer_to_hn(split_name, split_kernel_shape, layer, output_layer_ordinals[output_index])
            del(self._hn_modified["layers"][layer])

    def split_polylanenet_hn_layer(self, layer):
        kernel_shape = self._hn_modified["layers"][layer]["params"]["kernel_shape"]
        split_names = []
        split_kernel_shapes = []

        confs_name = layer + "_confs"
        confs_kernel_shape = kernel_shape[:1]
        confs_kernel_shape.append(5)
        self._hn_modified["net_params"]["output_layers_order"].append(confs_name)
        split_names.append(confs_name)
        split_kernel_shapes.append(confs_kernel_shape)

        lower_upper_name = layer + "_lower_upper"
        lower_upper_kernel_shape = kernel_shape[:1]
        lower_upper_kernel_shape.append(10)
        self._hn_modified["net_params"]["output_layers_order"].append(lower_upper_name)
        split_names.append(lower_upper_name)
        split_kernel_shapes.append(lower_upper_kernel_shape)

        coeffs_1_2_name = layer + "_coeffs_1_2"
        coeffs_1_2_kernel_shape = kernel_shape[:1]
        coeffs_1_2_kernel_shape.append(10)
        self._hn_modified["net_params"]["output_layers_order"].append(coeffs_1_2_name)
        split_names.append(coeffs_1_2_name)
        split_kernel_shapes.append(coeffs_1_2_kernel_shape)

        coeffs_3_4_name = layer + "_coeffs_3_4"
        coeffs_3_4_kernel_shape = kernel_shape[:1]
        coeffs_3_4_kernel_shape.append(10)
        self._hn_modified["net_params"]["output_layers_order"].append(coeffs_3_4_name)
        split_names.append(coeffs_3_4_name)
        split_kernel_shapes.append(coeffs_3_4_kernel_shape)
        return split_names, split_kernel_shapes

    def split_yolov3_hn_layer(self, layer):
        kernel_shape = self._hn_modified["layers"][layer]["params"]["kernel_shape"]
        split_names = []
        split_kernel_shapes = []

        centers_name = layer + "_centers"
        centers_kernel_shape = kernel_shape[:3]
        centers_kernel_shape.append(2 * self._num_anchors)
        self._hn_modified["net_params"]["output_layers_order"].append(centers_name)
        split_names.append(centers_name)
        split_kernel_shapes.append(centers_kernel_shape)

        scales_name = layer + "_scales"
        scales_kernel_shape = kernel_shape[:3]
        scales_kernel_shape.append(2 * self._num_anchors)
        self._hn_modified["net_params"]["output_layers_order"].append(scales_name)
        split_names.append(scales_name)
        split_kernel_shapes.append(scales_kernel_shape)

        obj_name = layer + "_obj"
        obj_kernel_shape = kernel_shape[:3]
        obj_kernel_shape.append(1 * self._num_anchors)
        self._hn_modified["net_params"]["output_layers_order"].append(obj_name)
        split_names.append(obj_name)
        split_kernel_shapes.append(obj_kernel_shape)

        probs_name = layer + "_probs"
        probs_kernel_shape = kernel_shape[:3]
        probs_kernel_shape.append(self._num_classes * self._num_anchors)
        self._hn_modified["net_params"]["output_layers_order"].append(probs_name)
        split_names.append(probs_name)
        split_kernel_shapes.append(probs_kernel_shape)
        return split_names, split_kernel_shapes

    def split_smoke_hn_layer(self, layer):
        kernel_shape = self._hn_modified["layers"][layer]["params"]["kernel_shape"]
        split_names = []
        split_kernel_shapes = []

        depth_name = layer + "_depth"
        depth_kernel_shape = kernel_shape[:3]
        depth_kernel_shape.append(1)
        self._hn_modified["net_params"]["output_layers_order"].append(depth_name)
        split_names.append(depth_name)
        split_kernel_shapes.append(depth_kernel_shape)

        """adding conv68 although it's not split because it needs to secure an output layer:"""
        self._hn_modified["net_params"]["output_layers_order"].append(f'{self._get_network_name()}/conv68')

        offset_name = layer + "_offset"
        offset_kernel_shape = kernel_shape[:3]
        offset_kernel_shape.append(2)
        self._hn_modified["net_params"]["output_layers_order"].append(offset_name)
        split_names.append(offset_name)
        split_kernel_shapes.append(offset_kernel_shape)

        dims_name = layer + "_dims"
        dims_kernel_shape = kernel_shape[:3]
        dims_kernel_shape.append(3)
        self._hn_modified["net_params"]["output_layers_order"].append(dims_name)
        split_names.append(dims_name)
        split_kernel_shapes.append(dims_kernel_shape)

        sin_name = layer + "_sin"
        sin_kernel_shape = kernel_shape[:3]
        sin_kernel_shape.append(1)
        self._hn_modified["net_params"]["output_layers_order"].append(sin_name)
        split_names.append(sin_name)
        split_kernel_shapes.append(sin_kernel_shape)

        cos_name = layer + "_cos"
        cos_kernel_shape = kernel_shape[:3]
        cos_kernel_shape.append(1)
        self._hn_modified["net_params"]["output_layers_order"].append(cos_name)
        split_names.append(cos_name)
        split_kernel_shapes.append(cos_kernel_shape)
        return split_names, split_kernel_shapes

    def _add_layer_to_hn(self, name, shape, orig_layer, output_index):
        output_layer_name = "{}/output_layer{}".format(self._get_network_name(), output_index)
        self._hn_modified["layers"][name] = copy.deepcopy(self._hn_modified["layers"][orig_layer])
        self._hn_modified["layers"][name]["output"] = [output_layer_name]
        self._hn_modified["layers"][name]["output_shapes"][0][-1] = shape[-1]
        self._hn_modified["layers"][name]["params"]["kernel_shape"] = shape
        self._make_output_layer(name, output_layer_name, self._hn_modified["layers"][name]["output_shapes"])
        input_shape = self._hn_modified["layers"][name]["input_shapes"][0]
        self._modify_prev_layer_output_list(name, input_shape, orig_layer)

    def _modify_prev_layer_output_list(self, new_output_name, output_shape, orig_output_name):
        input_layer_name = self._hn_modified["layers"][new_output_name]["input"][0]
        self._hn_modified["layers"][input_layer_name]["output"].append(new_output_name)
        self._hn_modified["layers"][input_layer_name]["output_shapes"].append(output_shape)

    def _make_output_layer(self, input_layer_name, output_layer_name, input_shape):
        self._hn_modified["layers"][output_layer_name] = {}
        self._hn_modified["layers"][output_layer_name]["type"] = "output_layer"
        self._hn_modified["layers"][output_layer_name]["input"] = [input_layer_name]
        self._hn_modified["layers"][output_layer_name]["output"] = []
        self._hn_modified["layers"][output_layer_name]["input_shapes"] = input_shape
        self._hn_modified["layers"][output_layer_name]["output_shapes"] = input_shape
        self._hn_modified["layers"][output_layer_name]["original_names"] = ["out"]

    def _check_which_layers_to_remove(self):
        if 'smoke' in self._split_type:
            return [1]
        else:
            return None

    def _remove_output_layers(self):
        layer_list = self._check_which_layers_to_remove()
        output_layers = [layer for layer in self._hn_modified["layers"] if
                         self._hn_modified["layers"][layer]["type"] == "output_layer"]
        if layer_list:
            for layer in output_layers:
                if layer not in [f'{self._get_network_name()}/output_layer{ind}' for ind in layer_list]:
                    output_layers.remove(layer)

        while len(output_layers) > 0:
            output_layer = output_layers.pop()
            del(self._hn_modified["layers"][output_layer])

    def _clear_names_and_shapes_of_prev_layer_outputs(self):
        for layer in self._layer_names:
            for modified_l in self._hn_modified["layers"]:
                if layer in self._hn_modified["layers"][modified_l]["output"]:
                    self._hn_modified["layers"][modified_l]["output"] = \
                        [x for x in self._hn_modified["layers"][modified_l]["output"] if x != layer]
                    self._hn_modified["layers"][modified_l]["output_shapes"] = \
                        [x for x in self._hn_modified["layers"][modified_l]["output_shapes"] if x != layer]

    def _clear_net_params_output_layer_order(self):
        self._hn_modified["net_params"]["output_layers_order"] = []

    def _modify_npz(self):
        self._npz_modified = dict(copy.deepcopy(self._npz))
        for layer in self._layer_names:
            self._split_layer_in_npz(layer)

    def _split_layer_in_npz(self, layer):
        """replacing a layer entry in the npz with 4 layers"""
        if 'polylanenet' in self._split_type:
            self._remodel_polylanenet_weights_and_biases_for_layer(layer)
        elif 'yolo' in self._split_type:
            self._remodel_yolov3_weights_and_biases_for_layer(layer)
        elif 'tiny_yolov4' in self._split_type:
            self._remodel_yolov3_weights_and_biases_for_layer(layer)
        elif 'smoke' in self._split_type:
            self._remodel_smoke_weights_and_biases_for_layer(layer)

        del(self._npz_modified[layer + '/kernel:0'])
        del(self._npz_modified[layer + '/bias:0'])

    def _remodel_polylanenet_weights_and_biases_for_layer(self, layer):
        """remodeling a single output layer"""
        confs_weight_list = []
        confs_bias_list = []
        lower_upper_weight_list = []
        lower_upper_bias_list = []
        coeffs_1_2_weight_list = []
        coeffs_1_2_bias_list = []
        coeffs_3_4_weight_list = []
        coeffs_3_4_bias_list = []
        weights = copy.deepcopy(self._npz_modified[layer + '/kernel:0'])
        biases = copy.deepcopy(self._npz_modified[layer + '/bias:0'])
        for anchor in range(self._num_anchors):
            """in polylanenet #anchors==#lanes"""
            """cherry-picking the weight elements for the 4 new output layer"""
            offset = anchor * (self._num_lane_coeffs + 3)
            """the weights:"""
            confs_weight_list.append(weights[:, offset: offset + 1])
            lower_upper_weight_list.append(weights[:, offset + 1: offset + 3])
            coeffs_1_2_weight_list.append(weights[:, offset + 3: offset + 5])
            coeffs_3_4_weight_list.append(weights[:, offset + 5: offset + 7])
            """the biases:"""
            confs_bias_list.append(biases[offset: offset + 1])
            lower_upper_bias_list.append(biases[offset + 1: offset + 3])
            coeffs_1_2_bias_list.append(biases[offset + 3: offset + 5])
            coeffs_3_4_bias_list.append(biases[offset + 5: offset + 7])

        confs_weights = np.concatenate(confs_weight_list, 1)
        lower_upper_weights = np.concatenate(lower_upper_weight_list, 1)
        coeffs_1_2_weights = np.concatenate(coeffs_1_2_weight_list, 1)
        coeffs_3_4_weights = np.concatenate(coeffs_3_4_weight_list, 1)
        confs_biases = np.concatenate(confs_bias_list, 0)
        lower_upper_biases = np.concatenate(lower_upper_bias_list, 0)
        coeffs_1_2_biases = np.concatenate(coeffs_1_2_bias_list, 0)
        coeffs_3_4_biases = np.concatenate(coeffs_3_4_bias_list, 0)

        self._npz_modified[layer + '_confs/kernel:0'] = confs_weights
        self._npz_modified[layer + '_confs/bias:0'] = confs_biases

        self._npz_modified[layer + '_lower_upper/kernel:0'] = lower_upper_weights
        self._npz_modified[layer + '_lower_upper/bias:0'] = lower_upper_biases

        self._npz_modified[layer + '_coeffs_1_2/kernel:0'] = coeffs_1_2_weights
        self._npz_modified[layer + '_coeffs_1_2/bias:0'] = coeffs_1_2_biases

        self._npz_modified[layer + '_coeffs_3_4/kernel:0'] = coeffs_3_4_weights
        self._npz_modified[layer + '_coeffs_3_4/bias:0'] = coeffs_3_4_biases

    def _remodel_yolov3_weights_and_biases_for_layer(self, layer):
        """remodeling a single output layer"""
        centers_weight_list = []
        centers_bias_list = []
        scales_weight_list = []
        scales_bias_list = []
        obj_weight_list = []
        obj_bias_list = []
        probs_weight_list = []
        probs_bias_list = []
        weights = copy.deepcopy(self._npz_modified[layer + '/kernel:0'])
        biases = copy.deepcopy(self._npz_modified[layer + '/bias:0'])
        for anchor in range(self._num_anchors):
            """cherry-picking the weight elements for the 4 new output layer"""
            offset = anchor * (self._num_classes + 5)
            """the weights:"""
            centers_weight_list.append(weights[:, :, :, offset: offset + 2])
            scales_weight_list.append(weights[:, :, :, offset + 2: offset + 4])
            obj_weight_list.append(weights[:, :, :, offset + 4: offset + 5])
            probs_weight_list.append(weights[:, :, :, offset + 5: offset + 5 + self._num_classes])
            """the biases:"""
            centers_bias_list.append(biases[offset: offset + 2])
            scales_bias_list.append(biases[offset + 2: offset + 4])
            obj_bias_list.append(biases[offset + 4: offset + 5])
            probs_bias_list.append(biases[offset + 5: offset + 5 + self._num_classes])

        centers_weights = np.concatenate(centers_weight_list, 3)
        scales_weights = np.concatenate(scales_weight_list, 3)
        obj_weights = np.concatenate(obj_weight_list, 3)
        probs_weights = np.concatenate(probs_weight_list, 3)
        centers_biases = np.concatenate(centers_bias_list, 0)
        scales_biases = np.concatenate(scales_bias_list, 0)
        obj_biases = np.concatenate(obj_bias_list, 0)
        probs_biases = np.concatenate(probs_bias_list, 0)

        self._npz_modified[layer + '_centers/kernel:0'] = centers_weights
        self._npz_modified[layer + '_centers/bias:0'] = centers_biases

        self._npz_modified[layer + '_scales/kernel:0'] = scales_weights
        self._npz_modified[layer + '_scales/bias:0'] = scales_biases

        self._npz_modified[layer + '_obj/kernel:0'] = obj_weights
        self._npz_modified[layer + '_obj/bias:0'] = obj_biases

        self._npz_modified[layer + '_probs/kernel:0'] = probs_weights
        self._npz_modified[layer + '_probs/bias:0'] = probs_biases

    def _remodel_smoke_weights_and_biases_for_layer(self, layer, rescale=1.0):
        """remodeling a single output layer"""
        weights = copy.deepcopy(self._npz_modified[layer + '/kernel:0'])
        biases = copy.deepcopy(self._npz_modified[layer + '/bias:0'])
        """the weights:"""
        depth_weights = weights[:, :, :, :1]
        offset_weights = weights[:, :, :, 1:3]
        dims_weights = weights[:, :, :, 3:6]
        sin_weights = weights[:, :, :, 6:7]
        cos_weights = weights[:, :, :, 7:] * rescale
        """the biases:"""
        depth_biases = biases[:1]
        offset_biases = biases[1:3]
        dims_biases = biases[3:6]
        sin_biases = biases[6:7]
        cos_biases = biases[7:] * rescale

        self._npz_modified[layer + '_depth/kernel:0'] = depth_weights
        self._npz_modified[layer + '_depth/bias:0'] = depth_biases

        self._npz_modified[layer + '_offset/kernel:0'] = offset_weights
        self._npz_modified[layer + '_offset/bias:0'] = offset_biases

        self._npz_modified[layer + '_dims/kernel:0'] = dims_weights
        self._npz_modified[layer + '_dims/bias:0'] = dims_biases

        self._npz_modified[layer + '_sin/kernel:0'] = sin_weights
        self._npz_modified[layer + '_sin/bias:0'] = sin_biases

        self._npz_modified[layer + '_cos/kernel:0'] = cos_weights
        self._npz_modified[layer + '_cos/bias:0'] = cos_biases

    def _get_network_name(self):
        hn_dict = self._hn
        return hn_dict['name']
