import tensorflow as tf
import numpy as np

from tensorflow.image import combined_non_max_suppression
from hailo_model_zoo.core.postprocessing.detection.detection_common import tf_postproc_nms


def collect_box_class_predictions(output_branches, num_classes, type):
    # RESHAPING AND CONCAT RESULTS BEFORE RUNNING THROUGH POST PROCESSING STAGE:
    box_predictors_list = []
    class_predictors_list = []
    sorted_output_branches = output_branches
    for i, BoxTensor in enumerate(sorted_output_branches):
        num_of_batches, branch_h, branch_w, branch_features = tf.unstack(tf.shape(BoxTensor))
        # Odd locations are the box predictors
        if i % 2 == 0:
            if type == 'palm':
                # Extract only the box features (remaining features are 7 key points (x,y))
                key_points_size = 18
                # reorder [y, x, h, w] to [x, y, w, h]
                indices = sum([[key_points_size * x + 1, key_points_size * x,
                                key_points_size * x + 3, key_points_size * x + 2]
                              for x in range(BoxTensor.shape[-1] // key_points_size)], [])
                BoxTensor = tf.gather(BoxTensor, tf.constant(indices, tf.int32), axis=-1)
                num_of_batches, branch_h, branch_w, branch_features = tf.unstack(tf.shape(BoxTensor))
            reshaped_tensor = tf.reshape(BoxTensor,
                                         shape=[num_of_batches,
                                                branch_h * branch_w * tf.cast(branch_features / 4, tf.int32),
                                                4])
            box_predictors_list.append(reshaped_tensor)
        # Even locations are the class preidctors
        else:
            reshaped_tensor = \
                tf.reshape(BoxTensor,
                           shape=[num_of_batches,
                                  branch_h * branch_w * tf.cast(branch_features / (num_classes + 1), tf.int32),
                                  num_classes + 1])
            class_predictors_list.append(reshaped_tensor)

    box_predictors = tf.concat(box_predictors_list, axis=1)
    class_predictors = tf.concat(class_predictors_list, axis=1)
    return box_predictors, class_predictors


class BoxSpecsCreator(object):

    def __init__(self, anchors):
        predefined_anchors_flag = anchors.get('predefined', None)
        if predefined_anchors_flag:
            pass
        else:
            self._type = anchors['type']
            self._scales = anchors['scales']
            self._num_layers = anchors['num_layers']
            self._aspect_ratios = anchors['aspect_ratios']
            self._scale_factors = anchors['scale_factors']

            if self._type == 'fpn':
                self._scales_per_octave = anchors['scales_per_octave']
            else:
                self._min_scale = anchors['min_scale']
                self._max_scale = anchors['max_scale']
                self._interpolated_scale_aspect_ratio = anchors['interpolated_scale_aspect_ratio']

    def create_box_specs_list(self):
        self._box_specs_list = []
        if self._type == 'fpn':
            for scale in self._scales:
                layer_box_specs = []
                for aspect in self._aspect_ratios:
                    for octave in range(self._scales_per_octave):
                        layer_box_specs.append((scale * np.sqrt(2**octave), aspect))
                self._box_specs_list.append(layer_box_specs)
        elif self._type == 'palm':
            if self._scales is None or not self._scales:
                self._scales = [self._min_scale + (self._max_scale - self._min_scale) * i / (self._num_layers - 1)
                                for i in range(self._num_layers)] + [1.0]
            for layer, scale, scale_next in zip(range(self._num_layers), self._scales[:-1], self._scales[1:]):
                layer_box_specs = []
                layer_box_specs.append((scale, self._aspect_ratios[0]))
                layer_box_specs.append((np.sqrt(scale * scale_next),
                                        self._interpolated_scale_aspect_ratio))
                self._box_specs_list.append(layer_box_specs)
        else:
            if self._scales is None or not self._scales:
                self._scales = [self._min_scale + (self._max_scale - self._min_scale) * i / (self._num_layers - 1)
                                for i in range(self._num_layers)] + [1.0]
            for layer, scale, scale_next in zip(range(self._num_layers), self._scales[:-1], self._scales[1:]):
                layer_box_specs = []
                if layer == 0:
                    layer_box_specs = [(0.1, 1.0), (scale, 2.0), (scale, 0.5)]
                else:
                    for aspect_ratio in self._aspect_ratios:
                        layer_box_specs.append((scale, aspect_ratio))
                    # Add one more anchor, with a scale between the current scale, and the
                    # scale for the next layer, with a specified aspect ratio (1.0 by
                    # default).
                    if self._interpolated_scale_aspect_ratio > 0.0:
                        layer_box_specs.append((np.sqrt(scale * scale_next),
                                                self._interpolated_scale_aspect_ratio))
                self._box_specs_list.append(layer_box_specs)
        return self._box_specs_list


class SSDPostProc(object):
    # The following params are corresponding to those used for training the model
    label_offset = 1

    def __init__(self, img_dims=(300, 300), nms_iou_thresh=0.6,
                 score_threshold=0.3, anchors=None, classes=90,
                 should_clip=True, **kwargs):
        self._image_dims = img_dims
        self._nms_iou_thresh = nms_iou_thresh
        self._score_threshold = score_threshold
        self._num_classes = classes
        if anchors is None:
            raise ValueError('Missing detection anchors metadata')
        self._anchors = BoxSpecsCreator(anchors)
        self._nms_on_device = False
        self._should_clip = should_clip
        if kwargs["device_pre_post_layers"] and kwargs["device_pre_post_layers"].get('nms', False):
            self._nms_on_device = True
        self.hpp = kwargs.get("hpp", False)
        self.sigmoid = kwargs["device_pre_post_layers"].get("sigmoid", False)

    @staticmethod
    def expanded_shape(orig_shape, start_dim, num_dims):
        with tf.name_scope('ExpandedShape'):
            start_dim = tf.expand_dims(start_dim, 0)  # scalar to rank-1
            before = tf.slice(orig_shape, [0], start_dim)
            add_shape = tf.ones(tf.reshape(num_dims, [1]), dtype=tf.int32)
            after = tf.slice(orig_shape, start_dim, [-1])
            new_shape = tf.concat([before, add_shape, after], 0)
            return new_shape

    @staticmethod
    def meshgrid(x, y):
        with tf.name_scope('Meshgrid'):
            x = tf.convert_to_tensor(x)
            y = tf.convert_to_tensor(y)
            x_exp_shape = SSDPostProc.expanded_shape(tf.shape(x), 0, tf.rank(y))
            y_exp_shape = SSDPostProc.expanded_shape(tf.shape(y), tf.rank(y), tf.rank(x))

            xgrid = tf.tile(tf.reshape(x, x_exp_shape), y_exp_shape)
            ygrid = tf.tile(tf.reshape(y, y_exp_shape), x_exp_shape)
            new_shape = y.get_shape().concatenate(x.get_shape())
            xgrid.set_shape(new_shape)
            ygrid.set_shape(new_shape)

            return xgrid, ygrid

    @staticmethod
    def feature_map_shapes_tensor(endnodes):
        with tf.name_scope('FeatureMapShapes'):
            return [tf.slice(tf.shape(output_branch), [1], [3]) for output_branch in endnodes[0::2]]

    def extract_anchors(self, endnodes):
        # base_anchor_size = np.array([1., 1.])
        bboxes_list = []
        im_height, im_width = self._image_dims
        min_im_shape = float(min(im_height, im_width))
        scale_height = min_im_shape / im_height
        scale_width = min_im_shape / im_width
        box_specs_list = self._anchors.create_box_specs_list()
        feature_map_shape_list = self.feature_map_shapes_tensor(endnodes)
        anchor_strides = [(1.0 / tf.cast(pair[0], tf.float32),
                           1.0 / tf.cast(pair[1], tf.float32)) for pair in feature_map_shape_list]
        anchor_offsets = [(0.5 * stride[0], 0.5 * stride[1]) for stride in anchor_strides]

        for feature_map_index, (grid_size, anchor_stride, anchor_offset, box_spec) in enumerate(
                zip(feature_map_shape_list, anchor_strides, anchor_offsets, box_specs_list)):
            grid_height, grid_width = grid_size[0], grid_size[1]
            scale, aspect_ratio = zip(*box_spec)

            ratio_sqrts = tf.sqrt(aspect_ratio)
            # TODO: HAVE A FACTOR FOR FIXING THE SCALE ACCORDING TO THE REALATION
            # BETWEEN NEW IMAGE SIZE TO TRAINED IMAGE SIZE
            heights = scale / ratio_sqrts * scale_height * 1.
            # TODO: HAVE A FACTOR FOR FIXING THE SCALE ACCORDING TO THE REALATION
            # BETWEEN NEW IMAGE SIZE TO TRAINED IMAGE SIZE
            widths = scale * ratio_sqrts * scale_width * 1.
            # Get a grid of box centers
            y_centers = tf.cast(tf.range(grid_height), tf.float32)
            y_centers = y_centers * anchor_stride[0] + anchor_offset[0]
            x_centers = tf.cast(tf.range(grid_width), tf.float32)
            x_centers = x_centers * anchor_stride[1] + anchor_offset[1]
            x_centers, y_centers = self.meshgrid(x_centers, y_centers)
            widths_grid, x_centers_grid = self.meshgrid(widths, x_centers)
            heights_grid, y_centers_grid = self.meshgrid(heights, y_centers)
            bbox_centers = tf.stack([y_centers_grid, x_centers_grid], axis=3)
            bbox_sizes = tf.stack([heights_grid, widths_grid], axis=3)
            bbox_centers = tf.reshape(bbox_centers, [-1, 2])
            bbox_sizes = tf.reshape(bbox_sizes, [-1, 2])
            bboxes_tensor = tf.concat([bbox_centers, bbox_sizes], axis=1)
            if self._anchors._type == 'palm' and feature_map_index == 0:
                # first output is duplicated by 3
                bboxes_tensor = tf.repeat(bboxes_tensor, 3, axis=0)
            bboxes_list.append(bboxes_tensor)
        anchors_tensor = tf.concat(bboxes_list, axis=0, name='Anchors')

        return anchors_tensor

    def _decode_boxes(self, rel_codes, anchors):
        # ycenter_a, xcentera, ha, wa = anchors.get_center_coordinates_and_sizes()
        ycenter_a, xcenter_a, ha, wa = tf.unstack(tf.transpose(anchors))

        ty, tx, th, tw = tf.unstack(tf.transpose(rel_codes))
        if self._anchors._scale_factors:
            # default factors: [10., 10., 5., 5.]
            ty /= self._anchors._scale_factors[0]
            tx /= self._anchors._scale_factors[1]
            th /= self._anchors._scale_factors[2]
            tw /= self._anchors._scale_factors[3]
        w = tf.exp(tw) * wa
        h = tf.exp(th) * ha
        ycenter = ty * ha + ycenter_a
        xcenter = tx * wa + xcenter_a
        ymin = ycenter - h / 2.
        xmin = xcenter - w / 2.
        ymax = ycenter + h / 2.
        xmax = xcenter + w / 2.
        return tf.transpose(tf.stack([ymin, xmin, ymax, xmax]))

    def tf_preproc(self, startnode, size=[300, 300]):
        with tf.name_scope('Preprocessor'):
            image_tensor_resized = tf.image.resize(startnode,
                                                   size=size,
                                                   method='bicubic',
                                                   align_corners=True)
            image_tensor_resized = tf.add(tf.multiply(image_tensor_resized, tf.constant(2. / 255)), tf.constant(-1.))
            return image_tensor_resized

    def tf_postproc(self, endnodes):

        with tf.name_scope('Postprocessor'):
            # Collect all output branches into Boxes/classes objects
            box_predictions, classes_predictions = \
                collect_box_class_predictions(endnodes, self._num_classes, self._anchors._type)
            # Score Conversion using Sigmoid function
            if not self.sigmoid:
                detection_scores = tf.sigmoid(classes_predictions)

            # detection_scores = tf.identity(classes_predictions_sigmoid, 'raw_box_scores')
            # Slicing Background class score (for a single class no need to slice)
            if self._num_classes > 0:
                detection_scores = tf.slice(detection_scores, [0, 0, 1], [-1, -1, -1])

            batch_size, num_proposals = tf.unstack(tf.slice(tf.shape(box_predictions), [0], [2]))

            # batch_size = self.batch_size if self.batch_size else batch_size
            # num_proposals = self.num_proposals if self.num_proposals else num_proposals

            anchors = self.extract_anchors(endnodes)
            # anchors = self.extract_anchors()

            tiled_anchor_boxes = tf.tile(tf.expand_dims(anchors, 0), [batch_size, 1, 1])
            tiled_anchors_boxlist = tf.reshape(tiled_anchor_boxes, [-1, 4])
            # Decode all predicted boxes to the following presentation [ymin,xmin,ymax,xmax] using the
            # anchors centers/width/hieght
            decoded_boxes = self._decode_boxes(
                tf.reshape(box_predictions, [-1, 4]),
                tiled_anchors_boxlist)
            detection_boxes = tf.reshape(decoded_boxes, [batch_size, num_proposals, 4])
            # detection_boxes = tf.identity(tf.expand_dims(detection_boxes, axis=[2]), 'raw_box_locations')
            detection_boxes = tf.expand_dims(detection_boxes, axis=[2])

            (nmsed_boxes, nmsed_scores, nmsed_classes, num_detections) = \
                combined_non_max_suppression(boxes=detection_boxes,
                                             scores=detection_scores,
                                             score_threshold=self._score_threshold,
                                             iou_threshold=self._nms_iou_thresh,
                                             max_output_size_per_class=100,
                                             max_total_size=100,
                                             clip_boxes=self._should_clip)
            # adding offset to the class prediction and cast to integer
            nmsed_classes = tf.cast(tf.add(nmsed_classes, self.label_offset), tf.int16)

        return {'detection_boxes': nmsed_boxes,
                'detection_scores': nmsed_scores,
                'detection_classes': nmsed_classes,
                'num_detections': num_detections}

    def postprocessing(self, endnodes, **kwargs):
        if self._nms_on_device or self.hpp:
            return tf_postproc_nms(endnodes,
                                   labels_offset=kwargs['labels_offset'],
                                   score_threshold=self._score_threshold,
                                   coco_2017_to_2014=False)
        else:
            return self.tf_postproc(endnodes)
