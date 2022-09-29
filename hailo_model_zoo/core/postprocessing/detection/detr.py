import tensorflow as tf
import pickle as pkl
from hailo_model_zoo.utils import path_resolver


class DetrPostProc(object):
    def __init__(self, **kwargs):
        if kwargs['meta_arch'] == 'detr_resnet_v1_50_bn_wo_heads':
            path = ('models_files/ObjectDetection/Detection-COCO/detr/detr_r50/detr_resnet_v1_50/2022-08-28/'
                    'detr_resnet_v1_50_bn_postproc_params.pkl')
        elif kwargs['meta_arch'] == 'detr_resnet_v1_18_bn_wo_heads':
            path = ('models_files/ObjectDetection/Detection-COCO/detr/detr_r18/detr_resnet_v1_18/2022-08-29/'
                    'detr_resnet_v1_18_bn_postproc_params.pkl')
        with open(path_resolver.resolve_data_path(path), 'rb') as fp:
            parameters = pkl.load(fp)
        self.parameters = {}
        self.parameters['class_embed_weight'] = tf.convert_to_tensor(parameters['class_embed.weight'])
        self.parameters['class_embed_bias'] = tf.convert_to_tensor(parameters['class_embed.bias'])
        self.parameters['bbox_embed_layers_0_weight'] = tf.convert_to_tensor(parameters['bbox_embed.layers.0.weight'])
        self.parameters['bbox_embed_layers_0_bias'] = tf.convert_to_tensor(parameters['bbox_embed.layers.0.bias'])
        self.parameters['bbox_embed_layers_1_weight'] = tf.convert_to_tensor(parameters['bbox_embed.layers.1.weight'])
        self.parameters['bbox_embed_layers_1_bias'] = tf.convert_to_tensor(parameters['bbox_embed.layers.1.bias'])
        self.parameters['bbox_embed_layers_2_weight'] = tf.convert_to_tensor(parameters['bbox_embed.layers.2.weight'])
        self.parameters['bbox_embed_layers_2_bias'] = tf.convert_to_tensor(parameters['bbox_embed.layers.2.bias'])

    def detr_postprocessing(self, endnodes, **kwargs):
        # endnoes = [batch, 1, 100, 256]
        x = tf.squeeze(endnodes, [1])               # [batch, 1, 100, 256] -> [batch, 100, 256]
        boxes, logits = self.boxes_logits_heads(x)  # boxes = [batch, 100, 4], logits = [batch, 100, 92]

        # SoftMax to get scores and class predictions
        probs = tf.nn.softmax(logits, axis=-1)                   # [batch, queries, 92]
        labels = tf.math.argmax(probs[:, :, 0:-1], axis=-1)      # [batch, queries]
        scores = tf.gather(probs, labels, axis=2, batch_dims=2)  # [batch, queries] = sample probs at the label location

        # [Cx, Cy, W, H] -> [y_min, x_min, y_max, x_max]
        x_c, y_c, w, h = tf.split(boxes, 4, axis=2)    # [batch, 100, 1] [batch, 100, 1] [batch, 100, 1] [batch, 100, 1]
        x_min = x_c - 0.5 * w
        y_min = y_c - 0.5 * h
        x_max = x_c + 0.5 * w
        y_max = y_c + 0.5 * h
        boxes = tf.concat([y_min, x_min, y_max, x_max], axis=2)   # [batch, 100, 4]

        # num detection -> [batch, 100]
        num_detections = tf.repeat(tf.constant([100]), tf.shape(boxes)[0], axis=0)

        return {'detection_boxes': boxes,
                'detection_scores': scores,
                'detection_classes': labels,
                'num_detections': num_detections}

    def boxes_logits_heads(self, x):

        # get logits
        logits = tf.linalg.matmul(x, self.parameters['class_embed_weight'], transpose_b=True) \
            + self.parameters['class_embed_bias']

        # get boxes
        boxes = tf.linalg.matmul(x, self.parameters['bbox_embed_layers_0_weight'], transpose_b=True) \
            + self.parameters['bbox_embed_layers_0_bias']
        boxes = tf.nn.relu(boxes)
        boxes = tf.linalg.matmul(boxes, self.parameters['bbox_embed_layers_1_weight'], transpose_b=True) \
            + self.parameters['bbox_embed_layers_1_bias']
        boxes = tf.nn.relu(boxes)
        boxes = tf.linalg.matmul(boxes, self.parameters['bbox_embed_layers_2_weight'], transpose_b=True) \
            + self.parameters['bbox_embed_layers_2_bias']
        boxes = tf.math.sigmoid(boxes)
        return boxes, logits

    def postprocessing(self, endnodes, **kwargs):
        return self.detr_postprocessing(endnodes, **kwargs)
