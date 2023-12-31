import numpy as np
from collections import OrderedDict
import os
from hailo_model_zoo.core.eval.eval_base_class import Eval
from sklearn.linear_model import LinearRegression
from scipy.interpolate import CubicSpline


class LaneDetectionEval(Eval):
    """lane evaluation metric class."""

    def __init__(self, **kwargs):
        """Constructs a lane detection evaluation class.

        The class provides the interface to metrics_fn. The
        _update_op() takes detections from each image and pushes them to
        calculates the per-image accuracy. It uses a tfrecord file for the images and image names
        and a json for the labels.
        """
        self._conf_threshold = 0.5
        self._gt_json = os.path.join(kwargs['gt_labels_path'], 'test_label.json')
        self._gt_generator = self._gt_gen()
        self._metric_names = ['Average Accuracy', 'Average FP', 'Average FN']
        self._metrics_vals = [0., 0., 0.]
        self._pix_thresh = 20  # acceptable x-val distance for a vertical section of the lane.
        self._pt_thresh = 0.85  # percentage of good-precision coors between the best detection
        # and a gt lane to be considered a match.
        self._accuracy = 0.
        self._fp = 0.
        self._fn = 0.
        self._images_seen = 0
        self.get_lanes_from_segmap = kwargs.get('meta_arch', '') == 'laneaf'
        self.discarded_lanes = 0
        self.reset()

    def reset(self):
        return

    def _gt_gen(self):
        """returning string with the data dict for the current image"""
        with open(self._gt_json, 'r') as anno_obj:
            gt_list = anno_obj.readlines()
        for im_dict in gt_list:
            yield im_dict

    def extract_gt_for_image(self, image_path):
        im_data = eval(next(self._gt_generator))
        assert im_data['raw_file'] == image_path, 'Image pathes in label json and tfrecord do not match'
        y_samples = im_data['h_samples']
        gt_lanes = im_data['lanes']
        lanes = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
        lanes = [lane for lane in lanes if len(lane) > 0]
        return {'y_samples': y_samples,
                'gt_lanes': gt_lanes,
                'lanes': lanes}

    def _parse_net_output(self, net_output):
        return net_output['predictions']

    def update_op(self, net_output, gt_labels):
        """we assume that the net_output here is already postprocessed which in this case also means we have actual
        lane coordinates and not polynom coefficients as is in the original project.
        The x-predictions are given densely (for all possible y's) - i.e.,
        without knowledge of the resolution that the gt labels are given at.
        """
        net_output = self._parse_net_output(net_output)
        for image_num in range(net_output.shape[0]):
            net_output_for_image = net_output[image_num, :, :]
            gt_image_path = gt_labels['image_name'][image_num].decode('utf-8')
            gt_labels_for_image = self.extract_gt_for_image(gt_image_path)
            gt_labels_for_image['height'] = gt_labels['height'][image_num]
            gt_labels_for_image['width'] = gt_labels['width'][image_num]
            if self.get_lanes_from_segmap:
                samp_factor = 8.0
                h_samples = gt_labels_for_image['y_samples']
                net_output_for_image = self._get_lanes_tusimple(net_output_for_image, h_samples, samp_factor)
                self._bench_one_submit_tusimple(net_output_for_image, gt_labels_for_image)
            else:
                update_inputs = [net_output_for_image, gt_labels_for_image]
                self._bench_one_submit(*update_inputs)

    def _get_gt_angle(self, single_lane_tuples):
        ys = [tup[1] for tup in single_lane_tuples]
        xs = [tup[0] for tup in single_lane_tuples]
        ys = np.array(ys).reshape(-1, 1)
        xs = np.array(xs).reshape(-1, 1)

        lr = LinearRegression()
        lr.fit(ys, xs)
        k = lr.coef_[0]
        theta = np.arctan(k)
        return theta

    def _get_pixel_threshold(self, angle):
        """thresh assumes original image size"""
        return self._pix_thresh / np.cos(angle)

    def _reduce_lanes_to_labels(self, pred_lanes, y_samples, img_h):
        """predicted lanes are given at high x sampling resolution. here we lower the resolution to that of the gt."""
        """pred_lanes is a list of 1d arrays
        pred_ys is a list of 1d arraysi"""
        h_range_int = np.arange(img_h)
        reduced_pred_lanes = []
        for idx, lane in enumerate(pred_lanes):
            reduced_lane = []
            for reduced_y in y_samples:
                if reduced_y in h_range_int:
                    reduced_x = lane[h_range_int == reduced_y][0]
                reduced_lane.append(reduced_x)

            reduced_pred_lanes.append(reduced_lane)
        return reduced_pred_lanes

    def _filter_by_conf_threshold(self, pred_lanes, pred_confs):
        pred_lanes_filtered = []
        for i, lane in enumerate(pred_lanes):
            if pred_confs[i] > self._conf_threshold:
                pred_lanes_filtered.append(lane)
        return pred_lanes_filtered

    def get_line_accuracy(self, pred_lane, gt_lane, thresh, img_w):
        """ i want the gt_lane and pred_lanes to be the x vals for a single lane."""
        pred = np.array([p if (p >= 0 and p < img_w) else -100 for p in pred_lane])
        gt = np.array([g if g >= 0 else -100 for g in gt_lane])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)

    def _bench(self, pred_lanes, gt, gt_org_lanes, y_samples, img_w, pred_confs):
        # gt is a list of lists, each containing (y,x) tuples
        pred_lanes = [np.array(lane).astype(int) for lane in pred_lanes]
        pred_lanes = self._filter_by_conf_threshold(pred_lanes, pred_confs)

        angles = [self._get_gt_angle(gt_lane) for gt_lane in gt]
        thresholds = [self._get_pixel_threshold(angle) for angle in angles]
        line_accs = []
        fp, fn = 0., 0.
        matched = 0.
        """now, running over the gt lanes, each time trying to match a detection with the gt lane:"""
        for gt_org_lane, thresh in zip(gt_org_lanes, thresholds):
            """generating a detection accuracy list for the predictions list over that specific gt lane:"""
            accs = [self.get_line_accuracy(pred_xvals, gt_org_lane, thresh, img_w) for pred_xvals in pred_lanes]

            max_acc = np.max(accs) if len(accs) > 0 else 0.  # the prediction with the highest acc is chosen.
            if max_acc < self._pt_thresh:  # the detection is considered good only if acc>threshold = 0.85
                fn += 1  # if even the detection with best acc is lower than threshold,
                # then the gt lane was not detected. it's a false negative.
            else:
                matched += 1  # if a detection for that lane is found then it is a match.
            line_accs.append(max_acc)  # the best detection accuracy for that gt
            # lane is appended, whether a match or not.
            """note: theoretically, this metric allows for a detection to be used for several gt lanes!"""
        fp = len(pred_lanes) - matched  # a detection that does not correspond to a gt lane is a false positive.
        if len(gt_org_lanes) > 4 and fn > 0:  # not sure why it's used.
            # maybe if 5 lanes are used they allow a low accuracy for 1 lane.
            fn -= 1
        s = sum(line_accs)
        if len(gt_org_lanes) > 4:
            s -= min(line_accs)  # together with the strangeness above: in case of 5 lanes,
            # they allow low-acc for 1 lane and remove it from calc.
        return [s / max(min(4.0, len(gt_org_lanes)), 1.),
                fp / len(pred_lanes) if len(pred_lanes) > 0 else 0.,
                fn / max(min(len(gt_org_lanes), 4.), 1.)]

    def _bench_one_submit(self, pred, gt):
        # this function is for a single image. that's the accuracy update. we will later modify it to be for a batch.
        """gt includes:
           h_samples for image
           lanes: x values for full h_range, for all lanes in image.
           pred is a tuple of 2 lists
           pred[0] is a list of x-value lists for all the lanes in the image
           pred[1] is a list ox y-value lists for all the lanes in the image"""

        pred_lanes = [lane[:-1] for lane in pred]
        pred_confs = [lane[-1] for lane in pred]
        # this representation is only relevant for the threshold calc.
        y_samples = gt['y_samples']
        gt_org_lanes = gt['gt_lanes']  # only x values, including negativ
        gt_lanes = gt['lanes']  # a list of (x,y)-tuple (in orig image coors).
        img_h = gt['height']
        img_w = gt['width']
        lanes = self._reduce_lanes_to_labels(pred_lanes, y_samples, img_h)
        # lowering resolution of predictions to fit gt.
        a, p, n = self._bench(lanes, gt_lanes, gt_org_lanes, y_samples, img_w, pred_confs)
        # in the end we need to divide by the num of images so far:
        self._images_seen += 1
        self._accuracy += a
        self._fp += p
        self._fn += n

    def _bench_one_submit_tusimple(self, pred_lanes, gt):
        gt_org_lanes = gt['gt_lanes']
        gt_lanes = gt['lanes']
        img_w = gt['width']
        angles = [self._get_gt_angle(gt_lane) for gt_lane in gt_lanes]
        thresholds = [self._get_pixel_threshold(angle) for angle in angles]
        line_accs = []
        fp, fn = 0., 0.
        matched = 0.
        for gt_org_lane, thresh in zip(gt_org_lanes, thresholds):
            """generating a detection accuracy list for the predictions list over that specific gt lane:"""
            accs = [self.get_line_accuracy(pred_xvals, gt_org_lane, thresh, img_w) for pred_xvals in pred_lanes]

            max_acc = np.max(accs) if len(accs) > 0 else 0.  # the prediction with the highest acc is chosen.
            if max_acc < self._pt_thresh:  # the detection is considered good only if acc>threshold = 0.85
                fn += 1  # if even the detection with best acc is lower than threshold,
                # then the gt lane was not detected. it's a false negative.
            else:
                matched += 1  # if a detection for that lane is found then it is a match.
            line_accs.append(max_acc)  # the best detection accuracy for that gt
            # lane is appended, whether a match or not.
            """note: theoretically, this metric allows for a detection to be used for several gt lanes!"""
        fp = len(pred_lanes) - matched  # a detection that does not correspond to a gt lane is a false positive.
        if len(gt_org_lanes) > 4 and fn > 0:  # not sure why it's used.
            # maybe if 5 lanes are used they allow a low accuracy for 1 lane.
            fn -= 1
        s = sum(line_accs)
        if len(gt_org_lanes) > 4:
            s -= min(line_accs)  # together with the strangeness above: in case of 5 lanes,
        # they allow low-acc for 1 lane and remove it from calc.
        _acc = s / max(min(4.0, len(gt_org_lanes)), 1.)
        _fp = fp / len(pred_lanes) if len(pred_lanes) > 0 else 0.
        _fn = fn / max(min(len(gt_org_lanes), 4.), 1.)
        self._images_seen += 1
        self._accuracy += _acc
        self._fp += _fp
        self._fn += _fn

    def _get_lanes_tusimple(self, seg_out, h_samples, samp_factor):
        pred_ids = np.unique(seg_out[seg_out > 0])  # find unique pred ids
        # sort lanes based on their size
        lane_num_pixels = [np.sum(seg_out == ids) for ids in pred_ids]
        ret_lane_ids = pred_ids[np.argsort(lane_num_pixels)[::-1]]
        # retain a maximum of 4 lanes
        if ret_lane_ids.size > 4:
            print("Detected more than 4 lanes")
            for rem_id in ret_lane_ids[4:]:
                seg_out[seg_out == rem_id] = 0
            ret_lane_ids = ret_lane_ids[:4]

        # fit cubic spline to each lane
        cs = []
        lane_ids = np.unique(seg_out[seg_out > 0])
        for idx, t_id in enumerate(lane_ids):
            xs, ys = [], []
            for y_op in range(seg_out.shape[0]):
                x_op = np.where(seg_out[y_op, :] == t_id)[0]
                if x_op.size > 0:
                    x_op = np.mean(x_op)
                    x_ip, y_ip = self.coord_op_to_ip(x_op, y_op, samp_factor)
                    xs.append(x_ip)
                    ys.append(y_ip)
            if len(xs) >= 10:
                cs.append(CubicSpline(ys, xs, extrapolate=False))
            else:
                cs.append(None)
        # get x-coordinates from fitted spline
        lanes = []
        for idx, t_id in enumerate(lane_ids):
            if cs[idx] is not None:
                x_out = cs[idx](np.array(h_samples))
                x_out[np.isnan(x_out)] = -2
                lanes.append(x_out.tolist())
            else:
                print("Lane too small, discarding...")
                self.discarded_lanes += 1
        return lanes

    def coord_op_to_ip(self, x, y, scale):
        # (160*scale, 88*scale) --> (160*scale, 88*scale+16=720) --> (1280, 720)
        if x is not None:
            x = int(scale * x)
        if y is not None:
            y = int(scale * y + 16)
        return x, y

    def evaluate(self):
        """Evaluates with detections from all images in our data set with COCO API.

        Returns:
          coco_metric: float numpy array with shape [12] representing the
            coco-style evaluation metrics.
        """
        self._metrics_vals[0] = np.array(self._accuracy / self._images_seen)
        self._metrics_vals[1] = np.array(self._fp / self._images_seen)
        self._metrics_vals[2] = np.array(self._fn / self._images_seen)

    def _get_accuracy(self):
        if self.get_lanes_from_segmap:
            print(f"Overall discarded lanes = {self.discarded_lanes}")
        return OrderedDict([(self._metric_names[0], self._metrics_vals[0]),
                            (self._metric_names[1], self._metrics_vals[1])])
