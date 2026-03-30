import collections

import cv2
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

STANDARD_COLORS = [
    "AliceBlue",
    "Chartreuse",
    "Aqua",
    "Aquamarine",
    "Azure",
    "Beige",
    "Bisque",
    "BlanchedAlmond",
    "BlueViolet",
    "BurlyWood",
    "CadetBlue",
    "AntiqueWhite",
    "Chocolate",
    "Coral",
    "CornflowerBlue",
    "Cornsilk",
    "Crimson",
    "Cyan",
    "DarkCyan",
    "DarkGoldenRod",
    "DarkGrey",
    "DarkKhaki",
    "DarkOrange",
    "DarkOrchid",
    "DarkSalmon",
    "DarkSeaGreen",
    "DarkTurquoise",
    "DarkViolet",
    "DeepPink",
    "DeepSkyBlue",
    "DodgerBlue",
    "FireBrick",
    "FloralWhite",
    "ForestGreen",
    "Fuchsia",
    "Gainsboro",
    "GhostWhite",
    "Gold",
    "GoldenRod",
    "Salmon",
    "Tan",
    "HoneyDew",
    "HotPink",
    "IndianRed",
    "Ivory",
    "Khaki",
    "Lavender",
    "LavenderBlush",
    "LawnGreen",
    "LemonChiffon",
    "LightBlue",
    "LightCoral",
    "LightCyan",
    "LightGoldenRodYellow",
    "LightGray",
    "LightGrey",
    "LightGreen",
    "LightPink",
    "LightSalmon",
    "LightSeaGreen",
    "LightSkyBlue",
    "LightSlateGray",
    "LightSlateGrey",
    "LightSteelBlue",
    "LightYellow",
    "Lime",
    "LimeGreen",
    "Linen",
    "Magenta",
    "MediumAquaMarine",
    "MediumOrchid",
    "MediumPurple",
    "MediumSeaGreen",
    "MediumSlateBlue",
    "MediumSpringGreen",
    "MediumTurquoise",
    "MediumVioletRed",
    "MintCream",
    "MistyRose",
    "Moccasin",
    "NavajoWhite",
    "OldLace",
    "Olive",
    "OliveDrab",
    "Orange",
    "OrangeRed",
    "Orchid",
    "PaleGoldenRod",
    "PaleGreen",
    "PaleTurquoise",
    "PaleVioletRed",
    "PapayaWhip",
    "PeachPuff",
    "Peru",
    "Pink",
    "Plum",
    "PowderBlue",
    "Purple",
    "Red",
    "RosyBrown",
    "RoyalBlue",
    "SaddleBrown",
    "Green",
    "SandyBrown",
    "SeaGreen",
    "SeaShell",
    "Sienna",
    "Silver",
    "SkyBlue",
    "SlateBlue",
    "SlateGray",
    "SlateGrey",
    "Snow",
    "SpringGreen",
    "SteelBlue",
    "GreenYellow",
    "Teal",
    "Thistle",
    "Tomato",
    "Turquoise",
    "Violet",
    "Wheat",
    "White",
    "WhiteSmoke",
    "Yellow",
    "YellowGreen",
]


def _get_multiplier_for_color_randomness():
    """Returns a multiplier to get semi-random colors from successive indices.

    This function computes a prime number, p, in the range [2, 17] that:
    - is closest to len(STANDARD_COLORS) / 10
    - does not divide len(STANDARD_COLORS)

    If no prime numbers in that range satisfy the constraints, p is returned as 1.

    Once p is established, it can be used as a multiplier to select
    non-consecutive colors from STANDARD_COLORS:
    colors = [(p * i) % len(STANDARD_COLORS) for i in range(20)]
    """
    num_colors = len(STANDARD_COLORS)
    prime_candidates = [5, 7, 11, 13, 17]

    # Remove all prime candidates that divide the number of colors.
    prime_candidates = [p for p in prime_candidates if num_colors % p]
    if not prime_candidates:
        return 1

    # Return the closest prime number to num_colors / 10.
    abs_distance = [np.abs(num_colors / 10.0 - p) for p in prime_candidates]
    num_candidates = len(abs_distance)
    inds = [i for _, i in sorted(zip(abs_distance, range(num_candidates), strict=True))]
    return prime_candidates[inds[0]]


def save_image_array_as_png(image, output_path):
    """Saves an image (represented as a numpy array) to PNG.

    Args:
      image: a numpy array with shape [height, width, 3].
      output_path: path to which image should be written.
    """
    image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
    # Save using PIL directly (no TensorFlow file API)
    image_pil.save(output_path, "PNG")


def draw_bounding_box_on_image_array(
    image,
    ymin,
    xmin,
    ymax,
    xmax,
    angle=0.0,
    color="red",
    thickness=4,
    display_str_list=(),
    use_normalized_coordinates=True,
):
    """Adds a bounding box to an image (numpy array).

    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    Args:
      image: a numpy array with shape [height, width, 3].
      ymin: ymin of bounding box.
      xmin: xmin of bounding box.
      ymax: ymax of bounding box.
      xmax: xmax of bounding box.
      angle: angle of bounding box (in degrees). Default is 0.
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list: list of strings to display in box
                        (each to be shown on its own line).
      use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
    """
    image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
    draw_bounding_box_on_image(
        image_pil, ymin, xmin, ymax, xmax, angle, color, thickness, display_str_list, use_normalized_coordinates
    )
    np.copyto(image, np.array(image_pil))


def draw_bounding_box_on_image(
    image,
    ymin,
    xmin,
    ymax,
    xmax,
    angle=0.0,
    color="red",
    thickness=4,
    display_str_list=(),
    use_normalized_coordinates=True,
):
    """Adds a bounding box to an image.

    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input 'color'.
    If the top of the bounding box extends to the edge of the image, the strings
    are displayed below the bounding box.

    Args:
      image: a PIL.Image object.
      ymin: ymin of bounding box.
      xmin: xmin of bounding box.
      ymax: ymax of bounding box.
      xmax: xmax of bounding box.
      angle: angle of bounding box (in degrees). Default is 0.
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list: list of strings to display in box
                        (each to be shown on its own line).
      use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size

    # Calculate center, width, and height of the unrotated box
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin
    if use_normalized_coordinates:
        center_x *= im_width
        center_y *= im_height
        width *= im_width
        height *= im_height

    # Create a rotated rectangle object
    rotated_rect = ((center_x, center_y), (width, height), angle)

    # Get the four corner points of the rotated rectangle
    box_points = cv2.boxPoints(rotated_rect)
    box_points = np.int0(box_points)  # Convert to integer coordinates
    if thickness > 0:
        draw.line([tuple(xy) for xy in box_points] + [tuple(box_points[0])], width=thickness, fill=color)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except (IOError, ImportError):
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    top = np.max(box_points[:, 1])
    left = np.min(box_points[:, 0])
    bottom = np.min(box_points[:, 1])

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)], fill=color)
        draw.text((left + margin, text_bottom - text_height - margin), display_str, fill="black", font=font)
        text_bottom -= text_height - 2 * margin


def draw_bounding_boxes_on_image_array(image, boxes, color="red", thickness=4, display_str_list_list=()):
    """Draws bounding boxes on image (numpy array).

    Args:
      image: a numpy array object.
      boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
             The coordinates are in normalized format between [0, 1].
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list_list: list of list of strings.
                             a list of strings for each bounding box.
                             The reason to pass a list of strings for a
                             bounding box is that it might contain
                             multiple labels.

    Raises:
      ValueError: if boxes is not a [N, 4] array
    """
    image_pil = Image.fromarray(image)
    draw_bounding_boxes_on_image(image_pil, boxes, color, thickness, display_str_list_list)
    np.copyto(image, np.array(image_pil))


def draw_bounding_boxes_on_image(image, boxes, color="red", thickness=4, display_str_list_list=()):
    """Draws bounding boxes on image.

    Args:
      image: a PIL.Image object.
      boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
             The coordinates are in normalized format between [0, 1].
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list_list: list of list of strings.
                             a list of strings for each bounding box.
                             The reason to pass a list of strings for a
                             bounding box is that it might contain
                             multiple labels.

    Raises:
      ValueError: if boxes is not a [N, 4] array
    """
    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError("Input must be of size [N, 4]")
    for i in range(boxes_shape[0]):
        display_str_list = ()
        if display_str_list_list:
            display_str_list = display_str_list_list[i]
        draw_bounding_box_on_image(
            image, boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3], color, thickness, display_str_list
        )


def draw_keypoints_on_image_array(
    image,
    keypoints,
    keypoint_scores=None,
    min_score_thresh=0.5,
    color="red",
    radius=2,
    use_normalized_coordinates=True,
    keypoint_edges=None,
    keypoint_edge_color="green",
    keypoint_edge_width=2,
):
    """Draws keypoints on an image (numpy array).

    Args:
      image: a numpy array with shape [height, width, 3].
      keypoints: a numpy array with shape [num_keypoints, 2].
      keypoint_scores: a numpy array with shape [num_keypoints]. If provided, only
        those keypoints with a score above score_threshold will be visualized.
      min_score_thresh: A scalar indicating the minimum keypoint score required
        for a keypoint to be visualized. Note that keypoint_scores must be
        provided for this threshold to take effect.
      color: color to draw the keypoints with. Default is red.
      radius: keypoint radius. Default value is 2.
      use_normalized_coordinates: if True (default), treat keypoint values as
        relative to the image.  Otherwise treat them as absolute.
      keypoint_edges: A list of tuples with keypoint indices that specify which
        keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
        edges from keypoint 0 to 1 and from keypoint 2 to 4.
      keypoint_edge_color: color to draw the keypoint edges with. Default is red.
      keypoint_edge_width: width of the edges drawn between keypoints. Default
        value is 2.
    """
    image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
    draw_keypoints_on_image(
        image_pil,
        keypoints,
        keypoint_scores=keypoint_scores,
        min_score_thresh=min_score_thresh,
        color=color,
        radius=radius,
        use_normalized_coordinates=use_normalized_coordinates,
        keypoint_edges=keypoint_edges,
        keypoint_edge_color=keypoint_edge_color,
        keypoint_edge_width=keypoint_edge_width,
    )
    np.copyto(image, np.array(image_pil))


def draw_keypoints_on_image(
    image,
    keypoints,
    keypoint_scores=None,
    min_score_thresh=0.5,
    color="red",
    radius=2,
    use_normalized_coordinates=True,
    keypoint_edges=None,
    keypoint_edge_color="green",
    keypoint_edge_width=2,
):
    """Draws keypoints on an image.

    Args:
      image: a PIL.Image object.
      keypoints: a numpy array with shape [num_keypoints, 2].
      keypoint_scores: a numpy array with shape [num_keypoints].
      min_score_thresh: a score threshold for visualizing keypoints. Only used if
        keypoint_scores is provided.
      color: color to draw the keypoints with. Default is red.
      radius: keypoint radius. Default value is 2.
      use_normalized_coordinates: if True (default), treat keypoint values as
        relative to the image.  Otherwise treat them as absolute.
      keypoint_edges: A list of tuples with keypoint indices that specify which
        keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
        edges from keypoint 0 to 1 and from keypoint 2 to 4.
      keypoint_edge_color: color to draw the keypoint edges with. Default is red.
      keypoint_edge_width: width of the edges drawn between keypoints. Default
        value is 2.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    keypoints = np.array(keypoints)
    keypoints_x = [k[1] for k in keypoints]
    keypoints_y = [k[0] for k in keypoints]
    if use_normalized_coordinates:
        keypoints_x = tuple([im_width * x for x in keypoints_x])
        keypoints_y = tuple([im_height * y for y in keypoints_y])
    if keypoint_scores is not None:
        keypoint_scores = np.array(keypoint_scores)
        valid_kpt = np.greater(keypoint_scores, min_score_thresh)
    else:
        valid_kpt = np.where(
            np.any(np.isnan(keypoints), axis=1), np.zeros_like(keypoints[:, 0]), np.ones_like(keypoints[:, 0])
        )
    valid_kpt = list(valid_kpt)

    for keypoint_x, keypoint_y, valid in zip(keypoints_x, keypoints_y, valid_kpt, strict=True):
        if valid:
            draw.ellipse(
                [(keypoint_x - radius, keypoint_y - radius), (keypoint_x + radius, keypoint_y + radius)],
                outline=color,
                fill=color,
            )
    if keypoint_edges is not None:
        for keypoint_start, keypoint_end in keypoint_edges:
            if (
                keypoint_start < 0
                or keypoint_start >= len(keypoints)
                or keypoint_end < 0
                or keypoint_end >= len(keypoints)
            ):
                continue
            if not (valid_kpt[keypoint_start] and valid_kpt[keypoint_end]):
                continue
            edge_coordinates = [
                keypoints_x[keypoint_start],
                keypoints_y[keypoint_start],
                keypoints_x[keypoint_end],
                keypoints_y[keypoint_end],
            ]
            draw.line(edge_coordinates, fill=keypoint_edge_color, width=keypoint_edge_width)


def visualize_boxes_and_labels_on_image_array(
    image,
    boxes,
    classes,
    scores,
    category_index,
    keypoints=None,
    keypoint_scores=None,
    keypoint_edges=None,
    track_ids=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=0.5,
    agnostic_mode=False,
    line_thickness=4,
    groundtruth_box_visualization_color="black",
    skip_boxes=False,
    skip_scores=False,
    skip_labels=False,
    skip_track_ids=False,
):
    """Overlay labeled boxes on an image with formatted scores and label names.

    This function groups boxes that correspond to the same location
    and creates a display string for each detection and overlays these
    on the image. Note that this function modifies the image in place, and returns
    that same image.

    Args:
      image: uint8 numpy array with shape (img_height, img_width, 3)
      boxes: a numpy array of shape [N, 4] or [N, 5] for orieneted bounding boxes
      classes: a numpy array of shape [N]. Note that class indices are 1-based,
        and match the keys in the label map.
      scores: a numpy array of shape [N] or None.  If scores=None, then
        this function assumes that the boxes to be plotted are groundtruth
        boxes and plot all boxes as black with no classes or scores.
      category_index: a dict containing category dictionaries (each holding
        category index `id` and category name `name`) keyed by category indices.
      keypoints: a numpy array of shape [N, num_keypoints, 2], can
        be None.
      keypoint_scores: a numpy array of shape [N, num_keypoints], can be None.
      keypoint_edges: A list of tuples with keypoint indices that specify which
        keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
        edges from keypoint 0 to 1 and from keypoint 2 to 4.
      track_ids: a numpy array of shape [N] with unique track ids. If provided,
        color-coding of boxes will be determined by these ids, and not the class
        indices.
      use_normalized_coordinates: whether boxes is to be interpreted as
        normalized coordinates or not.
      max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
        all boxes.
      min_score_thresh: minimum score threshold for a box or keypoint to be
        visualized.
      agnostic_mode: boolean (default: False) controlling whether to evaluate in
        class-agnostic mode or not.  This mode will display scores but ignore
        classes.
      line_thickness: integer (default: 4) controlling line width of the boxes.
      mask_alpha: transparency value between 0 and 1 (default: 0.4).
      groundtruth_box_visualization_color: box color for visualizing groundtruth
        boxes
      skip_boxes: whether to skip the drawing of bounding boxes.
      skip_scores: whether to skip score when drawing a single detection
      skip_labels: whether to skip label when drawing a single detection
      skip_track_ids: whether to skip track id when drawing a single detection

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
    """
    # Create a display string (and color) for every box location, group any boxes
    # that correspond to the same location.
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_keypoints_map = collections.defaultdict(list)
    box_to_keypoint_scores_map = collections.defaultdict(list)
    box_to_track_ids_map = {}
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(boxes.shape[0]):
        if max_boxes_to_draw == len(box_to_color_map):
            break
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if keypoints is not None:
                box_to_keypoints_map[box].extend(keypoints[i])
            if keypoint_scores is not None:
                box_to_keypoint_scores_map[box].extend(keypoint_scores[i])
            if track_ids is not None:
                box_to_track_ids_map[box] = track_ids[i]
            if scores is None:
                box_to_color_map[box] = groundtruth_box_visualization_color
            else:
                display_str = ""
                if not skip_labels:
                    if not agnostic_mode:
                        if classes[i] in category_index:
                            class_name = category_index[classes[i]]["name"]
                        else:
                            class_name = "N/A"
                        display_str = str(class_name)
                if not skip_scores:
                    if not display_str:
                        display_str = "{}%".format(round(100 * scores[i]))
                    else:
                        display_str = "{}: {}%".format(display_str, round(100 * scores[i]))
                if not skip_track_ids and track_ids is not None:
                    if not display_str:
                        display_str = "ID {}".format(track_ids[i])
                    else:
                        display_str = "{}: ID {}".format(display_str, track_ids[i])
                box_to_display_str_map[box].append(display_str)
                if agnostic_mode:
                    box_to_color_map[box] = "DarkOrange"
                elif track_ids is not None:
                    prime_multiple = _get_multiplier_for_color_randomness()
                    box_to_color_map[box] = STANDARD_COLORS[(prime_multiple * track_ids[i]) % len(STANDARD_COLORS)]
                else:
                    box_to_color_map[box] = STANDARD_COLORS[classes[i] % len(STANDARD_COLORS)]

    # Draw all boxes onto image.
    obb = boxes[0].shape[0] == 5
    for box, color in box_to_color_map.items():
        ymin, xmin, ymax, xmax = box[:4]
        # Convert radians to degrees if OBB
        angle = 0.0 if not obb else box[4] * 180.0 / np.pi
        draw_bounding_box_on_image_array(
            image,
            ymin,
            xmin,
            ymax,
            xmax,
            angle=angle,
            color=color,
            thickness=0 if skip_boxes else line_thickness,
            display_str_list=box_to_display_str_map[box],
            use_normalized_coordinates=use_normalized_coordinates,
        )
        if keypoints is not None:
            keypoint_scores_for_box = None
            if box_to_keypoint_scores_map:
                keypoint_scores_for_box = box_to_keypoint_scores_map[box]
            draw_keypoints_on_image_array(
                image,
                box_to_keypoints_map[box],
                keypoint_scores_for_box,
                min_score_thresh=min_score_thresh,
                color=color,
                radius=line_thickness / 2,
                use_normalized_coordinates=use_normalized_coordinates,
                keypoint_edges=keypoint_edges,
                keypoint_edge_color=color,
                keypoint_edge_width=line_thickness // 2,
            )

    return image
