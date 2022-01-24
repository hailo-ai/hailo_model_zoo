# Hailo Models

Here, we give the full list of models trained in-house for specific applications.

- Supported tasks:
    - [Object Detection](#object-detection)

<br>

## Object Detection

Network Name | mAP<sup>*</sup> | Input Resolution (HxWxC) | Params (M) | FLOPs (G) |
--- | --- | --- | --- | --- |
[yolov5m_vehicles](../hailo_models/vehicle_detection/README.md) | 46.5 | 640x640x3 | 21.47 | 25.63 |
[tiny_yolov4_license_plates](../hailo_models/license_plate_detection/README.md) | 73.45|  416x416x3 | 5.87 | 3.4 |

\* Evaluated on internal dataset  
\** FLOPs in the table are counted as MAC operations.