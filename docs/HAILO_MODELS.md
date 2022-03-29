# Hailo Models

Here, we give the full list of models trained in-house for specific use-cases.
Each model is accompanied with its own README, retraining docker and retraining guide.

- FLOPs in the table are counted as MAC operations.
- Supported tasks:
    - [Object Detection](#object-detection)
    - [License Plate Recognition](#license-plate-recognition)

<br>

## Object Detection

Network Name | mAP<sup>*</sup> | Input Resolution (HxWxC) | Params (M) | FLOPs (G) |
--- | --- | --- | --- | --- |
[yolov5m_vehicles](../hailo_models/vehicle_detection/README.md) | 46.5 | 640x640x3 | 21.47 | 25.63 |
[tiny_yolov4_license_plates](../hailo_models/license_plate_detection/README.md) | 73.45|  416x416x3 | 5.87 | 3.4 |

<br>

## License Plate Recognition

Network Name | Accuracy<sup>*</sup> | Input Resolution (HxWxC) | Params (M) | FLOPs (G) |
--- | --- | --- | --- | --- |
[lprnet](../hailo_models/license_plate_recognition/README.md) | 53.82|  75x300x3 | 7.14 | 18.29 |

\* Evaluated on internal dataset