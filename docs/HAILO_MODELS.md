# Hailo Models

Here, we give the full list of models trained in-house for specific use-cases.
Each model is accompanied with its own README, retraining docker and retraining guide.

- FLOPs in the table are counted as MAC operations.
- Supported tasks:
    - [Object Detection](#object-detection)
    - [Person & Face Detection](#object-detection)
    - [License Plate Recognition](#license-plate-recognition)

> **Important**:  
    Retraining is not available inside the docker version of Hailo Software Suite. In case you use it, clone the hailo_model_zoo outside of the docker, and perform the retraining there:  
    ```git clone https://github.com/hailo-ai/hailo_model_zoo.git``` 

<br>

## Object Detection

Network Name | mAP<sup>*</sup> | Input Resolution (HxWxC) | Params (M) | FLOPs (G) |
--- | --- | --- | --- | --- |
[yolov5m_vehicles](../hailo_models/vehicle_detection/README.md) | 46.5 | 640x640x3 | 21.47 | 25.63 |
[tiny_yolov4_license_plates](../hailo_models/license_plate_detection/README.md) | 73.45|  416x416x3 | 5.87 | 3.4 |
[yolov5s_personface](../hailo_models/personface_detection/README.md) | 47.5|  640x640x3 | 7.25 | 8.38 |

<br>

## License Plate Recognition

Network Name | Accuracy<sup>*</sup> | Input Resolution (HxWxC) | Params (M) | FLOPs (G) |
--- | --- | --- | --- | --- |
[lprnet](../hailo_models/license_plate_recognition/README.md) | 99.96|  75x300x3 | 7.14 | 18.29 |

\* Evaluated on internal dataset