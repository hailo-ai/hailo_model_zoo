# Use Your Own Network

In this document, we describe the process of training a new deep learning network and add it to the Hailo Model Zoo. We choose YOLOv3/4/5 as examples due to their popularity and easy-to-use frameworks for training. Learn more about YOLOv3/4 in [**here**](https://github.com/AlexeyAB/darknet) and YOLOv5 in [**here**](https://github.com/ultralytics/yolov5/tree/v2.0).

<details>
    <summary>YOLOv3</summary>

## Training YOLOv3
To train your YOLOv3 network follow these steps (full instructions can be found in [**here**](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects)):
1. Clone the Darknet framework:
```
git clone https://github.com/AlexeyAB/darknet.git; cd darknet
```
2. build the framework using <code>make</code> (it is recommended to build with CUDA support by setting <code>GPU=1</code> in the Makefile)
3. Download pretrained weights for YOLOv3 model from [**here**](https://pjreddie.com/media/files/darknet53.conv.74)
4. Create a new cfg file. This file contain the information about your model: input resolution, number of classes and so on. The default cfg file for YOLOv3 can be found in [**here**](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov3.cfg)
5. Add information about your data:
    * Create <code>obj.names</code> and <code>obj.data</code> in <code>build\darknet\x64\data\ </code>
    * Place your jpg images in <code>build\darknet\x64\data\obj\ </code>
    * Generate a txt for each image (in the same directory) containing the annotations in the format of <code>\<object-class> \<x_center> \<y_center> \<width> \<height></code>. For example: for img1.jpg create img1.txt containing:
```
1 0.716797 0.395833 0.216406 0.147222
0 0.687109 0.379167 0.255469 0.158333
1 0.420312 0.395833 0.140625 0.166667
```
6. Create <code>train.txt</code> in directory <code>build\darknet\x64\data\ </code> with filenames of your images. For example:
```
data/obj/img1.jpg
data/obj/img2.jpg
data/obj/img3.jpg
```
7. Start training:
```
./darknet detector train build/darknet/x64/data/obj.data cfg/yolov3.cfg yolov3.conv.74
```
8. Final product would be available in <code>build\darknet\x64\backup\ </code>


## Export to ONNX
To export the trained YOLOv3 network to ONNX follow these steps:
1. Clone the following repo:
```
git clone https://github.com/nivosco/pytorch-YOLOv4.git;cd pytorch-YOLOv4
```
2. Install onnxruntime:
```
pip install onnxruntime
```
3. Run python script to generate the ONNX model (pretrained <code>yolov3.weights</code> can be downloaded from [**here**](https://pjreddie.com/media/files/yolov3.weights)):
```
python demo_darknet2onnx.py cfg/yolov3.cfg yolov3.weights image.jpg 1
```

4. (optional) Using your own cfg file might require adding <code>scale_x_y=1.0</code> under each <code>[yolo]</code> block in the cfg file. Check <code>cfg/yolov3.cfg</code> for an example.


## Add the Model to the Hailo Model Zoo
In this section we can use the ONNX model to generate an HEF file to infer on the Hailo-8.
1. Add a new YAML file of YOLOv3 (for example, check <code>hailo_model_zoo/cfg/networks/yolov3.yaml</code>) with the new input resolution, number of classes and so on.
2. (optional) Generate TFRECORD files of your data for evaluation/calibration. For more information check [**DATA.md**](DATA.md).
3. Run the full precision evaluation to reproduce the accuracy of the model:
```
python hailo_model_zoo/main.py eval yolov3 --ckpt yolov3.onnx --data-path val.tfrecord --yaml yolov3.yaml
```
4. Measure the quantized accuracy of your model:
```
python hailo_model_zoo/main.py eval yolov3 --target emulator --ckpt yolov3.onnx --calib-path calib.tfrecord --data-path val.tfrecord --yaml yolov3.yaml
```
5. Compile the model and generate the HEF file:
```
python hailo_model_zoo/main.py compile yolov3 --ckpt yolov3.onnx --calib-path calib.tfrecord --yaml yolov3.yaml
```
6. (optional) Measure the accuracy on the Hailo-8:
```
python hailo_model_zoo/main.py eval yolov3 --target hailo8 --hef yolov3.hef  --data-path val.tfrecord --yaml yolov3.yaml
```
</details>

<details>
    <summary>YOLOv4</summary>

## Training YOLOv4-leaky
To train your YOLOv4-leaky network follow these steps (full instructions can be found in [**here**](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects)):
1. Clone the Darknet framework:
```
git clone https://github.com/AlexeyAB/darknet.git; cd darknet
```
2. build the framework using <code>make</code> (it is recommended to build with CUDA support by setting <code>GPU=1</code> in the Makefile)
3. Download pretrained weights for YOLOv4-leaky model from [**here**](https://drive.google.com/open?id=1dW-Sd70aTmXuFvYspiY85SwWdn0cr447)
4. Create a new cfg file. This file contain the information about your model: input resolution, number of classes and so on. The default cfg file for YOLOv4-leaky can be found in [**here**](https://drive.google.com/open?id=1C4w_2loEpi-MznqMgKo16oZD7BcvgsCF).
5. Add information about your data:
    * Create <code>obj.names</code> and <code>obj.data</code> in <code>build\darknet\x64\data\ </code>
    * Place your jpg images in <code>build\darknet\x64\data\obj\ </code>
    * Generate a txt for each image (in the same directory) containing the annotations in the format of <code>\<object-class> \<x_center> \<y_center> \<width> \<height></code>. For example: for img1.jpg create img1.txt containing:
```
1 0.716797 0.395833 0.216406 0.147222
0 0.687109 0.379167 0.255469 0.158333
1 0.420312 0.395833 0.140625 0.166667
```
6. Create <code>train.txt</code> in directory <code>build\darknet\x64\data\ </code> with filenames of your images. For example:
```
data/obj/img1.jpg
data/obj/img2.jpg
data/obj/img3.jpg
```
7. Start training:
```
./darknet detector train build/darknet/x64/data/obj.data cfg/yolov4-leaky.cfg yolov4-leaky.weights
```
8. Final product would be available in <code>build\darknet\x64\backup\ </code>


## Export to ONNX
To export the trained YOLOv4-leaky network to ONNX follow these steps:
1. Clone the following repo:
```
git clone https://github.com/nivosco/pytorch-YOLOv4.git;cd pytorch-YOLOv4
```
2. Install onnxruntime:
```
pip install onnxruntime
```
3. Run python script to generate the ONNX model (pretrained <code>yolov4-leaky.weights</code> can be downloaded from [**here**](https://drive.google.com/open?id=1dW-Sd70aTmXuFvYspiY85SwWdn0cr447)):
```
python demo_darknet2onnx.py cfg/yolov4-leaky.cfg yolov4-leaky.weights image.jpg 1
```


## Add the Model to the Hailo Model Zoo
In this section we can use the ONNX model to generate an HEF file to infer on the Hailo-8.
1. Add a new YAML file of YOLOv4 (for example, check <code>hailo_model_zoo/cfg/networks/yolov4.yaml</code>) with the new input resolution, number of classes and so on.
2. (optional) Generate TFRECORD files of your data for evaluation/calibration. For more information check [**DATA.md**](DATA.md).
3. Run the full precision evaluation to reproduce the accuracy of the model:
```
python hailo_model_zoo/main.py eval yolov4 --ckpt yolov4.onnx --data-path val.tfrecord --yaml yolov4.yaml
```
4. Measure the quantized accuracy of your model:
```
python hailo_model_zoo/main.py eval yolov4 --target emulator --ckpt yolov4.onnx --calib-path calib.tfrecord --data-path val.tfrecord --yaml yolov4.yaml
```
5. Compile the model and generate the HEF file:
```
python hailo_model_zoo/main.py compile yolov4 --ckpt yolov4.onnx --calib-path calib.tfrecord --yaml yolov4.yaml
```
6. (optional) Measure the accuracy on the Hailo-8:
```
python hailo_model_zoo/main.py eval yolov4 --target hailo8 --hef yolov4.hef  --data-path val.tfrecord --yaml yolov4.yaml
```
</details>


<details>
    <summary>YOLOv5</summary>

## Training YOLOv5s
To train your YOLOv5s network follow these steps (full instructions can be found in [**here**](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)):
1. Clone the YOLOv5 repo:
```
git clone https://github.com/ultralytics/yolov5.git; cd yolov5; git checkout v2.0
```
2. Install (Python>=3.6.0 and Pytorch>=1.7):
```
pip install -r requirements.txt
```
3. Create dataset.yaml file (for example, <code>data/coco128.yaml</code>) which defines the path to your images/annotation files.

3. Add information about your data:
    * Create <code>dataset.yaml</code> (for example, <code>data/coco128.yaml</code>) which defines the path to your images/annotation files.
    * Generate a txt for each image containing the annotations in the format of <code>\<object-class> \<x_center> \<y_center> \<width> \<height></code>. For example: for img1.jpg create img1.txt containing:
```
1 0.716797 0.395833 0.216406 0.147222
0 0.687109 0.379167 0.255469 0.158333
1 0.420312 0.395833 0.140625 0.166667
```
4. Start training (pretrained weights can be found [**here**](https://github.com/ultralytics/yolov5/releases/download/v2.0/yolov5s.pt)):
```
python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt
```


## Export to ONNX
To export the trained YOLOv5 network to ONNX:
1. Install export requirements:
```
pip install -U coremltools>=4.1 onnx>=1.9.0 scikit-learn==0.19.2
```
2. Run the following script:
```
python export.py --weights yolov5s.pt --img 640 --batch 1  # export at 640x640 with batch size 1
```


## Add the Model to the Hailo Model Zoo
In this section we can use the ONNX model to generate an HEF file to infer on the Hailo-8.
1. Add a new YAML file of YOLOv5s (for example, check <code>hailo_model_zoo/cfg/networks/yolov5s.yaml</code>) with the new input resolution, number of classes and so on.
2. (optional) Generate TFRECORD files of your data for evaluation/calibration. For more information check [**DATA.md**](DATA.md).
3. Run the full precision evaluation to reproduce the accuracy of the model:
```
python hailo_model_zoo/main.py eval yolov5s --ckpt yolov5s.onnx --data-path val.tfrecord --yaml yolov5s.yaml
```
4. Measure the quantized accuracy of your model:
```
python hailo_model_zoo/main.py eval yolov5s --target emulator --ckpt yolov5s.onnx --calib-path calib.tfrecord --data-path val.tfrecord --yaml yolov5s.yaml
```
5. Compile the model and generate the HEF file:
```
python hailo_model_zoo/main.py compile yolov5s --ckpt yolov5s.onnx --calib-path calib.tfrecord --yaml yolov5s.yaml
```
6. (optional) Measure the accuracy on the Hailo-8:
```
python hailo_model_zoo/main.py eval yolov5s --target hailo8 --hef yolov5s.hef  --data-path val.tfrecord --yaml yolov5s.yaml
```

</details>

The compiled model can be used in the [**Hailo TAPPAS**](https://hailo.ai/developer-zone/tappas-apps-toolkit/) to generate a full application.