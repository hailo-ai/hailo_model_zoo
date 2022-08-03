# Getting Started

This document provides install instructions and basic usage examples of the Hailo Model Zoo.

<br>

## System Requirements

- Ubuntu 18.04 and Python 3.6 or Ubuntu 20.04 and Python 3.8
  Note: Ubuntu 18.04 will be deprecated in Hailo Model Zoo future version
  Note: Python 3.6 will be deprecated in Hailo Model Zoo future version


- Hailo Dataflow Compiler v3.18.0 (Obtain from [**hailo.ai**](http://hailo.ai))
- HailoRT 4.8.0 (Obtain from [**hailo.ai**](http://hailo.ai)) - required only for inference on Hailo-8.
- The Hailo Model Zoo supports Hailo-8 connected via PCIe only.

<br>

## Install Instructions

### Hailo Software Suite

The [**Hailo Software Suite**](https://hailo.ai/developer-zone/sw-downloads/) includes all of Hailo's SW components and insures compatibility across products versions. The Hailo Model Zoo is already installed and ready to be used within the virtualenv of it.

### Manual Installation

1. Install the Hailo Dataflow compiler and enter the virtualenv (visit [**hailo.ai**](http://hailo.ai) for further instructions).
2. Install the HailoRT - required only for inference on Hailo-8 (visit [**hailo.ai**](http://hailo.ai) for further instructions).
3. Clone the Hailo Model Zoo repo:
```
git clone https://github.com/hailo-ai/hailo_model_zoo.git
```
4. Run the setup script:
```
cd hailo_model_zoo; pip install -e .
```
5. For setting up datasets please see [**DATA.md**](DATA.md).
6. Verify Hailo-8 is connected through PCIe (required only to run on Hailo-8. Full-precision / emulation run on GPU.)
```
hailo fw-control identify
```
Expected output:
```
(hailo) Running command 'fw-control' with 'hailortcli'
Identifying board
Control Protocol Version: 2
Firmware Version: 4.6.0 (release,app)
Logger Version: 0
Board Name: Hailo-8
Device Architecture: HAILO8_B0
Serial Number: HLUTM20204900071
Part Number: HM218B1C2FA
Product Name: HAILO-8 AI ACCELERATOR M.2 MODULE
```

### Upgrade Instructions

If you want to upgrade to a specific Hailo Model Zoo version within a suite or on top of a previous installation not in the suite.
1. Pull the specific repo branch:
```
git clone -b v2.2 https://github.com/hailo-ai/hailo_model_zoo.git
```
2. Run the setup script:
```
cd hailo_model_zoo; pip install -e .
```

<br>

## Usage

### Flow Diagram

The following scheme shows high-level view of the model-zoo evaluation process, and the different stages in between.

<p align="center">
  <img src="images/usage_flow.svg" />
</p>

By default, each stage executes all of its previously necessary stages according to the above diagram. The post-parsing stages also have an option to start from the product of previous stages (i.e., the Hailo Archive (HAR) file), as explained below. The operations are configured through a YAML file that exist for each model in the cfg folder. For a description of the YAML structure please see [**YAML.md**](YAML.md).
### Parsing

The pre-trained models are stored on AWS S3 and will be downloaded automatically when running the model zoo into your data directory. To parse models into Hailo's internal representation and generate the Hailo Archive (HAR) file:
```
hailomz parse <model_name>
```

### Profiling

To generate the Hailo profiler report:
```
hailomz profile <model_name>
```
To generate the Hailo profiler report using a previously generated HAR file:
```
hailomz profile <model_name> --har /path/to/model.har
```
\* The report contains information about your model and expected performance on the Hailo hardware.

### Optimize

To optimize models, convert them from full precision into integer representation and generate a quantized Hailo Archive (HAR) file:
```
hailomz optimize <model_name>
```
To optimize the model starting from a previously generated HAR file:
```
hailomz optimize <model_name> --har /path/to/model.har
```
You can use your own images by giving a directory path to the optimization process, with the following supported formats (.jpg,.jpeg,.png):
```
hailomz optimize <model_name> --calib-path /path/to/calibration/imgs/dir/
```
\* This step requires data for calibration. For additional information please see [**OPTIMIZATION.md**](OPTIMIZATION.md).

### Compile

To run the Hailo compiler and generate the Hailo Executable Format (HEF) file:
```
hailomz compile <model_name>
```
To generate the HEF starting from a previously generated HAR file:
```
hailomz compile <model_name> --har /path/to/model.har
```
### Evaluation

To evaluate models in full precision:
```
hailomz eval <model_name>
```
To evaluate models starting from a previously generated Hailo Archive (HAR) file:
```
hailomz eval <model_name> --har /path/to/model.har
```
To evaluate models with the Hailo emulator (after quantization to integer representation - fast_numeric):
```
hailomz eval <model_name> --target emulator
```
To evaluate models on Hailo-8:
```
hailomz eval <model_name> --target hailo8
```
If multiple Hailo-8 devices are available, it's possible to select a specific one
```
# Device id looks something like 0000:41:00.0
hailomz eval <model_name> --target <device_id>
# This command can be used to list available devices
hailomz eval --help
```
To limit the number of images for evaluation use the following flag:
```
hailomz eval <model_name> --eval-num <num-images>
```
To explore other options (for example: changing the default batch-size) use:
```
hailomz eval --help
```

### Visualization

To run visualization (without evaluation) and generate the output images:
```
hailomz eval <model_name> --visualize
```
To create a video file from the network predictions:
```
hailomz eval <model_name> --visualize --video-outpath /path/to/video_output.mp4
```

### Info

You can easily print information of any network exists in the model zoo, to get a sense of its input/output shape, parameters, operations, framework etc.

To print a model-zoo network information:
```
hailomz info <model_name>
```

Here is an example for printing information about mobilenet_v1:
```
hailomz info mobilenet_v1
```
Expected output:
```
<Hailo Model Zoo Info> Printing mobilenet_v1 Information
<Hailo Model Zoo Info>
        task:                    classification
        input_shape:             224x224x3
        output_shape:            1x1x1001
        operations:              0.57G
        parameters:              4.22M
        framework:               tensorflow
        training_data:           imagenet train
        validation_data:         imagenet val
        eval_metric:             Accuracy (top1)
        full_precision_result:   71.02
        source:                  https://github.com/tensorflow/models/tree/v1.13.0/research/slim
        license_url:             https://github.com/tensorflow/models/blob/v1.13.0/LICENSE

```

### Compile multiple networks together
We can use multiple disjoint models in the same binary.
This is useful for running several small models on the device.
```
python hailo_model_zoo/multi_main.py <config_name>
```

### TFRecord to NPY conversion
In some situations you might want to convert the tfrecord file to npy file (for example, when explicitly using the Dataflow Compiler for quantization). In order to do so, run the command:
```
python hailo_model_zoo/tools/conversion_tool.py /path/to/tfrecord_file resnet_v1_50 --npy
```
