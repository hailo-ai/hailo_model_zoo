# Benchmarks

In order to measure FPS, power and latency of the Hailo Model Zoo networks you can use the HailoRT command line interface.
For more information please refer to the HailoRT documentation in [**hailo.ai**](https://hailo.ai/).

<br>

## Example

The HailoRT command line interface works with the Hailo Executable File (HEF) of the model. To generate the HEF file use the following command:
```
python hailo_model_zoo/main.py compile <model_name>
```
After building the HEF you will be able to measure the performance of the model by using the HailoRT command line interface. Example for measuring performance of resnet_v1_50:
```
hailortcli benchmark resnet_v1_50.hef
```

Example output: 
```
=======
Summary
=======
FPS     (hw_only)                 = 1328.83
        (streaming)               = 1328.8
Latency (hw)                      = 2.93646 ms
Power in streaming mode (average) = 3.19395 W
                        (max)     = 3.20456 W

```