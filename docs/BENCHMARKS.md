# Benchmarks

All model zoo networks can be measured by the hailortcli tool in order to measure FPS, power and latency.  
For more information please refer to the **HailoRT documentation**.
<br><br>
## Example

In order to measure performance using hailortcli, you will first need to generate the corresponding Hailo Executable File (HEF) for the model.  It can be done easily by using the following hailo model zoo command:  
```
python hailo_model_zoo/main.py compile <model_name>
```
After building the HEF you will be able to measure the performance of the model.  
Here is an example for measureing the performance of resnet_v1_50:
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