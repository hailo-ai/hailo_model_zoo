import queue
from collections.abc import Callable
from contextlib import contextmanager

import tensorflow as tf
import torch
from tqdm import tqdm
from verboselogs import VerboseLogger

from hailo_model_optimization.flows.inference_flow import TorchInferenceModel
from hailo_model_optimization.saitama.tools.async_postprocessing import create_async_postprocessor
from hailo_model_optimization.saitama.tools.dataset_generator import DatasetGenerator
from hailo_model_optimization.saitama.tools.prefetch_dataloader import create_prefetch_dataloader
from hailo_model_optimization.saitama.utils.runner_builder import build_saitama_model_from_runner
from hailo_sdk_client.exposed_definitions import ContextInfo
from hailo_sdk_client.runner.client_runner import ClientRunner

from hailo_model_zoo.core.factory import INFER_FACTORY
from hailo_model_zoo.core.infer.infer_utils import aggregate, log_accuracy, visualize, write_results


@contextmanager
def cuda_config(device: torch.device):
    # Configure TF32 for Ampere+ GPUs for faster performance
    if device.type == "cuda":
        original_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("high")
    try:
        yield
    finally:
        if device.type == "cuda":
            torch.set_float32_matmul_precision(original_precision)


@INFER_FACTORY.register
def torch_infer(
    runner: ClientRunner,
    context: ContextInfo,
    logger: VerboseLogger,
    eval_num_examples: int | None,
    print_num_examples: int,
    batch_size: int,
    dataset: tf.data.Dataset,
    postprocessing_callback: Callable,
    eval_callback: Callable,
    visualize_callback: Callable | None = None,
    model_augmentation_callback: Callable | None = None,
    video_outpath: str | None = None,
    dump_results: bool = False,
    results_path: str | None = None,
):
    """
    Run inference using PyTorch/Saitama model.

    This function handles inference for Saitama models with proper tensor format conversion
    between TensorFlow dataset format (NHWC) and PyTorch format (NCHW).

    Args:
        runner: ClientRunner instance
        context: ContextInfo
        logger: Logger instance
        eval_num_examples: Number of examples to evaluate (None for all)
        print_num_examples: Print progress every N examples
        batch_size: Batch size for inference
        dataset: TensorFlow dataset providing preprocessed data
        postprocessing_callback: Callback for postprocessing outputs
        eval_callback: Callback to create evaluation metric
        visualize_callback: Optional callback for visualization
        model_augmentation_callback: Optional callback to augment model
        video_outpath: Optional path for video output
        dump_results: Whether to dump raw results
        results_path: Path to save results

    Returns:
        accuracy: Evaluation accuracy/metrics
    """
    eval_metric = eval_callback()
    if eval_num_examples:
        dataset = dataset.take(eval_num_examples)

    # Determine device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create prefetch dataloader with GPU-ready batches
    # This handles all TF→Numpy→Torch conversion in background thread
    # Automatically applies TF prefetch (AUTOTUNE) for I/O overlap
    dataloader = create_prefetch_dataloader(
        tf_dataset=dataset,
        batch_size=batch_size,
        device=device,
        buffer_size=2,  # 2 batches ahead for large images
    )
    # Create async postprocessor to overlap postprocessing with inference
    # Note: buffer_size=2 matches dataloader prefetch and absorbs postprocessing variance
    # This allows TWO batches to be queued, preventing inference stalls on slow postprocessing
    # Postprocess can be a bottleneck when processing large images, so we use an large buffer to absorb the variance.
    # Could cause memory issues if the buffer is too large.
    async_postproc = create_async_postprocessor(
        postprocessing_callback=postprocessing_callback,
        buffer_size=512,  # Buffer to absorb postprocessing variance.
        visualize_callback=visualize_callback,  # Determines if img cleanup is needed
    )

    logger.info("Running inference with prefetch dataloader...")

    # Initialize result containers
    num_of_images = 0
    logits = []
    gt = []

    with torch.no_grad(), cuda_config(device), context as ctx:
        # Get the torch/saitama model from runner
        model = build_saitama_model_from_runner(runner, ctx, device=device)
        # Apply model augmentation if provided
        model = model_augmentation_callback(model)
        # Set to evaluation mode and compile the model
        model.eval()
        model.compile(dynamic=False)

        try:
            with (
                tqdm(
                    total=None,
                    desc="Processed",
                    unit="images",
                    disable=None if not print_num_examples < 1 else True,
                ) as pbar,
            ):  # Disable gradient computation for inference
                for preprocessed_data, img_info in dataloader:
                    output_tensors = model(preprocessed_data)

                    # Ensure GPU inference is complete
                    if device.type == "cuda":
                        torch.cuda.synchronize()

                    # Convert to numpy in main thread (faster than cloning!)
                    # The ~6ms conversion is much cheaper than cloning large GPU tensors
                    # CRITICAL: This also protects against CUDA graph overwrites since
                    # numpy arrays are on CPU and torch.compile can't overwrite them
                    output_tensors_np = DatasetGenerator.torch_to_np(output_tensors)
                    # Submit numpy arrays for async postprocessing
                    async_postproc.submit(output_tensors_np, img_info)
                    # Retrieve completed results immediately to keep pipeline flowing
                    # This prevents result_queue from filling up and blocking the worker
                    while async_postproc.has_pending_results():
                        try:
                            # Non-blocking get with short timeout
                            logits_batch, img_info_np = async_postproc.get_results(block=False)
                        except queue.Empty:
                            # Result not ready yet, will get it later
                            break
                        logits.append(logits_batch)
                        gt.append(img_info_np)

                    # Get batch size from output for progress tracking
                    current_batch_size = TorchInferenceModel.get_batch_size(output_tensors)
                    num_of_images += current_batch_size
                    pbar.update(current_batch_size)

                # Collect all remaining postprocessed results (in order)
                while async_postproc.has_pending_results():
                    # Returns tuple (logits_batch, img_info_np) with process_img_info=True
                    logits_batch, img_info_np = async_postproc.get_results(block=True)
                    logits.append(logits_batch)
                    gt.append(img_info_np)

        except KeyboardInterrupt:
            pbar.close()
            logger.info("Inference interrupted by user, displaying partial results")
        finally:
            # Clean up the prefetch dataloader and async postprocessor
            dataloader.shutdown()
            async_postproc.shutdown()

    # Check if any data was processed
    if not gt or not logits:
        logger.warning("No data was processed during inference")
        return None

    # Aggregate results
    labels_keys = list(gt[0])
    labels = {k: aggregate([p[k] for p in gt]) for k in labels_keys}
    probs = {k: aggregate([p[k] for p in logits]) for k in logits[0]}

    accuracy = None
    if not visualize_callback and not dump_results:
        eval_metric.update_op(probs, labels)
        eval_metric.evaluate()
        accuracy = eval_metric.get_accuracy()
        log_accuracy(logger, num_of_images, accuracy)

    if dump_results:
        write_results(probs, labels, results_path)

    if visualize_callback:
        img_info_per_image = [x[1] for x in dataset]
        visualize(probs, img_info_per_image, visualize_callback, video_outpath)

    return accuracy
