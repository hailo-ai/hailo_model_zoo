import tensorflow as tf
from tqdm import tqdm
import numpy as np
import onnxruntime
from hailo_model_zoo.core.infer.infer_utils import log_accuracy, write_results, aggregate, visualize, to_numpy
from hailo_model_zoo.utils.path_resolver import resolve_data_path


ITERATIONS = 20
TIMESTEPS = np.linspace(999, 0, ITERATIONS, dtype=np.float32).reshape(-1, 1)
GUIDANCE_SCALE = 7.5
SIGMAS = np.array([14.614647, 10.74685, 8.081536, 6.204952, 4.855665,
                   3.8653843, 3.123769, 2.5571685, 2.1156619, 1.7648253,
                   1.480583, 1.2458155, 1.0481429, 0.87842995, 0.7297196,
                   0.59643507, 0.47358626, 0.35554704, 0.23216821, 0.02916753,
                   0.], dtype=np.float32)


VAE_CONFIG_SCALING_FACTOR = 0.18215
VAE_ONNX_PATH = resolve_data_path('models_files/stable_diffusion_v2/vae/stable_diffusion_2_base_vae_decoder.sim.onnx')


def scheduler_step(model_output, t, sample, step_index, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0):

    sigma = SIGMAS[step_index]
    gamma = min(s_churn / (len(SIGMAS) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0
    noise = np.random.normal(0, 1, (model_output.shape)).astype(np.float16)

    eps = noise * s_noise
    sigma_hat = sigma * (gamma + 1)

    if gamma > 0:
        sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

    # 1.
    pred_original_sample = sample - sigma_hat * model_output

    # 2. Convert to an ODE derivative
    derivative = (sample - pred_original_sample) / sigma_hat

    dt = SIGMAS[step_index + 1] - sigma_hat

    prev_sample = sample + derivative * dt

    return prev_sample


def unet_infer(runner, context, logger, eval_num_examples, print_num_examples,
               batch_size, dataset, postprocessing_callback,
               eval_callback, visualize_callback, model_augmentation_callback,
               video_outpath, dump_results, results_path, *, np_infer=False):
    eval_metric = eval_callback()
    postprocessing_callback = tf.function(postprocessing_callback, reduce_retracing=True)
    if eval_num_examples:
        dataset = dataset.take(eval_num_examples)
    batched_dataset = dataset.batch(batch_size)
    logger.info('Running inference...')
    with context as ctx, tqdm(total=None, desc="Processed", unit="images",
                              disable=None if not print_num_examples < 1 else True) as pbar:
        model = runner.get_keras_model(ctx)
        model = model_augmentation_callback(model)

        @tf.function()
        def predict_function(data):
            return model(data, training=False)

        num_of_images = 0
        logits = []
        gt = []
        for preprocessed_data, img_info in batched_dataset:

            # unet stable diffusion flow
            # dict_keys(['sd2_unet/input_layer1', 'sd2_unet/input_layer2', 'sd2_unet/input_layer3'])
            # TensorShape([1, 2, 64, 64, 4]), TensorShape([1, 2, 1, 77, 1024]), TensorShape([1, 20, 2, 1, 1, 320])

            latent_model_input = preprocessed_data['sd2_unet/input_layer1'][0]
            latents = latent_model_input[0:1].numpy() * np.sqrt(SIGMAS[0]**2 + 1)
            for i, t in enumerate(TIMESTEPS):

                unet_input = {'sd2_unet/input_layer1': latent_model_input,
                              'sd2_unet/input_layer2': preprocessed_data['sd2_unet/input_layer2'][0],
                              'sd2_unet/input_layer3': preprocessed_data['sd2_unet/input_layer3'][0, i]}

                unet_output = predict_function(unet_input)

                noise_pred_uncond, noise_pred_text = np.split(unet_output.numpy(), 2)   # [1, 4, 64, 46], [1, 4, 64, 64]
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)
                latents = scheduler_step(noise_pred, t, latents, i)

                latent_model_input = np.concatenate([latents, latents])
                sigma = SIGMAS[i + 1]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            session_vae = onnxruntime.InferenceSession(VAE_ONNX_PATH.as_posix())
            output_tensors = session_vae.run(
                None, {'latent_sample': latents.transpose(0, 3, 1, 2) / VAE_CONFIG_SCALING_FACTOR}
            )[0]

            output_tensors = tf.clip_by_value(output_tensors / 2 + 0.5, 0.0, 1.0)
            logits_batch = postprocessing_callback(output_tensors, gt_images=img_info)
            current_batch_size = (output_tensors[0].shape[0] if isinstance(output_tensors, list)
                                  else output_tensors.shape[0])
            num_of_images += current_batch_size
            pbar.update(current_batch_size)
            logits.append(logits_batch)
            if not visualize_callback:
                if "img_orig" in img_info:
                    del img_info["img_orig"]
                if "img_resized" in img_info:
                    del img_info["img_resized"]
            gt.append(to_numpy(img_info))
    labels_keys = list(gt[0].keys())
    labels = {k: aggregate([p[k] for p in gt]) for k in labels_keys}
    probs = {k: aggregate([p[k] for p in logits]) for k in logits[0].keys()}
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
