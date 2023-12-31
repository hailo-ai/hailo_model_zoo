import tensorflow as tf


def vae(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information
    """
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'prompt': tf.io.FixedLenFeature([], tf.string),
            'unet_latent_input': tf.io.FixedLenFeature([], tf.string),
            'unet_prompt_embeds_input': tf.io.VarLenFeature(tf.string),
            'unet_t_emb_input': tf.io.VarLenFeature(tf.string),
            'unet_latent_output': tf.io.VarLenFeature(tf.string),
            'image_float': tf.io.FixedLenFeature([], tf.string),
            'vae_input': tf.io.FixedLenFeature([], tf.string),
        })

    prompt = tf.cast(features['prompt'], tf.string)
    image_float = tf.reshape(tf.io.decode_raw(features['image_float'], tf.float32), [1, 512, 512, 3])
    vae_input = tf.reshape(tf.io.decode_raw(features['vae_input'], tf.float32), [1, 64, 64, 4])

    image_info = {'prompt': prompt,
                  'img_orig': image_float,
                  'img_float': image_float}
    return [vae_input, image_info]


def unet(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information
    """
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'prompt': tf.io.FixedLenFeature([], tf.string),
            'unet_latent_input': tf.io.FixedLenFeature([], tf.string),
            'unet_prompt_embeds_input': tf.io.FixedLenFeature([], tf.string),
            'unet_t_emb_input': tf.io.FixedLenFeature([], tf.string),
            'unet_latent_output': tf.io.FixedLenFeature([], tf.string),
            'image_float': tf.io.FixedLenFeature([], tf.string),
            'vae_input': tf.io.FixedLenFeature([], tf.string),
        })

    prompt = tf.cast(features['prompt'], tf.string)
    image_float = tf.reshape(tf.io.decode_raw(features['image_float'], tf.float32), [1, 512, 512, 3])
    vae_input = tf.reshape(tf.io.decode_raw(features['vae_input'], tf.float32), [1, 64, 64, 4])

    unet_latent_input = tf.reshape(tf.io.decode_raw(features['unet_latent_input'], tf.float32), [20, 2, 64, 64, 4])
    unet_prompt_embeds_input = tf.reshape(
        tf.io.decode_raw(features['unet_prompt_embeds_input'], tf.float32),
        [2, 1, 77, 1024]
    )

    unet_t_emb_input = tf.reshape(tf.io.decode_raw(features['unet_t_emb_input'], tf.float32), [20, 2, 1, 1, 320])

    image = {'sd2_unet/input_layer1': unet_latent_input[0], 'sd2_unet/input_layer2': unet_prompt_embeds_input,
             'sd2_unet/input_layer3': unet_t_emb_input}

    image_info = {'prompt': prompt,
                  'img_orig': image_float,
                  'img_float': image_float,
                  'vae_gt_input': vae_input}
    return [image, image_info]
