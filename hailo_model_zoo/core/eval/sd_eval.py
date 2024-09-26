from hailo_model_zoo.core.eval.low_light_enhancement_evaluation import LowLightEnhancementEval
from hailo_model_zoo.core.factory import EVAL_FACTORY

# Intentionally registering in a separate file so this won't get released
EVAL_FACTORY.register(LowLightEnhancementEval, name="stable_diffusion_v1.5_decoder")
EVAL_FACTORY.register(LowLightEnhancementEval, name="stable_diffusion_v1.5_encoder")
