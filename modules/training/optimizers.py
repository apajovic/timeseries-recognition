from tensorflow import keras

class LearningRateOptimizerFactory:
    @staticmethod
    def get(optimizer_type, **kwargs):
        if optimizer_type == "cosine":
            steps = kwargs.get("steps", 100)
            return keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate = kwargs.get("initial_learning_rate", 1e-4),
                    warmup_target = kwargs.get("warmup_target", 1e-3),
                    warmup_steps = kwargs.get("warmup_steps", int(steps * 0.1)),
                    alpha = kwargs.get("alpha", 1e-4),
                    decay_steps = kwargs.get("decay_steps", steps * 0.7)
                )
        elif optimizer_type == "stair":
            return keras.optimizers.schedules.InverseTimeDecay(
                    initial_learning_rate = kwargs.get("initial_learning_rate", 1e-3),
                    decay_steps = kwargs.get("decay_steps", 200),
                    decay_rate = kwargs.get("decay_rate", 0.3),
                    staircase = kwargs.get("staircase", True)
                    )
        else:
            raise ValueError("Invalide optimizer_type provided!")