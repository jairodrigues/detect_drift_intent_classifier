import tensorflow as tf

checkpoint_path = "./checkpoint"


class MyCustomCallback(tf.keras.callbacks.Callback):

    def __init__(self, model):
        ckpt = tf.train.Checkpoint(Dcnn=model)
        self.ckpt_manager = tf.train.CheckpointManager(
            ckpt,  checkpoint_path, max_to_keep=1)
        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print("Latest checkpoint restored!!")

    def on_epoch_end(self, epoch, logs=None):
        self.ckpt_manager.save()
        print("Checkpoint saved at {}.".format(checkpoint_path))
