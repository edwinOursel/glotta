import os
import tensorflow as tf

CHECKPOINT_EPOCH = 50
EPOCHS = 60


def fit(model, dataset):
    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    if CHECKPOINT_EPOCH > 0:
        model.load_weights(os.path.join(checkpoint_dir, "ckpt_" + str(CHECKPOINT_EPOCH)))

    if CHECKPOINT_EPOCH < EPOCHS:
        history = model.fit(dataset, epochs=EPOCHS - CHECKPOINT_EPOCH, callbacks=[checkpoint_callback])
