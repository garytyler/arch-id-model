from pathlib import Path

import models
import tensorflow as tf
import training


def test(args):
    trainer = training.Trainer()

    def _get_run_checkpoints_dir():
        for f in training.CP_DIR.iterdir():
            if f.name.startswith("run-"):
                n = int(f.name.replace("run-", "")[0])
            else:
                n = int(f.name[0])
            if n == args.run_number:
                return Path(f)

    run_checkpoints_dir = _get_run_checkpoints_dir()
    if not run_checkpoints_dir:
        print(f"run '{args.run_number}' not found")
    else:
        latest_checkpoint_path = tf.train.latest_checkpoint(run_checkpoints_dir)
        model = tf.keras.models.load_model(f"{latest_checkpoint_path}.ckpt")
        cnn_app = models.CNN_APPS["InceptionResNetV2"]
        test_ds = trainer.get_dataset(
            split="test",
            image_size=cnn_app["image_size"],
            batch_size=cnn_app["batch_size"],
            preprocessor=cnn_app["preprocessor"],
        )
        loss, accuracy = model.evaluate(test_ds)
        print(f"Test loss: {loss}\nTest accuracy: {accuracy}")


def train(args):
    # Configure eager execution of tf.function calls
    tf.config.run_functions_eagerly(args.eager)
    # Start training
    trainer = training.Trainer()
    trainer.train(
        data_proportion=args.data_proportion,
        max_epochs=args.max_epochs,
        profile=args.profile,
    )
