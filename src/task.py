from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import data
import model


def parse_flags(args_parser):
    """Returns a dictionary with the command line arguments."""

    args_parser.add_argument(
        "--train-files",
        help="GCS or local paths to training data",
        nargs="+",
        required=True,
    )
    args_parser.add_argument(
        "--eval-files",
        help="GCS or local paths to evaluation data",
        nargs="+",
        required=True,
    )
    args_parser.add_argument(
        "--verbosity",
        choices=["DEBUG", "ERROR", "FATAL", "INFO", "WARN"],
        default="INFO",
    )

    return args_parser.parse_args()


def main():
    tf.logging.set_verbosity(FLAGS.verbosity)
    regressor = model.get_regressor()
    train_spec = tf.estimator.TrainSpec(
        input_fn=data.get_input_fn(
            FLAGS.train_files, shuffle=True, repeat=1000, batch_size=FLAGS.batch_size
        ),
        max_steps=FLAGS.max_steps,
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=data.get_input_fn(FLAGS.eval_files),
        steps=None,
        start_delay_secs=10,
        throttle_secs=FLAGS.throttle_secs,
    )
    tf.estimator.train_and_evaluate(regressor, train_spec, eval_spec)


_args_parser = argparse.ArgumentParser()
FLAGS = parse_flags(_args_parser)

if __name__ == "__main__":
    # The FLAGS global variable is intialized by this point.
    main()
