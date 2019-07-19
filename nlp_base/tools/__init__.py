import yaml

def get_tensorflow_conf(tf):
  with open('../../conf/settings.yml', 'r', encoding='utf8') as fin:
    conf = yaml.load(fin, Loader=yaml.FullLoader)

  with open('../../conf/settings.yml', 'r', encoding='utf8') as fin:
    conf = yaml.load(fin, Loader=yaml.FullLoader)

  # Data loading params
  tf.app.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
  tf.app.flags.DEFINE_string("train_file", conf['train_file'], "Train file source.")
  tf.app.flags.DEFINE_string("target_file", conf['target_file'], "Train file source.")

  tf.app.flags.DEFINE_integer("num_tag", 5, "Train file source.")

  # Model Hyperparameters
  tf.app.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
  tf.app.flags.DEFINE_integer("rnn_units", 128, "Number of filters per filter size (default: 128)")
  tf.app.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 0.5)")
  tf.app.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
  tf.app.flags.DEFINE_float("lr", 0.0001, "Learning rate")

  # Training parameters
  tf.app.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
  tf.app.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
  tf.app.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
  tf.app.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
  tf.app.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

  # Misc Parameters
  tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
  tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

  tf.app.flags.DEFINE_string("checkpoint_path", conf['checkpoint_path'], "Model checkpoint path")

  FLAGS = tf.app.flags.FLAGS

  return FLAGS
