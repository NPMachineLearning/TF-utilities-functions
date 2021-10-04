# Tensorboard callback
from tensorflow.keras.callbacks import TensorBoard

def create_tensorboard_callback(log_dir='./tensorboard', exp_name='my experiment'):
  '''
  Create a simple and quick tensorboard callback.

  Args:
   * `log_dir`: `str` tensorbord log directory
   * `exp_name`: `str` your experiment name

  Return:
    A tensorboard callback

  Reference:
    https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard 
  '''
  cb = TensorBoard(log_dir + '/' + exp_name)
  return cb

# ModelCheck point callback
from tensorflow.keras.callbacks import ModelCheckpoint

def create_checkpoint_callback(dir='./checkpoints', monitor='val_loss'):
  '''
  Create a simple and quick model check point callback. Save occur when 
  model perform the bset and only save model's weights.
  
  Args:
    * `dir`: `str` checkpoint directory
    * `monitor`: `str` the metric name to monitor
  
  Return:
    A ModelCheckpoint callback
  
  Reference:
    https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
  '''
  cb = ModelCheckpoint(dir,
                       monitor=monitor,
                       save_best_only=True,
                       save_weights_only=True)
  return cb

# Eearly stopping callback
from tensorflow.keras.callbacks import EarlyStopping

def create_early_stop_callback(num_without_change=3, monitor='val_loss'):
  '''
  Create a simple and quick early stopping callback.
  
  Args:
    * `num_without_change`: `number` number of time model's metrics(monitor)
    without changing
    * `monitor`: `str` the metric name to monitor
  
  Return:
   A EearlyStopping callback
  
  Reference:
    https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
  '''
  cb = EarlyStopping(patience=num_without_change,
                     monitor=monitor)
  return cb

# Plot history
import matplotlib.pyplot as plt

def plot_history(model_history, accuracy_monitor='accuracy', fig_size=(7, 5)):
  '''
  Plot an model's training and validation history
  
  Args:
    * `model_history`: `History` tensorflow model's training and validation history, 
    it came from after calling `fit` on model
    https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
    * `accuracy_monitor`: `str` the accuracy metrics the model was mointoring with
    * `fig_size`: `tuple` size of figure for plot. (width, height)
  '''
  # Get history
  history = model_history.history
  
  # Get accuracy history
  acc = history[accuracy_monitor]
  
  # Get loss history
  loss = history['loss']

  # Get validation accuracy history
  val_acc_monitor = 'val_' + accuracy_monitor
  val_acc = history[val_acc_monitor]

  # Get validation loss history
  val_loss = history['val_loss']

  plt.figure(figsize=fig_size)

  # Plot accuracy
  fig, ax_acc = plt.subplots()
  ax_acc.plot(acc, label=accuracy_monitor)
  ax_acc.plot(val_acc, label=val_acc_monitor)
  ax_acc.set_title(accuracy_monitor)
  ax_acc.set_xlabel('Epochs')
  ax_acc.set_ylabel(accuracy_monitor)
  ax_acc.legend(bbox_to_anchor=(1, 1));

  # Plot loss
  fig, ax_loss = plt.subplots()
  ax_loss.plot(loss, label='train_loss')
  ax_loss.plot(val_loss, label='val_loss')
  ax_loss.set_title('Loss')
  ax_loss.set_xlabel('Epochs')
  ax_loss.set_ylabel('Loss')
  ax_loss.legend(bbox_to_anchor=(1, 1));