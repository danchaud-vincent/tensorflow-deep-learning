### Python files containing helpful functions 
import tensorflow as tf
import zipfile
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def unzip_data(filename):
    """
    Unzip filename into the current directory 
    
    Arguments:
    - filename (str): a filepath to a target zip folder to be unzipped
    """
    zip_ref = zipfile.ZipFile(filename,"r")
    zip_ref.extractall()
    zip_ref.close()
 

def walk_through_dir(dir_path):
    """
    Walk through a directory and list the number of files and returning its contents.
    
    Arguments:
    - dirname (str): target directory
    """
    
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"{dirpath}: There are {len(dirnames)} directories and {len(filenames)} files")
    

def create_tensorboard_callback(dir_name, experiment_name):
    """
    Creates a TensorBoard callback instand to store log files.

    Arguments:
    - dir_name (str): target directory to stores TensorBoard log files
    - experiment_name (str): name of the experiment directory (ex: efficientNet_model1)
    
    Returns:
    - tensorboard_callback
    """
    time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    
    # path of the directory where to save the log files to be parsed by TensorBoard
    log_dir = f"{dir_name}/{experiment_name}/{time_now}"
    
    # tensorboard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    
    print(f"Saving TensorBoard log files to :{log_dir}")
    
    return tensorboard_callback
    

def plot_loss_curves(history):

    """
    Returns separate loss curves for training and validation metrics
    
    Arguments:
    - history : TensorFlow model history object
    """
    
    # Loss
    loss_train = history.history["loss"]
    loss_validation = history.history["val_loss"]
    
    # accuracy
    acc_train = history.history["accuracy"]
    acc_validation = history.history["val_accuracy"]
    
    # figure
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,7))
    
    # accuracy axe
    axs[0].plot(acc_train, label="train accuracy")
    axs[0].plot(acc_validation, label="validation accuracy")
    axs[0].set_title("Accuracy")
    axs[0].set_xlabel("Epochs")
    axs[0].legend()
    
    # loss axe
    axs[1].plot(loss_train, label="train loss")
    axs[1].plot(loss_validation, label="validation loss")
    axs[1].set_title("Loss")
    axs[1].set_xlabel("Epochs")
    axs[1].legend()
    
    plt.show()
    
def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compares two TensorFlow model History objects.
    
    Args:
      original_history: History object from original model (before new_history)
      new_history: History object from continued model training (after original_history)
      initial_epochs: Number of epochs in original_history (new_history plot starts from here) 
    """
    
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
