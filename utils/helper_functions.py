### Python files containing helpful functions 

import tensorflow as tf
import zipfile
import os
import datetime

def unzip_data(filename):
	"""
	Unzip filename into the current directory 
    
    Arguments:
    - filename (str): a filepath to a target zip folder to be unzipped
	"""
    
    zip_ref = zipfile.ZipFile(filename,"r")
    zip_ref = zip_ref.extractall()
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
    

