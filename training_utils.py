"""Training utilities for neural network models."""

import os
import pickle
from tqdm import tqdm
from flax import nnx


def save_checkpoint(model, optimizer_state, loss, checkpoint_path):
    """Save model and optimizer state to checkpoint."""
    checkpoint = {
        'model_state': nnx.state(model),
        'optimizer_state': nnx.state(optimizer_state),
        'loss': loss
    }
    checkpoint_dir = os.path.dirname(checkpoint_path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # If this is a best model checkpoint, delete previous best model checkpoints
    if 'best_model_epoch_' in checkpoint_path:
        # Find and delete previous best model files
        for filename in os.listdir(checkpoint_dir):
            if filename.startswith('best_model_epoch_') and filename.endswith('.pkl'):
                old_checkpoint = os.path.join(checkpoint_dir, filename)
                if old_checkpoint != checkpoint_path:
                    try:
                        os.remove(old_checkpoint)
                        print(f'Deleted old best checkpoint: {filename}')
                    except OSError:
                        pass
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f'Checkpoint saved to {checkpoint_path} (loss: {loss:.4f})')


def load_checkpoint(model, optimizer_state, checkpoint_path):
    """Load model and optimizer state from checkpoint."""
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    nnx.update(model, checkpoint['model_state'])
    nnx.update(optimizer_state, checkpoint['optimizer_state'])
    print(f'Checkpoint loaded from {checkpoint_path}')
    return checkpoint['loss']


def train_model(model, optimizer, loss_fn, checkpoint_dir, num_epochs=1000, checkpoint_interval=50, start_epoch=0):
    """
    Train a neural network model with checkpointing.
    
    Parameters
    ----------
    model : nnx.Module
        The neural network model to train
    optimizer : nnx.ModelAndOptimizer
        The optimizer for the model
    loss_fn : callable
        Loss function that takes the model as input and returns a scalar loss
    checkpoint_dir : str
        Directory to save checkpoints
    num_epochs : int
        Number of training epochs
    checkpoint_interval : int
        Interval (in epochs) for saving periodic checkpoints
    start_epoch : int
        Epoch to start training from.
        If 0, starts from scratch.
        If -1, resumes from the latest checkpoint.
        If > 0, resumes from that specific epoch checkpoint.
    
    Returns
    -------
    best_loss : float
        The best loss achieved during training
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_loss = float('inf')
    best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model_epoch_{:04d}.pkl')
    latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_model_epoch_{:04d}.pkl')
    
    # Handle resuming from a checkpoint
    current_epoch_start = 0
    if start_epoch == -1:
        # Resume from latest checkpoint
        latest_checkpoint, latest_epoch = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint is not None:
            print(f'Found latest checkpoint at epoch {latest_epoch}. Resuming training...')
            best_loss = load_checkpoint(model, optimizer, latest_checkpoint)
            current_epoch_start = latest_epoch + 1
        else:
            print('No latest checkpoint found. Starting from scratch...')
            current_epoch_start = 0
    elif start_epoch > 0:
        # Resume from specific epoch
        checkpoint_file = best_checkpoint_path.format(start_epoch)
        if os.path.exists(checkpoint_file):
            print(f'Resuming from epoch {start_epoch}...')
            best_loss = load_checkpoint(model, optimizer, checkpoint_file)
            current_epoch_start = start_epoch
        else:
            raise FileNotFoundError(f'Checkpoint for epoch {start_epoch} not found at {checkpoint_file}')
    else:
        # start_epoch == 0: Start from scratch
        current_epoch_start = 0
    
    for epoch in tqdm(range(current_epoch_start, num_epochs), desc="Training", initial=current_epoch_start, total=num_epochs):
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        tqdm.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')
        
        # Save best model
        if loss < best_loss:
            best_loss = loss
            save_checkpoint(model, optimizer, loss, best_checkpoint_path.format(epoch + 1))
        
        # Save checkpoint periodically
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, loss, latest_checkpoint_path.format(epoch + 1))
    
    return best_loss


def find_best_checkpoint(checkpoint_dir):
    """
    Find the best checkpoint file in a directory based on filename pattern.
    
    Returns
    -------
    checkpoint_path : str or None
        Path to the best checkpoint file, or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    best_checkpoint = None
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith('best_model_epoch_'):
            if best_checkpoint is None or filename > best_checkpoint:
                best_checkpoint = filename
    
    if best_checkpoint is not None:
        return os.path.join(checkpoint_dir, best_checkpoint)
    return None


def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint file (by epoch number) in a directory.
    Searches through both latest_model_epoch_* and best_model_epoch_* files.
    
    Returns
    -------
    checkpoint_path : str or None
        Path to the latest checkpoint file, or None if no checkpoints found
    epoch : int or None
        The epoch number of the latest checkpoint
    """
    if not os.path.exists(checkpoint_dir):
        return None, None
    
    latest_checkpoint = None
    latest_epoch = -1
    
    for filename in os.listdir(checkpoint_dir):
        # Check both latest and best model checkpoints
        if (filename.startswith('latest_model_epoch_') or filename.startswith('best_model_epoch_')) and filename.endswith('.pkl'):
            # Extract epoch number from filename
            try:
                # Remove prefix and .pkl extension
                if filename.startswith('latest_model_epoch_'):
                    epoch_str = filename.replace('latest_model_epoch_', '').replace('.pkl', '')
                elif filename.startswith('best_model_epoch_'):  # best_model_epoch_
                    epoch_str = filename.replace('best_model_epoch_', '').replace('.pkl', '')
                else:
                    raise ValueError('Filename does not match expected patterns')
                epoch = int(epoch_str)
                if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_checkpoint = filename
            except ValueError:
                continue
    
    if latest_checkpoint is not None:
        return os.path.join(checkpoint_dir, latest_checkpoint), latest_epoch
    return None, None
