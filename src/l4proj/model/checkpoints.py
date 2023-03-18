from datetime import datetime, date
import numpy as np
import torch

def load_checkpoint(filename, device='cuda'):
  """
  Loading a checkpoint
  """
  try:
    filepath = "../checkpoints/"+filename
    
    checkpoint = torch.load(filepath, map_location=device)

    t_history = checkpoint['train_loss_history']
    v_history = checkpoint['val_loss_history']
    train_time = checkpoint['train_time']

    t_losses = np.array([x[1] for x in t_history])
    v_losses = np.array([x[1] for x in v_history])

   
    t_min_arg = np.argmin(t_losses)
    v_min_arg = np.argmin(v_losses)

    print('-' * 50)
    print("Checkpoint data:")
    print("Best train loss: {:.3f} at epoch {}/{}".format(t_history[t_min_arg][1], t_history[t_min_arg][0], len(t_history)-1))
    print("Best valid loss: {:.3f} at epoch {}/{}".format(v_history[v_min_arg][1], v_history[v_min_arg][0], len(t_history)-1))
    # print("Total training time: {:.0f}h {:.0f}m".format(train_time // 3600, train_time // 60))
    print('-' * 50)

    model_weights = checkpoint['model_weights']
    best_model_weights = checkpoint['best_weights']
    optimizer_state = checkpoint['optimizer']

    return model_weights, best_model_weights, t_history, v_history, optimizer_state, train_time
  except Exception as e:
    print("ERROR could not load checkpoint:", e)

def save_checkpoint(model, best_weights, train_loss, valid_loss, optimizer, batch_size, train_time, description="", path=""):
    """
    Saving a checkpoint
    """
    try:
        checkpoint = {
            'description': description,
            'train_loss_history': train_loss,
            'val_loss_history': valid_loss,
            'batch_size': batch_size,
            'optimizer': optimizer.state_dict(),
            'model_weights': model.state_dict(),
            'best_weights': best_weights,
            'train_time': train_time
            }
  
        TODAY = date.today()
        TIME = datetime.now().strftime("%H%M")
        FILENAME = 'checkpoint_'+str(TODAY)+'_'+str(TIME)+'.pth'
        if description:
          FILENAME = description+'_checkpoint_'+str(TODAY)+'_'+str(TIME)+'.pth'
        torch.save(checkpoint, path+FILENAME)
        print("Saving", FILENAME)
    except Exception as e:
        print("ERROR could not save checkpoint:", e)
