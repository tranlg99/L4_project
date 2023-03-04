import torch
import torchvision.transforms as transforms
from model.custom_dataset_loader import TaiChiDataset, ToTensor, ShiftData, AugmentData, get_dataloader
from model.helper_functions import initialise_model, train_model
from model.checkpoints import load_checkpoint
import torch.optim as optim


# DEFINE PATH TO DATASET
TRAIN_DATA="../dataset/train_data"
VALID_DATA="../dataset/valid_data"

# DEFINE PATH TO CHECKPOINT
CHECKPOINT_FOLDER = "../checkpoints/"
CHECKPOINT = CHECKPOINT_FOLDER+"model_4_dropout_checkpoint_2023-03-03_0550.pth"
CONTINUE_TRAIN = True # Continue training by loading best train model weights, else best val model weights will be used

# DEFINE HYPERPARAMETERS
B = 16 # Batch size
E = 100 # Number of epochs
C = 1.0 # Gradient clip value
EVAL_FREQ = 10 # Evaluation frequency
LR = 0.06 # Learning rate
M = 0.9 # Momentum


# LOAD DATASETS and CREATE DATALOADERS
train_dataset = TaiChiDataset(log_file=TRAIN_DATA+'/sample_ids.txt',
                        root_dir=TRAIN_DATA,
                        check=True,
                        transform=transforms.Compose([AugmentData(),
                                                      ShiftData(),
                                                      ToTensor()]))

valid_dataset = TaiChiDataset(log_file=VALID_DATA+'/sample_ids.txt',
                        root_dir=VALID_DATA,
                        check=True,
                        transform=transforms.Compose([ToTensor()]))

train_dataloader = get_dataloader(train_dataset, batch_size=B, shuffle=True)
valid_dataloader = get_dataloader(valid_dataset, batch_size=B, shuffle=True)
dataloaders_dict = {'train': train_dataloader, 'val': valid_dataloader}
print("Train dataset: {}, Validation dataset: {}".format(len(train_dataset), len(valid_dataset)))

# INITIALISE MODEL
device = torch.device('cuda')
my_model = initialise_model(device, to_learn=['all'], drop_out=1)
params_to_update = my_model.parameters()

# LOAD MODEL WEIGHTS FROM CHECKPOINT
if CHECKPOINT:
    model_weights, best_valid_weights, train_past_loss, validation_past_loss, optimizer_state, train_time = load_checkpoint(CHECKPOINT)
    if CONTINUE_TRAIN:
        my_model.load_state_dict(model_weights)
    else:
        my_model.load_state_dict(best_valid_weights)
else:
    best_valid_weights = []
    train_past_loss = []
    validation_past_loss = []


# OPTIMIZER
# TODO: Load optimizer state?
my_optimizer = optim.SGD(params_to_update, lr=LR, momentum=M)

# RUN TRAINING
# saving checkpoints during training, at every evaluation
new_model, best_weights, model_t_loss, model_v_loss, model_train_time = train_model(
    device,
    my_model,
    dataloaders_dict,
    my_optimizer,
    train_past_loss,
    validation_past_loss,
    best_weights = best_valid_weights,
    num_epochs = E,
    verbose = True,
    clip = C,
    eval_freq=EVAL_FREQ,
    description="model_4_dropout",
    path=CHECKPOINT_FOLDER
)