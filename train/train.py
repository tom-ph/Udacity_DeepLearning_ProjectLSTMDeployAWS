import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data

from model import LSTMClassifier

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(model_info['embedding_dim'], model_info['hidden_dim'], model_info['vocab_size'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Load the saved word_dict.
    word_dict_path = os.path.join(model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'rb') as f:
        model.word_dict = pickle.load(f)

    model.to(device).eval()

    print("Done loading model.")
    return model

def _get_train_data_loader(batch_size, training_dir):
    print("Get train data loader.")

    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
    train_X = torch.from_numpy(train_data.drop([0], axis=1).values).long()

    train_ds = torch.utils.data.TensorDataset(train_X, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)

def _get_valid_data_loader(batch_size, training_dir):
    print("Get validation data loader.")

    valid_data = pd.read_csv(os.path.join(training_dir, "validation.csv"), header=None, names=None)

    valid_y = torch.from_numpy(valid_data[[0]].values).float().squeeze()
    valid_X = torch.from_numpy(valid_data.drop([0], axis=1).values).long()

    valid_ds = torch.utils.data.TensorDataset(valid_X, valid_y)

    return torch.utils.data.DataLoader(valid_ds, batch_size=batch_size)

def train(model, train_loader, valid_loader, epochs, early_stopping_rounds, optimizer, loss_fn, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    valid_loader - The PyTorch DataLoader that should be used for validation.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    loss_fn      - The loss function used for training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    
    # TODO: Paste the train() method developed in the notebook here.
    batch_size = next(iter(train_loader))[0].shape[0]
    #TF: for early stopping
    bad_rounds=0
    last_valid_loss=np.Inf
    #TF: for checkpoint
    best_valid_loss=np.Inf
    checkpoint_valid_acc=0
    model_checkpoint=model
    
    for epoch in range(1, epochs + 1):
        if bad_rounds == early_stopping_rounds:
            break
        sys.stdout.write("Starting Epoch {}\n".format(epoch))
        model.train()
        total_loss = 0
        for batch in train_loader:         
            batch_X, batch_y = batch
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # TODO: Complete this train method to train the model provided.
            model.zero_grad()
            pred_y = model(batch_X)
            loss = loss_fn(pred_y, batch_y)
            loss.backward()
            optimizer.step()           
            total_loss += loss.data.item()            
        sys.stdout.write("Epoch: {}, Training_Loss: {}\n".format(epoch, total_loss / len(train_loader)))
        
        model.eval()
        total_loss = 0
        num_correct = 0
        for batch in valid_loader:         
            batch_X, batch_y = batch
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            pred_y = model(batch_X)
            loss = loss_fn(pred_y, batch_y)                    
            total_loss += loss.data.item()            
            # convert output probabilities to predicted class (0 or 1)
            pred = torch.round(pred_y.squeeze())
            pred = pred.to(device)
    
            # compare predictions to true label
            correct_tensor = pred.eq(batch_y.float().view_as(pred))
            correct = np.squeeze(correct_tensor.cpu().numpy())
            num_correct += np.sum(correct)
            valid_loss = total_loss / len(valid_loader)
            valid_acc = num_correct / len(valid_loader.dataset)
        sys.stdout.write("Epoch: {}, Validation_Loss: {}\n".format(epoch, valid_loss))
        sys.stdout.write("Epoch: {}, Validation_Accuracy: {}\n".format(epoch, valid_acc))
        if valid_loss <= last_valid_loss:
            bad_rounds = 0
        else:
            bad_rounds += 1
        last_valid_loss = valid_loss
        if valid_loss <= best_valid_loss:
            best_valid_loss=valid_loss
            checkpoint_valid_acc=valid_acc
            model_checkpoint=model
    sys.stdout.write("Final_Validation_Loss: {}\n".format(best_valid_loss))
    sys.stdout.write("Final_Validation_Accuracy: {}\n".format(checkpoint_valid_acc))
    model=model_checkpoint


if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments when the script
    # is executed. Here we set up an argument parser to easily access the parameters.

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='L',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--early_stopping_rounds', type=int, default=99, metavar='E',
                        help='number of rounds the model have to worse validation loss to stop earlier (default: 99)')

    # Model Parameters
    parser.add_argument('--embedding_dim', type=int, default=32, metavar='N',
                        help='size of the word embeddings (default: 32)')
    parser.add_argument('--hidden_dim', type=int, default=100, metavar='N',
                        help='size of the hidden dimension (default: 100)')
    parser.add_argument('--vocab_size', type=int, default=5000, metavar='N',
                        help='size of the vocabulary (default: 5000)')

    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--valid-dir', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader = _get_train_data_loader(args.batch_size, args.train_dir)
    
    # TF: Load the validation data.
    valid_loader = _get_valid_data_loader(args.batch_size, args.valid_dir)

    # Build the model.
    model = LSTMClassifier(args.embedding_dim, args.hidden_dim, args.vocab_size).to(device)

    with open(os.path.join(args.train_dir, "word_dict.pkl"), "rb") as f:
        model.word_dict = pickle.load(f)

    print("Model loaded with embedding_dim {}, hidden_dim {}, vocab_size {}.".format(
        args.embedding_dim, args.hidden_dim, args.vocab_size
    ))

    # Train the model.
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCELoss()

    train(model, train_loader, valid_loader, args.epochs, args.early_stopping_rounds, optimizer, loss_fn, device)

    # Save the parameters used to construct the model
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'vocab_size': args.vocab_size,
            'learning_rate': args.lr,
        }
        torch.save(model_info, f)

	# Save the word_dict
    word_dict_path = os.path.join(args.model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'wb') as f:
        pickle.dump(model.word_dict, f)

	# Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)