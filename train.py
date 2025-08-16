"""
EEG Classification Model Training Script
Clean version with essential functionality for model training and evaluation.
"""

import os
import time
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import precision_score, recall_score, f1_score

from dotmap import DotMap
from myutils import getData, save_model, load_model, save_load_name
from model_epsanet import epsanet50_spd

# Environment setup
os.environ["NUMEXPR_MAX_THREADS"] = "16"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Logging setup
logging.basicConfig(level=logging.INFO)
result_logger = logging.getLogger('result')


def setup_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ModelTrainer:
    """Main class for model training and evaluation"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        self.model = epsanet50_spd().to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=0.001,
            weight_decay=0.00005
        )

        print(f"Model initialized with {count_parameters(self.model):,} trainable parameters")

    def train_epoch(self, train_loader):
        """Train model for one epoch"""
        self.model.train()
        train_acc_sum = 0
        train_loss_sum = 0
        batch_size = train_loader.batch_size

        for batch_data in train_loader:
            train_data, train_adj, train_label = batch_data
            train_label = train_label.squeeze(-1)
            train_data, train_adj, train_label = (
                train_data.to(self.device),
                train_adj.to(self.device),
                train_label.to(self.device)
            )

            # Forward pass
            preds, _ = self.model(train_data, train_adj)
            loss = self.criterion(preds, train_label.long())

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            # Statistics
            with torch.no_grad():
                train_loss_sum += loss.item() * batch_size
                predicted = preds.data.max(1)[1]
                train_acc_sum += predicted.eq(train_label).cpu().sum()

        return train_loss_sum / len(train_loader.dataset), train_acc_sum / len(train_loader.dataset)

    def evaluate(self, data_loader):
        """Evaluate model on given data loader"""
        self.model.eval()
        total_loss = 0.0
        acc_sum = 0
        all_preds = []
        all_labels = []
        batch_size = data_loader.batch_size

        with torch.no_grad():
            for batch_data in data_loader:
                test_data, test_adj, test_label = batch_data
                test_label = test_label.squeeze(-1)
                test_data, test_adj, test_label = (
                    test_data.to(self.device),
                    test_adj.to(self.device),
                    test_label.to(self.device)
                )

                preds, _ = self.model(test_data, test_adj)

                total_loss += self.criterion(preds, test_label.long()).item() * batch_size
                predicted = preds.data.max(1)[1]
                acc_sum += predicted.eq(test_label).cpu().sum()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(test_label.cpu().numpy())

        avg_loss = total_loss / len(data_loader.dataset)
        avg_acc = acc_sum / len(data_loader.dataset)

        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return avg_loss, avg_acc, precision, recall, f1

    def train(self, train_loader, valid_loader, test_loader):
        """Main training loop"""
        train_losses = []
        valid_losses = []

        epochs_without_improvement = 0
        best_epoch = 1
        best_valid_loss = float('inf')

        for epoch in tqdm(range(1, self.args.max_epoch + 1), desc='Training'):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validation
            val_loss, val_acc, val_precision, val_recall, val_f1 = self.evaluate(valid_loader)

            train_losses.append(train_loss)
            valid_losses.append(val_loss)

            print(f'Epoch {epoch:2d} | Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | '
                  f'Valid Loss {val_loss:.4f} | Valid Acc {val_acc:.4f} | '
                  f'Precision {val_precision:.4f} | Recall {val_recall:.4f} | F1 {val_f1:.4f}')

            # Early stopping
            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                epochs_without_improvement = 0
                best_epoch = epoch
                print(f"Saved model at pre_trained_models/{save_load_name(self.args, name=self.args.name)}.pt!")
                save_model(self.args, self.model, name=self.args.name)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement > 10:
                    print("Early stopping triggered")
                    break

        # Load best model and test
        self.model = load_model(self.args, name=self.args.name)
        test_loss, test_acc, test_precision, test_recall, test_f1 = self.evaluate(test_loader)

        print(f'Best epoch: {best_epoch}')
        print(f"Test Results - Acc: {test_acc:.4f}, Precision: {test_precision:.4f}, "
              f"Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

        return train_losses, valid_losses, test_loss, test_acc, test_precision, test_recall, test_f1


def plot_loss_curves(train_losses, valid_losses, subject_name):
    """Plot and save training/validation loss curves"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.title(f'Loss Curve for Subject {subject_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Create directory if it doesn't exist
    os.makedirs('loss', exist_ok=True)
    plt.savefig(f'loss/loss_curve_subject_{subject_name}.png')
    plt.close()


def train_subject(subject_name, time_len=1, dataset="KUL"):
    """Train model for a single subject"""
    setup_seed(2024)

    # Setup arguments
    args = DotMap()
    args.name = subject_name
    args.max_epoch = 100
    args.random_seed = 1234

    # Load data
    train_loader, valid_loader, test_loader = getData(subject_name, time_len, dataset)
    print(f'Subject: {subject_name}, Data shape: {train_loader.dataset.data.shape}')

    # Train model
    trainer = ModelTrainer(args)
    train_losses, valid_losses, test_loss, test_acc, test_precision, test_recall, test_f1 = \
        trainer.train(train_loader, valid_loader, test_loader)

    # Plot loss curves
    plot_loss_curves(train_losses, valid_losses, subject_name)

    # Log results
    info_msg = (f'{dataset}_{subject_name}_{time_len}s '
                f'loss:{test_loss:.4f} acc:{test_acc:.4f} '
                f'precision:{test_precision:.4f} recall:{test_recall:.4f} f1:{test_f1:.4f}')
    result_logger.info(info_msg)

    return test_loss, test_acc, test_precision, test_recall, test_f1


def main():
    """Main function to train models for all subjects"""
    setup_seed(14)

    # Setup logging
    os.makedirs('log', exist_ok=True)
    file_handler = logging.FileHandler('log/result.log')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    result_logger.addHandler(file_handler)

    print(f"CUDA available: {torch.cuda.is_available()}")

    # Configuration
    dataset = 'KUL'
    time_len = 1
    subjects = ['S1', 'S2', 'S3', 'S4']  # Adjust as needed

    # Store results
    all_results = {
        'accuracies': [],
        'precisions': [],
        'recalls': [],
        'f1_scores': []
    }

    # Train for each subject
    for subject in subjects:
        print(f"\nTraining for subject: {subject}")
        _, acc, precision, recall, f1 = train_subject(subject, time_len, dataset)

        all_results['accuracies'].append(acc)
        all_results['precisions'].append(precision)
        all_results['recalls'].append(recall)
        all_results['f1_scores'].append(f1)

    # Calculate and print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY RESULTS")
    print("=" * 60)

    for metric_name, values in all_results.items():
        values_tensor = torch.tensor(values)
        mean_val = torch.mean(values_tensor)
        std_val = torch.std(values_tensor)
        print(f'{metric_name.capitalize()}: {mean_val:.4f} ± {std_val:.4f}')

    # Log summary
    summary_msg = (f'{dataset} Summary - '
                   f'Avg Accuracy: {torch.mean(torch.tensor(all_results["accuracies"])):.4f} '
                   f'Avg Precision: {torch.mean(torch.tensor(all_results["precisions"])):.4f} '
                   f'Avg Recall: {torch.mean(torch.tensor(all_results["recalls"])):.4f} '
                   f'Avg F1: {torch.mean(torch.tensor(all_results["f1_scores"])):.4f}')
    result_logger.info(summary_msg)


if __name__ == "__main__":
    main()