import wandb
import torch
import numpy as np
import sys
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
sys.path.append('./GRASP-main/DERpp/')
from MNModel_derpp import MNModel, _set_seed
from utils_dr import Counter
import seaborn as sns


# ImageFolder wrapper that returns the index of the image
class IndexedImageFolder(ImageFolder):
    def __getitem__(self, index):
        # Retrieve the image and label from the parent class
        image, label = super().__getitem__(index)
        # Return the image, label, and index
        return image, label, index
    
def main():
    # Initialize wandb
    wandb.init(project="GRASP-2", name="difficulty-distribution")

    # Prep data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and data loader for train data
    folder_paths = ['./imagenet100/train.X1', './imagenet100/train.X2', './imagenet100/train.X3', './imagenet100/train.X4']
    datasets = [IndexedImageFolder(root=path, transform=transform) for path in folder_paths]
    train_dataset = ConcatDataset(datasets)  # Combine the datasets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    ######### EXPERIMENT SETUP #########
    data_loader = train_loader
    SEED = 1998 # Make sure it's the same seed as in the MNModel
    _set_seed(SEED)

    # Initialize model
    num_classes = 100
    model = MNModel(num_classes=num_classes, seed = SEED)

    # Initialize buffer components for rehearsal
    latent_dict = {} #embeddings for images
    rehearsal_ixs = [] #list to store indices of samples in the rehearsal buffer
    class_id_to_item_ix_dict = {i: [] for i in range(num_classes)} #maps class ID to a list of sample indices for that class
    class_id_dist = {i: [] for i in range(num_classes)} #holds the difficulty (distance) values for each sample within each class
    counter = Counter() # counter to keep track of buffer updates

    # Perform buffer update to initialize rehearsal set and difficulty scores
    latent_dict, rehearsal_ixs, class_id_to_item_ix_dict, class_id_dist, recent_class_list = model.update_buffer(
        curr_loader=data_loader, 
        latent_dict=latent_dict, 
        rehearsal_ixs=rehearsal_ixs, 
        class_id_to_item_ix_dict=class_id_to_item_ix_dict, 
        class_id_dist=class_id_dist, 
        counter=counter
    )

    # Initial difficulty distribution histogram
    difficulty_scores = [score for scores in class_id_dist.values() for score in scores]
    bins = int(np.sqrt(len(difficulty_scores)))
    plt.figure(figsize=(8, 6))
    plt.hist(difficulty_scores, bins=bins, edgecolor='black')
    plt.xlabel("Difficulty Score")
    plt.ylabel("Frequency")
    plt.title("Initial Difficulty Distribution (Absolute Frequency)")
    wandb.log({"Initial Difficulty Distribution": wandb.Image(plt)})
    plt.close()
    
    # Track the average difficulty per iteration
    iterations = 10
    avg_difficulties = []
    num_bins = 5
    bin_edges = np.linspace(min(difficulty_scores), max(difficulty_scores), num_bins + 1)
    heatmap_data = np.zeros((iterations, num_bins)) # 10 iterations x num_bins bins
    
    # Run GRASP and log difficulty distribution
    for iteration in range(iterations):
        selected_indices = model.grasp_sampling(class_id_to_item_ix_dict, class_id_dist, recent_class_list, num_iters=1)
        
        # Capture difficulty scores for selected samples
        sampled_difficulties = []
        for idx in selected_indices:
            class_id = None
            for c, indices in class_id_to_item_ix_dict.items():
                if idx in indices:
                    class_id = c
                    break
            # Add the corresponding difficulty score
            if class_id is not None:
                sample_difficulty = class_id_dist[class_id][indices.index(idx)]
                sampled_difficulties.append(sample_difficulty)
        
        # Calculate avg. difficulty for the current iteration
        avg_difficulty = np.mean(sampled_difficulties)
        avg_difficulties.append(avg_difficulty)
        
        # Log the average difficulty to wandb
        wandb.log({
            "Average Difficulty": avg_difficulty,
            "Iteration": iteration
        })
        
        # Log iteration-specific difficulty distribution as histogram
        bins = int(np.sqrt(len(sampled_difficulties)))
        plt.figure(figsize=(8, 6))
        plt.hist(sampled_difficulties, bins=bins, edgecolor='black')
        plt.xlabel("Difficulty Score")
        plt.ylabel("Frequency")
        plt.title(f"Difficulty Distribution - Iteration {iteration}")
        wandb.log({f"Difficulty Distribution Iteration {iteration}": wandb.Image(plt)})
        plt.close()
    
        # Update heatmap data for current iteration
        counts, _ = np.histogram(sampled_difficulties, bins=bin_edges)
        heatmap_data[iteration] = counts
        
        
    # Log the full trend of average difficulties as a line plot
    wandb.log({"Average Difficulty Evolution": wandb.plot.line_series(
        xs=[list(range(10))],
        ys=[avg_difficulties],
        keys=["Average Difficulty"],
        title="Average Difficulty Evolution",
        xname="Iteration"
    )})
    # Create and log heat map showing the selection frequency across difficulty bins
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap="YlGnBu", xticklabels=np.round(bin_edges, 2), yticklabels=range(10))
    plt.xlabel("Difficulty Bins")
    plt.ylabel("Iteration")
    plt.title("Sample Selection Across Difficulty Spectrum Over Iterations")
    wandb.log({"Difficulty Selection Heatmap": wandb.Image(plt)})
    plt.close()
    
    # Combined difficulty distribution histogram across all iterations
    all_sampled_difficulties = [score for iteration in range(10) for idx in model.grasp_sampling(class_id_to_item_ix_dict, class_id_dist, recent_class_list, num_iters=1)
                                for c, indices in class_id_to_item_ix_dict.items() if idx in indices
                                for score in [class_id_dist[c][indices.index(idx)]]]
    bins = int(np.sqrt(len(all_sampled_difficulties)))
    plt.figure(figsize=(8, 6))
    plt.hist(all_sampled_difficulties, bins=bins, edgecolor='black')
    plt.xlabel("Difficulty Score")
    plt.ylabel("Frequency")
    plt.title("Combined Difficulty Distribution Across All Iterations")
    wandb.log({"Combined Difficulty Distribution": wandb.Image(plt)})
    plt.close()
    
if __name__ == '__main__':
    main()