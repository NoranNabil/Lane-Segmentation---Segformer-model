import os
from PIL import Image  # Add this import
import cv2
import numpy as np
import copy
from tqdm import tqdm
import requests

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
from torchvision import transforms as TF
import torch.nn.functional as F
from torch.optim import AdamW
import evaluate
from transformers import SegformerForSemanticSegmentation
from transformers import get_scheduler
from transformers import TrainingArguments,Trainer

from sklearn.metrics import jaccard_score

class BDDDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = [img for img in os.listdir(images_dir) if img.endswith('.png')]
        self.masks = [mask.replace('.jpg', '.png') for mask in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert('L')  # Convert mask to grayscale

        # Convert mask to binary format with 0 and 1 values
        mask = np.array(mask)
        mask = (mask > 0).astype(np.uint8)  # Assuming non-zero pixels are lanes

        # Convert to PIL Image for consistency in transforms
        mask = Image.fromarray(mask)

        if self.transform:
            image = self.transform(image)
            # Assuming to_tensor transform is included which scales pixel values between 0-1
            # mask = to_tensor(mask)  # Convert the mask to [0, 1] range
        mask = TF.functional.resize(img=mask, size=[360, 640], interpolation=Image.NEAREST)
        mask = TF.functional.to_tensor(mask)
        mask = (mask > 0).long()  # Threshold back to binary and convert to LongTensor


        return image, mask

def mean_iou(preds, labels, num_classes):
    # Flatten predictions and labels
    preds_flat = preds.view(-1)
    labels_flat = labels.view(-1)

    # Check that the number of elements in the flattened predictions
    # and labels are equal
    if preds_flat.shape[0] != labels_flat.shape[0]:
        raise ValueError(f"Predictions and labels have mismatched shapes: "
                         f"{preds_flat.shape} vs {labels_flat.shape}")

    # Calculate the Jaccard score for each class
    iou = jaccard_score(labels_flat.cpu().numpy(), preds_flat.cpu().numpy(),
                        average=None, labels=range(num_classes))

    # Return the mean IoU
    return np.mean(iou)

# Define the appropriate transformations
transform = TF.Compose([
    TF.Resize((360, 640)),
    TF.ToTensor(),
    TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create the dataset
train_dataset = BDDDataset(images_dir='/lfs01/workdirs/benha012u1/N_N/train/image',
                           masks_dir='/lfs01/workdirs/benha012u1/N_N/train/gt_binary_image',
                           transform=transform)
valid_dataset = BDDDataset(images_dir='/lfs01/workdirs/benha012u1/N_N/valid/image',
                           masks_dir='/lfs01/workdirs/benha012u1/N_N/valid/gt_binary_image',
                           transform=transform)

# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False, num_workers=6)

valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=False, num_workers=6)
# Load the pre-trained model

#nvidia/segformer-b0-finetuned-ade-512-512
#model = SegformerForSemanticSegmentation.from_pretrained("/lfs01/workdirs/benha009u1/Lanes/seg/save2_11/",cache_dir = "/lfs01/workdirs/benha009u1/Lanes/seg/best_model.pth",ignore_mismatched_sizes=True)#("/lfs01/workdirs/benha009u1/Lanes/seg/best_model_weights")#
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
#model = SegformerForSemanticSegmentation.from_pretrained(pretrained_model_name_or_path ="/lfs01/workdirs/benha009u1/Lanes/seg/save_id2label/")
#                                                         #state_dict =Dict["/lfs01/workdirs/benha009u1/Lanes/seg/save/best_model.pth", torch.Tensor])
# Print some parameters before loading weights
#print("Parameters before loading weights:")
#print(model.state_dict())  # Print some parameter to compare later

metric = evaluate.load('mean_iou')
#model.load_state_dict(torch.load('/lfs01/workdirs/benha009u1/Lanes/seg/best_model.pth'))
#Adjust the number of classes for BDD dataset
model.config.num_labels = 2  # Replace with the actual number of classes
new_id2label = {0: "background", 1: "lane"}  # New id2label mapping
model.config.id2label = new_id2label

# Print some parameters after loading weights
#print("Parameters after loading weights:")
#print(model.state_dict())  # Print the same parameter again to compare


#training_args = TrainingArguments("/lfs01/workdirs/benha009u1/Lanes/seg/best_model_weights")

#trainer = Trainer(
#    model=model,
#    args=training_args)
# Check for CUDA acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device);
# Define the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5) 

# Define the learning rate scheduler
num_epochs = 200
num_training_steps = num_epochs * len(train_loader)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# Placeholder for best mean IoU and best model weights
best_iou = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
model.train()
for epoch in range(num_epochs):

    train_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
    #print(train_dataset.__len__())
    for idx, batch in enumerate (train_iterator): #range (train_dataset.__len__()):
        
        images, masks = train_dataset.__getitem__(idx)
        images = images.to(device)
        masks = masks.to(device).long()  # Ensure masks are LongTensors

        # Remove the channel dimension from the masks tensor
        masks = masks.squeeze(1)  # This changes the shape from [batch, 1, H, W] to [batch, H, W]
        optimizer.zero_grad()
        images = images.unsqueeze(0)
        # Pass pixel_values and labels to the model
        outputs = model(pixel_values=images, labels=masks)

        loss = outputs["loss"]
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        outputs = F.interpolate(outputs["logits"], size=masks.shape[-2:], mode="bilinear", align_corners=False)

        train_iterator.set_postfix(loss=loss.item())

    # Evaluation loop for each epoch
    model.eval()
    total_iou = 0
    num_batches = 0
    valid_iterator = tqdm(valid_loader, desc="Validation", unit="batch")
    for idx, batch in enumerate (valid_iterator):
        images, masks = valid_dataset.__getitem__(idx)
        images = images.to(device)
        masks = masks.to(device).long()
        masks = masks.squeeze(1)  # This changes the shape from [batch, 1, H, W] to [batch, H, W]
        optimizer.zero_grad()
        images = images.unsqueeze(0)

        with torch.no_grad():
            # Get the logits from the model and apply argmax to get the predictions
            print ("Pixel_values" , images.shape )
            outputs = model(pixel_values=images,return_dict=True)
            outputs = F.interpolate(outputs["logits"], size=masks.shape[-2:], mode="bilinear", align_corners=False)
            preds = torch.argmax(outputs, dim=1)
            preds = torch.unsqueeze(preds, dim=1)

        preds = preds.view(-1)
        masks = masks.view(-1)

        # Compute IoU
        iou = mean_iou(preds, masks, model.config.num_labels)
        total_iou += iou
        num_batches += 1
        valid_iterator.set_postfix(mean_iou=iou)

    epoch_iou = total_iou / num_batches
    print(f"Epoch {epoch+1}/{num_epochs} - Mean IoU: {epoch_iou:.4f}")
    print("Learning rate:", optimizer.param_groups[0]['lr'])

    # Check for improvement
    if epoch_iou > best_iou:
        print(f"Validation IoU improved from {best_iou:.4f} to {epoch_iou:.4f}")
        best_iou = epoch_iou
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, 'best_model.pth')


#model.save_pretrained(save_directory="/lfs01/workdirs/benha009u1/Lanes/seg/load_id2label")
