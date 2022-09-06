import copy
import os
import time

import argparse
import torch
from torchvision import transforms, datasets
import torchvision.models.quantization as models
from torch import nn
import torch.optim as optim


ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input", required=True)  # dataset path

args = vars(ap.parse_args())

# Data augmentation and normalization for training # Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = args['input']

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=8) for x in
               ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # We will need the number of filters in the `fc` for future use.
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_fe = models.resnet18(pretrained=True, progress=True, quantize=True)

    new_model = create_combined_model(model_fe)
    new_model = new_model.to('cpu')

    criterion = nn.CrossEntropyLoss()

    # Note that we are only training the head.
    optimizer_ft = optim.SGD(new_model.parameters(), lr=0.01, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    new_model = train_model(new_model, criterion, optimizer_ft, exp_lr_scheduler,
                            num_epochs=25, device='cpu')

    torch.save(new_model.state_dict(), './model/check1.pth')

    # notice `quantize=False`
    model = models.resnet18(pretrained=True, progress=True, quantize=False)
    num_ftrs = model.fc.in_features

    # Step 1
    model.train()
    model.fuse_model()
    # Step 2
    model_ft = create_combined_model(model)
    model_ft[0].qconfig = torch.quantization.default_qat_qconfig  # Use default QAT configuration
    # Step 3
    model_ft = torch.quantization.prepare_qat(model_ft, inplace=True)

    for param in model_ft.parameters():
        param.requires_grad = True

    model_ft.to(device)  # We can fine-tune on GPU if available

    criterion = nn.CrossEntropyLoss()

    # Note that we are training everything, so the learning rate is lower
    # Notice the smaller learning rate
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.1)

    # Decay LR by a factor of 0.3 every several epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.3)

    model_ft_tuned = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                 num_epochs=25, device=device)

    from torch.quantization import convert

    model_ft_tuned.cpu()

    model_quantized_and_trained = convert(model_ft_tuned, inplace=False)
    torch.save(model_quantized_and_trained.state_dict(), './model/check2.pth')


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, device='cpu'):
    """
    Support function for model training.

    Args:
      model: Model to be trained
      criterion: Optimization criterion (loss)
      optimizer: Optimizer to use for training
      scheduler: Instance of ``torch.optim.lr_scheduler``
      num_epochs: Number of epochs
      device: Device to run the training on. Must be 'cpu' or 'cuda'
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def create_combined_model(model_fe):
    num_ftrs = model_fe.fc.in_features

    # Step 1. Isolate the feature extractor.
    model_fe_features = nn.Sequential(
        model_fe.quant,  # Quantize the input
        model_fe.conv1,
        model_fe.bn1,
        model_fe.relu,
        model_fe.maxpool,
        model_fe.layer1,
        model_fe.layer2,
        model_fe.layer3,
        model_fe.layer4,
        model_fe.avgpool,
        model_fe.dequant,  # Dequantize the output
    )

    # Step 2. Create a new "head"
    new_head = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 2),
    )

    # Step 3. Combine, and don't forget the quant stubs.
    new_model = nn.Sequential(
        model_fe_features,
        nn.Flatten(1),
        new_head,
    )
    return new_model

if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    main()  # execute this only when run directly, not when imported!
