import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

#epsilons = [0.4, .05]#, .1, .15, .2, .25, .3]

epsilons = [0.0045]
pretrained_model = "lenet_mnist_model.pth"
# Set random seed for reproducibility
torch.manual_seed(42)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    

# MNIST Test dataset and dataloader declaration
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ])),
        batch_size=1, shuffle=True)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Initialize the network
model = Net().to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location=device, weights_only=True))

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

# restores the tensors to their original scale
def denorm(batch, mean=[0.1307], std=[0.3081]):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


######################################################################
# Testing Function
# ~~~~~~~~~~~~~~~~
#
# Finally, the central result of this tutorial comes from the ``test``
# function. Each call to this test function performs a full test step on
# the MNIST test set and reports a final accuracy. However, notice that
# this function also takes an *epsilon* input. This is because the
# ``test`` function reports the accuracy of a model that is under attack
# from an adversary with strength :math:`\epsilon`. More specifically, for
# each sample in the test set, the function computes the gradient of the
# loss w.r.t the input data (:math:`data\_grad`), creates a perturbed
# image with ``fgsm_attack`` (:math:`perturbed\_data`), then checks to see
# if the perturbed example is adversarial. In addition to testing the
# accuracy of the model, the function also saves and returns some
# successful adversarial examples to be visualized later.
#

def project(imagen_original, imagen_adversarial):
    
    radio=0.075

    lower_bound = imagen_original - radio
    upper_bound = imagen_original + radio
    projected = torch.clamp(imagen_adversarial, min=lower_bound, max=upper_bound)
    
    return projected

def test( model, device, test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []
    iteracion=0

    print("Total test set",len(test_loader))

    # Loop over all examples in test set
    for data, target in test_loader:

        print("Iteracion",iteracion)

        data_iteracion=data
        iteracion+=1

        #if(iteracion==1000): break

        for k in range(35):

            #print("Iteracion",k)

            # Send the data and label to the device
            data_en_device, target = data_iteracion.to(device), target.to(device)

            # Set requires_grad attribute of tensor. Important for Attack
            data_en_device.requires_grad = True

            # Forward pass the data through the model
            output = model(data_en_device)
            
            if(k==0):
            
                init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

                # If the initial prediction is wrong, don't bother attacking, just move on
                if (init_pred.item() != target.item()): break

            # Calculate the loss
            loss = F.nll_loss(output, target)

            # Zero all existing gradients
            model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Restore the data to its original scale
            data_denorm = denorm(data_en_device)

            imagen_original = data_denorm
            perturbed_image = data_denorm

            data_grad = data_en_device.grad.data

            sign_data_grad = data_grad.sign()
            perturbed_image = perturbed_image + epsilon*sign_data_grad

            perturbed_image = project(imagen_original,perturbed_image)

            perturbed_data = torch.clamp(perturbed_image, 0, 1)

            perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)

            data_iteracion = perturbed_data_normalized.detach()


        # Re-classify the perturbed image
        output = model(data_iteracion.to(device))

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if epsilon == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


######################################################################
# Run Attack
# ~~~~~~~~~~
#
# The last part of the implementation is to actually run the attack. Here,
# we run a full test step for each epsilon value in the *epsilons* input.
# For each epsilon we also save the final accuracy and some successful
# adversarial examples to be plotted in the coming sections. Notice how
# the printed accuracies decrease as the epsilon value increases. Also,
# note the :math:`\epsilon=0` case represents the original test accuracy,
# with no attack.
#

accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)


plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()

cnt = 0

for i in range(len(epsilons)):
    eps_val = epsilons[i]
    folder_name = f"final_epsilon_{eps_val:.4f}"
    
    # Create the folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)

    for j in range(len(examples[i])):
        cnt += 1
        orig_label, adv_label, ex_image = examples[i][j]

        # Convert image to 0-255 and uint8 if needed
        if isinstance(ex_image, torch.Tensor):
            ex_image = ex_image.detach().cpu().numpy()

        # Scale if it's in [0,1] float
        if ex_image.max() <= 1.0:
            ex_image = (ex_image * 255).astype(np.uint8)

        # Save image
        img = Image.fromarray(ex_image.squeeze())  # squeeze in case of shape (1,H,W)
        img.save(os.path.join(folder_name, f"{orig_label}_to_{adv_label}_{j}.png"))
