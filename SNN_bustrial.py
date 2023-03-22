"""
Trying to implement the Siamese Neural Network
"""
from __future__ import print_function
import argparse, random, copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T
from torch.optim.lr_scheduler import StepLR

# Modified version
import general_scripts as gs
import torchvision.transforms as transforms
from PIL import Image
import os
from torch.utils.data import random_split

torch.cuda.is_available()

def check_mem_alloc():
    print("Checking mem usage...\n" + torch.cuda.memory_summary(device=None, abbreviated=False))

# print("Emptying cache...")
# torch.cuda.empty_cache()


class SiameseNetwork(nn.Module):
    """
        Siamese network for image similarity estimation.
        The network is composed of two identical networks, one for each input.
        The output of each network is concatenated and passed to a linear layer. 
        The output of the linear layer passed through a sigmoid function.
        `"FaceNet" <https://arxiv.org/pdf/1503.03832.pdf>`_ is a variant of the Siamese network.
        This implementation varies from FaceNet as we use the `ResNet-18` model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_ as our feature extractor.
        In addition, we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # get resnet model
        self.resnet = torchvision.models.resnet18(pretrained=False)

        # over-write the first conv layer to be able to read MNIST images
        # as resnet18 reads (3,x,x) where 3 is RGB channels
        # whereas MNIST has (1,x,x) where 1 is a gray-scale channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features
        
        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()

        # initialize the weights
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)
        
        return output

def prepare_dataset(save=False):
    os.chdir("D:\BusXray\Compiling_All_subfolder_images\Compiled_Clean_Images")
    print("Current working directory: {0}".format(os.getcwd()))
    temp_dataset = gs.load_images_from_folder("D:\BusXray\Compiling_All_subfolder_images\Compiled_Clean_Images")
    temp_dataset = [ x for x in temp_dataset if "Monochrome" in x ]

    # Define a transform to convert the image to tensor
    transform = transforms.ToTensor()
    compiled_tensorised_dict = {}

    for index, first_image in enumerate(temp_dataset):
        bus_model = ' '.join(first_image.split(" ")[0:-3])
        image = Image.open(first_image)
        tensor = transform(image)

        # Instantialise new key if dictionary didn't have it previously.
        if compiled_tensorised_dict.get(f"{bus_model}") == None:
            print(f"New model, {bus_model} found, creating new list")
            compiled_tensorised_dict[bus_model] = []

        compiled_tensorised_dict[bus_model].append([index, tensor])
    
    if save:
        torch.save(compiled_tensorised_dict, 'C:\\Users\\User1\\Desktop\\alp\\compiled_tensorised_dict.pt')
    return compiled_tensorised_dict

def transform_dataset_into_tensor_data(dataset):
    tensor_list = []
    for big_list in dataset.values():
        for individual_list in big_list:
            # individual_list[0] contains their index, individual_list[1] contains their respective tensor
            tensor_list.append(individual_list[1])
    
    tensor = torch.stack(tensor_list)
    return tensor

class APP_MATCHER(Dataset):
    def __init__(self, root, dataset):
        super(APP_MATCHER, self).__init__()
        
        self.dataset = dataset
        self.data = transform_dataset_into_tensor_data(self.dataset)
        self.group_examples()

    def group_examples(self):
        """
            To ease the accessibility of data based on the class, we will use `group_examples` to group 
            examples based on class. 
            
            Every key in `grouped_examples` corresponds to a class in MNIST dataset. For every key in 
            `grouped_examples`, every value will conform to all of the indices for the MNIST 
            dataset examples that correspond to that key.
        """
        
        # group examples based on class
        self.grouped_examples = {}
        
        # This groups up the index of each tensor to it's respective groups according to the blog's format.
        for key, value in self.dataset.items():
            self.grouped_examples[key] = np.array([each_tensor[0] for each_tensor in value])
        
        # print("self.grouped_examples:", self.grouped_examples)
        
        
    
    def __len__(self):
        return len(self.grouped_examples)
    
    def __getitem__(self, index):
        """
            For every example, we will select two images. There are two cases, 
            positive and negative examples. For positive examples, we will have two 
            images from the same class. For negative examples, we will have two images 
            from different classes.
            Given an index, if the index is even, we will pick the second image from the same class, 
            but it won't be the same image we chose for the first class. This is used to ensure the positive
            example isn't trivial as the network would easily distinguish the similarity between same images. However,
            if the network were given two different images from the same class, the network will need to learn 
            the similarity between two different images representing the same class. If the index is odd, we will 
            pick the second image from a different class than the first image.
        """

        # Get all the respective bus models in a list
        list_of_bus_model = list(self.dataset)

        # pick some random class for the first image
        selected_class = random.randint(0, len(list_of_bus_model)-1)

        # pick a random index for the first image in the grouped indices based of the label
        # of the class
        random_index_1 = random.randint(0, len(self.grouped_examples[list_of_bus_model[selected_class]])-1)
        
        # pick the index to get the first image
        index_1 = self.grouped_examples[list_of_bus_model[selected_class]][random_index_1]

        # get the first image
        image_1 = self.data[index_1].clone().float()

        # same class
        if index % 2 == 0:
            # pick a random index for the second image
            random_index_2 = random.randint(0, len(self.grouped_examples[list_of_bus_model[selected_class]])-1)
            
            # ensure that the index of the second image isn't the same as the first image
            while random_index_2 == random_index_1:
                random_index_2 = random.randint(0, len(self.grouped_examples[list_of_bus_model[selected_class]])-1)
            
            # pick the index to get the second image
            index_2 = self.grouped_examples[list_of_bus_model[selected_class]][random_index_2]

            # get the second image
            image_2 = self.data[index_2].clone().float()

            # set the label for this example to be positive (1)
            target = torch.tensor(1, dtype=torch.float)
        
        # different class
        else:
            # pick a random class
            other_selected_class = random.randint(0, len(list_of_bus_model)-1)

            # ensure that the class of the second image isn't the same as the first image
            while other_selected_class == selected_class:
                other_selected_class = random.randint(0, len(list_of_bus_model)-1)

            
            # pick a random index for the second image in the grouped indices based of the label
            # of the class
            random_index_2 = random.randint(0, len(self.grouped_examples[list_of_bus_model[selected_class]])-1)

            # pick the index to get the second image
            index_2 = self.grouped_examples[list_of_bus_model[selected_class]][random_index_2]

            # get the second image
            image_2 = self.data[index_2].clone().float()

            # set the label for this example to be negative (0)
            target = torch.tensor(0, dtype=torch.float)

        return image_1, image_2, target



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    # we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    criterion = nn.BCELoss()

    for batch_idx, (images_1, images_2, targets) in enumerate(train_loader):
        images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images_1, images_2).squeeze()
        targets = torch.squeeze(targets)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images_1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    # we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    criterion = nn.BCELoss()

    with torch.no_grad():
        for (images_1, images_2, targets) in test_loader:
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            outputs = model(images_1, images_2).squeeze()
            targets = torch.squeeze(targets)
            test_loss += criterion(outputs, targets).sum().item()  # sum up batch loss
            pred = torch.where(outputs > 0.5, 1, 0)  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    # for the 1st epoch, the average loss is 0.0001 and the accuracy 97-98%
    # using default settings. After completing the 10th epoch, the average
    # loss is 0.0000 and the accuracy 99.5-100% using default settings.
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Siamese network Example')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
#    parser.add_argument('--no-mps', action='store_true', default=False,
#                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--save-preprocessing', action='store_true', default=False,
                        help='saves pre-processing of images data')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
#    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
#    elif use_mps:
#        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using {device}")
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # train_dataset = APP_MATCHER(prepare_dataset(save=True))
    # test_dataset = APP_MATCHER(prepare_dataset())
    preprocessed_data = torch.load('compiled_tensorised_dict.pt')
    train_data = {}
    test_data = {}
    # Split dataset into 80/20 train/test
    for bus_model, value in preprocessed_data.items():
        temp_train_list = []
        temp_test_list = []
        for index, items in enumerate(value):
            if index < len(value)*80/100-1:
                temp_train_list.append(items)
            else:
                temp_test_list.append(items)
        train_data[bus_model] = temp_train_list
        test_data[bus_model] = temp_test_list

    print("The length of train data is:",len(train_data))
    print("The length of test data is:",len(test_data))
    cwd = os.getcwd
    train_dataset = APP_MATCHER(root=cwd, dataset=train_data)
    test_dataset = APP_MATCHER(root=cwd, dataset=test_data)
    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = SiameseNetwork().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        check_mem_alloc()
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "siamese_network.pt")


if __name__ == '__main__':
    # Run this to prep the images to be stored in a dataset
    # prepare_dataset(save=True)
    main()






