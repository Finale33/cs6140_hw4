import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib as mpl
import const

mpl.use('TkAgg')


# building the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)   # 300-5+1= 296 * 296 * 10
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # 148-5+1= 144 * 144 * 20
        self.conv3 = nn.Conv2d(20, 30, kernel_size=5)  # 72-5+1= 68 * 68 * 30
        self.conv4 = nn.Conv2d(30, 40, kernel_size=5)  # 34-5+1= 30 * 30 * 40
        self.conv5 = nn.Conv2d(40, 50, kernel_size=5)  # 15-5+1= 11 * 11 * 50
        self.conv5_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(6050, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))                      # 296/2 = 148 * 148 * 10
        x = F.relu(F.max_pool2d(self.conv2(x), 2))                      # 144/2 = 72 * 72 * 20
        x = F.relu(F.max_pool2d(self.conv3(x), 2))                      # 68/2 = 34*34*30
        x = F.relu(F.max_pool2d(self.conv4(x), 2))                      # 30/2 = 15*15*50
        x = F.relu(self.conv5_drop(self.conv5(x)))                      # 11 * 11 * 50
        x = x.view(-1, 6050)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def MINIST_Tutorial():
    # greek data set transform
    class GreekTransform:
        def __init__(self):
            pass

        def __call__(self, x):
            x = torchvision.transforms.functional.rgb_to_grayscale(x)
            x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36 / 128, 0)
            x = torchvision.transforms.functional.center_crop(x, (300, 300))
            return torchvision.transforms.functional.invert(x)

    # setting up hyper parameters
    n_epochs = 10
    batch_size_train = 1
    batch_size_test = 15
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # DataLoader for the Greek data set
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(const.TRAININT_SET_PATH_ALPHABET,
                                         transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                   GreekTransform(),
                                                                                   torchvision.transforms.Normalize(
                                                                                       (0.1307,), (0.3081,))])),
        batch_size=batch_size_train,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(const.TESTING_SET_PATH_ALPHABET,
                                         transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                   GreekTransform(),
                                                                                   torchvision.transforms.Normalize(
                                                                                       (0.1307,), (0.3081,))])),
        batch_size=batch_size_test,
        shuffle=True)

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    print(example_data.shape)

    # plotting some examples
    plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])

    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                          momentum=momentum)

    # train the model
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    def train(epoch):
        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
                torch.save(network.state_dict(), const.TASK1_MODEL_PATH)
                torch.save(optimizer.state_dict(), const.TASK1_OPTIMIZER_PATH)

    def test():
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = network(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    test()
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()

    # evaluate the model by printing the loss
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')

    with torch.no_grad():
        output = network(example_data)

    # evaluate the model by plotting 8 examples
    plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
            output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    MINIST_Tutorial()