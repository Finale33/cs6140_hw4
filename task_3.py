import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from task_1 import Net

# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import const


def Greek_Letter():
    # greek data set transform
    class GreekTransform:
        def __init__(self):
            pass

        def __call__(self, x):
            x = torchvision.transforms.functional.rgb_to_grayscale(x)
            x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36 / 128, 0)
            x = torchvision.transforms.functional.center_crop(x, (28, 28))
            return torchvision.transforms.functional.invert(x)

    n_epochs = 100
    batch_size_train = 1
    batch_size_test = 9
    learning_rate = 0.5
    momentum = 0.5
    log_interval = 10

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # DataLoader for the Greek data set
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(const.TRAININT_SET_PATH,
                                         transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                   GreekTransform(),
                                                                                   torchvision.transforms.Normalize(
                                                                                       (0.1307,), (0.3081,))])),
        batch_size=batch_size_train,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(const.TESTING_SET_PATH,
                                         transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                   GreekTransform(),
                                                                                   torchvision.transforms.Normalize(
                                                                                       (0.1307,), (0.3081,))])),
        batch_size=batch_size_test,
        shuffle=True)

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    print(example_data.shape)

    network = Net()

    print(network)

    network.load_state_dict(torch.load(const.TASK1_MODEL_PATH))
    # freezes the parameters for the whole network
    for param in network.parameters():
        param.requires_grad = False

    network.fc2 = nn.Linear(50, 3)

    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                          momentum=momentum)

    print(network)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    def train(epoch):
        network.train()
        print("Train Epoch:", epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, batch_idx * len(data), len(train_loader.dataset),
                #            100. * batch_idx / len(train_loader), loss.item()))

                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
                torch.save(network.state_dict(), const.TASK3_MODEL_PATH)
                torch.save(optimizer.state_dict(), const.TASK3_OPTIMIZER_PATH)

    def test():
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output1 = network(data)
                test_loss += F.nll_loss(output1, target, size_average=False).item()
                pred = output1.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    test()
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()

    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()

    with torch.no_grad():
        output = network(example_data)

    fig = plt.figure()
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
    Greek_Letter()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
