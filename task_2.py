import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib as mpl
import time
from sklearn.metrics import confusion_matrix
import const

mpl.use('TkAgg')


def MINIST_Fashion(n_epochs, batch_size_train, dropout_rate, learning_rate, num_filters_1, num_filters_2,
                   is_3_layers=False):
    # setting up hyper parameters
    batch_size_test = 1000
    momentum = 0.5
    log_interval = 10

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # loading dataset
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST(const.DATA_ROOT, train=True, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize(
                                                  (0.5,), (0.5,))
                                          ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST(const.DATA_ROOT, train=False, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize(
                                                  (0.5,), (0.5,))
                                          ])),
        batch_size=batch_size_test, shuffle=True)

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    print(example_data.shape)

    # Network with 2 layers
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, num_filters_1, kernel_size=5)
            self.conv2 = nn.Conv2d(num_filters_1, num_filters_2, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(num_filters_2 * 16, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, num_filters_2 * 16)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, p=dropout_rate, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x)

    # Network with 3 layers
    class Net2(nn.Module):
        def __init__(self):
            super(Net2, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=3)  # 28-3+1 = 26 * 26 * 10
            self.conv2 = nn.Conv2d(10, 20, kernel_size=2)  # 13-2+1 = 12 * 12 * 20
            self.conv3 = nn.Conv2d(20, 30, kernel_size=3)  # 6-3+1 = 4*4*30
            self.conv3_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(480, 50)  # 4*4*30
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))  # 26/2 = 13 * 13 * 10
            x = F.relu(F.max_pool2d(self.conv2(x), 2))  # 12/2 = 6 * 6 * 20
            x = F.relu(self.conv3_drop(self.conv3(x)))  # 4*4*30
            x = x.view(-1, 480)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x)

    network = Net()
    if is_3_layers:
        network = Net2()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                          momentum=momentum)

    # train the model
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
    time_start = time.time()

    def train(epoch):
        network.train()
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
                torch.save(network.state_dict(), const.TASK2_MODEL_PATH)
                torch.save(optimizer.state_dict(), const.TASK2_OPTIMIZER_PATH)

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
        time_stop = time.time()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        cur_accuracy = torch.round(100. * correct / len(test_loader.dataset))

        # evaluate the model by printing confusion matrix
        y_true = []
        y_pred = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = network(images)
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        cm = confusion_matrix(y_true, y_pred)
        print(cm)

        # evaluate the model by printing time used
        cur_time_spent = round(time_stop - time_start)
        print(f"time spent: {cur_time_spent} s")
        return cur_accuracy, cur_time_spent

    test()
    accuracy = 0
    time_spent = 0
    for epoch in range(1, n_epochs + 1):
        print("Epoch times with 2 layers: ", epoch)
        train(epoch)
        accuracy, time_spent = test()
        print()

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
        plt.imshow(example_data[i + 20][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
            output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    return accuracy, time_spent


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    DEFAULT_N_EPOCH = 3
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_DROPOUT_RATE = 0.5
    DEFAULT_LEARNING_RATE = 0.01
    DEFAULT_NUM_FILTERS_1 = 10
    DEFAULT_NUM_FILTERS_2 = 20

    accuracy, time_spent = MINIST_Fashion(DEFAULT_N_EPOCH, DEFAULT_BATCH_SIZE, DEFAULT_DROPOUT_RATE,
                                          DEFAULT_LEARNING_RATE,
                                          DEFAULT_NUM_FILTERS_1, DEFAULT_NUM_FILTERS_2)
    print(f"The result for using all default values, accuracy is: {accuracy}%, time spent is: {time_spent}")

    # try number of epochs = 20 to find out the optimal number of epoch.
    MINIST_Fashion(20, DEFAULT_BATCH_SIZE, DEFAULT_DROPOUT_RATE, DEFAULT_LEARNING_RATE, DEFAULT_NUM_FILTERS_1,
                   DEFAULT_NUM_FILTERS_2)

    # try different drop out rate to find the optimal dropout rate
    best_dropout_rate = DEFAULT_DROPOUT_RATE
    best_accuracy = 0.00
    for dropout_rate in [0.01, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5]:
        accuracy, time_spent = MINIST_Fashion(DEFAULT_N_EPOCH, DEFAULT_BATCH_SIZE, dropout_rate, DEFAULT_LEARNING_RATE,
                                              DEFAULT_NUM_FILTERS_1, DEFAULT_NUM_FILTERS_2)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_dropout_rate = dropout_rate
    print(f"the best dropout rate is: {best_dropout_rate}, and the accuracy is: {best_accuracy}%")

    # try different learning rate to find the optimal
    best_learning_rate = DEFAULT_LEARNING_RATE
    best_accuracy = 0
    for learning_rate in range(1, 15):
        learning_rate = learning_rate * 0.01
        accuracy, time_spent = MINIST_Fashion(DEFAULT_N_EPOCH, DEFAULT_BATCH_SIZE, DEFAULT_DROPOUT_RATE, learning_rate,
                                              DEFAULT_NUM_FILTERS_1, DEFAULT_NUM_FILTERS_2)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_learning_rate = learning_rate
    print(f"the best learning rate is: {best_learning_rate}, and the accuracy is: {best_accuracy}%")

    # try different batch size to the find the optimal
    best_batch_size = DEFAULT_BATCH_SIZE
    batch_size = DEFAULT_BATCH_SIZE
    accuracy, time_spent = MINIST_Fashion(DEFAULT_N_EPOCH, batch_size, DEFAULT_DROPOUT_RATE, DEFAULT_LEARNING_RATE,
                                          DEFAULT_NUM_FILTERS_1, DEFAULT_NUM_FILTERS_2)
    prev_time_spent = time_spent
    prev_accuracy = accuracy
    while True:
        batch_size = batch_size * 2
        try:
            accuracy, time_spent = MINIST_Fashion(DEFAULT_N_EPOCH, batch_size, DEFAULT_DROPOUT_RATE,
                                                  DEFAULT_LEARNING_RATE, DEFAULT_NUM_FILTERS_1, DEFAULT_NUM_FILTERS_2)
            if (prev_time_spent - time_spent) > 2 and accuracy >= prev_accuracy:
                prev_time_spent = time_spent
                best_batch_size = batch_size
                prev_accuracy = accuracy
            else:
                break
        except:
            break
    print(f"the best batch size is: {best_batch_size}")

    # try different number of filters
    best_num_filters_1 = DEFAULT_NUM_FILTERS_1
    best_num_filters_2 = DEFAULT_NUM_FILTERS_2
    num_filters_1 = DEFAULT_NUM_FILTERS_1 / 2
    num_filters_2 = DEFAULT_NUM_FILTERS_2 / 2
    best_accuracy = 0
    time_spent = 0
    for i in range(1, 10):
        num_filters_1 = int(num_filters_1 * 2)
        num_filters_2 = int(num_filters_2 * 2)
        accuracy, time_spent = MINIST_Fashion(DEFAULT_N_EPOCH, DEFAULT_BATCH_SIZE, DEFAULT_DROPOUT_RATE,
                                              DEFAULT_LEARNING_RATE, num_filters_1, num_filters_2)
        if time_spent > 300:
            break
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_num_filters_1 = num_filters_1
            best_num_filters_2 = num_filters_2
    print(
        f"the best number of filter is: {best_num_filters_1} and {best_num_filters_2}, and the best accuracy is {best_accuracy}%")

    # try different learning rate with the optimal dropout rate
    best_learning_rate_2 = best_learning_rate
    best_accuracy = 0
    for learning_rate in range(1, 15):
        learning_rate = learning_rate * 0.01
        accuracy, time_spent = MINIST_Fashion(DEFAULT_N_EPOCH, DEFAULT_BATCH_SIZE, best_dropout_rate, learning_rate,
                                              DEFAULT_NUM_FILTERS_1, DEFAULT_NUM_FILTERS_2)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_learning_rate_2 = learning_rate
    print(
        f"the best learning rate with optimal dropout rate is: {best_learning_rate_2}, and the accuracy is: {best_accuracy}%")

    # try different number of filters using optimal dropout rate and best learning rate
    best_num_filters_1_2 = best_num_filters_1
    best_num_filters_2_2 = best_num_filters_2
    num_filters_1_2 = DEFAULT_NUM_FILTERS_1 / 2
    num_filters_2_2 = DEFAULT_NUM_FILTERS_2 / 2
    best_accuracy = 0
    time_spent = 0
    for i in range(1, 10):
        num_filters_1_2 = int(num_filters_1_2 * 2)
        num_filters_2_2 = int(num_filters_2_2 * 2)
        accuracy, time_spent = MINIST_Fashion(DEFAULT_N_EPOCH, DEFAULT_BATCH_SIZE, best_dropout_rate,
                                              best_learning_rate_2, num_filters_1_2, num_filters_2_2)
        if time_spent > 300:
            break
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_num_filters_1_2 = num_filters_1_2
            best_num_filters_2_2 = num_filters_2_2
    print(
        f"the best number of filter with optimal dropout rate and learning rate is: {best_num_filters_1_2} and {best_num_filters_2_2}, and the best accuracy is {best_accuracy}%")

    # try all best dimensions
    accuracy, time_spent = MINIST_Fashion(15, best_batch_size, best_dropout_rate, best_learning_rate_2,
                                          best_num_filters_1_2, best_num_filters_2_2)
    print(f"the best combined result: accuracy is {accuracy}, and time spent is {time_spent}s")

    # try all best dimensions with one more layer
    accuracy, time_spent = MINIST_Fashion(15, 128, 0.001, 0.11, 40, 80, True)
    print(f"the best combined result with 3 layers: accuracy is {accuracy}, and time spent is {time_spent}s")
