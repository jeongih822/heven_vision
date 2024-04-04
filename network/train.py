#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import torch
import numpy as np
import random
import torch.nn as nn
import torchvision
from torchvision import transforms as tr
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
import time
import net  #the network architecture
import tqdm
import os
from matplotlib import pyplot as plt
import shutil
import argparse

def main(args):
    # Parse command-line arguments
    modelName = args.model
    numEpoch = args.epoch
    savingPeriod = args.savePeriod
    batchSize = args.batchSize
    learningRate = args.lr
    loss = args.loss

    print("... RUNNING ...")
    print(f"> modelName: {modelName}")
    print(f"> numEpoch: {numEpoch}")
    print(f"> savingPeriod: {savingPeriod}")
    print(f"> learningRate: {learningRate}")
    print(f"> batchSize: {batchSize}")
    print(f"> lossFunction: {loss}\n")

    numClass = 10
    randomSeed = 1234
    torch.manual_seed(randomSeed)
    np.random.seed(randomSeed)
    random.seed(randomSeed)

    # Set device (cuda or cpu) based on availability
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed(randomSeed)
        torch.cuda.manual_seed_all(randomSeed)
    else:
        device = "cpu"

    # Instantiate the neural network model
    network = net.NET()
    network.to(device)

    lossTrain = []
    lossValid = []
    epochList = []

    print("Device:", device)

    # Choose the appropriate loss function based on user input
    if loss == "L1":
        lossFunction = nn.L1Loss().to(device)
    elif loss == "L2":
        lossFunction = nn.MSELoss().to(device)
    elif loss == "XEntropy":
        lossFunction = nn.CrossEntropyLoss().to(device)
    else:
        print("unexpected Loss function")
        return

    # Set up the Adam optimizer
    optimizer = optim.Adam(network.parameters(), lr=learningRate)

    # Create a directory to save the model checkpoints
    os.makedirs(modelName, exist_ok=True)

    # Define image transformations
    trans = tr.Compose([tr.Grayscale(1), tr.ToTensor()])

    # Create training and validation datasets
    trainSet = torchvision.datasets.ImageFolder(root="trainingSet", transform=trans)
    validSet = torchvision.datasets.ImageFolder(root="validSet", transform=trans)

    print("# of Training Image:", len(trainSet))
    print("# of Validation Image:", len(validSet))

    # Create data loaders for training and validation datasets
    trainLoader = DataLoader(trainSet, batch_size=batchSize, shuffle=True, num_workers=4)
    validLoader = DataLoader(validSet, batch_size=batchSize, shuffle=True, num_workers=4)

    totalTime = 0
    bestValidLoss = 9999.0
    saveEpoch = 0

    # Training loop
    for epoch in range(numEpoch):
        start = time.time()
        msg = "Epoch " + str(epoch + 1) + "/" + str(numEpoch)

        network.train()
        trainLoss = 0.0
        numCorrect = 0
        total = 0

        # Iterate over batches in the training data
        for i, (images, labels) in enumerate(tqdm.tqdm(trainLoader, desc=msg)):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad() # Initialize gradient
            outputs = network(images) # input images to network
            lossBatch = lossFunction(outputs, labels) # Loss function
            lossBatch.backward() # Backpropagation
            optimizer.step() # Update weights
            trainLoss += lossBatch.item() * images.size(0)

            total += outputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            numCorrect += (predicted == labels).sum().item()

        trainAccuracy = numCorrect / total * 100

        end = time.time()
        totalTime += end - start
        averageTime = totalTime / (epoch + 1)

        print(
            "Avg. loss: %.6f, Training Acc.: %.2f%% // %.1fsec. Expected: %dh %dm."
            % (
                trainLoss / total,
                trainAccuracy,
                end - start,
                averageTime * (numEpoch - epoch - 1) / 3600,
                (averageTime * (numEpoch - epoch - 1) % 3600) / 60,
            )
        )

        if (epoch + 1) % savingPeriod == 0 or epoch == numEpoch - 1:
            network.eval()
            validLoss = 0.0
            numCorrect = 0
            totalV = 0

            # Validate the model on the validation set
            with torch.no_grad():
                for j, (images, labels) in enumerate(
                    tqdm.tqdm(validLoader, desc="Validation Set"), 1
                ):
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = network(images)
                    lossBatch = lossFunction(outputs, labels)
                    validLoss += lossBatch.item() * images.size(0)
                    totalV += outputs.size(0)

                    _, predicted = torch.max(outputs.data, 1)
                    numCorrect += (predicted == labels).sum().item()

                validAccuracy = numCorrect / totalV * 100

                # Save the model if the current validation loss is the best so far
                if validLoss / totalV < bestValidLoss:
                    bestValidLoss = validLoss / totalV
                    saveEpoch = epoch + 1
                    if saveEpoch < 10:
                        PATH = modelName + "/epoch000" + str(saveEpoch) + ".pth"
                    elif saveEpoch < 100:
                        PATH = modelName + "/epoch00" + str(saveEpoch) + ".pth"
                    elif saveEpoch < 1000:
                        PATH = modelName + "/epoch0" + str(saveEpoch) + ".pth"
                    else:
                        PATH = modelName + "/epoch" + str(saveEpoch) + ".pth"
                    torch.save(network.state_dict(), PATH)

                # Save the model at the last epoch
                if epoch + 1 == numEpoch:
                    PATH = modelName + "/lastEpoch.pth"
                    torch.save(network.state_dict(), PATH)

                print(
                    "=======[Valid] Avg. loss: %.6f, Valid Acc.: %.2f%% // Best: %.6f (Epoch %d).======="
                    % (validLoss / totalV, validAccuracy, bestValidLoss, saveEpoch)
                )

                # Record training and validation losses for plotting
                lossTrain.append(trainLoss / total)
                lossValid.append(validLoss / totalV)
                epochList.append(epoch + 1)

    print("Finish")

    # Plot the training and validation losses
    plt.title(modelName)
    plt.plot(epochList, lossTrain)
    plt.plot(epochList, lossValid)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Loss Train", "Loss Valid"])
    plt.savefig(modelName + "/" + modelName + ".png")

    # Write information about the training process to a text file
    f = open(modelName + "/Info.txt", "a")

    f.write("\n====================================================\n")
    data = "numEpoch: " + str(numEpoch) + "\n"
    f.write(data)
    data = "batchSize: " + str(batchSize) + "\n"
    f.write(data)
    data = "learningRate: " + str(learningRate) + "\n"
    f.write(data)
    data = "bestEpoch: " + str(saveEpoch) + "\n"
    f.write(data)
    data = "bestValidLoss: " + str(bestValidLoss) + "\n"
    f.write(data)
    f.write("====================================================\n")

    for i in range(len(epochList)):
        data = (
            str(epochList[i]) + "," + str(lossTrain[i]) + "," + str(lossValid[i]) + "\n"
        )
        f.write(data)

    f.close()

if __name__ == "__main__":
    # Parse command-line arguments and run the main function
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--savePeriod", type=int, default=5)
    parser.add_argument("--batchSize", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.00)
    parser.add_argument("--model", type=str, default="model")
    parser.add_argument("--loss", type=str, default="XEntropy")

    args = parser.parse_args()

    main(args)
