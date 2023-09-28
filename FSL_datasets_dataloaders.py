# pytorch modules
import torch 
from torch.utils.data import WeightedRandomSampler, DataLoader # wraps the data so its iterable
from torch import nn # nn class our model inherits from
import torch.optim as optim
from torchmetrics import Accuracy

import tqdm

import FSL_models as fsl

#############################
### saving function
#############################
# Function to save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)


#############################
### training and testing simpleshot
#############################

# test accuracy function
# test accuracy function
def testAccuracy(model, dataloader, device):
    
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in dataloader:
            crops, labels = data
            total += labels.size(0)
            crops, labels = crops.to(device), labels.to(device)
            # run the model on the test set to predict labels
            outputs = model(crops.float())
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            labels = labels# torch.max(labels.data, 1)
            accuracy += (predicted == labels).sum().item()
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    
    return(accuracy)

def train(model, dataloader_train, dataloader_test, loss_, num_epochs, path, device, optimizer):
    # define lists to store accuracy gain as we train
    train_acc_gain = []
    test_acc_gain = []
    
    best_accuracy = 0
    best_loss = float('inf')

    model = model.to(torch.float)
    model.train()
    # Iterate over the training data
    for epoch in range(num_epochs):
        running_train_loss = 0.0
        running_test_loss = 0.0
        
        # train the model
        model.train()
        for i, (crops, labels) in enumerate(dataloader_train):
            # Get the crops and labels
            crops, labels = crops.to(device), labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # get prediction
            outputs = model(crops.float())
            loss = loss_(outputs.double(), labels)
            running_train_loss += loss.item()
        
            # Backward pass
            loss.backward()
            optimizer.step()

        accuracy = testAccuracy(model, dataloader_train, device)
        print('epoch', epoch, 'train accuracy over whole train set: %d %%' % (accuracy))
            
        # save the model if the accuracy is the best
        #if accuracy > best_accuracy:
        #    save_model(model, path)
        #    best_accuracy = accuracy
                
        # get the test accuracy
        model.eval()
        for i, (crops, labels) in enumerate(dataloader_test):
            # Get the crops and labels
            crops, labels = crops.to(device), labels.to(device)
            # get prediction and loss
            pred = model(crops.float())
            loss = loss_(pred.double(), labels)
            
            running_test_loss += loss.item()
            
        accuracy = testAccuracy(model, dataloader_test, device)
        print('epoch', epoch, 'test accuracy over whole test set: %d %%' % (accuracy))

        # save the model if the accuracy is the best
        if accuracy > best_accuracy:
            print('Saving model from epoch', epoch)
            save_model(model, path)
            best_accuracy = accuracy
        elif accuracy == best_accuracy:
            if running_test_loss<best_loss:
                print('Saving model from epoch', epoch)
                save_model(model, path)
                best_loss = running_test_loss
        
        

        print('Epoch: %d loss: %.3f' % (epoch + 1, running_test_loss / len(dataloader_test)))


#############################
### training and testing proto/match/relation nets
#############################

def train_evaluate(fslNet, file_name, train_loader, test_loader, epochs = 1, onehot=True):
    # define the FSL
    learner = fsl.FewShotLearner(fslNet)

    # train
    trainer = pl.Trainer(accelerator="gpu", devices = 1, max_epochs=epochs)
    trainer.fit(learner, train_loader, val_dataloaders=test_loader)

    # save
    save_model(learner, file_name + '.pth')

    # evaluate
    learner.eval()
    learner = learner.to(DEVICE)
    # instantiate the accuracy metric
    metric = Accuracy(task = 'multiclass', num_classes=n_way).to(DEVICE)
    # collect all the embeddings in the test set
    # so we can plot them later
    embedding_table = []
    pbar = tqdm.tqdm(range(len(test_episodes)))
    for episode_idx in pbar:
        support, query = test_episodes[episode_idx]

    # get the embeddings
    logits = learner.FSLnet(query, support)
    if not onehot:
        logits = torch.round(logits)
    #print(query['target'].device)
    # compute the accuracy
    acc = metric(logits, query["target"].to(DEVICE))
    pbar.set_description(f"Episode {episode_idx} // Accuracy: {acc.item():.2f}")
    # compute the total accuracy across all episodes
    total_acc = metric.compute()
    print(f"Total accuracy, averaged across all episodes: {total_acc}")

    return learner, metric, total_acc

def evaluate(learner, ds, type_net, n_way, n_support, device, embeddings=True):
    # type_net should be one of 'protonet', 'matchnet', 'relnet', 'simple shot', or 'knn_net'
    # evaluate
    learner.eval()
    learner = learner.to(device)
    # instantiate the accuracy metric
    metric = Accuracy(task = 'multiclass', num_classes=n_way).to(device)
    # collect all the embeddings in the test set
    # so we can plot them later
    embedding_table = []
    pbar = tqdm.tqdm(range(len(ds)))
    for episode_idx in pbar:
        support, query = ds[episode_idx]
    
        if type_net == 'simpleshot':
            # get the predictions
            logits = learner.FSLnet(query, support, n_way)
        elif type_net=='knn_net':
            logits = learner.FSLnet(query, support, neighbors=n_support, embeddings = embeddings)
        else:
            logits = learner.FSLnet(query, support)
        
        #print(query['target'].device)
        # compute the accuracy
        #print(logits.shape, query["target"].shape)
        acc = metric(logits, query["target"].to(device))
        pbar.set_description(f"Episode {episode_idx} // Accuracy: {acc.item():.2f}")
        # compute the total accuracy across all episodes
        total_acc = metric.compute()
    print(f"Total accuracy, averaged across all episodes: {total_acc}")

    return learner, metric, total_acc
