import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math

# pytorch modules
import torch
import torch.nn as nn 
from torchmetrics import Accuracy
import pytorch_lightning as pl
import tqdm
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchvision import transforms

# debugging module
from icecream import ic


###################################
### embedding network which is kept the same for every FSL network
### we use a standard conv4-net 
###################################

def init_layer(L):
    """
    Initialize the weights of a layer using fan-in initialization.

    Args:
        L (nn.Module): The layer to initialize.

    Returns:
        None
    """
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class ConvBlock(nn.Module):
    """
    Convolutional block module used in the embedding network.

    Args:
        indim (int): Number of input channels.
        outdim (int): Number of output channels.
        pool (bool): Whether to apply max pooling after convolution. Default is True.
        padding (int): Amount of padding. Default is 1.

    Attributes:
        indim (int): Number of input channels.
        outdim (int): Number of output channels.
        C (nn.Conv2d): Convolutional layer.
        BN (nn.BatchNorm2d): Batch normalization layer.
        relu (nn.ReLU): ReLU activation layer.
        parametrized_layers (list): List of layers with learnable parameters.
        pool (nn.MaxPool2d): Max pooling layer.
        trunk (nn.Sequential): Sequential container for the layers.

    Methods:
        forward(x): Forward pass through the block.

    """
    def __init__(self, indim, outdim, pool=True, padding=1):
        super(ConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
        self.BN = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]

        if pool:
            self.pool = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)


    def forward(self, x):
        """
        Forward pass through the ConvBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.trunk(x)
        return out

class ConvNet(nn.Module):
    """
    Convolutional network module used in the embedding network.

    Attributes:
        trunk (nn.Sequential): Sequential container for the ConvBlocks.

    Methods:
        forward(x): Forward pass through the ConvNet.

    """
    def __init__(self):
        super(ConvNet, self).__init__()
        trunk = []
        for i in range(4):
            indim = 2 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool=True)
            trunk.append(B)
        trunk.append(nn.Flatten())

        self.trunk = nn.Sequential(*trunk)


    def forward(self, x):
        """
        Forward pass through the ConvNet.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.trunk(x)
        return out

###################################
### embedding network which is kept the same for every FSL network 
###################################
class EmbeddingNetwork(nn.Module):
    """
    Embedding network module.

    Args:
        channels (int): Number of input channels.
        crop_size (int): Size of the input image after cropping.

    Attributes:
        convolutional_relu_stack (nn.Sequential): Sequential container for the convolutional layers and ReLU activations.
        linear_relu_stack (nn.Sequential): Sequential container for the linear layers and ReLU activations.

    Methods:
        forward(x): Forward pass through the EmbeddingNetwork.

    """
    def __init__(self, channels, crop_size):
        super(EmbeddingNetwork, self).__init__()
        fc_nodes = 100
        dropout = 0.2
        self.convolutional_relu_stack = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=1, padding='valid'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding='valid'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding='valid'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=(crop_size-6)**2 * 128, out_features=fc_nodes),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(fc_nodes),
        )

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(fc_nodes, fc_nodes),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(fc_nodes)
        )

    def forward(self, x):
        """
        Forward pass through the EmbeddingNetwork.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.convolutional_relu_stack(x)
        for i in range(2):
            x = self.linear_relu_stack(x)

        return x
    
###################################
### Matching network implementation
###################################
class MatchingNetwork(nn.Module):
    """
    Matching network module.

    Args:
        device (torch.device): Device to run the module on.

    Attributes:
        embedding_layer (ConvNet): Embedding network.
        cos_dist (nn.CosineSimilarity): Cosine similarity module.
        softmax (nn.Softmax): Softmax activation module.
        device (torch.device): Device to run the module on.

    Methods:
        forward(query, support): Forward pass through the MatchingNetwork.

    """
    def __init__(self, device):
        super(MatchingNetwork, self).__init__()
        self.embedding_layer = ConvNet()
        self.cos_dist = nn.CosineSimilarity(dim=1)
        self.softmax = nn.Softmax(dim=0)
        self.device = device

    def forward(self, query, support):
        """
        Forward pass through the MatchingNetwork.

        Args:
            query (dict): Dictionary containing the query data.
            support (dict): Dictionary containing the support data.

        Returns:
            torch.Tensor: Output tensor.
        """
        support["embeddings"] = self.embedding_layer(support["image"])
        query["embeddings"] = self.embedding_layer(query["image"])

        cos_distances = []
        for embedding in support["embeddings"]:
            cos_distances.append(torch.exp(self.cos_dist(query["embeddings"], embedding)))

        cos_distances = torch.stack(cos_distances).squeeze(-1)
        attentions = self.softmax(cos_distances)

        support["attentions"] = attentions

        y = torch.matmul(support["attentions"].T, torch.nn.functional.one_hot(support["target"]).float().to(self.device))

        return y

###################################
### Prototypical network implementation
###################################
class PrototypicalNetwork(nn.Module):
    """
    Prototypical network module.

    Attributes:
        embedding_layer (ConvNet): Embedding network.

    Methods:
        forward(query, support): Forward pass through the PrototypicalNetwork.

    """
    def __init__(self):
        super(PrototypicalNetwork, self).__init__()
        self.embedding_layer = ConvNet()

    def forward(self, query, support):
        """
        Forward pass through the PrototypicalNetwork.

        Args:
            query (dict): Dictionary containing the query data.
            support (dict): Dictionary containing the support data.

        Returns:
            torch.Tensor: Output tensor.
        """
        support["embeddings"] = self.embedding_layer(support["image"])
        query["embeddings"] = self.embedding_layer(query["image"])

        support_embeds = []
        for idx in range(len(support["classlist"])):
            embeds = support["embeddings"][support["target"] == idx]
            support_embeds.append(embeds)

        support_embeds = torch.stack(support_embeds)
        prototypes = support_embeds.mean(dim=1)
        support["prototypes"] = prototypes

        distances = torch.cdist(query["embeddings"].unsqueeze(0), prototypes.unsqueeze(0), p=2).squeeze(0)
        distances = distances ** 2

        logits = -distances

        return logits

###################################
### Relation network implementation
###################################
class RelationNetwork(nn.Module):
    """
    Relation network module.

    Attributes:
        embedding_layer (ConvNet): Embedding network.
        fc_nodes (int): Number of nodes in the fully connected layers.
        relation_module (nn.Sequential): Sequential container for the relation module layers.

    Methods:
        forward(query, support): Forward pass through the RelationNetwork.

    """
    def __init__(self):
        super(RelationNetwork, self).__init__()
        self.embedding_layer = ConvNet()
        self.fc_nodes = 100
        self.relation_module = nn.Sequential(
            nn.Linear(512, self.fc_nodes),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.BatchNorm1d(self.fc_nodes),
            nn.Linear(self.fc_nodes, self.fc_nodes),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.BatchNorm1d(self.fc_nodes),
            nn.Linear(self.fc_nodes, 1),
            nn.Dropout(0.2),
            nn.ReLU(),
        )

    def forward(self, query, support):
        """
        Forward pass through the RelationNetwork.

        Args:
            query (dict): Dictionary containing the query data.
            support (dict): Dictionary containing the support data.

        Returns:
            torch.Tensor: Output tensor.
        """
        support["embeddings"] = self.embedding_layer(support["image"])
        query["embeddings"] = self.embedding_layer(query["image"])

        support_embeds = []
        for idx in range(len(support["classlist"])):
            embeds = support["embeddings"][support["target"] == idx]
            support_embeds.append(embeds)

        support_embeds = torch.stack(support_embeds)
        sums = support_embeds.sum(dim=1)
        support["sums"] = sums / torch.sum(sums)

        relation_scores = {}
        for qvector in query['embeddings']:
            relation_scores[qvector] = []
            concats = []
            for svector in sums:
                concat = torch.cat((qvector, svector))
                concats.append(concat)
            relation_scores[qvector] = self.relation_module(torch.stack(concats)).squeeze(1)

        fin_rel_scores = torch.stack([rel_score for rel_score in relation_scores.values()])

        return fin_rel_scores

###################################
### pytorch lightning network which trains using episodic training
### it can take any of the three networks above (matching, prototypical, or relation)
### as the FS network
###################################
class FewShotLearner(pl.LightningModule):

    def __init__(self,
        FSLnet: nn.Module,
        learning_rate: float = 0.001,
        n_way: int = 5
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['matchnet'])
        self.FSLnet = FSLnet
        self.learning_rate = learning_rate

        self.loss = nn.CrossEntropyLoss()
        self.metrics = nn.ModuleDict({
            'accuracy': Accuracy(task="multiclass", num_classes=n_way)
        })

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def step(self, batch, batch_idx, tag: str):
        support, query = batch

        logits = self.FSLnet(query, support)
        loss = self.loss(logits, query["target"])

        output = {"loss": loss}
        for k, metric in self.metrics.items():
            output[k] = metric(logits, query["target"])

        for k, v in output.items():
            self.log(f"{k}/{tag}", v)
        return output

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test")
    
###################################
### simple shot classifier
###################################
class SimpleShotEmbed(nn.Module):
    def __init__(self, classes, crop_size=20):
        super().__init__()
        self.embedding_layer = ConvNet()
        if crop_size == 20:
            self.dense = nn.Linear(256, classes)
        if crop_size == 40:
            self.dense = nn.Linear(256, classes)

    def forward(self, x, training = True):
        x = self.embedding_layer(x)
        if training == True:
            x = self.dense(x)
            logits = torch.nn.functional.softmax(x,dim=1)
            return logits
        else: 
            x = torch.nn.functional.normalize(x)
            return x

###################################
### simple shot fsl network
###################################
class SimpleShot(nn.Module):
    '''
    This class implements a simpleshot FSL network. It takes a model as input and uses it to compute
    embeddings for the query and support sets. It then computes the euclidean distance between the
    query embeddings and the average support embeddings for each class. It then returns the class
    with the smallest distance.

    '''
    def __init__(self, model):#, channels, crop_size, n_outputs, fc_layers, fc_nodes, dropout):
        super().__init__()
        self.classifier = model
        

    def forward(self, query, support, n_way):
        # compute embeddings for query and support sets
        support["embeddings"] = self.classifier(support["image"], training=False)
        query["embeddings"] = self.classifier(query["image"], training=False)
        
        # make an average feature vector for each class
        average_support_vecs = []
        for i in range(n_way):
            average_support_vecs.append( support["embeddings"][support["target"]==i,:].mean(dim=0) )
        
        average_support_vecs = torch.stack(average_support_vecs)
        
        # L2 normalise output
        support["embeddings"] = torch.nn.functional.normalize(support["embeddings"])
        average_support_vecs = torch.nn.functional.normalize(average_support_vecs)
        # support["embeddings"] i of shape (n_query, dimension of embedding vector)
        # average_support_vecs is of shape (n_way, dimension of embedding vector)
        
        # find euclidean distances between them
        distances = torch.cdist(query["embeddings"], average_support_vecs, p=2).squeeze(0)     
        # distances is a tensor of dimensions (n_query, n_ways)
        distances = distances ** 2
        
        logits = torch.argmin(distances, dim=1)
        
        support["embeddings_norm"] = torch.nn.functional.normalize(support["embeddings"]) 
        query["embeddings_norm"] = torch.nn.functional.normalize(query["embeddings"]) 
        
        return logits #, distances
    

class knnNet(nn.Module):    
    '''
    '''
    def __init__(self, device):
        super().__init__()
        self.classifier = simpleShotembed
        self.device = device

    def forward(self, query, support, neighbors=1, embeddings=True):
        KNN = KNeighborsClassifier(n_neighbors=neighbors)
        
        if embeddings:
            # compute embeddings for query and support sets
            support["embeddings"] = self.classifier(support["image"], training=False)
            query["embeddings"] = self.classifier(query["image"], training=False)
        
            KNN.fit(support["embeddings"].detach().numpy(), support["target"].detach().numpy())
            y = KNN.predict(query["embeddings"].detach().numpy())
        else:
            KNN.fit(torch.flatten(support["image"], start_dim=1, end_dim=-1).detach().numpy(), support["target"].detach().numpy() )
            y = KNN.predict(torch.flatten(query["image"], start_dim=1, end_dim=-1).detach().numpy())
            
        return torch.tensor(y).float().to(self.device)
        
###################################
### UNet implementation
###################################

class DoubleConv(nn.Module):
    """
    Double convolution that does not change the resolution.
    Args:
        n_channels1: number of channels in the input
        n_channels2: number of channels after first conv layer
        n_channels3: number of channels after second conv layer
    
    Methods:
        forward: applies the double conv.
    """

    def __init__(self, n_channels1, n_channels2, n_channels3):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(n_channels1, n_channels2, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(n_channels2, eps=0.001, momentum = 0.99),
            nn.Conv2d(n_channels2, n_channels3, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(n_channels3, eps=0.001, momentum = 0.99))

    def forward(self, x):
        return self.double_conv(x)

class downsample(nn.Module):
  """
  A double convolution followed by a maxpooling layer to downsample.
    Args:
    n_channels1: number of channels in the input
    n_channels2: number of channels after first conv layer
    n_channels3: number of channels after second conv layer
  """
  def __init__(self, n_channels1, n_channels2, n_channels3):
      super().__init__()
      self.doubleconvDown = DoubleConv(n_channels1, n_channels2, n_channels3)


  def forward(self, x):
      skip = self.doubleconvDown(x)
      y = nn.MaxPool2d(2)(skip)
      y = nn.Dropout(0.3)(y)
      return skip, y

class upsample(nn.Module):
  """
  A transpose convolution (to upsample) followed by a double convolution.
  Args:
    n_channels1: number of channels in the input
    n_channels2: number of channels after first conv layer
    n_channels3: number of channels after second conv layer
  """
  def __init__(self, n_channels1, n_channels2, n_channels3):
    super().__init__()
    self.convTranspose = nn.Sequential(nn.ConvTranspose2d(n_channels1, n_channels2, 2 , stride=2))

    self.doubleConv = DoubleConv(n_channels1, n_channels2, n_channels3) #n_channels2 is first because it's concatenated with skip

  def forward(self, skip, x):
    # upsample
    x = self.convTranspose(x)
    # concatenate with the skip
    x = torch.concatenate([x, skip], axis=1)
    # dropout
    x = nn.Dropout(0.3)(x)
    # double convolution
    x = self.doubleConv(x)
    return x

class UNet(nn.Module):
    """
    A fully convolutional network that outputs a binary map.
    """
    def __init__(self):
        super().__init__()
        # downsample 1
        self.downsample1 = downsample(1, 32, 32)
        # downsample 2
        self.downsample2 = downsample(32, 64, 64)
        # downsample 3
        self.downsample3 = downsample(64, 128, 128)

        # bottleneck
        self.bottleneck = DoubleConv(128,256,256)
    
        # upsample 1
        self.upsample1 = upsample(256, 128, 128)
        # upsample 2
        self.upsample2 = upsample(128, 64, 64)
        # upsample 3
        self.upsample3 = upsample(64, 32, 32)

        # final layer
        self.output = nn.Sequential(nn.Conv2d(32,2, kernel_size=1, padding="same"))

    def forward(self, x):
        # downsample 1
        skip1, x1 = self.downsample1(x)
        # downsample 2
        skip2, x2 = self.downsample2(x1)
        # downsample 3
        skip3, x3 = self.downsample3(x2)
        # bottleneck
        bottleneck = self.bottleneck(x3)
        # upsample 1
        x4 = self.upsample1(skip3, bottleneck)
        # upsample 2
        x5 = self.upsample2(skip2, x4)
        # upsample 3
        x6 = self.upsample3(skip1, x5)

        outputs = self.output(x6)

        return outputs

###################################
