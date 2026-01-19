import math

# pytorch modules
import torch
import torch.nn as nn # nn class our model inherits from
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torchmetrics import Accuracy
import pytorch_lightning as pl

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class ConvBlock(nn.Module):
    '''
    A convolutional block consisting of a convolutional layer,
    batch normalization, ReLU activation, and optional max pooling.
    '''
    def __init__(self, indim, outdim, pool = True, padding = 1):
        super(ConvBlock, self).__init__()
        self.indim  = indim
        self.outdim = outdim
        self.C      = nn.Conv2d(indim, outdim, 3, padding= padding)
        self.BN     = nn.BatchNorm2d(outdim)
        self.relu   = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]

        self.pool   = nn.MaxPool2d(2)
        self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)


    def forward(self,x):
        out = self.trunk(x)
        return out

class ConvNet(nn.Module):
    '''
    The Conv4 backbone network used for few-shot learning. 
    '''
    def __init__(self):
        super(ConvNet,self).__init__()
        trunk = []
        for i in range(4):
            indim = 2 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = True)
            trunk.append(B)
        trunk.append(nn.Flatten())

        self.trunk = nn.Sequential(*trunk)


    def forward(self,x):
        out = self.trunk(x)
        return out

class ResNet18Embedding(nn.Module):
    '''
    A ResNet18 embedding layer adapted for 2-channel input and
    pretrained on ImageNet.
    '''
    def __init__(self, input_channels=2):
        super().__init__()        
        # Load pretrained model
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)


        # 0. Input Normalization
        # Since we can't use ImageNet stats for 2-channel STM data, we use a 
        # BatchNorm layer to learn the normalization parameters (mean/std) from the data.
        self.input_norm = nn.BatchNorm2d(input_channels)

        # 1. Handle Input Channels
        # ConvNet uses indim=2. ResNet expects 3. We modify the first layer.
        if input_channels != 3:
            old_conv = self.resnet.conv1
            self.resnet.conv1 = nn.Conv2d(
                input_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias
            )

        # 2. Remove Classification Head to get embeddings (512 dim)
        self.resnet.fc = nn.Identity()

        # 3. Freeze all weights
        for param in self.resnet.parameters():
            param.requires_grad = False

        # 4. Unfreeze the last few layers for fine-tuning
        # Unfreeze the modified input layer
        if input_channels != 3:
            for param in self.resnet.conv1.parameters():
                param.requires_grad = True
        
        # Unfreeze the input normalization layer so it can learn
        for param in self.input_norm.parameters():
            param.requires_grad = True
        
        # Unfreeze the last residual block (layer4)
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

    def forward(self, x):
        # Upsample to 224x224 as ResNet was trained on this resolution
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Normalize input
        x = self.input_norm(x)
        
        return self.resnet(x)

def get_resnet18_embedding_layer():
    """
    Returns a pretrained ResNet18 embedding layer with frozen weights 
    (except the last block) adapted for 2-channel input.
    """
    return ResNet18Embedding(input_channels=2)

class MatchingNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # define the embedding layer

        self.embedding_layer = ConvNet()

       # self.embedding_layer = EmbeddingNetwork(channels, crop_size)
        self.cos_dist = nn.CosineSimilarity(dim=1)
        self.softmax = nn.Softmax(dim=0)
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, query, support):
        # compute embeddings for query and support sets
        support["embeddings"] = self.embedding_layer(support["image"]) # f(x)
       # print(support['embeddings'].shape)
        query["embeddings"] = self.embedding_layer(query["image"]) # g(x_i), for us g = f


        # compute the cosine distances between the query embeddings and the support
        # query['embeddings'] is a tensor of shape (n_samples, dimensions of embedding vector space)
        cos_distances = []
        for embedding in support["embeddings"]:
          cos_distances.append(torch.exp(self.cos_dist(query["embeddings"], embedding)))
         # cos_distances.append(torch.cdist(query["embeddings"].unsqueeze(0), embedding.unsqueeze(0), p=2).squeeze(0)) # c(f(x), g(x_i))
        '''
        # support["prototypes"] is a tensor of shape
        # (n_way, dimensions of embedding vector space)

        # compute the distances between the query embeddings and the prototypes
        # query['embeddings'] is a tensor of shape (n_samples, dimensions of embedding vector space)
        distances = torch.cdist(query["embeddings"].unsqueeze(0), prototypes.unsqueeze(0), p=2).squeeze(0)
        '''

        cos_distances = torch.stack(cos_distances).squeeze(-1) # c(f(x),g(x_i))
        # cos_distances is of shape (n_support, n_query). We have a cosine distance vector between each
        # of the support embeddings and the query embeddings and then we take the exponential of it.
        attentions = self.softmax(cos_distances)

        support["attentions"] = attentions # a = e^c(f(x),g(x_i))/sum_(j=1)^k e^c(f(x),g(x_j))

        # output using integer labels
      #  y = torch.matmul(support["target"].float().to(DEVICE), support["attentions"]).float()

        # output using one hot encoding for targets (got better accuracy)
        y = torch.matmul( support["attentions"].T, torch.nn.functional.one_hot(support["target"]).float().to(self.DEVICE) )

        # the final predictions should be (where we use einstein summation convention):
        # y = a(x,x_i)y_i. With a(x,x_i) = e^{c(f(x),g(x_i))}/sum_{j=1}^{k}e^{c(f(x),g(x_j))}

        return y

class RelationNetwork(nn.Module):

    def __init__(self, res):
        super().__init__()
        # define the embedding layer
        self.embedding_layer = ConvNet()

        self.fc_nodes = 100

        # the embedding vectors are of size
        if res==20:
          start = 128
        elif res==40:
          start = 512

        self.relation_module = nn.Sequential(
                               nn.Linear(start, self.fc_nodes),
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
        # compute embeddings for query and support sets
        # input is a (num_channels, res, res)
        support["embeddings"] = self.embedding_layer(support["image"]) # f(x)
        query["embeddings"] = self.embedding_layer(query["image"]) # g(x_i), for us g = f

        # sum up the embeddings of the support vectors in the same class
        support_embeds = []
        for idx in range(len(support["classlist"])):
            embeds = support["embeddings"][support["target"] == idx]
            support_embeds.append(embeds)

        # support_embeds is a list of torch tensors of shape
        # (n_support, dimensions of embedding vector space)

        support_embeds = torch.stack(support_embeds)
        # support_embeds now a tensor of shape
        # (n_way, n_support, dimensions of embedding vector space)

        # we compute the sums of these support vectors
        # sums has shape (n_way, dimensions of embedding vector)
        sums = support_embeds.sum(dim=1)
        support["sums"] = sums/torch.sum(sums)

        relation_scores = {}
        for qvector in query['embeddings']:
            # qvector.shape = (dim_emb)
            relation_scores[qvector] = []
            concats = []
            for svector in sums:
                # svector.shape = (dim_emb)
                concat = torch.cat((qvector,svector))
                # concat.shape = (2*dim_emb)
                concats.append(concat)
            relation_scores[qvector] = self.relation_module(torch.stack(concats)).squeeze(1)
            # relation_scores[qvector].shape = (n_way)

        # relation_scores is a dictionary that has the query vectors as keys and their relation scores as values
        fin_rel_scores = torch.stack([rel_score for rel_score in relation_scores.values()])
        # fin_rel_scores.shape = (n_way*n_query, n_way)

        return fin_rel_scores

class PrototypicalNetwork(nn.Module):

    def __init__(self, embedding_layer = ConvNet()):
        super().__init__()
        # define the embedding layer

        self.embedding_layer = embedding_layer

    def forward(self, query, support):
        # compute embeddings for query and support sets
        support["embeddings"] = self.embedding_layer(support["image"])
        query["embeddings"] = self.embedding_layer(query["image"])

        # now we need to compute the prototype for each class
        # this was the 'average' class member
        support_embeds = []
        for idx in range(len(support["classlist"])):
            embeds = support["embeddings"][support["target"] == idx]
            support_embeds.append(embeds)
        # support_embeds is a list of torch tensors of shape
        # (n_support, dimensions of embedding vector space)

        support_embeds = torch.stack(support_embeds)
        # support_embeds now a tensor of shape
        # (n_way, n_support, dimensions of embedding vector space)

        # we compute the mean of these support vectors to get prototypes
        prototypes = support_embeds.mean(dim=1)
        support["prototypes"] = prototypes

        # support["prototypes"] is a tensor of shape
        # (n_way, dimensions of embedding vector space)

        # compute the distances between the query embeddings and the prototypes
        # query['embeddings'] is a tensor of shape (n_samples, dimensions of embedding vector space)
        distances = torch.cdist(query["embeddings"].unsqueeze(0), prototypes.unsqueeze(0), p=2).squeeze(0)
        # distances is a tensor of dimensions (n_samples, n_ways)
        distances = distances ** 2

        # the negative of the distances give the final output logits
        logits = - distances

        return logits

class FewShotLearner(pl.LightningModule):

    def __init__(self,
        FSLnet: nn.Module,
        n_way: int,
        learning_rate: float = 0.001,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['FSLnet'])
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


class NeuralNetwork(nn.Module):
    def __init__(self, crop_size, n_outputs, fc_layers, fc_nodes, dropout):
        super().__init__()
        
        self.fc_layers = fc_layers
        
        self.conv4 = ConvNet()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(256, fc_nodes),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(fc_nodes)
        )
        
        self.linear_relu_stack_last = nn.Sequential(
            nn.Linear(fc_nodes, n_outputs)
        )
        
    def forward(self, x, training = True):
        x = self.conv4(x)
        for i in range(self.fc_layers-1):
            x = self.linear_relu_stack(x)
        if training == True:
            x = self.linear_relu_stack_last(x)
            return x
        else: 
            x = torch.nn.functional.normalize(x)
            return x

class SimpleShot(nn.Module):
    def __init__(self, model):#, channels, crop_size, n_outputs, fc_layers, fc_nodes, dropout):
        super().__init__()
        self.classifier = model
        

    def forward(self, query, support, support_labels, n_way):
        # find embeddings of query set
        x_q = self.classifier(query, training=False)
        # find embeddings of support
        x_s = self.classifier(support, training=False)
        


        # make an average feature vector for each class
        average_support_vecs = []
        for i in range(n_way):
            support_embeddings = x_s[support_labels==i]
            average_embedding = support_embeddings.mean(dim=0)
            average_support_vecs.append(average_embedding)

        average_support_vecs = torch.stack(average_support_vecs)
        
        # L2 normalise output
        x_q = torch.nn.functional.normalize(x_q)
        average_support_vecs = torch.nn.functional.normalize(average_support_vecs)
        # x_q i of shape (n_query, dimension of embedding vector)
        # average_support_vecs is of shape (n_way, dimension of embedding vector)
        
        # find euclidean distances between them
        distances = torch.cdist(x_q, average_support_vecs, p=2).squeeze(0)     
        # distances is a tensor of dimensions (n_query, n_ways)
        distances = distances ** 2
        
        logits = torch.argmin(distances, dim=1)
        
        x_s_norm = torch.nn.functional.normalize(x_s)
        
        return logits, distances, x_q, x_s_norm

        
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
