# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import gzip
import json

class MLPClassifier(nn.Module):
    def __init__(self,
                 pretrained_embeddings_path,
                 token_to_index,
                 n_labels,
                 hidden_layers=[256, 128],
                 dropout=0.3,
                 vector_size=300,
                 freeze_embedings=True):
        super().__init__()
        with gzip.open(token_to_index, "rt") as fh:
            token_to_index = json.load(fh)
        embeddings_matrix = torch.randn(len(token_to_index), vector_size)
        embeddings_matrix[0] = torch.zeros(vector_size)
        with gzip.open(pretrained_embeddings_path, "rt",encoding='utf-8') as fh:
            next(fh)
            for line in fh:
                word, vector = line.strip().split(None, 1)
                if word in token_to_index:
                    embeddings_matrix[token_to_index[word]] =\
                        torch.FloatTensor([float(n) for n in vector.split()])
        self.embeddings = nn.Embedding.from_pretrained(embeddings_matrix,
                                                       freeze=freeze_embedings,
                                                       padding_idx=0)
        self.hidden_layers = [
            nn.Linear(vector_size, hidden_layers[0])
        ]
        for input_size, output_size in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.hidden_layers.append(
                nn.Linear(input_size, output_size)
            )
        self.dropout = dropout
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.output = nn.Linear(hidden_layers[-1], n_labels)
        self.vector_size = vector_size

    def forward(self, x):
        x = self.embeddings(x)
        x = torch.mean(x, dim=1)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            if self.dropout:
                x = F.dropout(x, self.dropout)
        x = self.output(x)
        return x
    
class NNClassifier(nn.Module):
    def __init__(self,
                 pretrained_embeddings_path,
                 token_to_index,
                 n_labels,
                 FILTERS_COUNT=200,
                 FILTERS_LENGTH = [2, 3, 4, 5],
                 hidden_layers=[256, 128],
                 dropout=0.5,
                 vector_size=300,
                 freeze_embedings=True):   

        super().__init__()
         
        with gzip.open(token_to_index, "rt") as fh:
              token_to_index = json.load(fh)
        embeddings_matrix = torch.randn(len(token_to_index), vector_size)
        embeddings_matrix[0] = torch.zeros(vector_size)
        with gzip.open(pretrained_embeddings_path, "rt",encoding='utf-8') as fh:
              next(fh)
              for line in fh:
                  word, vector = line.strip().split(None, 1)
                  if word in token_to_index:
                       embeddings_matrix[token_to_index[word]] =\
                          torch.FloatTensor([float(n) for n in vector.split()])
        self.embeddings = nn.Embedding.from_pretrained(embeddings_matrix,
                                                       freeze=freeze_embedings,
                                                       padding_idx=0)
        self.convs = []
        for filter_lenght in FILTERS_LENGTH:
            self.convs.append(nn.Conv1d (vector_size , FILTERS_COUNT, filter_lenght))
         
        self.convs = nn.ModuleList(self.convs)
        self.fc = nn.Linear(FILTERS_COUNT * len(FILTERS_LENGTH), 250)
        self.output = nn.Linear(250, n_labels)
        self.vector_size = vector_size
             
        self.droupout_layer = nn.Dropout(dropout)

    @staticmethod
    def conv_global_max_pool(x, conv):
        return F.relu(conv(x).transpose(1, 2).max(1)[0])

    def forward(self, x):
        x = self.embeddings(x).transpose(1, 2)  # Conv1d takes (batch, channel, seq_len)
        x = [self.conv_global_max_pool(x, conv) for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = F.relu(self.fc(x))
        x = self.output(self.droupout_layer(x))
        return x


class LSTM_class(nn.Module):
    def __init__(self,
                pretrained_embeddings_path, 
                token_to_index,
                n_labels,
                vector_size,
                hidden_layer=32,
                num_layers=1, dropout=0., bias=True,
                bidirectional=False,
                freeze_embedings=True):

        super(LSTM_class,self).__init__()
        output_size = n_labels # dataset.target.nunique()

            # Create the Embeddings layer and add pre-trained weights
        with gzip.open(token_to_index, "rt") as fh:
            token_to_index = json.load(fh)
        embeddings_matrix = torch.randn(len(token_to_index), vector_size)
        embeddings_matrix[0] = torch.zeros(vector_size)
        with gzip.open(pretrained_embeddings_path, "rt",encoding='utf-8') as fh:
            for line in fh:
                word, vector = line.strip().split(None, 1)
                if word in token_to_index:
                    embeddings_matrix[token_to_index[word]] =\
                        torch.FloatTensor([float(n) for n in vector.split()])
        
        
        
        self.embeddings = nn.Embedding.from_pretrained(embeddings_matrix,
                                                        freeze=freeze_embedings,
                                                        padding_idx=0)      
            # Set our LSTM parameters
        self.lstm_config = {'input_size': vector_size,
                            'hidden_size': hidden_layer,
                            'num_layers': num_layers,
                            'bias': bias,
                            'batch_first': True,
                            'dropout': dropout,
                            'bidirectional': bidirectional}        
        # Set our fully connected layer parameters
        k=1 
        if bidirectional: k=2
        self.linear_config = {'in_features':k * hidden_layer,
                                'out_features': output_size,
                                'bias': bias}

        self.droupout_layer = nn.Dropout(dropout)        
        self.lstm = nn.LSTM(**self.lstm_config)

        self.classification_layer= nn.Linear(**self.linear_config)

    def forward (self, inputs):
        emb = self.embeddings(inputs)
        # emb=self.droupout_layer(emb)
        lstm_out,_ = self.lstm(emb)
        lstm_out= lstm_out[:, -1, :].squeeze()        
        lstm_out= self.droupout_layer(lstm_out)        
        predictions = self.classification_layer(lstm_out)

        return predictions