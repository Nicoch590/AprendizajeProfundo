import argparse
import gzip
import json
import logging
import mlflow
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from .dataset import MeliChallengeDataset
from .utils import PadSequences
from .Models import MLPClassifier, NNClassifier, LSTM_class


logging.basicConfig(
    format="%(asctime)s: %(levelname)s - %(message)s",
    level=logging.INFO
)



#pad_sequences = PadSequences(min_length=max(FILTERS_LENGTH))
#train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
#                          collate_fn=pad_sequences, drop_last=False)
#test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False,
#                         collate_fn=pad_sequences, drop_last=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data",
                        help="Path to the the training dataset",
                        required=True)
    parser.add_argument("--sample_size",
                         default=None,
                         type=int )

    parser.add_argument("--token-to-index",
                        help="Path to the the json file that maps tokens to indices",
                        required=True)
    parser.add_argument("--pretrained-embeddings",
                        help="Path to the pretrained embeddings file.",
                        required=True)
    parser.add_argument("--language",
                        help="Language working with",
                        required=True)
    parser.add_argument("--test-data",
                        help="If given, use the test data to perform evaluation.")
    parser.add_argument("--validation-data",
                        help="If given, use the validation data to perform evaluation.")
    parser.add_argument("--embeddings-size",
                        default=300,
                        help="Size of the vectors.",
                        type=int)
    parser.add_argument("--hidden-layers",
                        help="Sizes of the hidden layers of the MLP (can be one or more values)",
                        nargs="+",
                        default=[256, 128],
                        type=int)
    parser.add_argument("--dropout",
                        help="Dropout to apply to each hidden layer",
                        default=0.3,
                        type=float)
    parser.add_argument("--epochs",
                        help="Number of epochs",
                        default=3,
                        type=int)

    parser.add_argument("--model",
                        help="MLPClassifier, NNClassifier, LSTM_class",
                        default='MLPClassifier'
                        )
                    

    args = parser.parse_args()

    BATCH_SIZE=256

    pad_sequences = PadSequences(
        pad_value=0,
        max_length=12 ,# if args.model=='LSTM_class' else None,
        min_length=1
    )

    logging.info("Building training dataset")
    train_dataset = MeliChallengeDataset(
        dataset_path=args.train_data,
        random_buffer_size=2048,  # This can be a hypterparameter
        sample_size=args.sample_size
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,  # This can be a hyperparameter
        shuffle=False,
        collate_fn=pad_sequences,
        drop_last=False
    )

    if args.validation_data:
        logging.info("Building validation dataset")
        validation_dataset = MeliChallengeDataset(
            dataset_path=args.validation_data,
            random_buffer_size=1,
            sample_size=args.sample_size 
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=pad_sequences,
            drop_last=False
        )
    else:
        validation_dataset = None
        validation_loader = None

    if args.test_data:
        logging.info("Building test dataset")
        test_dataset = MeliChallengeDataset(
            dataset_path=args.test_data,
            random_buffer_size=1,
            )#sample_size=int(args.sample_size * 0.1))
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=pad_sequences,
            drop_last=False
        )
    else:
        test_dataset = None
        test_loader = None

    mlflow.set_experiment(f"diplodatos.Meli2019") #{args.language}

    with mlflow.start_run():
        logging.info("Starting experiment")

        mlflow.log_params({
                "Sample_size":args.sample_size ,              
            })
        if args.model=='MLPClassifier':
            # Log all relevent hyperparameters
            mlflow.log_params({
                "model_type": "Multilayer Perceptron",
                "embeddings": args.pretrained_embeddings,
                "hidden_layers": args.hidden_layers,
                "dropout": args.dropout,
                "embeddings_size": args.embeddings_size,
                "epochs": args.epochs
            })
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

            logging.info(args.model)
            model = MLPClassifier(
                pretrained_embeddings_path=args.pretrained_embeddings,
                token_to_index=args.token_to_index,
                n_labels=train_dataset.n_labels,
                hidden_layers=args.hidden_layers,
                dropout=args.dropout,
                vector_size=args.embeddings_size,
                freeze_embedings=True  # This can be a hyperparameter
            )
        if args.model=='NNClassifier':

            Filters_count= 200
            Filters_lenght = [2, 3, 4, 5]
            # Log all relevent hyperparameters
            mlflow.log_params({
                "model_type": "Red convolucional",
                "embeddings": args.pretrained_embeddings,
                "FILTERS_COUNT":Filters_count,
                "FILTERS_LENGTH": Filters_lenght,
                "hidden_layers": args.hidden_layers,
                "dropout": args.dropout,
                "embeddings_size": args.embeddings_size,
                "epochs": args.epochs
            })
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

            logging.info(args.model)
            model = NNClassifier(pretrained_embeddings_path=args.pretrained_embeddings,
                    token_to_index=args.token_to_index,
                    n_labels=train_dataset.n_labels,
                    FILTERS_COUNT=Filters_count,
                    FILTERS_LENGTH = Filters_lenght,
                    # hidden_layers=[256, 128],
                    dropout=args.dropout,
                    vector_size=args.embeddings_size,
                    freeze_embedings=True )

        if args.model=='LSTM_class':
            HID_layers=64
            Num_layers=1
            # Log all relevent hyperparameters
            
            mlflow.log_params({
                "model_type": "Red LSTM",
                "embeddings": args.pretrained_embeddings,
                "hidden_layer": HID_layers,
                "num_layers": Num_layers,
                "dropout": args.dropout,
                "bidirectional":True ,
                "embeddings_size": args.embeddings_size,
                "epochs": args.epochs
            })
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

            logging.info(args.model)
            model = LSTM_class(pretrained_embeddings_path=args.pretrained_embeddings,
                    token_to_index=args.token_to_index,
                    n_labels=train_dataset.n_labels,
                    hidden_layer=HID_layers,
                    num_layers=Num_layers,
                    dropout= args.dropout,
                    vector_size=args.embeddings_size,
                    freeze_embedings=True,
                    bidirectional=True )          

                   
        

        model = model.to(device)
        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=1e-3,  # This can be a hyperparameter
            weight_decay=1e-5  # This can be a hyperparameter
        )

        logging.info("Training classifier")
        for epoch in trange(args.epochs):
            model.train()
            running_loss = []
            for idx, batch in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                data = batch["data"].to(device)
                target = batch["target"].to(device)
                output = model(data)
                loss_value = loss(output, target)
                loss_value.backward()
                optimizer.step()
                running_loss.append(loss_value.item())
            mlflow.log_metric("train_loss", sum(running_loss) / len(running_loss), epoch)

            if validation_dataset:
                logging.info("Evaluating model on validation")
                model.eval()
                running_loss = []
                targets = []
                predictions = []
                with torch.no_grad():
                    for batch in tqdm(validation_loader):
                        data = batch["data"].to(device)
                        target = batch["target"].to(device)
                        output = model(data)
                        running_loss.append(
                            loss(output, target).item()
                        )
                        targets.extend(batch["target"].numpy())
                        predictions.extend(output.argmax(axis=1).detach().cpu().numpy())
                    mlflow.log_metric("validation_loss", sum(running_loss) / len(running_loss), epoch)
                    mlflow.log_metric("validation_bacc", balanced_accuracy_score(targets, predictions), epoch)

        if test_dataset:
            logging.info("Evaluating model on test")
            model.eval()
            running_loss = []
            targets = []
            predictions = []
            with torch.no_grad():
                for batch in tqdm(test_loader):
                    data = batch["data"].to(device)
                    target = batch["target"].to(device)
                    output = model(data)
                    running_loss.append(
                        loss(output, target).item()
                    )
                    targets.extend(batch["target"].numpy())
                    predictions.extend(output.argmax(axis=1).detach().cpu().numpy())
                mlflow.log_metric("test_loss", sum(running_loss) / len(running_loss), epoch)
                mlflow.log_metric("test_bacc", balanced_accuracy_score(targets, predictions), epoch)
