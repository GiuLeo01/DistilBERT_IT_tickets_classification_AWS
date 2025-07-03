import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import argparse
import os
from torchmetrics.classification import MulticlassF1Score


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--val_batch_size', type=int, default=2)
    parser.add_argument('--dev_run', type=bool, default=False)
    parser.add_argument('--max_time', type=str, default="00:01:30:00")
    parser.add_argument('--timestamp', type=str, default="0")
    args = parser.parse_args()

    # get output directory from sagemaker environment
    output_dir = os.environ['SM_MODEL_DIR']

    # set training parameters
    TRAIN_BATCH_SIZE = args.train_batch_size
    VALID_BATCH_SIZE = args.val_batch_size
    TEST_BATCH_SIZE = 1
    LR = args.lr
    MODEL_NAME = 'distilbert-base-uncased'
    EPOCHS = args.epochs
    DEV_RUN = args.dev_run
    MAX_TIME = args.max_time
    TIMESTAMP = args.timestamp
    
    # load training data from s3
    s3_path_train = 's3://placeholder/data/train.csv'
    train = pd.read_csv(s3_path_train)
    train = train[['Document', 'Topic_group']]
    
    # load validation data from s3
    s3_path_val = 's3://placeholder/data/val.csv'
    val = pd.read_csv(s3_path_val)
    val = val[['Document', 'Topic_group']]
    
    # load test data from s3
    s3_path_test = 's3://placeholder/data/test.csv'
    test = pd.read_csv(s3_path_test)
    test = test[['Document', 'Topic_group']]
    
    # standardize column names
    train.columns = ['text', 'label']
    val.columns = ['text', 'label']
    test.columns = ['text', 'label']
    
    # encode labels as integers
    enc = LabelEncoder().fit(train['label'])
    train['label'] = enc.transform(train['label'])
    val['label'] = enc.transform(val['label'])
    test['label'] = enc.transform(test['label'])
    label_decoder = dict(zip(range(0, len(enc.classes_)), enc.classes_))
    
    # initialize model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=8)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # create datasets
    training_set = TicketsDataset(train, tokenizer, 512)
    validation_set = TicketsDataset(val, tokenizer, 512)
    test_set = TicketsDataset(test, tokenizer, 512)
    
    # create data loaders
    train_dataloader = DataLoader(training_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(validation_set, batch_size=VALID_BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_set, batch_size=TEST_BATCH_SIZE, shuffle=False)
    
    # create lightning module
    lit_model = LitDistilBert(model, LR, label_decoder)  
    
    # configure model checkpointing
    top1_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename=f"DistilBertTickets-{TIMESTAMP}-ckp",
        verbose=True,
        dirpath=output_dir
    )
    
    # setup tensorboard logging and trainer
    tb_logger = TensorBoardLogger(save_dir="/opt/ml/output/tensorboard", name=f"DistilBertTickets-{TIMESTAMP}")
    trainer = L.Trainer(max_epochs=EPOCHS, log_every_n_steps=1000, logger=tb_logger, callbacks=[top1_checkpoint], max_time=MAX_TIME)
    
    # start training
    trainer.fit(model=lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


class TicketsDataset(Dataset):
    """custom dataset for text classification with tokenization"""
    
    def __init__(self, df, tokenizer, maxlen):
        self.len = len(df)
        self.data = df
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __getitem__(self, index):
        text = self.data['text'][index]
        label = self.data['label'][index]

        # tokenize text with padding and truncation
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens = True,
            max_length = self.maxlen,
            padding = 'max_length',
            return_token_type_ids = True,
            truncation = True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'masks': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(label, dtype=torch.long),
        }

    def __len__(self):
        return self.len


class LitDistilBert(L.LightningModule):
    """pytorch lightning module for distilbert classification"""
    
    def __init__(self, model, lr, label_decoder):
        super().__init__()
        self.lr = lr
        self.model = model
        self.f1_score = MulticlassF1Score(num_classes=8, average='weighted')
        self.label_decoder = label_decoder

    def forward(self, x):
        ids = x['ids']
        masks = x['masks']
        y_pred = self.model(ids, masks).logits
        return y_pred

    def training_step(self, batch, batch_idx):
        # extract batch data
        ids = batch['ids']
        masks = batch['masks']
        targets = batch['targets']
        
        # forward pass
        y_pred = self.model(ids, masks).logits
        loss = torch.nn.functional.cross_entropy(y_pred, targets)
        
        # get predictions
        y_pred_bin = torch.nn.functional.softmax(y_pred, 1)
        y_pred_bin = y_pred_bin.argmax(dim=1)

        # calculate metrics
        accuracy = (y_pred_bin == targets).float().mean()
        f1 = self.f1_score(y_pred_bin, targets)

        # log metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", accuracy, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_f1_weighted", f1, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # extract batch data
        ids = batch['ids']
        masks = batch['masks']
        targets = batch['targets']
        
        # forward pass
        y_pred = self.model(ids, masks).logits
        loss = torch.nn.functional.cross_entropy(y_pred, targets)
        
        # get predictions
        y_pred_bin = torch.nn.functional.softmax(y_pred, 1)
        y_pred_bin = y_pred_bin.argmax(dim=1)

        # calculate metrics
        accuracy = (y_pred_bin == targets).float().mean()
        f1 = self.f1_score(y_pred_bin, targets)

        # log metrics
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_f1_weighted", f1, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        # extract batch data
        ids = batch['ids']
        masks = batch['masks']
        targets = batch['targets']
        
        # forward pass
        y_pred = self.model(ids, masks).logits
        loss = torch.nn.functional.cross_entropy(y_pred, targets)
        
        # get predictions
        y_pred_bin = torch.nn.functional.softmax(y_pred, 1)
        y_pred_bin = y_pred_bin.argmax(dim=1)

        # calculate metrics
        accuracy = (y_pred_bin == targets).float().mean()
        f1 = self.f1_score(y_pred_bin, targets)

        # log metrics
        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_accuracy", accuracy, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_f1_weighted", f1, prog_bar=False, on_step=False, on_epoch=True)

        # log per-class f1 scores
        for i in range(8):
            self.log(f"{self.label_decoder[i]}_f1", f1[i], prog_bar=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    print('start')
    main()
    print('stop')