# DistilBERT IT Tickets Classifier Training with Amazon Sagemaker Studio

This project trains a text classification model to automatically categorize IT service tickets into different topic groups. The model uses DistilBERT as the base architecture and is trained using PyTorch Lightning on Amazon SageMaker Studio.

## Features

- **Model**: DistilBERT for sequence classification
- **Framework**: PyTorch Lightning for structured training
- **Cloud Training**: Amazon SageMaker integration
- **Monitoring**: TensorBoard logging
- **Metrics**: Accuracy and weighted F1-score

## Dataset
The model is trained on the **IT Service Ticket Classification Dataset**: https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset

**Have a look at the "exploratory_data_analysis.ipynb" notebook for more details on the dataset!**

Data is splitted into:
- train.csv - Training data
- val.csv - Validation data  
- test.csv - Test data

## Usage

### Configure S3 Bucket

Update the S3 bucket paths in the scripts:
- Replace "placeholder" with your actual bucket name
- Ensure your data is uploaded to the "/data/" folder

### Run Training

Execute the training notebook to start a SageMaker training job.


### Logging

TensorBoard logs are automatically saved to S3 and can be accessed through SageMaker or downloaded locally.

## Model Architecture

- **Base Model**: distilbert-base-uncased
- **Task**: Multi-class classification (8 classes)
- **Max Sequence Length**: 512 tokens
- **Optimizer**: Adam
- **Loss Function**: Cross-entropy

## Hyperparameters

- Learning Rate: 1e-5
- Batch Size: 8 (training), 2 (validation)
- Epochs: 5
- Max Training Time: 1h 10m


## Requirements

- Python 3.11+
- PyTorch 2.3+
- Transformers 4.46+
- PyTorch Lightning
- SageMaker SDK
- AWS account

## License

Unlicense
