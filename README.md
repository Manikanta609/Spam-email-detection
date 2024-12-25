# Spam Email Detection Model

This repository contains a spam email detection model implemented using a fine-tuned BERT transformer. The model predicts whether an email is classified as "Spam" or "Ham" (not spam).

## Prerequisites

Before running the code, ensure you have the following installed:

1. Python 3.8 or higher
2. Required Python libraries:
   - `torch`
   - `transformers`
   - `pandas`
   - `scikit-learn`
   - `joblib`
   - `tqdm`

Install the dependencies using the following command:
```bash
pip install torch transformers pandas scikit-learn joblib tqdm
```

## Dataset

The dataset used for training and testing is a CSV file named `spam_ham_dataset.csv`. It should contain the following columns:

- **v1**: Label of the email (`spam` or `ham`)
- **v2**: The text content of the email

## Project Structure

- **Model Training**:
  - Tokenizes the dataset using `BertTokenizer`.
  - Fine-tunes a pre-trained BERT model (`bert-base-uncased`) for binary classification.
  - Trains the model for 3 epochs with a batch size of 8.
  - Saves the trained model and tokenizer to local directories (`bert_spam_detector` and `bert_tokenizer`).

- **Inference**:
  - Loads the saved model and tokenizer.
  - Predicts whether a given email is "Spam" or "Ham".

## File Structure

- `spam_email_detection.py`: Python script containing the training and prediction code.
- `spam_ham_dataset.csv`: Dataset file containing labeled email data.
- `bert_spam_detector`: Directory storing the trained model.
- `bert_tokenizer`: Directory storing the tokenizer configuration and vocabulary.

## Usage

### Training the Model
1. Place the dataset file `spam_ham_dataset.csv` in the specified directory.
2. Run the script to train the model:
   ```bash
   python spam_email_detection.py
   ```
3. The trained model and tokenizer will be saved in `bert_spam_detector` and `bert_tokenizer` directories respectively.

### Predicting Spam or Ham
1. Ensure the trained model and tokenizer directories are available.
2. Use the `predict_spam` function to classify emails. Example:
   ```python
   from transformers import BertTokenizer, BertForSequenceClassification
   import torch

   # Load the trained model and tokenizer
   model = BertForSequenceClassification.from_pretrained("bert_spam_detector")
   tokenizer = BertTokenizer.from_pretrained("bert_tokenizer")
   model.eval()
   model.to('cuda')

   def predict_spam(email_text):
       inputs = tokenizer(email_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
       inputs = {key: value.to('cuda') for key, value in inputs.items()}
       with torch.no_grad():
           outputs = model(**inputs)
           logits = outputs.logits
           predicted_class = torch.argmax(logits, dim=1).item()
       return "Spam" if predicted_class == 1 else "Ham"

   # Test the model
   example_email = "Subject: Special Offer - Limited Time Discount!"
   result = predict_spam(example_email)
   print(f"Prediction: {result}")
   ```

### Expected Output
For the example email, the output will be:
```
Prediction: Spam
```

## Notes

- This model is fine-tuned on a specific dataset. Performance may vary depending on the dataset used.
- To improve accuracy, consider training on a larger and more diverse dataset.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for providing pre-trained models and utilities.
- The dataset used in this project for providing labeled email data.

## License

This project is licensed under the MIT License. Feel free to use and modify it as needed.

