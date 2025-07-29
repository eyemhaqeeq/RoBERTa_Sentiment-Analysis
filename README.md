
## RoBERTa Sentiment Analysis on Amazon Reviews

This project demonstrates fine-tuning of a pre-trained RoBERTa model for binary sentiment classification using the Amazon Fine Food Reviews dataset. The implementation uses Hugging Face Transformers and PyTorch, optimized for Google Colab GPU.

## Model Summary

Model: RoBERTa (roberta-base)
Task: Binary Sentiment Classification (Positive vs Negative)
Dataset: Amazon Fine Food Reviews (Kaggle)
Sample Size Used: 15000 reviews
Max Sequence Length: 128 tokens
Training Epochs: 2
Batch Size: 16 (train), 64 (eval)
Tokenizer: RobertaTokenizer from Hugging Face

## Libraries Used

transformers
scikit-learn
pandas
torch
Google Colab (with GPU)

## Evaluation Results

Accuracy: 0.9515
Precision: 0.9726
Recall: 0.9701
F1 Score: 0.9714
Training Time: 534.08 seconds
Testing Time: 18.47 seconds

## Workflow

1. Load Amazon reviews dataset (e.g., Reviews.csv from Kaggle)
2. Remove neutral reviews (Score == 3)
3. Map scores to binary labels: 0 (negative), 1 (positive)
4. Tokenize using RoBERTa tokenizer (max_length = 128)
5. Create a PyTorch Dataset class for Trainer
6. Fine-tune RobertaForSequenceClassification with Trainer
7. Evaluate using scikit-learn metrics

## File Structure

- RoBERTa_Sentiment_Analysis.ipynb: Jupyter notebook with training and evaluation
- Reviews.csv: Input dataset (Amazon reviews)
- README.md: Project overview and performance

## Future Work

- Compare performance with BERT and DistilBERT
- Add LIME or SHAP for explainability
- Test with larger datasets or multilingual models

## License

This project is licensed under the MIT License — see the LICENSE file for details.

This project is for educational and research purposes only.
