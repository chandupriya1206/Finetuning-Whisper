# Fine-Tuning Whisper Model for Singlish Accented Speech

This project demonstrates fine-tuning OpenAI's Whisper model to transcribe Singlish-accented speech effectively. The notebook trains a smaller version of Whisper (`openai/whisper-tiny`) using Hugging Face's Transformers library and a Singlish dataset.

Use Google Colab or Jupyter Notebook to run the Fine_Tune_Whisper_Singlish.ipynb. Follow these key steps:

## Project Overview

This project aims to fine-tune the Whisper model for Singlish audio transcription. The process includes:
- Preparing and preprocessing audio and text data.
- Fine-tuning the Whisper model using the Hugging Face `Trainer` API.
- Evaluating the model's performance with the Word Error Rate (WER) metric.
- Performing real-time transcription for new Singlish audio inputs.

## Workflow

### 1. Data Preparation
1. **Load and preprocess the dataset**:
   - Audio files are resampled to 16kHz using `torchaudio`.
   - The dataset includes audio samples and corresponding text transcriptions. It is split into training and testing subsets for model evaluation.
2. **Preprocessing functions**:
   - Convert audio samples into the format required by the Whisper model using a `preprocess_function`.
   - Tokenize text transcriptions for training.

### 2. Fine-Tuning
1. Load the Whisper model and its processor.
2. Fine-tune the model using the following training arguments:
   - `output_dir`: Directory to save results.
   - `batch_size`: 2 (for training and evaluation).
   - `num_train_epochs`: 3.
   - `gradient_accumulation_steps`: 2.
   - `learning_rate`: 5e-5.

### 3. Evaluation
- Evaluate the model's performance using the Word Error Rate (WER) metric.
- Compute WER on the test dataset.

### 4. Inference
- Perform real-time transcription of new Singlish audio inputs using the fine-tuned Whisper model.

## Clone this repository
1. Clone this repository:
   ```bash
   git clone https://github.com/chandupriya1206/Finetuning-Whisper.git
   cd whisper-singlish-finetune

2. You can also simply download the file from this `link : https://colab.research.google.com/drive/1QZv1FE9L_SpsfQ_BkYW5HDVQscg-vi1B?usp=sharing` and then make a copy of the file then run the code . 
