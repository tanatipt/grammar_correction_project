# English Grammar Correction

This hands-on project explores the development of a grammar correction system using deep learning. The primary goal is to fine-tune an encoder-decoder model capable of automatically correcting grammatical errors in English sentences. This project serves as an exploratory exercise to apply and reinforce my understanding of encoder-decoder architectures and their applications in natural language processing (NLP) tasks.


## Data Source

For this project, we used the [juancavallotti/multilingual-gec](https://huggingface.co/datasets/juancavallotti/multilingual-gec) dataset available on Hugging Face. This dataset contains approximately **220,000 examples** of grammatically incorrect sentences paired with their corrected versions, spanning multiple languages. The language distribution is as follows:

- **German**: 32,282 examples  
- **English**: 51,393 examples  
- **Spanish**: 67,672 examples  
- **French**: 67,157 examples  

In addition to sentence pairs, each example is annotated with the type of grammatical transformation applied, such as `AdjectiveGenderChangeDestroyer`, `ProgressiveDestroyOperation`, and `RemovePunctuationDestroyer`.

Below are a few sample entries from the dataset:

| Language | Corrected Sentence | Incorrect Sentence | Transformation |
|----------|-------------------|-------------------|----------------|
| en       | Most of the recipients have been American. | fix grammar: Most of the recipients having been American. | ProgressiveDestroyOperation |
| fr       | Il est très important de parler une langue étrangère. | fix grammar: Il est très importante de parler une langue étrangère. | AdjectiveGenderChangeDestroyer |
| en       | Plants, obviously, cannot move after they have put down roots. | fix grammar: Plants, obviously, cannot moved after they hadn't put down roots. | VerbAgreementDestroyer |

This dataset was generated synthetically using code that introduces common grammatical errors into otherwise correct sentences. The grammatically correct sentences were initially sourced from various open-source datasets, such as **Tatoeba**. A synthetic data generation script then applied specific transformations to create incorrect versions, guided by a catalog of typical grammar mistakes collected from across the internet.


## Data Preprocessing

The data preprocessing process was straightforward. First, we filtered the dataset to retain only **English** examples. We then cleaned the text by:

- Stripping leading and trailing whitespace  
- Replacing multiple spaces with a single space  
- Removing unwanted characters  

After cleaning, we split the dataset into four subsets: **training**, **early stopping**, **validation**, and **test** sets. To ensure consistency, each set contained the same number of examples. We also applied **stratified sampling** to maintain the same proportion of grammar transformation types across all subsets.



## Model Architecture

For our grammar correction task, we used the [Flan-T5 Small](https://huggingface.co/google/flan-t5-sma;;) model from Hugging Face. At the time of building this project, the T5 model family represented one of the state-of-the-art architectures for encoder-decoder tasks, making it a strong choice for this application.

We specifically chose the **Small** version of Flan-T5 due to hardware limitations, as the project was run locally on a laptop with only **4GB of GPU memory**. The base model provided a good balance between performance and resource efficiency. 

Flan-T5 is an encoder-decoder model pre-trained on a wide range of tasks, making it well-suited for **text-to-text transformations**. This aligns closely with our objective: converting grammatically incorrect sentences into their corrected forms.

## Evaluation Metric

To evaluate our model’s performance, we used the **BLEU (Bilingual Evaluation Understudy)** score. BLEU is a widely used metric for assessing the quality of machine-generated text, particularly in tasks like machine translation and text generation. It compares the model-generated text (candidate) to one or more reference texts and outputs a score between **0 and 1**, where a score closer to 1 indicates higher similarity to the reference.

Below is a general interpretation of BLEU scores:

| BLEU Score | Interpretation                                      |
|------------|-----------------------------------------------------|
| < 0.1      | Almost useless                                      |
| 0.1–0.19   | Hard to understand the gist                         |
| 0.2–0.29   | The gist is clear, but contains significant grammar errors |
| 0.3–0.4    | Understandable to good translations                 |
| 0.4–0.5    | High-quality translations                           |
| 0.5–0.6    | Very high quality, adequate and fluent translations |
| > 0.6      | Often better than human translations                |

We used this metric to quantitatively assess how well our model corrected grammatical errors in comparison to the ground-truth reference sentences.

## Training Methodology

We trained the model using a batch size of 8. The training process began with hyperparameter tuning on the T5 model to identify the optimal settings. This tuning was conducted using the training, early stopping, and validation datasets. After selecting the best hyperparameters, we fine-tuned the T5 model again on the entire training dataset (including early stopping and validation sets) and finally evaluated its performance on the test set.

The hyperparameter tuning focused on optimizing the number of encoder and decoder transformer blocks to unfreeze, as well as the learning rate. The search space for each hyperparameter is summarized in the following table:

| Hyperparameter           | Search Space    |
|-------------------------|-----------------|
| Unfrozen Decoder Layers  | 0, 1, 2            |
| Unfrozen Encoder Layers  | 0, 1, 2            |
| Learning Rate            | 1e-4, 3e-4      |

In total, we experimented with 4 different configurations sampled from these search spaces. For each combination, the T5 model was fine-tuned on the training set for up to 10 epochs, applying early stopping to prevent overfitting. During early stopping, after each epoch, we evaluated the model on the early stopping set and computed the cross-entropy loss. If the loss did not decrease for more than 2 consecutive epochs, training was halted early. After training, the model was evaluated on the validation set. The best hyperparameter configuration was selected based on the highest BLEU score achieved on the validation data.



## Repository Structure

- `grammar_correction_trainer.py`: Contains the Python class for training and evaluating the T5 model.
- `preprocess.py`: Python file for preprocessing the data source.
- `utils.py`: Contains utility function that is used in the project.
- `main.py`: Main file for running the fine-tuning experiments.
- `preprocessed/`: Contains the preprocessed train, early stopping, validation and test dataset.
- `source/`: Contains the raw dataset.
- `results/`: Contains the result of the fine-tuning of the T5 model.