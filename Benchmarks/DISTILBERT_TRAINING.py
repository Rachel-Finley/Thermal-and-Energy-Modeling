# !pip install datasets
import warnings # to ignore tqdm warning (irrelevant/ annoying warning)
warnings.filterwarnings("ignore") # set parameter to ignore warnings
import datasets # to get datasets from huggingface's api
from datasets import load_dataset # for loading in those datasets
from tqdm.auto import tqdm # for progress bar
from transformers import DistilBertTokenizerFast # BERT tokenizer to get encodings
import torch # NN Framework
from transformers import AdamW # Optimizer function from huggingface
from transformers import DistilBertForQuestionAnswering # QA model


def add_end_idx(answers, contexts):
    new_answers = []
    # loop through all of the answer-context pairs
    for answer, context in tqdm(zip(answers, contexts)):
        # reformat to remove lists
        answer['text'] = answer['text'][0]
        answer['answer_start'] = answer['answer_start'][0]
        # gold text is the answer we are looking to find in the context
        gold_text = answer['text']
        # assign start index for answer span and end index with len of answer span
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # accounting for squad dataset not always being correct with answer span indices
        if context[start_idx:end_idx] == gold_text:
            # if the indexing is correct
            answer['answer_end'] = end_idx
        else:
            # if the answer index length is off by a couple of tokens, check for matches in near indices
            for n in [1, 2]:
                # +- the answer span gets saved
                if context[start_idx - n: end_idx - n] == gold_text:
                    answer['answer_start'] = start_idx - n
                    answer['answer_end'] = end_idx - n

        # append either answer span to a list
        new_answers.append(answer)
    return new_answers


def prep_data(dataset):
    # create dictionary out of q/a/context groupings
    questions = dataset['question']
    contexts = dataset['context']
    answers = add_end_idx(dataset['answers'], contexts)

    return {'question': questions, 'context': contexts, 'answers': answers}


def add_token_positions(encodings, answers, tokenizer):
    # initialize lists to hold start/end indicies of tokenized answer spans
    start_positions = []
    end_positions = []

    for i in tqdm(range(len(answers))):
        # append start/ end token position using char_to_token
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

        # if start position is Nonetype, the anwer has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        # end position cannot be found so shift the index
        shift = 1
        while end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - shift)
            shift += 1
    # update encodings object with new start/end answer spans
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # Initialize the dataset with the provided encodings
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {}
        for key, val in self.encodings.items():
            if isinstance(val[idx], torch.Tensor) and val[idx].dtype.is_floating_point:
                item[key] = val[idx].clone().detach().requires_grad_(True)
            else:
                item[key] = val[idx]
        return item

    def __len__(self):
        # Return the length of the dataset, which is determined by the number of input_ids
        return len(self.encodings.input_ids)

def main():
    # Load in and format dataset for QA
    # Disable annoying progress bar
    # datasets.disable_progress_bar()
    # Load the dataset from the 'oscar' dataset in the 'unshuffled_deduplicated_en' split
    dataset = datasets.load_dataset('oscar', 'unshuffled_deduplicated_en', split='train', streaming = True)
    # Load the SQuAD dataset
    data = load_dataset('squad')
    # Preprocess the SQuAD dataset
    dataset = prep_data(data['train'])

    # Load the pretrained tokenizer for BERT from Hugging Face
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    # Tokenize the dataset using the tokenizer
    train = tokenizer(dataset['context'], dataset['question'], truncation=True, padding='max_length', max_length=512, return_tensors='pt')

    # Create token-based start/end indices
    add_token_positions(train, dataset['answers'], tokenizer)

    # Build the dataset using the SquadDataset class
    train_dataset = SquadDataset(train)

    # Create a data loader for batch processing
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Load the DistilBERT model for Question Answering
    model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')
    model.to(device)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Training loop
    for epoch in range(3):
        loop = tqdm(loader)
        for batch in loop:
            optim.zero_grad()

            # Move the input tensors to the appropriate device (CPU or GPU)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            # Clear any previously calculated gradients
            optim.zero_grad()

            # Forward pass through the model
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)

            # Compute the loss
            loss = outputs.loss

            # Perform backpropagation and update the model's parameters
            loss.backward()
            optim.step()

            # Update the progress bar description and display the current loss
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

    # Save the trained model
    model.save_pretrained('./distilbert-qa-test')


main()
