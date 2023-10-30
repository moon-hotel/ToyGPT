from transformers import OpenAIGPTModel
from transformers import OpenAIGPTConfig
from transformers import OpenAIGPTLMHeadModel
from transformers import GPTJForCausalLM
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


from datasets import load_dataset

eli5 = load_dataset("eli5", split="train_asks[:5000]")
eli5 = eli5.train_test_split(test_size=0.2)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
#
eli5 = eli5.flatten()
#
def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["answers.text"]])

tokenized_eli5 = eli5.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=eli5["train"].column_names,
)

block_size = 10


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)

print(type(lm_dataset))
from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


out = data_collator([lm_dataset["train"][i] for i in range(2)])
for key in out:
    print(f"{key} shape: {out[key].shape}")
    print(out[key])
    if key == 'labels':
        shift_labels = out[key][..., 1:]
        print(shift_labels)
    print("========")


# iter_data = DataLoader(lm_dataset['train'],collate_fn=data_collator,batch_size=2)
# for x in iter_data:
#     print(x)
#     break
# print()
# print(lm_dataset['train'])
# print(type(lm_dataset['train']))
# print(lm_dataset['train']['input_ids'][:2])
#
# print(lm_dataset['train']['labels'][:2])
# print(lm_dataset['train']['attention_mask'][:2])

