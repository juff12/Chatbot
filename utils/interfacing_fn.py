from deepmultilingualpunctuation import PunctuationModel

# define the punctuation model
punc_model = PunctuationModel()

# define the punctuation endsings
punctuation_ends = ['.', '!', '?']

def output_formating(text):
    # replace the ellipsis with '--'
    text = text.replace('...', '--')
    # restore the punctuation
    text = punc_model.restore_punctuation(text)
    # remove the last sentence if it ends with a punctuation
    for i in range(len(text)-1, -1, -1):
        if text[i] in punctuation_ends:
            return text[:i+1].replace('--','...')
    # restore the ellipsis
    return text.replace('--','...') + '...'

def generate(prompt, pipe):
    result = pipe(
        prompt,
        do_sample=True,
        temperature=0.75,
        penalty_alpha=0.65,
        top_k=4,
        max_length=64,
        num_return_sequences=1,
        no_repeat_ngram_size=2
    )
    return result[0]['generated_text']