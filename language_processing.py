import numpy as np
import nltk
nltk.download('treebank')
from nltk.corpus import treebank as ptb 

## Show first few sentences in corpus
prep_ptb = [[w.lower() for w in s] for s in ptb.sents()]
print("The PTB has {} sentences, here the first few:".format(len(prep_ptb)))
for i, s in zip(range(5), prep_ptb):
    print(s)
    
## Check if my beginning/ending of sentence symbols are not already in corpus
from itertools import chain
ptb_original_vocab = set(chain(*prep_ptb))
assert "<s>" not in ptb_original_vocab, "Our choice of BOS symbol should not clash with an existing token" 
assert "</s>" not in ptb_original_vocab, "Our choice of EOS symbol should not clash with an existing token"

## Randomply split list of sentences into 2 sets given a ration
import numpy as np
def split_corpus(sentences, ratio=0.8):
    """
    :param sentences: tokenized sentences (list of strings)
    :returns: portion containing (ratio)*100% of the data, portion containing (1-ratio)*100% of the data
    """
    # guarantee that the permutation is the same every time (which is important for reproducibility)
    rng = np.random.RandomState(42)
    rng.permutation(5)
    indices = rng.permutation(len(sentences))
    n = int(indices.size * ratio)
    return [sentences[i] for i in indices[:n]], [sentences[i] for i in indices[n:]]

# 80% training, 10% development, 10% test split
prep_training, prep_test = split_corpus(prep_ptb)
prep_dev, prep_test = split_corpus(prep_test, ratio=0.5)

print("Number of observations: training={} development={} test={}".format(len(prep_training), len(prep_dev), len(prep_test)))

## Count the unigrams
from collections import Counter
def count_unigrams(sentences, EOS="</s>"):
    """
    input: preprocessed sentences
        - a preprocessed sentence is a list of lowercased tokens
    output: 
        unigram_counts: dictionary of frequency of each word
    """    
    unigram_counts = Counter()
    for sentence in sentences:
        unigram_counts.update(sentence + [EOS])
    return unigram_counts

unigram_count_table =  count_unigrams(prep_training)
assert unigram_count_table['</s>'] == len(prep_training), "EOS should occur as many times as there are sentences in the corpus"
#Check how many times man, old and </s> happen
print('unigram=cat count={}'.format(unigram_count_table['cat']))
print('unigram=mat count={}'.format(unigram_count_table['mat']))
print('unigram=</s> count={}'.format(unigram_count_table['</s>']))

def unigram_mle(unigram_counts: Counter):
    """
    input: unigram_count: dictionary of frequency of each word       
    output: unigram_prob: dictionary with the probabilty of each word 
            (parameters of the model)
    """
    total_count = sum(unigram_counts.values())
    unigram_probs = dict()
    
    for word, count in unigram_counts.items():
        unigram_probs[word] = float(count) / total_count             
    return unigram_probs

# Let's check the MLE parameters associated with 'cat' and 'mat' by querying their unigram probabilities
unigram_prob_table = unigram_mle(unigram_count_table)
assert all(0 <= p <= 1. for p in unigram_prob_table.values()), "Probabilities are between 0 and 1"
assert np.isclose(sum(unigram_prob_table.values()), 1., 0.0001), "The coordinates of a probability vector add up to 1.0"

# Use table to check probability of a token
print('unigram=cat prob=%f' % unigram_prob_table.get('cat', 0.0))  # 0.0 is the default, returned in case the key is not in the dict
print('unigram=mat prob=%f' % unigram_prob_table.get('mat', 0.0))
print('unigram=bgafsiu prob=%f' % unigram_prob_table.get('bgafsiu', 0.0))
print('unigram=</s> prob=%f' % unigram_prob_table.get('</s>', 0.0))

# now we calculate the log probability
def calculate_sentence_unigram_log_probability(sentence, word_probs, EOS="</s>", UNK="<unk>"):
    """
    input: list of words in a sentence
    word_probs: MLE paremeters
    output:
            sentence_probability_sum: log probability of the sentence
    """
    sentence_log_probability = 0.
    # we first get the probability of unknown words
    #  which by default is 0. in case '<unk>' is not in the support
    unk_probability = word_probs.get(UNK, 0.)
    for word in sentence + [EOS]:
        # this will return `unk_probability` if the word is not in the support
        word_probability = word_probs.get(word, unk_probability)  
        # it is a sum of log pboabilities
        # we use np.log because it knows that log(0) is float('-inf')
        sentence_log_probability += np.log(word_probability)
    return sentence_log_probability

print(calculate_sentence_unigram_log_probability(['the', 'cat', 'sat', 'on', 'the', 'cat'], unigram_prob_table))
# Can now be used to assess the log of joint probability of subsets in, for example, training data
sum(calculate_sentence_unigram_log_probability(s, unigram_prob_table) for i, s in zip(range(10), prep_training))

## Smoothing and normalizing the distribution into probabilities
def unigram_smoothed_mle(unigram_counts: Counter, alpha=1.0, UNK="<unk>"):
    """
    input: unigram_count: dictionary of frequency of each word
           
    output: unigram_prob: dictionary with the (smoothed) probabilty of each word 
            (parameters of the model)
    """
    unigram_probs = dict()

    total_count = sum(unigram_counts.values())
    vocab_size = len(unigram_counts) + 1  
    denominator = total_count + alpha * vocab_size 

    for word, count in chain([(UNK, 0)], unigram_counts.items()):
        unigram_probs[word] = (float(count) + alpha) / denominator    

    return unigram_probs

unigram_smoothed_prob_table = unigram_smoothed_mle(unigram_count_table)
assert all(0 <= p <= 1. for p in unigram_prob_table.values()), "Probabilities are between 0 and 1"
assert np.isclose(sum(unigram_smoothed_prob_table.values()), 1., 0.0001), "The coordinates of a probability vector add up to 1.0"

# check if it worked
log_prob_first10_before = sum(calculate_sentence_unigram_log_probability(s, unigram_prob_table) for i, s in zip(range(10), prep_training))
log_prob_first10_after  = sum(calculate_sentence_unigram_log_probability(s, unigram_smoothed_prob_table) for i, s in zip(range(10), prep_training))
assert log_prob_first10_after > log_prob_first10_before, "Smoothing generalise improves the probability of observed sentences"