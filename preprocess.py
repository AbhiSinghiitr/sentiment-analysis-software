import re
import pandas as pd
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

vocab_size = 20000
embedding_dim = 32
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000

def preprocess(text):
    text=' '.join(text)
    # Replace repeated punctuation signs with labels and add spaces
    text = re.sub(r'(\.{2,})', r' multistop ', text)
    text = re.sub(r'(\!{2,})', r' multiexclamation ', text)
    text = re.sub(r'(\?{2,})', r' multiquestion ', text)
    # Add spaces before and after single punctuation signs
    text = re.sub(r'(\.|\!|\?|\,)', r' ', text)
    
    # Lower case the text
    text = text.lower()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if not w in stop_words]
    text = ' '.join(words)
    
    # Lemmatize text
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words]
    text = ' '.join(words)

    return text

def load_emoticons(emo_filename):
    # Load emoticons and their polarity from a file
    emoticon_dict = {}
    with open(emo_filename, 'r', encoding='latin-1') as file:
        for line in file:
            emoticon, polarity = line.strip().split('\t')
            emoticon_dict[emoticon] = polarity
    return emoticon_dict

# Load emoticons and their polarity from a file
emoticon_dict = load_emoticons('./content/EmoticonLookupTable.txt')

def replace_emoticons(text, emoticon_dict=emoticon_dict):
    # Replace emoticons with their polarity and delete neutral ones
    for emoticon, polarity in emoticon_dict.items():
        pattern = re.compile(re.escape(emoticon), re.IGNORECASE)
        if polarity == '1':
            text = pattern.sub("positive", text)
        elif polarity == '-1':
            text = pattern.sub("negative", text)
        else:
            text = pattern.sub('', text)
            
    text = re.sub(r'[^a-zA-Z\s]', '', text)
            
    return text.split()

def load_slang(slang_filename):
    # Load emoticons and their polarity from a file
    slang_dict = {}
    with open(slang_filename, 'r', encoding='latin-1') as file:
        for line in file:
            slang, meaning = line.strip().split('\t')
            slang_dict[slang] = meaning
    return slang_dict

# Load emoticons and their polarity from a file
slang_dict = load_slang('./content/SlangLookupTable.txt')

def replace_slang(tokens, slang_dict=slang_dict):
    # Replace emoticons with their polarity and delete neutral ones
    for i, token in enumerate(tokens):
        if token in slang_dict:
            tokens[i] = slang_dict[token]
            
    return tokens

def label_user_topic(tokens):
    labeled_tokens = []
    for token in tokens:
        if token.startswith("@"):
            labeled_tokens.append("PERSON")
        elif token.startswith("#"):
            labeled_tokens.append("TOPIC")
        elif token.startswith("http"):
            labeled_tokens.append("URL")
        else:
            labeled_tokens.append(token)
    return labeled_tokens


def reduce_word(word):
    # Check if the word is in Roget's Thesaurus
    synsets = wn.synsets(word)
    if synsets:
        return word
    
    # Iterate over the letters in the word, starting from the end
    for i in range(len(word)-1, 1, -1):
        # If the current letter is the same as the previous one,
        # remove the current letter and check if the resulting word
        # is in Roget's Thesaurus
        if word[i] == word[i-1]:
            word = word[:i] + word[i+1:]
            synsets = wn.synsets(word)
            if synsets:
                return "STRESSED " + word

        # If the current and previous letters are the same as the one before them,
        # remove the current letter and check if the resulting word
        # is in Roget's Thesaurus
        elif i > 2 and word[i] == word[i-2]:
            word = word[:i-1] + word[i:]
            synsets = wn.synsets(word)
            if synsets:
                return "STRESSED " + word
    
    # If no match is found, return the original word
    return word


def normalize_words(tokens):
    normalized_tokens = []
    for token in tokens:
        # Check if the token is a word
        if re.match(r'\b\w+\b', token):
            # Normalize the word
            normalized_word = reduce_word(token.lower())
            # If the normalized word is different from the original word,
            # add both versions to the list of tokens
            if normalized_word != token.lower():
                normalized_tokens.append(normalized_word)
            else:
                normalized_tokens.append(token)
        else:
            normalized_tokens.append(token)
            
#     normalized_tokens = [token.split() if 'STRESSED' in token else token for token in normalized_tokens]
#     normalized_tokens = [item if not isinstance(item, list) else item for sublist in normalized_tokens for item in sublist]
    return normalized_tokens

# define the lists of negation, intensification and diminishment expressions
negation_list = ["no", "not", "never", "none", "nobody", "nowhere", "nothing", "neither", "nor", "cannot", "can't", "don't", "doesn't", "didn't", "won't", "wouldn't", "shouldn't", "couldn't", "isn't", "aren't", "ain't", "hate", "dislike", "disapprove", "disapprove of", "disagree", "disagree with", "reject", "rejects", "rejected", "refuse", "refuses", "refused", "never", "rarely", "seldom", "hardly", "scarcely", "barely"]
intensification_list = ["very", "extremely", "super", "really", "quite", "most", "more", "quite", "too", "enough", "so", "such", "just", "almost", "absolutely", "completely", "totally", "utterly", "highly", "deeply", "greatly", "seriously", "intensely", "especially", "exceedingly", "exceptionally", "particularly", "unusually", "incredibly", "undeniably", "undeniable", "emphatically", "decidedly", "really", "truly", "hugely", "mega", "ultra", "majorly", "extraordinarily", "mightily", "fully", "mightily", "perfectly", "thoroughly", "utterly", "all", "way", "significantly", "terribly", "awfully", "fantastically"]
diminishment_list = ["little", "slightly", "somewhat", "kind", "sort", "bit", "little", "moderately", "marginally", "fairly", "reasonably", "comparatively", "relatively", "tad", "touch", "extent"]


def match_modifier_words(tokens):
    matched_tokens = []
    for token in tokens:
        if token in negation_list:
            matched_tokens.append("negator")
        elif token in intensification_list:
            matched_tokens.append("intensifier")
        elif token in diminishment_list:
            matched_tokens.append("diminisher")
        else:
            matched_tokens.append(token)
    return matched_tokens

def process2(df):
    df['preprocessed_text']=df['text'].str.lower()
    df['preprocessed_text']=df['text'].str.split()
    text=df['preprocessed_text'].copy()
    text = text.apply(match_modifier_words)
    text = text.apply(preprocess)
    text = text.apply(replace_emoticons)
    text = text.apply(replace_slang)
    text = text.apply(label_user_topic)
    text = text.apply(normalize_words)
    df['preprocessed_text']=text
    return df



def process_link():
    df2 = pd.read_csv('comment.csv')
    df2.head()
    df2.columns = ['text']
    df2 = df2.reset_index(drop=True)
    # apply preprocess function to text column and store the result in new column
    df2 = process2(df2)
    df2['preprocessed_text'] = df2['preprocessed_text'].apply(lambda x: ' '.join(x))

    
    load_model = tf.keras.models.load_model('./model.keras')
    
    comments = df2['preprocessed_text']
    
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

    tokenizer.fit_on_texts(comments)
    test_text = tokenizer.texts_to_sequences(comments)
    test_text_padded = pad_sequences(test_text, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    prediction = load_model.predict(test_text_padded)

    df2['prediction'] = prediction
    # function to replace values greater than 0.5 with 1 and others with 0
    replace_func = lambda x: 1 if x > 0.5 else 0

    # apply the function to the column and store the result in a new column
    df2['sentiment'] = df2['prediction'].apply(replace_func)
    df2.to_csv('comment.csv', index=False)
    print(df2['sentiment'])
    return 


def p_text(text):
  text = text.lower()
  text = text.split()
  text = match_modifier_words(text)
  text = preprocess(text)
  text = replace_emoticons(text)
  text = replace_slang(text)
  text = label_user_topic(text)
  text = normalize_words(text)
  text = ' '.join(text)
  entry = [text]
  
  load_model = tf.keras.models.load_model('./model.keras')
  tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

  tokenizer.fit_on_texts(entry)
  test_text = tokenizer.texts_to_sequences(entry)
  test_text_padded = pad_sequences(test_text, maxlen=max_length, padding=padding_type, truncating='post')

  prediction = load_model.predict(test_text_padded)
  
  print(prediction)
  return prediction
