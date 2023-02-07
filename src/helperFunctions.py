import math
import nltk
import re
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# function to help print time elapsed
def stringTime(start, end, show_ms=False):
    """
    Formats a given number of seconds into hours, minutes, seconds and milliseconds.

    Args:
        start (float): The start time (in seconds)
        end (float): The end time (in seconds)

    Returns:
        t (str): The time elapsed between start and end, formatted nicely.
    """
    h = "{0:.0f}".format((end-start)//3600)
    m = "{0:.0f}".format(((end-start)%3600)//60)
    s = "{0:.0f}".format(math.floor(((end-start)%3600)%60))
    ms = "{0:.2f}".format((((end-start)%3600)%60 - math.floor(((end-start)%3600)%60))*1000) # remember s = math.floor(((end-start)%3600)%60
    h_str = f"{h} hour{'' if float(h)==1 else 's'}"
    m_str = f"{'' if float(h)==0 else ', '}{m} minute{'' if float(m)==1 else 's'}"
    s_str = f"{'' if (float(h)==0 and float(m)==0) else ', '}{s} second{'' if float(s)==1 else 's'}"
    ms_str = f"{'' if (float(h)==0 and float(m)==0 and float(s)==0) else ', '}{ms} ms"

    t = f"{h_str if float(h) != 0 else ''}{m_str if float(m) != 0 else ''}{s_str if float(s) != 0 else ''}{ms_str if show_ms else ''}"
    return t

def stem_text(text, stopwords=False):
    # split sentence into individual words
    words = nltk.word_tokenize(text)

    # stem each word in list of words
    s = nltk.stem.SnowballStemmer("english", ignore_stopwords=stopwords)
    stemmed_words = [s.stem(word) for word in words]

    # recombine into sentence
    text = ' '.join(word for word in stemmed_words)

    return text

def lemmatize_text(text):
    # split sentence into individual words
    word_tokens = nltk.word_tokenize(text)

    # lemmatize each word in list of words
    l = nltk.stem.WordNetLemmatizer()
    lemmatized_words = [l.lemmatize(word) for word in word_tokens]

    # recombine into sentence
    text = ' '.join(word for word in lemmatized_words)

    return text

def remove_stopwords(text):
    # get list of stopwords from nltk package
    s = nltk.corpus.stopwords.words("english")

    # split sentence into individual words
    word_tokens = nltk.word_tokenize(text)

    # remove stopwords
    text = ' '.join(word for word in word_tokens if word not in s)

    return text

def clean_text(text):
    # remove line breaks
    text = re.sub(r'\n', ' ', text)

    # remove digits
    text = ''.join(c for c in text if not c.isdigit())

    # remove punctuation
    text = re.sub(r'[!"#$%&\'()*+,-.\/:;<=>?@^_`{|}~]+', ' ', text)

    # remove duplicate spaces
    text = re.sub(r' +', ' ', text)

    # remove standalone 's' or 't' letters, usually from apostrophe situations
    text = re.sub(r'\ss\s|\st\s', ' ', text)

    return text

def preprocess_doc(doc, stemming=False, lemmatization=False, stopwords=False):
    processed_doc = []
    for text in doc:
        text = clean_text(text)
        if (stemming):
            text = stem_text(text, stopwords=stopwords)
        if (lemmatization):
            text = lemmatize_text(text)
        if (stopwords):
            text = remove_stopwords(text)
        processed_doc.append(text)

    return processed_doc

# adapted from https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
def plot_confusion_matrix(y_test, y_pred):
    # get confusion matrix from predictions
    cm = confusion_matrix(y_test, y_pred)

    # plot confusion matrix
    ax = plt.subplot()
    heatmap = sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='Greens')
    fig = heatmap.get_figure()
    
    # labels, title and ticks
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.xaxis.set_ticklabels(['negative', 'positive'])
    ax.yaxis.set_ticklabels(['negative', 'positive'])
    ax.set_title('Confusion Matrix')
    plt.show()
    return fig

# function to get the percent change from one value to another
def percentChange(a, b):
    return ((a - b)*100)/a