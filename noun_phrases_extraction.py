### Adapted from https://gist.github.com/alexbowe/879414

import nltk
from nltk.corpus import stopwords
import tagger


###############################################################################
## Lemmatizer, Stemmer, Chunker, Stopwords, Tagger, etc.
###############################################################################

sentence_re = r'''(?x)      # set flag to allow verbose regexps
      ([A-Z])(\.[A-Z])+\.?  # abbreviations, e.g. U.S.A.
    | \w+(-\w+)*            # words with optional internal hyphens
    | \$?\d+(\.\d+)?%?      # currency and percentages, e.g. $12.40, 82%
    | \.\.\.                # ellipsis
    | [][.,;"'?():-_`]      # these are separate tokens
'''

lemmatizer = nltk.WordNetLemmatizer()
stemmer = nltk.stem.porter.PorterStemmer()

# Taken from Su Nam Kim Paper
grammar = r"""
    # Nouns and Adjectives, terminated with Nouns
    NBAR:
        {<NN.*|JJ>*<NN.*>}
        
    # Above, connected with preposition or subordinating conjunction (in, of, etc...)
    NP:
        {<NBAR>}
        {<NBAR><IN><NBAR>}
"""
chunker = nltk.RegexpParser(grammar)
stopwords = stopwords.words('english')
tagger = tagger.tagger()

###############################################################################
## Helper function for normalizing words and extracting noun phrases from
## the Syntax Tree
###############################################################################

def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter = lambda t: t.node=='NP'):
        yield subtree.leaves()


def normalize(word):
    """Normalizes words to lowercase and stems and lemmatizes it."""
    word = word.lower()
    #word = stemmer.stem_word(word)
    word = lemmatizer.lemmatize(word)
    return word


def acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword."""
    accepted = bool(2 <= len(word) <= 40
        and word.lower() not in stopwords)
    return accepted


def get_terms(tree):
    """Get all the acceptable noun_phrase term from the syntax tree"""
    for leaf in leaves(tree):
        term = [ normalize(w) for w,t in leaf if acceptable_word(w) ]
        yield term


def extract_noun_phrases(text):
    """Extract all noun_phrases from the given text"""
    toks = nltk.regexp_tokenize(text, sentence_re)
    postoks = tagger.tag(toks)

    # Build a POS tree
    tree = chunker.parse(postoks)
    terms = get_terms(tree)

    # Extract Noun Phrase
    noun_phrases = []
    for term in terms:
        np = ""
        for word in term:
            np += word + " "
        if np != "":
            noun_phrases.append(np.strip())
    return noun_phrases


if __name__ == "__main__":
    text = """Pokemon is a media franchise published and owned by Japanese 
    video game company Nintendo and created by Satoshi Tajiri in 1996. 
    Originally released as a pair of interlinkable Game Boy role-playing 
    video games developed by Game Freak, Pokemon has since become the 
    second-most successful and lucrative video game-based media franchise 
    in the world, behind only Nintendo's own Mario franchise."""

    noun_phrases = extract_noun_phrases(text)
    print noun_phrases
