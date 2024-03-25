
##
import os.path
import math
import sys
import random
from operator import itemgetter
from collections import defaultdict
from nltk import bigrams
from collections import Counter
#----------------------------------------
#  Data input 
#----------------------------------------

# Read a text file into a corpus (list of sentences (which in turn are lists of words))
# (taken from nested section of HW0)
def readFileToCorpus(f):
    """ Reads in the text file f which contains one sentence per line.
    """
    if os.path.isfile(f):
        file = open(f, "r") # open the input file in read-only mode
        i = 0 # this is just a counter to keep track of the sentence numbers
        corpus = [] # this will become a list of sentences
        print("Reading file ", f)
        for line in file:
            i += 1
            sentence = line.split() # split the line into a list of words
            #append this lis as an element to the list of sentences
            corpus.append(sentence)
            if i % 1000 == 0:
    	#print a status message: str(i) turns int i into a string
    	#so we can concatenate it
                sys.stderr.write("Reading sentence " + str(i) + "\n")
        #endif
    #endfor
        return corpus
    else:
    #ideally we would throw an exception here, but this will suffice
        print("Error: corpus file ", f, " does not exist")
        sys.exit() # exit the script
    #endif
#enddef


# Preprocess the corpus
def preprocess(corpus):
    #find all the rare words
    freqDict = defaultdict(int)
    for sen in corpus:
	    for word in sen:
	       freqDict[word] += 1
	#endfor
    #endfor
           
    # replace rare words with unk
    for sen in corpus:
        for i in range(0, len(sen)):
            word = sen[i]
            #print(word)
            #print(freqDict[word])
            if freqDict[word] < 2:

                sen[i] = UNK
	    #endif
	#endfor
    #endfor

    #bookend the sentences with start and end tokens
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)
    #endfor
    
    return corpus
#enddef

def preprocessTest(vocab, corpus):
    #replace test words that were unseen in the training with unk
    for sen in corpus:
        for i in range(0, len(sen)):
            word = sen[i]
            if word not in vocab:
                sen[i] = UNK
	    #endif
	#endfor
    #endfor
    
    #bookend the sentences with start and end tokens
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)
    #endfor

    return corpus
#enddef

# Constants 
UNK = "UNK"     # Unknown word token
start = "<s>"   # Start-of-sentence token
end = "</s>"    # End-of-sentence-token


#--------------------------------------------------------------
# Language models and data structures
#--------------------------------------------------------------

# Parent class for the three language models you need to implement
class LanguageModel:
    # Initialize and train the model (ie, estimate the model's underlying probability
    # distribution from the training corpus)
    def __init__(self, corpus):
        print("""Your task is to implement four kinds of n-gram language models:
      a) an (unsmoothed) unigram model (UnigramModel)
      b) a unigram model smoothed using Laplace smoothing (SmoothedUnigramModel)
      c) an unsmoothed bigram model (BigramModel)
      d) a bigram model smoothed using linear interpolation smoothing (SmoothedBigramModelInt)
      """)
    #enddef

    # Generate a sentence by drawing words according to the 
    # model's probability distribution
    # Note: think about how to set the length of the sentence 
    #in a principled way
    def generateSentence(self):
        print("Implement the generateSentence method in each subclass")
        return "mary had a little lamb ."
    #emddef

    # Given a sentence (sen), return the probability of 
    # that sentence under the model
    def getSentenceProbability(self, sen):
        print("Implement the getSentenceProbability method in each subclass")
        return 0.0
    #enddef

    # Given a corpus, calculate and return its perplexity 
    #(normalized inverse log probability)
    def getCorpusPerplexity(self, corpus):
        print("Implement the getCorpusPerplexity method")
        return 0.0
    #enddef

    # Given a file (filename) and the number of sentences, generate a list
    # of sentences and write each to file along with its model probability.
    # Note: you shouldn't need to change this method
    def generateSentencesToFile(self, numberOfSentences, filename):
        filePointer = open(filename, 'w+')
        for i in range(0,numberOfSentences):
            sen = self.generateSentence()
            prob = self.getSentenceProbability(sen)

            stringGenerated = str(prob) + " " + " ".join(sen) 
            print(stringGenerated, end="\n", file=filePointer)
            
	#endfor
    #enddef
#endclass

# Unigram language model
class UnigramModel(LanguageModel):
    def __init__(self, corpus):
        print("Unigram language model....")
        self.unigramDist = UnigramDist(corpus)
    #endddef
        
    def generateSentence(self):
        sentence = "<s> "
        while True:
            word = self.unigramDist.draw()
            if word == '</s>':
                sentence += word
                break
            sentence  += word + " "
        return sentence
    
    def getSentenceProbability(self,sen):
        sentence_prob = 1
        for word in sen.split():
            sentence_prob *= self.unigramDist.prob(word)
        #print(sentence_prob)
        return sentence_prob
    
    def getCorpusPerplexity(self, corpus):
        numWords = 0
        corpus_prob = 1
        for word in corpus.split():
            numWords += 1
            corpus_prob *= self.unigramDist.prob(word)
        try:
            pp = (1.0/corpus_prob)**(1.0/numWords)
            return pp
        except:
            print("Perplexity cannot be calculated as probability reaches to very small value nearly 0")
#endclass

#Smoothed unigram language model (use laplace for smoothing)
class SmoothedUnigramModel(LanguageModel):
    def __init__(self, corpus):
        print("Smoothed Unigram language model.....")
        self.smoothedUnigramDist = SmoothedUnigramDist(corpus)
    #endddef
        
    def generateSentence(self):
        sentence = "<s> "
        while True:
            word = self.smoothedUnigramDist.draw()
            if word == '</s>':
                sentence += word
                break
            sentence += word + " "
        return sentence
    
    def getSentenceProbability(self,sen):
        sentence_prob = 1
        for word in sen.split():
            sentence_prob *= self.smoothedUnigramDist.prob(word)
        #print(sentence_prob)
        return sentence_prob
    
    def getCorpusPerplexity(self, corpus):
        numWords = 0
        corpus_prob = 1
        for word in corpus.split():
            numWords += 1
            corpus_prob *= self.smoothedUnigramDist.prob(word)
        try:
            pp = (1.0/corpus_prob)**(1.0/numWords)
            return pp
        except:
            print("Perplexity cannot be calculated as probability reaches to very small value nearly 0")
#endclass

# Unsmoothed bigram language model
class BigramModel(LanguageModel):
    def __init__(self, corpus):
        print("Bigram language model....")
        self.bigramDist = BigramDist(corpus)
    #endddef
        
    def generateSentence(self):
        sentence = "<s> "
        for _ in range(20):
            bigram = self.bigramDist.draw()
            if bigram[1] == '</s>' or bigram[0] == '</s>':
                sentence += bigram[0] + " " + bigram[1] + " "
                break
            sentence += bigram[0] + " " + bigram[1] + " "
        
        return sentence + "</s>"

    def getSentenceProbability(self, sentence):
        sen_bigrams = list(bigrams(sentence.split()))
        sentence_prob = 1
        for bigram in sen_bigrams:
            sentence_prob *= self.bigramDist.prob(bigram)
        
        return sentence_prob
    
    def getCorpusPerplexity(self, corpus):
        corpus_bigrams = list(bigrams(corpus.split()))
        numWords = len(corpus.split())
        corpus_prob = 1
        for bigram in corpus_bigrams:
            corpus_prob *= self.bigramDist.prob(bigram)
        try:
            pp = (1.0/corpus_prob)**(1.0/numWords)
            return pp
        except:
            print("Perplexity cannot be calculated because of 0 probability due to unseen Bigram")
#endclass

# Smoothed bigram language model (use linear interpolation for smoothing, set lambda1 = lambda2 = 0.5)
class SmoothedBigramModelKN(LanguageModel):
    def __init__(self, corpus):
        print("Smoothed Bigram Model....")
        self.smoothedBigramDist = SmoothedBigramDist(corpus)
    #endddef
        
    def generateSentence(self):
        sentence = "<s> "
        for _ in range(20):
            bigram = self.smoothedBigramDist.draw()
            if bigram[1] == '</s>' or bigram[0] == '</s>':
                sentence += bigram[0] + " " + bigram[1] + " "
                break
            sentence += bigram[0] + " " + bigram[1] + " "
        
        return sentence + "</s>"

    def getSentenceProbability(self, sentence):
        sen_bigrams = list(bigrams(sentence.split()))
        sentence_prob = 1
        for bigram in sen_bigrams:
            sentence_prob *= self.smoothedBigramDist.prob(bigram)
        
        return sentence_prob
    
    def getCorpusPerplexity(self, corpus):
        corpus_bigrams = list(bigrams(corpus.split()))
        numWords = len(corpus.split())
        corpus_prob = 1
        for bigram in corpus_bigrams:
            corpus_prob *= self.smoothedBigramDist.prob(bigram)
        try:
            pp = (1.0/corpus_prob)**(1.0/numWords)
            return pp
        except:
            print("Perplexity cannot be calculated because of 0 probability due to unseen Bigram")
#endclass
#endclass



# Sample class for a unsmoothed unigram probability distribution
# Note: 
#       Feel free to use/re-use/modify this class as necessary for your 
#       own code (e.g. converting to log probabilities after training). 
#       This class is intended to help you get started
#       with your implementation of the language models above.
class UnigramDist:
    def __init__(self, corpus):
        self.counts = defaultdict(float)
        self.total = 0.0
        self.train(corpus)
    #endddef

    # Add observed counts from corpus to the distribution
    def train(self, corpus):
        for sen in corpus:
            for word in sen:
                if word == start:
                    continue
                self.counts[word] += 1.0
                self.total += 1.0
        self.counts["<s>"] = self.total
            #endfor
        #endfor
    #enddef

    # Returns the probability of word in the distribution
    def prob(self, word):
        p = self.counts[word] / self.total
        return p
    #enddef

    # Generate a single random word according to the distribution
    def draw(self):
        rand = random.uniform(0.0,0.8)
        for word in self.counts.keys():
            rand -= self.prob(word)
            if rand <= 0.0:
                return word
	    #endif
	#endfor
    #enddef
#endclass
            
class SmoothedUnigramDist(UnigramDist):

    def prob(self, word):
        p = (self.counts[word]+1) / (self.counts[word] + self.total)
        return p
    
class BigramDist:

    def __init__(self, corpus):
        self.counts = defaultdict(float)
        self.total = 0.0
        self.bigram_counts = None
        self.corpus_bigrams = None
        self.train(corpus)
    
    def train(self, corpus):
        for sen in corpus:
            for word in sen:
                if word == start:
                    continue
                self.counts[word] += 1.0
                self.total += 1.0
        self.counts["<s>"] = self.total

        flat_corpus = [word for sentence in corpus for word in sentence]
        self.corpus_bigrams = list(bigrams(flat_corpus))
        self.bigram_counts = Counter(self.corpus_bigrams)

    def prob(self, bigram):
        p = self.bigram_counts[bigram] / self.counts[bigram[0]]
        return p
    
    def draw(self):
        rand = random.uniform(1.0, 3.0)
        for bigram in self.bigram_counts.keys():
            rand -= self.prob(bigram)
            if rand <= 0.0:
                return bigram

class SmoothedBigramDist(BigramDist):

    def prob(self, bigram):
        p =  (0.5)*(self.bigram_counts[bigram] / self.counts[bigram[0]]) + (0.5)*(self.counts[bigram[1]] / self.total) 
        return p
    
    def draw(self):
        rand = random.uniform(0.3, 2.0)
        for bigram in self.bigram_counts.keys():
            rand -= self.prob(bigram)
            if rand <= 0.0:
                return bigram


def list_to_str(list_corpus):
    single_string = ""
    for sentence in list_corpus:
        for word in sentence:
            single_string += word + " "

    return single_string

#-------------------------------------------
# The main routine
#-------------------------------------------
if __name__ == "__main__":
    #read your corpora
    trainCorpus = readFileToCorpus('train.txt')
    trainCorpus = preprocess(trainCorpus)

    #print(trainCorpus)
    
    posTestCorpus = readFileToCorpus('pos_test.txt')
    negTestCorpus = readFileToCorpus('neg_test.txt')

    flat_corpus = [word for sentence in trainCorpus for word in sentence]

    vocab = set(flat_corpus)

    # Please write the code to create the vocab over here before the function preprocessTest
    #print("""Task 0: create a vocabulary(collection of word types) for the train corpus""")


    posTestCorpus = preprocessTest(vocab, posTestCorpus)
    str_posTestCorpus = list_to_str(posTestCorpus)
    negTestCorpus = preprocessTest(vocab, negTestCorpus)
    str_negTestCorpus = list_to_str(negTestCorpus)

    test_sen = [['interstellar', 'is', 'one', 'of', 'my', 'favourite', 'movies', '.']]
    test = list_to_str(preprocessTest(vocab, test_sen))
    print()

    unigram_model = UnigramModel(trainCorpus)
    unigram_model.generateSentencesToFile(5,"unigramModel.txt")
    print("Unigram Model Generated Sentence : ",unigram_model.generateSentence())
    print("Unigram Sentence Prob: ",unigram_model.getSentenceProbability(test))
    print("Given Sentence Perplexity using Unigram Model: ",unigram_model.getCorpusPerplexity(test))
    print("posTestCorpus Prob using Unigram Model: ",unigram_model.getSentenceProbability(str_posTestCorpus))
    print("posTestCorpus Perplexity using Unigram Model: ",unigram_model.getCorpusPerplexity(str_posTestCorpus))
    print("negTestCorpus Prob using Unigram Model: ",unigram_model.getSentenceProbability(str_negTestCorpus))
    print("negTestCorpus Perplexity using Unigram Model: ",unigram_model.getCorpusPerplexity(str_negTestCorpus),"\n")


    smoothedUnigramModel = SmoothedUnigramModel(trainCorpus) 
    smoothedUnigramModel.generateSentencesToFile(5,"smoothedUnigramModel.txt")
    print("Smoothed Unigram Model Generated Sentence: ",smoothedUnigramModel.generateSentence())
    print("Smoothed Unigram Sentence Prob: ",smoothedUnigramModel.getSentenceProbability(test))
    print("Given Sentence Perplexity using Smoothed Unigram Model: ",smoothedUnigramModel.getCorpusPerplexity(test))
    print("posTestCorpus Prob using Smoothed Unigram Model: ",smoothedUnigramModel.getSentenceProbability(str_posTestCorpus))
    print("posTestCorpus Perplexity using Smoothed Unigram Model: ",smoothedUnigramModel.getCorpusPerplexity(str_posTestCorpus))
    print("negTestCorpus Prob using Smoothed Unigram Model: ",smoothedUnigramModel.getSentenceProbability(str_negTestCorpus))
    print("negTestCorpus Perplexity using Smoothed Unigram Model: ",smoothedUnigramModel.getCorpusPerplexity(str_negTestCorpus),"\n")


    bigram_model = BigramModel(trainCorpus)
    bigram_model.generateSentencesToFile(5, "bigramModel.txt")
    print("Bigram Model Generated Sentence : ",bigram_model.generateSentence())
    print("Bigram Sentence Prob: ",bigram_model.getSentenceProbability(test))
    print("Given Sentence Perplexity using Bigram Model: ",bigram_model.getCorpusPerplexity(test))
    print("posTestCorpus Prob using Bigram Model: ",bigram_model.getSentenceProbability(str_posTestCorpus))
    print("posTestCorpus Perplexity using Bigram Model: ",bigram_model.getCorpusPerplexity(str_posTestCorpus))
    print("negTestCorpus Prob using Bigram Model: ",bigram_model.getSentenceProbability(str_negTestCorpus))
    print("negTestCorpus Perplexity using Bigram Model: ",bigram_model.getCorpusPerplexity(str_negTestCorpus),"\n")

    smooth_bigram_model = SmoothedBigramModelKN(trainCorpus)
    smooth_bigram_model.generateSentencesToFile(5, "smoothBigramModel.txt")
    print("Smoothed bigram Model Generated Sentence : ",smooth_bigram_model.generateSentence())
    print("Smoothed Bigram Sentence Prob: ",smooth_bigram_model.getSentenceProbability(test))
    print("Given Sentence Perplexity using Smoothed Bigram Model: ",smooth_bigram_model.getCorpusPerplexity(test))
    print("posTestCorpus Prob using Smoothed Bigram Model: ",smooth_bigram_model.getSentenceProbability(str_posTestCorpus))
    print("posTestCorpus Perplexity using Smoothed Bigram Model: ",smooth_bigram_model.getCorpusPerplexity(str_posTestCorpus))
    print("negTestCorpus Prob using Smoothed Bigram Model: ",smooth_bigram_model.getSentenceProbability(str_negTestCorpus))
    print("negTestCorpus Perplexity using Smoothed Bigram Model: ",smooth_bigram_model.getCorpusPerplexity(str_negTestCorpus),"\n")



"""
Answers:

1- While generating sentences using Unigram Model, the sentence is generated based on single word probability. 
An end element (</s>) controls the length of sentence. When draw() function generates </s>, it terminates sentence that is without any context as it generates based on individual probabilities.
On the other hand, in Bigram sentence genration, It depends on conditional probability of occuring </s> in any bigram that results in sentence termination.
Therefore, bigram model contains more context than unigram.

2- Models assign different probabilities to different sentences because of their different nature and dependencies.
Unigram model only depends on a single word probability while bigram models depend on conditional probabilities between words.
So, probabilties are calculated according to their individual criteria therefore they are very different.

3- Smoothed Bigram Model produces better sentences but it goes on repeating many combination of words in same sentence which infact make 
sense as compared to case of any unigram model that generates totally random sentences.

4- I got 0 perplexity for both corpuses as due to large multiplication of very small probabilities, corpus probabilities ultimately reaches to zero
due to underflow. And when we have 0 probability of any corpus, we cant calculate its perplexity. I can explain it further in evaluation.
But I calculated perplexity on a single test sentence that is mentioned in code.

The test sentence is "Interstellar is one of my favourite movies."
- Unigram Model gives 353.11 perplexity for test sentence
- Smoothed Unigram Model gives 362.92 perplexity for test sentence
- Bigram Model gives 0 perplexity for test sentence because of an unseen bigram occured that results in 0 probabilty for the sentence.
- Smoothed Bigram Model gives 117.34 perplexity for test sentence

So, According to these figures, i can say that smoothed bigram model has lowest perplexity means good performance than others.
It is because it handles zero counts and more dependencies better than any other model.

"""
