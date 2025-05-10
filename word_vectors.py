
# now some imports for toolkits: numpy gives us fast
# vector and matrix operations;
import numpy as np

# now for nltk
import nltk

from collections import Counter

# this is the reuters data set
from nltk.corpus import reuters

from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plot

# many data sets use start and end delimiters
START_TOKEN = '<START>'
END_TOKEN = '<END>'

def read_reuters_corpus(category="gold", debug=False):
    """ Reads and parses the files from the specified Reuter's category.

    Params:
        category (str): category name from reuters/cats.txt
    Return: 
        list of lists with words from each processed file
    """
    files = reuters.fileids(category)
    file_words = []

    # walk through the documents and add the START_TOKEN and END_TOKEN to each document
    for file_id in files:
        # fetch the words for each file and prepend with START_TOKEN and append with END_TOKEN
        words = [START_TOKEN] + list(reuters.words(fileids=[file_id])) + [END_TOKEN]
        words = [word.lower() for word in words]
        # append the tokenized and delimited words to the file_words list
        file_words.append(words)

    # optionally output the first 3 examples if debug mode is enabled
    if debug:
        print(f'First 3 examples: {file_words[:3]}')
        
    return file_words


def get_vocabulary(word_lists : list[list[str]]) -> list[str]:
    """
    From the list of lists of words, create a sorted list of words

    Parameters:
        word_lists: list of list of words from the text

    Returns:
        A sorted list of unique words in the vocabulary 
    """
    # TODO: return a sorted list of all the unique words in all the documents



#generate vector terms off of the word lists and 
    all_words = [word for sublist in word_lists for word in sublist]
    frequency = Counter(all_words)
    unique_words = sorted(frequency, key=frequency.get, reverse=True)

    return unique_words, frequency
    

def create_indexed_vocabulary(vocab : list[str]) -> dict[str, int]:
    """
    Creates a dictionary of vocabulary words to indices.

    Parameters:
        vocab: a sorted list of unique words in the corpus

    Returns:
        A dictionary that maps word to index in vocabulary list
    """
    # TODO: return a dictionary of vocab to indices (assumg the vocab is already sorted)
    vocab_indices = {word: idx for idx, word in enumerate(vocab)}
    return vocab_indices


def get_co_occurrence_matrix(corpus : list[list[str]], vocab_indices: dict[str, int], vocab_size, window_size = 4, debug = False) -> np.ndarray[np.float64]:
    """
    Create the co-occurrence matrix. 
    
    Parameters:
        corpus: a list of each document, where each document is a list of words in that document
        and includes the start and end tokens (see read_reuters_corpus)

        vocab_indices: a dictionary of word to index mappings

        vocab_size: the number of words in the vocabulary

        window_size: the number of words + or - the center word

    """

    # TODO: walk through all the documents and then walk through each word as a center word
    # updating the matrix M with the count of the number of times a word was seen as a neighbor word
    # for a given center word

        # Create a matrix of zeros with dimensions vocab_size x vocab_size
    M = np.zeros((vocab_size, vocab_size), dtype=np.float64)

#each word in the corpus
    for document in corpus:
        # each word in the document
        for i, word in enumerate(document):
            # grab center word index from the vocab_indices
            center_word_index = vocab_indices.get(word, None)
            if center_word_index is not None:
                #find start and end index 
                start = max(0, i - window_size)
                end = min(len(document), i + window_size + 1)
                
                #iterate over each word in the window
                for j in range(start, end):
                    if i != j:  #center word shouldnt be counted as a neighbor
                        neighbor_word = document[j]
                        neighbor_index = vocab_indices.get(neighbor_word, None)
                        if neighbor_index is not None:
                            M[center_word_index][neighbor_index] += 1

    if debug:
        print(f'Co-occurrence matrix shape -> {M.shape}')

    return M


def reduce_to_k_dim(M: np.ndarray[np.float64], k: int = 2, iterations: int = 10) -> np.ndarray[np.float64]:
    """
    Reduce a co-occurrence matrix of dimensionality (vocab_size, vocab_size) to k dimensions
    using singular value decomposition (SVD). You will use TruncatedSVD.

    Parameters:
        M (numpy matrix of shape (vocab_size, vocab_size)): created from your sample
        k (int): the number of dimensions to reduce to, 2 by default
        iterations (int): number of iterations for SVD computation

    Return:
        M_reduced: (numpy matrix of shape (vocab_size, k)). This represents the matrix U * S.
    """
    print(f"Running Truncated SVD over {M.shape[0]} words...")
        # TODO: call TruncatedSVD with k components and the given number of iterations, 

    svd = TruncatedSVD(n_components=k, n_iter=iterations, random_state=42)
    
    # Perform the dimensionality reduction on the co-occurrence matrix
    M_reduced = svd.fit_transform(M)
    
    # now print out the work here

    print(f'Reduced shape: {M_reduced.shape}')
    print(f'Matrix:\n {M_reduced}')

    return M_reduced


def plot_embeddings(M_reduced : np.ndarray[np.float64], word_indices : dict[str, int], vocab : list[str], filename: str) -> None:
    """
    Scatter plot the embeddings on a 2D graph of the embeddings of the
    words specified in "words". Be sure to label each point.
    """
    for word in vocab:
        x = M_reduced[word_indices[word]][0]
        y = M_reduced[word_indices[word]][1]
        plot.scatter(x, y, marker='.', color='red')
        plot.text(x, y, word, fontsize=9)

    plot.savefig(filename)
    plot.close()

def main():
    # set up the figure size for plotting
    plot.rcParams['figure.figsize'] = [10, 5]

    # download 'reuters' dataset if you don't have it
    nltk.download('reuters')

    # Step 1: read the corpus
    corpus = read_reuters_corpus(category="gold", debug=True)

    #i need to figure out a freqencu
    # Step 2: Get the vocabular
    vocab, frequency = get_vocabulary(corpus)  #i tried to use the frequency to filter out the words that are not frequent enough but it was not working
    # Step 3: get the vocabulary size
    vocab_size = len(vocab)

    # Step 4: get the vocabulary indices
    vocab_indices = create_indexed_vocabulary(vocab)

    # Step 5: get the co-occurrence matrix
    co_matrix = get_co_occurrence_matrix(corpus, vocab_indices, vocab_size)
    print(f'Co-occurrence matrix: \n{co_matrix}')

    # Step 6: Reduce the matrix using SVD
    co_matrix_reduced = reduce_to_k_dim(co_matrix)

    #normalization of the matrix to get the cosine similarity
    M_lengths = np.linalg.norm(co_matrix_reduced, axis=1)
    normalized_mat = co_matrix_reduced / M_lengths[:, np.newaxis]

    #select words to visualize
    words_to_visualize = ['gold', 'market', 'trade', 'investment', 'currency'] #investment catergory
    words_to_visualize = [word for word in words_to_visualize if word in vocab_indices]  # Filter words actually in vocab

    # Step 7: Plot embeddings for the selected words and save the plot
    plot_embeddings(normalized_mat, vocab_indices, words_to_visualize, 'real_data_plotSecond.png')
    print("Visualization has been saved as 'real_data_plot.png'.")

if __name__ == '__main__':
    main()
