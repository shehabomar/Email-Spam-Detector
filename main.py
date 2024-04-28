# Import pandas library for data manipulation
import pandas as pd

# Load the dataset containing email data
email_data = pd.read_csv('spam_ham_dataset.csv')

# Select only the 'label' and 'text' columns from the dataset
email_data = email_data[['label', 'text']]

# Randomize the dataset to ensure randomness in data selection
data_randomized = email_data.sample(frac=1, random_state=1)

# Calculate the index for splitting the dataset into training and test sets (80% training, 20% test)
training_test_indx = round(len(data_randomized) * 0.8)

# Split the dataset into training and test sets
training_set = data_randomized[:training_test_indx].reset_index(drop=True)
test_set = data_randomized[training_test_indx:].reset_index(drop=True)

# Import the regular expression module for text processing
import re

# Define a function to filter text, removing non-alphabetic characters
def filter_text(text):
    pattern = r'[a-zA-Z]+'  # Regular expression pattern to match alphabetic characters
    matches = re.findall(pattern, text)  # Find all matches of alphabetic characters in the text
    return ' '.join(matches)  # Join the matches into a single string

# Convert text to lowercase and apply the filter_text function to remove non-alphabetic characters
training_set['text'] = training_set['text'].str.lower().apply(lambda x: filter_text(x))

# Split the text into individual words
training_set['text'] = training_set['text'].str.split()

# Create a vocab list containing unique words
vocab = [word for sms in training_set['text'] for word in sms]
vocab = list(set(vocab))  # Remove duplicates from the vocab list

# Initialize a dictionary to store word counts per email
word_counts_per_email = {unique_word: [0] * len(training_set['text']) for unique_word in vocab}

# Count occurrences of each word in each email
for index, email in enumerate(training_set['text']):
    for word in email:
        word_counts_per_email[word][index] += 1

# Convert word counts to DataFrame
word_counts = pd.DataFrame(word_counts_per_email)

# Drop the 'label' column from word_counts DataFrame
word_counts.drop(columns=['label'], inplace=True)

# Concatenate the training set DataFrame with word_counts DataFrame
trianing_set_cleaned = pd.concat([training_set, word_counts], axis=1)

# Separate spam and ham msgs from the cleaned training set
spam_msgs = trianing_set_cleaned[trianing_set_cleaned['label'] == 'spam']
ham_msgs = trianing_set_cleaned[trianing_set_cleaned['label'] == 'ham']

# Calculate probabilities: P(Spam) and P(Ham)
p_spam = len(spam_msgs) / len(trianing_set_cleaned)
p_ham = len(ham_msgs) / len(trianing_set_cleaned)

# Calculate total number of words in spam and ham msgs
n_words_per_msg = spam_msgs['text'].apply(len)
n_spam = n_words_per_msg.sum()
n_words_per_ham_msg = ham_msgs['text'].apply(len)
n_ham = n_words_per_ham_msg.sum()

# Calculate the total number of unique words in the vocab
n_vocab = len(vocab)

# Laplace smoothing parameter
alpha = 1

# Initialize dictionaries to store parameters 
parameters_spam = {unique_word: 0 for unique_word in vocab} # stores the values of probability of words given they are spam
parameters_ham = {unique_word: 0 for unique_word in vocab} # # stores the values of probability of words given they are ham

# Calculate parameters for each word in the vocab
for word in vocab[:5000]:  # Consider only the first 5000 words for efficiency
    # Calculate P(word|Spam) using Laplace smoothing
    n_word_given_spam = spam_msgs[word].sum()
    ####### formula for calculating probability of a word given it's spam is:
    p_word_given_spam = n_word_given_spam  / (n_spam + n_vocab) 
    parameters_spam[word] = p_word_given_spam
    
    # Calculate P(word|Ham) using Laplace smoothing
    n_word_given_ham = ham_msgs[word].sum()
    ####### formula for calculating probability of a word given it's ham is:
    p_word_given_ham = n_word_given_ham / (n_ham + n_vocab)
    parameters_ham[word] = p_word_given_ham

# Define a function to classify msgs as spam or ham
def classify(msg):
    msg = filter_text(msg)
    msg = msg.lower().split()
    
    # Initialize probabilities of spam and ham for the given msg
    p_spam_given_msg = p_spam
    p_ham_given_msg = p_ham

    # Update probabilities based on the presence of each word in the msg
    for word in msg:
        if word in parameters_spam and parameters_spam[word] != 0:
            p_spam_given_msg *= parameters_spam[word] # Calculates P(Spam|w1, w2, ..., wn)
            
        if word in parameters_ham and parameters_ham[word] != 0:
            p_ham_given_msg *= parameters_ham[word] # Calculates P(Ham|w1, w2, ..., wn)
            
    # Print probabilities and classify msg as spam or ham
    print('P(Spam|msg):', p_spam_given_msg)
    print('P(Ham|msg):', p_ham_given_msg)
    
    if p_ham_given_msg > p_spam_given_msg:
        print('Label: Ham')
    elif p_ham_given_msg < p_spam_given_msg:
        print('Label: Spam')
    else:
        print('Equal probabilities, have a human classify this!')

# Call the classify function with a sample msg to classify it as spam or ham
classify('Sounds good, Tom, then see u there')
