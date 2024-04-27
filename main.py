# Import pandas library for data manipulation
import pandas as pd

# Load the dataset containing email data
email_data = pd.read_csv('spam_ham_dataset.csv')

# Select only the 'label' and 'text' columns from the dataset
email_data = email_data[['label', 'text']]

# Randomize the dataset to ensure randomness in data selection
data_randomized = email_data.sample(frac=1, random_state=1)

# Calculate the index for splitting the dataset into training and test sets (80% training, 20% test)
training_test_index = round(len(data_randomized) * 0.8)

# Split the dataset into training and test sets
training_set = data_randomized[:training_test_index].reset_index(drop=True)
test_set = data_randomized[training_test_index:].reset_index(drop=True)

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

# Create a vocabulary list containing unique words
vocabulary = [word for sms in training_set['text'] for word in sms]
vocabulary = list(set(vocabulary))  # Remove duplicates from the vocabulary list

# Initialize a dictionary to store word counts per email
word_counts_per_email = {unique_word: [0] * len(training_set['text']) for unique_word in vocabulary}

# Count occurrences of each word in each email
for index, email in enumerate(training_set['text']):
    for word in email:
        word_counts_per_email[word][index] += 1

# Convert word counts to DataFrame
word_counts = pd.DataFrame(word_counts_per_email)

# Drop the 'label' column from word_counts DataFrame
word_counts.drop(columns=['label'], inplace=True)

# Concatenate the training set DataFrame with word_counts DataFrame
training_set_clean = pd.concat([training_set, word_counts], axis=1)

# Separate spam and ham messages from the cleaned training set
spam_messages = training_set_clean[training_set_clean['label'] == 'spam']
ham_messages = training_set_clean[training_set_clean['label'] == 'ham']

# Calculate probabilities: P(Spam) and P(Ham)
p_spam = len(spam_messages) / len(training_set_clean)
p_ham = len(ham_messages) / len(training_set_clean)

# Calculate total number of words in spam and ham messages
n_words_per_spam_message = spam_messages['text'].apply(len)
n_spam = n_words_per_spam_message.sum()
n_words_per_ham_message = ham_messages['text'].apply(len)
n_ham = n_words_per_ham_message.sum()

# Calculate the total number of unique words in the vocabulary
n_vocabulary = len(vocabulary)

# Laplace smoothing parameter
alpha = 1

# Initialize dictionaries to store parameters
parameters_spam = {unique_word: 0 for unique_word in vocabulary}
parameters_ham = {unique_word: 0 for unique_word in vocabulary}

# Calculate parameters for each word in the vocabulary
for word in vocabulary[:5000]:  # Consider only the first 5000 words for efficiency
    # Calculate P(word|Spam) using Laplace smoothing
    n_word_given_spam = spam_messages[word].sum()
    p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + alpha * n_vocabulary)
    parameters_spam[word] = p_word_given_spam
    
    # Calculate P(word|Ham) using Laplace smoothing
    n_word_given_ham = ham_messages[word].sum()
    p_word_given_ham = (n_word_given_ham + alpha) / (n_ham + alpha * n_vocabulary)
    parameters_ham[word] = p_word_given_ham

# Define a function to classify messages as spam or ham
def classify(message):
    message = filter_text(message)
    message = message.lower().split()
    
    # Initialize probabilities of spam and ham for the given message
    p_spam_given_message = p_spam
    p_ham_given_message = p_ham

    # Update probabilities based on the presence of each word in the message
    for word in message:
        if word in parameters_spam and parameters_spam[word] != 0:
            p_spam_given_message *= parameters_spam[word]
            
        if word in parameters_ham and parameters_ham[word] != 0:
            p_ham_given_message *= parameters_ham[word]
            
    # Print probabilities and classify message as spam or ham
    print('P(Spam|message):', p_spam_given_message)
    print('P(Ham|message):', p_ham_given_message)
    
    if p_ham_given_message > p_spam_given_message:
        print('Label: Ham')
    elif p_ham_given_message < p_spam_given_message:
        print('Label: Spam')
    else:
        print('Equal probabilities, have a human classify this!')

# Call the classify function with a sample message to classify it as spam or ham
classify('Sounds good, Tom, then see u there')
