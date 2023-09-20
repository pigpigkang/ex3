import nltk
import matplotlib.pyplot as plt
from nltk.corpus import gutenberg, stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK stop word and part-of-speech tagging data
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')  # Download the data you need for sentiment analysis

# Read Moby Dick text
moby_dick = gutenberg.raw('melville-moby_dick.txt')

# participle
tokens = word_tokenize(moby_dick)

# Unstop word
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalnum()]

# Part-of-speech tagging
pos_tags = nltk.pos_tag(filtered_tokens)

# Statistical frequency
pos_counts = Counter(tag for word, tag in pos_tags)
top_pos = pos_counts.most_common(5)

# Morphology reduction
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'  # adjective
    elif treebank_tag.startswith('V'):
        return 'v'  # verb
    elif treebank_tag.startswith('N'):
        return 'n'  # noun
    elif treebank_tag.startswith('R'):
        return 'r'  # adverb
    else:
        return 'n'  # Default to noun

top_lemmatized = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in pos_tags[:20]]

# Sentiment analysis using VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()
sentiment_scores = [analyzer.polarity_scores(word)['compound'] for word in filtered_tokens]
average_sentiment = sum(sentiment_scores) / len(sentiment_scores)

# Judge overall emotion
overall_sentiment = "positive" if average_sentiment > 0.05 else "negative"

# Map the distribution of parts of speech
plt.figure(figsize=(10, 5))
pos_freq_dist = FreqDist(tag for word, tag in pos_tags)
pos_freq_dist.plot(title="Part of Speech Frequency Distribution")
plt.show()

# Print result
print("Top 5 Parts of Speech and their counts:")
for pos, count in top_pos:
    print(f"{pos}: {count}")

print(f"Average Sentiment Score: {average_sentiment}")
print(f"Overall Text Sentiment: {overall_sentiment}")


# Sentiment analysis using VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()
sentiment_scores = [analyzer.polarity_scores(word)['compound'] for word in filtered_tokens]
average_sentiment = sum(sentiment_scores) / len(sentiment_scores)

# Judge overall emotion
overall_sentiment = "positive" if average_sentiment > 0.05 else "negative"

# Draw a bar chart of emotion scores
plt.figure(figsize=(8, 6))
plt.bar(range(len(sentiment_scores)), sentiment_scores, color=['green' if score >= 0 else 'red' for score in sentiment_scores])
plt.xlabel('Token Index')
plt.ylabel('Sentiment Score')
plt.title('Sentiment Analysis of Moby Dick Text')
plt.show()

# Print result
print("Top 5 Parts of Speech and their counts:")
for pos, count in top_pos:
    print(f"{pos}: {count}")

print(f"Average Sentiment Score: {average_sentiment}")
print(f"Overall Text Sentiment: {overall_sentiment}")
