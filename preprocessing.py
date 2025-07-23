import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word.isalnum()]
    lemmatized = [lemmatizer.lemmatize(word) for word in filtered if word not in stop_words]
    return ' '.join(lemmatized)
