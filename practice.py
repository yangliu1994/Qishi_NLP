
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

analyze = vectorizer.build_analyzer()

print(analyze('I love you'))
