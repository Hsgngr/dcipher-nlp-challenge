# Introduction
This report outlines my approach to the dcipher-nlp-challenge's Binary Classifier for academic articles. The goal of the project is to correctly classify academic article's label as
"Chemistry" or "Material Science". We are given 7494 article with their titles and abstracts. As we are using text as our input data, the project falls under the category of Text Classification - Natural Language Process (NLP) 

The main challenges of the project are three-fold:
 * There are two different inputs,'Title' and 'Abstract'. Title is short enough to improve the model on it however abstract has many valuable information about the article's label. It was challenging to select between two and hard to combine.
 * Since these are academic articles, many of them had unique words which are not part of the pre-trained embedding models such as GLOVE.
 * There are overwhelmingly many different models as open-source and lack of computation power was a huge struggle since it wasn't viable to try many of them and NLP models generally have many trainable parameters.

The next sections should give an insight into how I have adressed these challenges.

# Approach
To adress the problem's complexity, I started with an MVP where I was only using a small dataset of GLOVE embeddings and taking the average of each word's vector in the title of articles. Although this wasn't a huge success it helped me to create my pipeline around the project. Later I continued with Google's BERT model which has many options and easier to implement with TensorFlow.
# Methodology

# Results and Discussion

# Conclusion
