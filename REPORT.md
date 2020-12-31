# Introduction
This report outlines my approach to the dcipher-nlp-challenge's Binary Classifier for academic articles. The goal of the project is to correctly classify academic article's label as
"Chemistry" or "Material Science". We are given 7494 article with their titles and abstracts. As we are using text as our input data, the project falls under the category of Text Classification - Natural Language Process (NLP) 

The main challenges of the project are three-fold:
 * There are two different inputs,'Title' and 'Abstract'. Title is short enough to improve the model on it however abstract has many valuable information about the article's label. It was challenging to select between two and hard to combine.
 * Since these are academic articles, many of them had unique words which are not part of the pre-trained embedding models such as GLOVE.
 * There are overwhelmingly many different models as open-source and lack of computation power was a huge struggle since it wasn't viable to try many of them and NLP models generally have many trainable parameters.

The next sections should give an insight into how I have adressed these challenges.

# Approach

As the project required I split my dataset into training and test sets by stratifying the Label column of the data. I saved them in data folder.

```              
                       TRAIN       TEST
Material Science:      3033        759
Chemistry:             2962        740
```

To adress the problem's complexity, I started with an MVP where I was only using a small dataset of GLOVE embeddings (6B and 50d). As preprocessing, I split the sentences into word lists convert them to lower case and remove all punctuations from the Titles. When I saw some words like 'self-assembly', I understood that it was a mistake to delete '-' and some others so I customized the punctuations that I'm removing. I have found there were 14433 unique words just in Title and when I used Glove Embeddings, 7194 of them didn't have an embedding in the GLOVE model. So I was curious whether if I used really small dataset of GLOVE or these words or not common enough to be in the vocabularies of these models. I decided to use one of the GLOVE's bigger dataset which includes 840B words with 300 columns) and still got 5781 unknown words I embedded unknown words as 'unk' and you can see how many times did they used in titles. 

```    Total unique words:
                                 GLOVE_6B.50d       GLOVE_840B.300d
        Total unique words:      14433              14433      
       Total unknown words:      7194               5781
Total count of <unk> token:      12994              7365
```

As the most naive approach I took the average vector of each sentences by adding each words together and dividing them by the counter of the word.Although this wasn't a huge success it helped me to create my pipeline around the project. Since nearly half of the embeddings were unknown I decided to create my own embeddings from the scratch with TensorFlow.


Google's BERT model which has many options and easier to implement with TensorFlow.

# Results and Discussion

# Conclusion
