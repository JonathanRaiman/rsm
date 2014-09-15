## Report on Recommendation systems for Yelp (Personal Transport Vehicle) ##

Dataset
-------

Is collected from the Yelp website by *scraping* their pages systematically with a range of Seattle zipcodes. The result is an exhaustive (in January 2013) list of Seattle restaurants and venues with their rating, price, address, and the contents of their reviews.

It should be noted however, that the reviews are notoriously scarce in nutritional information.


Assumptions
-----------

Key assumptions for this task are that the review content contains both human-readable and interpretable information for judging and connecting different restaurants.

Semantic relaxation is reliant on a good metric that is a compromise of relevancy and quality. Quality for a venue is an appreciation either obtained through elicitation or through group statistics. In the case of Yelp the aggregate reviews can serve as a summary of the group opinion on a venue.

Finally, future data may not be taxonomically classified as it is in Yelp. The classifications may be arbitrary or irrelevant to the end user. A new classification built on the fly, hierarchically sorting venues into relevant categories for the end user is preferred to tailor a query to this user.


Recommendation Strategies
-------------------------

Several methods for doing unsupervised clustering of the restaurants were attempted.

### Autoencoders (from Ranzato, M. A., and Szummer, M. at ICML 2008, and earlier) ###

Classical non generative auto-regressive strategies based around autoencoders, and sparse autoencoders were attempted. In these strategies we find that a strategy sufficiently rapid for online parsing is possible, however the results are hard to interpret due to the lack of sparsity of the results. Bag of words models seem also to focus regression tasks onto non-descriptive portions of the reviews. Moreover, the dataset may be too poor to perform this task.

### RSM (from Salakhutdinov, R., and Hinton, G. at NIPS 2009)###

The Replicated Softmax Strategy is a generative neural net that uses fixed size vectors to perform an auto-regressive task resulting in a generative topic model for the reviews. No particular pattern was found between the original category labels and the unsupervised clusters. The clusters did exhibit the *starfish* pattern described in the litterature, thereby showing that unsupervised clusters did emerge, and did correlate with some feature set.


### Paragraph Vector Distributed Memory (from Le, Q.V., and Mikolov, T. at ICML 2014)

The PVDM strategy shows promise on short passages for finding correlations based on distributional properties of words and syntactic constructs (as visible from the results of word2vec by Mikolov et al.), however the learnt embeddings for the paragraphs in this do not yield credible distances (Domino's pizza is closer to many other Pho restaurants than to any other pizza restaurant). In appears that the deeply stochastic nature of this task makes the dataset pivotal in obtaining good embeddings. Thus the data may be too noisy and variable to be useful for a topic-model like strategy.


Next Steps
----------

The next step is to find a way of enriching the modeled text to confirm that data poverty is the cause of the discrepancies. Next we should also evaluate whether a more grammatical parse would be useful (perhaps by applying a gated memory model to the reviews to lift crucial parts from the text) to separate noise from data.
