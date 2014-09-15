

    from imp import reload
    import rsm
    import numpy as np
    import theano as T
    from matplotlib import pyplot as plt
    import utils
    import batch_data
    import time
    from batch_data import BatchData as Batch
    from tsne import bh_sne
    %matplotlib inline


    # if you need to reload the code shift-enter here !
    reload(utils)
    reload(batch_data)
    from batch_data import BatchData as Batch

Connect to **MongoDB**:


    # make sure mongo is running somewhere :
    utils.connect_to_database(database_name = 'yelp')

Use database to construct a lexicon (hash-table mapping words to vector
dimensions):


    lexicon, reverse_lexicon = utils.gather_lexicon('restaurants')

Create a **Replicated Softmax Machine**:


    # if you need to reload the replicated softmax code:
    reload(rsm);


    batch_size = 100
    learning_rate = 0.001 / batch_size
    encoder = rsm.RSM(momentum = 0.9, data = np.zeros([0,len(lexicon.items())]), hidden_units = 123, learning_rate=learning_rate)
    errors = np.zeros(0)

Create the stochastic batch element with 100 elements per mini-batch:


    rc = utils.ResourceConverter(lexicon = lexicon)
    batch = Batch(
        data=utils.mongo_database_global['restaurants'].find(), # from Mongo's cursor enumerator
        batch_size = batch_size,  # mini-batch
        shuffle = True, # stochastic
        conversion = rc.process # convert to matrices using lexicon)
    )

Start mini-batch learning for 1000 epochs:


    epochs = 1000
    batch.batch_size = 100
    new_errors = np.zeros(epochs)
    start_time = time.time()
    encoder.learning_rate = 0.001
    
    for epoch in range(epochs):
        if epoch > 0 and epoch % 50 == 0:
            encoder.k = int(1 + (float(epoch) / float(epochs)) * 4)
        if epoch > 0 and epoch % 100 == 0:
            encoder.learning_rate = max(0.00001, encoder.learning_rate.get_value()*0.98)
        encoder.data = batch.next()
        new_errors[epoch] = encoder.train()
        if epoch > 0 and epoch % 10 == 0:
            print("Epoch[%2d] : MSE = %.02f [# Gibbs steps=%d] elapsed = %.05fs" % (epoch, new_errors[epoch],encoder.k, time.time() - start_time))
    errors = np.append(errors, new_errors)

    Epoch[10] : MSE = 0.00 [# Gibbs steps=1] elapsed = 6.77807s



    ---------------------------------------------------------------------------
    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-11-197c176fcba3> in <module>()
         11         encoder.learning_rate = max(0.00001, encoder.learning_rate.get_value()*0.98)
         12     encoder.data = batch.next()
    ---> 13     new_errors[epoch] = encoder.train()
         14     if epoch > 0 and epoch % 10 == 0:
         15         print("Epoch[%2d] : MSE = %.02f [# Gibbs steps=%d] elapsed = %.05fs" % (epoch, new_errors[epoch],encoder.k, time.time() - start_time))


    /Users/jonathanraiman/Documents/Master/research/deep_learning/restaurant_rsm/rsm.py in train(self)
        432                                 visible_fantasy,
        433                                 hidden_fantasy,
    --> 434                                 scaling_factor
        435 				)
        436                         #if step == 0:


    /Users/jonathanraiman/Documents/Master/research/deep_learning/restaurant_rsm/rsm.py in contrastive_divergence(self, visible, hidden, scaling)
        348 		[visible_fantasy_pdf, likelihood] = self.propdown(
        349                         visible,
    --> 350 			hidden)
        351 
        352 		visible_fantasy = self.sample_visible(


    /usr/local/lib/python3.3/site-packages/theano/compile/function_module.py in __call__(self, *args, **kwargs)
        577         t0_fn = time.time()
        578         try:
    --> 579             outputs = self.fn()
        580         except Exception:
        581             if hasattr(self.fn, 'position_of_error'):


    KeyboardInterrupt: 


maybe this could work to **visualize** the categories ?


    cat_list = ["restaurants", "afghani", "african", "senegalese",
"southafrican", "New", "newamerican", "Traditional", "tradamerican", "arabian",
"argentine", "armenian", "asianfusion", "australian", "austrian", "bangladeshi",
"bbq", "basque", "belgian", "brasseries", "brazilian", "breakfast_brunch",
"british", "buffets", "burgers", "burmese", "cafes", "cafeteria", "cajun",
"cambodian", "caribbean", "dominican", "haitian", "puertorican", "trinidadian",
"catalan", "cheesesteaks", "chicken_wings", "chinese", "cantonese", "dimsum",
"shanghainese", "szechuan", "comfortfood", "creperies", "cuban", "czech",
"delis", "diners", "ethiopian", "hotdogs", "filipino", "fishnchips", "fondue",
"food_court", "foodstands", "french", "gastropubs", "german", "gluten_free",
"greek", "halal", "hawaiian", "himalayan", "hotdog", "hotpot", "hungarian",
"iberian", "indpak", "indonesian", "irish", "italian", "japanese", "korean",
"kosher", "laotian", "latin", "colombian", "salvadoran", "venezuelan",
"raw_food", "malaysian", "mediterranean", "falafel", "mexican", "mideastern",
"egyptian", "lebanese", "modern_european", "mongolian", "moroccan", "pakistani",
"persian", "peruvian", "pizza", "polish", "portuguese", "russian", "salad",
"sandwiches", "scandinavian", "scottish", "seafood", "singaporean", "slovakian",
"soulfood", "soup", "southern", "spanish", "steak", "sushi", "taiwanese",
"tapas", "tapasmallplates", "tex-mex", "thai", "turkish", "ukrainian", "vegan",
"vegetarian", "vietnamese"]
    #cat_list = ["$", "$$", "$$$"]
    (label_cat_lexicon, reverse_label_cat_lexicon) =
utils.create_lexicon_from_strings(cat_list)
    rl_cat = utils.ResourceLabeler(lexicon = label_cat_lexicon,
accessor="categories")
    categorized = rl_cat.process(some_batch)
    extended_cat_lexicon = reverse_label_cat_lexicon + ['unknown']
    reverse_point_categories = {}
    for index, category in enumerate(categorized):
        if reverse_point_categories.get(category) != None:
            reverse_point_categories[category] =
np.append(reverse_point_categories[category], X_2d[index, :].reshape(1, 2),
axis=0)
        else:
            reverse_point_categories[category] = np.zeros([1,2])
            reverse_point_categories[category][0, :] = X_2d[index, :]

    for key in reverse_point_categories.keys():
        plt.scatter(reverse_point_categories[key][:, 0],
reverse_point_categories[key][:, 1], cmap=mpl.cm.summer,
c=key*np.ones(len(reverse_point_categories[key])),
label=extended_cat_lexicon[key]);
    plt.legend(scatterpoints=1)


    # create a labeling lexicon:
    (label_lexicon, reverse_label_lexicon) = utils.create_lexicon_from_strings(["restaurants", "afghani", "african", "senegalese", "southafrican", "New", "newamerican", "Traditional", "tradamerican", "arabian", "argentine", "armenian", "asianfusion", "australian", "austrian", "bangladeshi", "bbq", "basque", "belgian", "brasseries", "brazilian", "breakfast_brunch", "british", "buffets", "burgers", "burmese", "cafes", "cafeteria", "cajun", "cambodian", "caribbean", "dominican", "haitian", "puertorican", "trinidadian", "catalan", "cheesesteaks", "chicken_wings", "chinese", "cantonese", "dimsum", "shanghainese", "szechuan", "comfortfood", "creperies", "cuban", "czech", "delis", "diners", "ethiopian", "hotdogs", "filipino", "fishnchips", "fondue", "food_court", "foodstands", "french", "gastropubs", "german", "gluten_free", "greek", "halal", "hawaiian", "himalayan", "hotdog", "hotpot", "hungarian", "iberian", "indpak", "indonesian", "irish", "italian", "japanese", "korean", "kosher", "laotian", "latin", "colombian", "salvadoran", "venezuelan", "raw_food", "malaysian", "mediterranean", "falafel", "mexican", "mideastern", "egyptian", "lebanese", "modern_european", "mongolian", "moroccan", "pakistani", "persian", "peruvian", "pizza", "polish", "portuguese", "russian", "salad", "sandwiches", "scandinavian", "scottish", "seafood", "singaporean", "slovakian", "soulfood", "soup", "southern", "spanish", "steak", "sushi", "taiwanese", "tapas", "tapasmallplates", "tex-mex", "thai", "turkish", "ukrainian", "vegan", "vegetarian", "vietnamese"])


    batch.batch_size = 2000
    some_batch = batch.get_raw_batch()
    (hidden, scaling) = encoder.project_into_hidden_layer(batch.conversion(some_batch))
    X_2d = bh_sne(hidden.astype('float64'))


    # Save the matrices:
    wm = encoder.weight_matrix.get_value()
    hb = encoder.hidden_bias.get_value()
    vb = encoder.visible_bias.get_value()


    encoder.weight_matrix.set_value(wm)
    encoder.hidden_bias.set_value(hb)
    encoder.visible_bias.set_value(vb)


    from mpl_toolkits.mplot3d.axes3d import Axes3D
    datum = bh_sne(hidden.astype('float64'), d=3)
    fig = plt.figure()
    axis = Axes3D(fig = fig)
    axis.scatter(datum[:, 0], datum[:, 1], datum[:, 2], c=y)




    <mpl_toolkits.mplot3d.art3d.Path3DCollection at 0x117354510>




![png](Mongo%20Stochastic%20Gradient%20Descent%20T-SNE_files/Mongo%20Stochastic%20Gradient%20Descent%20T-SNE_18_1.png)



    (label_lexicon, reverse_label_lexicon) = utils.create_lexicon_from_strings(["$", "$$", "$$$", "$$$$"])
    rl = utils.ResourceLabeler(lexicon = label_lexicon, accessor="price")
    (hidden, scaling) = encoder.project_into_hidden_layer(batch.conversion(some_batch))
    from mpl_toolkits.mplot3d.axes3d import Axes3D
    datum = bh_sne(hidden.astype('float64'), d=3)
    fig = plt.figure()
    axis = Axes3D(fig = fig, rect= [0.1, 0.1, 2.0, 2.0])
    axis.scatter(datum[:, 0], datum[:, 1], datum[:, 2], c=rl.process(some_batch))
    axis.legend(reverse_label_lexicon + ['unknown'])




    <matplotlib.legend.Legend at 0x11738b2d0>




![png](Mongo%20Stochastic%20Gradient%20Descent%20T-SNE_files/Mongo%20Stochastic%20Gradient%20Descent%20T-SNE_19_1.png)



    (rating_lexicon, reverse_rating_lexicon) = utils.create_lexicon_from_strings(["0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0"])
    rl = utils.ResourceLabeler(lexicon = rating_lexicon, accessor="rating", converter = lambda i: ("%.1f" % (round(i * 2) / 2.0)))
    (hidden, scaling) = encoder.project_into_hidden_layer(batch.conversion(some_batch))
    from mpl_toolkits.mplot3d.axes3d import Axes3D
    datum = bh_sne(hidden.astype('float64'), d=3)
    fig = plt.figure()
    axis = Axes3D(fig = fig, rect= [0.1, 0.1, 2.0, 2.0])
    axis.scatter(datum[:, 0], datum[:, 1], datum[:, 2], c=rl.process(some_batch), cmap=mpl.cm.binary)
    axis.legend(reverse_label_lexicon + ['unknown'])




    <matplotlib.legend.Legend at 0x1195f5350>




![png](Mongo%20Stochastic%20Gradient%20Descent%20T-SNE_files/Mongo%20Stochastic%20Gradient%20Descent%20T-SNE_20_1.png)



    fig = plt.figure()
    axis = Axes3D(fig = fig, rect= [0.1, 0.1, 2.0, 2.0])
    axis.scatter(datum[:, 0], datum[:, 1], datum[:, 2], c=rl.process(some_batch), cmap=mpl.cm.binary)
    axis.legend(reverse_label_lexicon + ['unknown'])




    <matplotlib.legend.Legend at 0x118e188d0>




![png](Mongo%20Stochastic%20Gradient%20Descent%20T-SNE_files/Mongo%20Stochastic%20Gradient%20Descent%20T-SNE_21_1.png)



    z = np.zeros([1,2])
    z[0,:] = [1, 2]
    np.append(z, np.array([[1, 2]]), axis=0)




    array([[ 1.,  2.],
           [ 1.,  2.]])




    np.array([1, 2]).reshape(1, 2)




    array([[1, 2]])




    
