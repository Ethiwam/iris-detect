# Iris Detect
This is a simple project meant to teach me how to build a neural network.
I was in a class on the application of AI and machine learning, when half way through the semester we had to learn about building neural networks. Before starting this project I thought, "wait, I also need to practice some PostgreSQL," and thus this project was created.

I used the classic Iris dataset (Fisher, 1936), saving it all to a PostgreSQL database and using SQL commands to interact with it. It's a small dataset, only with 150 samples, 4 features each, with 3 classes each sample could fall into.
I designed a deep neural network featuring 3 dense hidden layers, all of which used a sigmoid activation function. The output layer uses a softmax function to output to one of three classes: setosa, versicolor, or virginica. It's fairly straightforward, 
all samples were normalized and the labels were all one-hot encoded before training the model.

This does not save any keras models, though could easily be modified to do so. Every time the code is run, a new model is trained on the training data, fine tuned with the validation data, and tested on the test data, and will typically yield a model with between 85%-100% accuracy.
The training data, validation data, and test data all come from the original dataset, roughly 80% training, 15% validation, 5% testing.
