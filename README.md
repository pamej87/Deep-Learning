# Deep-Learning
Neural networks with multiple layers to perform image classifier model

To start, I have run a model with only 1 convolutional layer, and 2 dense layers, this model didn´t give pretty good results, overfitting a lot. 

Then I have added 3 convolutional layers, though each one adds more computation expenses, with this we tipically add the number of filters so the model can learn more complex representations.
I have added 2 pooling layers, It's important not to have too many pooling layers, as each pooling discards some data. Pooling too often will lead to there being almost nothing for the densely connected layers to learn about when the data reaches them.

In this case, since the images are small we didn't pool more than twice.

Then I added the first Dense layer. Here I included a kernel constraint to regularize the data as it learns, another thing that helps prevent overfitting. And finally the last dense layer with one neuron as this is a binomial classifier model.
I added dropout in all our layers too, at the begining I run a model without dropout in all the layers and it was overfitting, after add dropout I obtain a pretty good accuracy and the model was not overfitting as it can be seen with the metrics (91.83% for train and 85.92% for test). 

Finally the model was improved using the pretrained model VGG16, frizing all the convolutional layers and adding 3 more dense layers. Also we included early stopping to know the correct epochs that obtain the better accuracy and not overfit. The model stops in 3 epochs, achiving 90% in test and 92% in train (loss=0.24)

The difference between the model that i obtained with the VGG16 are the layers, VGG16 includes 16 layers, in terms of accuracy it doesnt grow that much, when i used only 4 layers I obtained a 85% and with 12 more we only grow 5 porcentual points. But the real advantage at running VGG16 are the computional costs, as the layers were frized the model was only train with the last dense layers, and that is an clear advantage in saving time and computational costs optimization.

