
# Neural Network How To Train Demo Using C#

## The Intuition Old vs. New Training
<p align="center">
  <img src="https://github.com/grensen/how_to_train/blob/main/figures/old_vs_new_training.png?raw=true">
</p>

In the last article, "[Good vs. Bad Code](https://github.com/grensen/good_vs_bad_code)" I demonstrated how to create a highly efficient and fast neural network. While a good implementation is crucial, improved training techniques can have an even greater impact, pushing our models into new spheres. Let's consider the previous result, a neural network with 100 hidden neurons achieving 98% accuracy. Better training techniques could potentially yield the same accuracy with only 50 hidden neurons. Or we reach more with 100 hidden neurons. Let's explore how this could work.

The new technique to improve neural network training drops just 50% of the input neuron signals. We called it input dropout. It is not a new technique overall, but if you search for information on how to drop input neurons, you may find advice either against dropping input neurons, or you may find information about the usual way of dropout with hidden neurons. The problem is that it doesn't work reliably in every aspect, and the result I present here may be more based on experience gained through trial and error. 

## The Demo
<p align="center">
  <img src="https://github.com/grensen/how_to_train/blob/main/figures/how_to_train_demo.png?raw=true">
</p>

The training in the demo is only modified with input dropout. The initial training runs with a 50% dropout rate for 200 epochs using the entire MNIST training dataset in parallel (multi-core). In the fine-tuning phase, which runs for 30 epochs, we reduce both the dropout rate and learning rate to achieve the final result. The reached test accuracy of 98.80% is extremely high for this small neural network.

## The Demo With .NET 7
<p align="center">
  <img src="https://github.com/grensen/how_to_train/blob/main/figures/how_to_train_demo_dotnet7.png?raw=true">
</p>

Here's the same demo in C# using .NET 7. The results from the parallel training segments vary slightly each time, resulting in a more normally distributed outcome.

## With More Neurons
<p align="center">
  <img src="https://github.com/grensen/how_to_train/blob/main/figures/how_to_train_higher.png?raw=true">
</p>

That's the result with 300 hidden neurons on both layers. Achieving close to 99%, I wondered what is required to reach 99% accuracy in the test.

## And Even More Neurons
<p align="center">
  <img src="https://github.com/grensen/how_to_train/blob/main/figures/how_to_train_highest.png?raw=true">
</p>

With 400 hidden neurons was it possible to reach 99% accuracy in the test. Great!

## Different Systems On MNIST
<p align="center">
  <img src="https://github.com/grensen/how_to_train/blob/main/figures/mnist_bench.png?raw=true">
</p>

Let's get a clearer picture. A fascinating ML system is the Tsetlin machine, perhaps something new to you. Instead of a neural network, we can also use a Tsetlin machine for nice predictions, as the comparison of different ML systems on MNIST demonstrates.

This is a nice [presentation of the Tsetlin machine](https://www.regjeringen.no/contentassets/7e8fd99613f04fe983e790607b7d0f40/01-granmo.pdf), and it shows also results of "Results on raw, unenhanced, and
unextended MNIST data" that we can take to compare our work in the MNIST test. And this [Tsetlin demo on MNIST](https://github.com/adrianphoulady/weighted-tsetlin-machine-cpp) validates the result, achieving a peak accuracy of 98.63%.

Maybe you're not impressed, but consider this: the Tsetlin machine is a truly remarkable system capable of making accurate predictions. Achieving a 98.60% accuracy without convolutional layers is quite cool and falls within the same range as a neural network with 2 layers and 800 hidden neurons. 

Without convolutional layers or other tricks like data augmentation and others, an accuracy of ~98.50%-98.60% seems to be the limit. Consider what we achieved with 100 hidden neurons on the hidden layers, 98.80%! But the network here isn't anything special. It's simply a better training method that could create this perception. 

## Weight Maps Input To Hidden1 
<p align="center">
  <img src="https://github.com/grensen/how_to_train/blob/main/figures/old_vs_new_weightmaps.png?raw=true">
</p>

These are the weight maps or the perception of what the neural network calculates on the first input to the hidden layer. Each map forms a kind of cluster that sends activations, which then propagate into further signals until the final output layer makes a prediction. However, let's remain focused on the first layer. Left, against my intuition, because this is the usual way to train with 100% of an input sample, the image of each map appears noisy with some offline maps. However, on the right side, with 50% dropout, a much clearer picture emerges. There's a high contrast between yes, no, and nothing, even though we can only speculate about which classes each weight map serves. 

It is perhaps more intuition than evidence, but optimizing in this way was what was needed to push the limit of what is possible a bit further. Take a closer look at the new maps, you can even detect some offline maps and likely a lot of noise that we don't need.

## System One Takes 80% Of The Data With 100% Accuracy
<p align="center">
  <img src="https://github.com/grensen/how_to_train/blob/main/figures/system1.png?raw=true">
</p>

It should end here, but one more thing. We have the first system (System 1) with 100 hidden neurons, which achieves a high accuracy of 98.8%, and our best model, roughly 4 times bigger, with 99%. Without any further training, we can use System 1 if the probability is high enough to predict 80% of the training data, but with 100% accuracy. With System 2, we can predict the rest of the data, and perhaps even better. We can even stack these predictions and explore other possibilities. Something like [this](https://github.com/grensen/ML-Art/raw/master/2022/reassessment_network.png) maybe. Keep coding! :)
