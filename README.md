# InverseActivation
This is a single (simple) Keras Layer that can be used with/in place of Relu (or similar) Activation functions. The goal is to reduce the number of computations required by reusing existing computations. 

Implemented for TF.Keras.

---

# Inspiration
While looking at the activation maps of early layers in DNNs I noticed (and read articles about) how it is often finding edges in the early layers. This led me to read about non-DNN edge detection, which led to the Sobel Kernel (which most DNNs recreate in some manner) https://en.wikipedia.org/wiki/Sobel_operator
The one striking thing with the Sobel Kernel is that it only outputs two linear values, Up-Down-ness, and left-right-ness. 

## Non-Leaniearity and Computational Simplicity
In DNNs we (almost) always want non-liniearities, and the Go-To for that is the ReLU. But, if I wanted to implement Sobel Kernel using ReLU, I'd have to create four non-linear outputs, Up-ness, Down-ness, Left-ness, and Right-ness. That's fine, and I believe that many DNNs learn this automatically, but it requires twice as much computations to create since each of those four must apply their kernel and activation, where their kernel should just be the inverse of the other Kernel. Even with a black and white 3x3, that's ~18 extra multiply accumulate operations per input pixel, and for RGB it's 54 extra MACs. When dealing with large images, that can end up being huge numbers of MACs (54*1024*1024) that really give no knowledge value. But using linear activation instead means we sacrifice non-linearity, and additionally, we have no way to know if a specific layer would benefit or not.

## My "Solution" the 'InverseActiviation' (Name subject to change)
My simple solution is to instead apply the ReLU twice, once to the positive filters outputs, and once to the negated filters outputs.
I belienve that in almost any case, the cost of one negation and one extra ReLU per filter, the Computational cost is much lower than the cost of recalculating the filters kernel a second time. I also believe that many "features/filters" provide value in A-ness and not-A-ness (where A is a generic "Feature").
Note: a dedicated Activation layer that did this may be faster, but this implementation uses as much pure Tensorflow/Keras Code as possible. 


### Does this actually help?
That is hard to define, because without this layer it is possible for it to learn the Inverse of a Kernel anyway, meaning that it's hard or impossible to create a "NULL Hypothesis Test" that is conclusive. 


What can be easily concluded is that the computations needed and the number of weights to train are vastly reduced. Additional testing is needed to validate that it is able to learn similar complexity and accuracy with a reduced number of filters compared to the standard ReLU.


### What about non-liniearity?
I believe that because the Filters linear values are split into multiple non-linear Outputs, that it maintains at least as much non-linearity as the stnadard ReLU (remenber, nothing stops a NN from learning the negative kernel as well). 


My initial experiments support this conclusion; When testing with the same network and comparing the standard ReLU vs InverseActivation vs Linear, the This activation and the ReLU score close enough to blame randomness, but the Linear lags behind significanly. 


For this test I tested the same archetecture on Fashion MNIST by removing all but the dense layers Relu and Softmax, note: I kept the concat of the negation to keep the params similar/same. I also tested with standard ReLU, but without the Negation (COncat the positive with itself). The results of that test show that this layer scored much better on fashion MNIST than the all Liniear version but was very similar for all ReLU.

Linear:

Test score: 0.21352055146694182

Test accuracy: 0.9235

VS

Inverse Activation:

Test score: 0.1780399688757956

Test accuracy: 0.9433





## Assumptions and Theories

My Big assumption is that even at deeper levels of the NN this functionality will be useful, if for example, a Layer learns Feather-ness or Beak-ness, then wouldn't it also be usefull to know Not-Beak-ness or Not-Feather-ness? 


Another Theory I have is that this may help with Back Prop, eliminating the need for fixes like leaky ReLU. But I believe that this still provides an appropriate level of non-linearity, because they are seperated into seperate outputs (so it functions the same as 2 kernels learning the inverse of eachother, except their learning is tied together).



---

# Ussage Note
This is coded for TF.Keras, though it would be trivial to adapt it to plain Keras.

## Note: Doubles Output Channel Count 
This will Double the number of channels of the output that it activates, and this can result in the output channel count being bigger than you expect. 

## Also Note
This needs to be applied as the Activation, such that the Input to it should be 'Linear' activation or None.

---

# Evaluation
My initial Evaluation is, this does not significantly harm the ability of a NN to learn and operate. It's possible that it improves the operation of the DNN, but, more testing is needed (it may simply be the increased number of parameters).


I've done simple evaluation using this code on MNIST and Fashion MNIST and it performs well. 
I will continue to use it on other harder problems and post their results here. 


If you use this on your own, please let me know your experince with it, and if you're willing, let me post your results here.


## Actual Results

MNIST (w/ data aug): Validation Accuracy: 99.66%

MNIST(w/ data aug): Validation Loss: 0.01109

Fashion MNIST(w/ data aug): Validation Accuracy: 93.9%

Fashion MNIST(w/ data aug): Validation Loss: 0.1759


## Results when combined with my other crazy ideas.

MNIST(w/ data aug): Validation Accuracy: 99.59%

MNIST(w/ data aug): Validation Loss: 0.0122

Fashion MNIST(w/ data aug): Validation Accuracy: 94.51%

Fashion MNIST(w/ data aug): Validation Loss: 0.1637






