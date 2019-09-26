# InverseActivation
This is a single (simple) Keras Layer that can be used with/in place of Relu (or similar) Activation functions. The goal is to reduce the number of computations required by reusing existing computations.

---

# Inspiration
While looking at the activation maps of early layers in DNNs I noticed (and read articles about) how it is often finding edges in the early layers. This led me to read about non-DNN edge detection, which led to the Sobel Kernel (which most DNNs recreate in some manner) https://en.wikipedia.org/wiki/Sobel_operator
The one striking thing with the Sobel Kernel is that it only outputs two linear values, Up-Down-ness, and left-right-ness. 

## Non-Leaniearity and Computational Simplicity
In DNNs though we (almost) always want non-liniearities, and the Go-To for that is the ReLU. But, if I wanted to implement Sobel Kernel using ReLU, I'd have to create four non-linear outputs, Up-ness, Down-ness, Left-ness, and Right-ness. That's fine, and I believe that many DNNs learn this automatically, but it requires twice as much computations to create since each of those four must apply their kernel and activation, where their kernel should just be the inverse of the other Kernel. Even with a black and white 3x3, that's ~18 extra multiply accumulate operations per input pixel, and for RGB it's 54 extra MACs. When dealing with large images, that can end up being huge numbers of MACs (54*1024*1024) that really give no value. So the simple action is to instead apply the ReLU twice, once to the positive outputs, and once to the negated outputs. And because negating is cheap, and ReLU's are cheap, this can save significantly on the computational workload, and still provide usefull outputs without lossing our non-liniearity. 

## Assumptions and Theories

My Big assumption is that even at deeper levels of the NN this functionality will be useful, if for example, a Layer learns Feather-ness or Beak-ness, then wouldn't it also be usefull to know Not-Beak-ness or Not-Feather-ness? 

Another Theory I have is that this may help with Back Prop, eliminating the need for fixes like leaky ReLU. But I believe that this still provides an appropriate level of non-linearity, because they are seperated into seperate outputs (so it functions the same as 2 kernels learning the inverse of eachother, except their learning is tied together).

---

# Ussage Note
This will Double the number of channels of the output that it activates, and this can result in the output channel count being bigger than you expect. 

## Also Note
This needs to be applied as the Activation, such that the Input to it should be 'Linear' activation or None.

---

# Evaluation
My initial Evaluation is, this does not significantly harm the ability of a NN to learn and operate. 



I've done simple evaluation using this code on MNIST and Fashion MNIST and it performs well. 
I will continue to use it on other harder problems and post their results here. 


If you use this on your own, please let me know your experince with it

## Actual Results
MNIST (w/ data aug): Validation Accuracy: 
Fashion MNIST(w/ data aug): Validation Accuracy:

## Results when combined with my other crazy idea.
MNIST(w/ data aug): Validation Accuracy: > 99.5%
MNIST(w/ data aug): Validation Loss: < 0.010








