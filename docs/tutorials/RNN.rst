

RECURRENT NEURAL NETWORKS TUTORIAL
==================================

Recurrent Neural Networks (RNNs) are popular models that have shown great promise in many NLP tasks. But despite their recent popularity I’ve only found a limited number of resources that throughly explain how RNNs work, and how to implement them. That’s what this tutorial is about.

Part 1 Introduction to RNNs
---------------------------

As part of the tutorial we will implement a recurrent neural network based language model. The applications of language models are two-fold: First, it allows us to score arbitrary sentences based on how likely they are to occur in the real world. This gives us a measure of grammatical and semantic correctness. Such models are typically used as part of Machine Translation systems. Secondly, a language model allows us to generate new text (I think that’s the much cooler application). Training a language model on Shakespeare allows us to generate Shakespeare-like text. This fun post by Andrej Karpathy demonstrates what character-level language models based on RNNs are capable of.

I’m assuming that you are somewhat familiar with basic Neural Networks. If you’re not, you may want to head over to Implementing A Neural Network From Scratch,  which guides you through the ideas and implementation behind non-recurrent networks.



