# **Neuro Evolution of Neural-Networks: Adaptive Resonance Theory (ART) case study** 

Overview:
---

Neural networks including **deep learning** have numerous hyperparameters that are critical to the network's performance, **Adaptive Resonance Theory (ART)** - A class of **self-organizing maps (MAPs)** that are brain-inspired are no exception. ART is a theory about how advanced primate brains learn to adapt & recognize objects & events in a dynamic world. The various forms of ART that have been developed continue to demonstrate exceptional performance on complex tasks requiring neural network systems capable of autonomous adaptation in response to environmental challenges. However, selecting suitable configuration of the network's parameters such as **vigilance** is often done empirically in an ad hoc manner - this doesn't effectively support the lifelong-learning capabilities of ART models in changing environments. Various approaches in isolated efforts have been applied to adapt these parameters (e.g vigilance adaptation) in response to changes in the environment as the network learns internal representations via long-term memory traces while interacting with the environment. This project applies **genetic algorithms (GA)**- a subclass of **evolutionary algorithms (EA)** to apply principles of evolution to evolve an ART network specifically, **Fuzzy ARTMAP**. Integrating evolutionary methods to biologically plausible networks seems to be conceptually aligned in that both are inspired by biological evolution to environmentally successful behaviors. I claim so because "brain evolution is shaped by behavioral success" [Stephen Grossberg](https://www.bu.edu/articles/2021/stephen-grossberg-conscious-mind-resonant-brain/) & EC in principle  applies evolutionary pressures leading to most successful behaviors. 

![elementary_artmap](https://user-images.githubusercontent.com/76077647/139080600-de59748c-59d1-405d-92cd-bd75f7ed19af.JPG)

Methodology:
---

Implementation of both GA & FuzzyARTMAP is done in python. Besides the classical GA **operators** including **Selection**, **Crossover**, & **Mutation**, **Elitism** is applied to guarantee that the best indivdual(s) always makes it to the next generation. The elite individual(s) however, are still eligible for the selection process.

<p align="center">
  <img src="https://user-images.githubusercontent.com/76077647/139085612-c0e5ec32-bb48-4b78-8a98-848ee2d65f50.JPG" />
</p>


Performance:
---

Experiments were performed using the [glass dataset](https://archive.ics.uci.edu/ml/datasets/glass+identification) which is publicly available. Even though this dataset isn't so complex for classification purposes, it does present a frequently recurring problem in production environments namely; imbalanced classes which inhibits the success application of most approaches including deep learning.

**NOTE** Please refer to the demo jupyter notebook for performance. I recommend using f1 score, precision, or recall for analysis. My demo uses accuracy for simplicity.

Future Steps:
---
1. Apply Niching & Sharing.
2. Apply both GA & Strongly Typed Genetic Programming (GP) to evolve both network parameters & part of the functonality of the network respectively.

References:
---
1. "Conscious Minds, Resonant Brain: How Each Brain Makes A Mind"  - Stephen Grossberg
2. "Evolutionary Computation for the Automated Design of Category Functions for Fuzzy ART: An Initial Exploration" - Wunsch, Islam & Tauritz
