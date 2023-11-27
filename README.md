# 4S for few-shot
Implementation of the ACM MM 2023 paper: Semantic-based Selection Synthesis and Supervision for few-shot learning

## Introduction
**S**emantic-based **S**election **S**ynthesis and **S**upervision is a new method for few-shot learning, where semantics provide more diverse and informative supervision for recognizing novel objects. In this work, we firstly utilize semantic knowledge to explore the correlation of categories in the textual space and select base categories related to the given novel category. Then we analyze the semantic knowledge to hallucinate the training samples by selectively synthesizing the contents from base and support samples. 
And finally we employ semantic knowledge as both soft and hard supervision to enrich the supervision for the fine-tuning procedure. Empirical studies on four FSL benchmarks demonstrate the effectiveness of 4S.

## Get Started
