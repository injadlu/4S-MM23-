# 4S: Semantic-based Selection Synthesis and Supervision for few-shot learning
Implementation of the ACM MM 2023 paper: [Semantic-based Selection Synthesis and Supervision for few-shot learning](https://dl.acm.org/doi/abs/10.1145/3581783.3611784)

## Introduction
**S**emantic-based **S**election **S**ynthesis and **S**upervision (**4S**) is a new method for few-shot learning, where semantics provide more diverse and informative supervision for recognizing novel objects. In this work, we firstly utilize semantic knowledge to explore the correlation of categories in the textual space and select base categories related to the given novel category, and then, we analyze the semantic knowledge to hallucinate the training samples by selectively synthesizing the contents from base and support samples (Distribution Exploration). Finally, we employ semantic knowledge as both soft and hard supervision to enrich the supervision for the fine-tuning procedure (Classifier Exploration). Empirical studies on four FSL benchmarks demonstrate the effectiveness of 4S.<br>
<div align=center>
  <img src="https://github.com/injadlu/4S-MM23-/blob/main/Figure-1.svg">
</div>

## Get Started
Our implementation consists of 2 steps.<br>
### Extract feature
1. download existing works with pre-trained backbone. <br>
2. extract base and novel features to your feature path. <br>
 ```
 python extract_feature.py
 ```
### Run 1-shot and 5-shot
1. modify the path to your feature path. <br>
2. run the Scripts_shot1.py & Scripts_shot5.py for 1-shot and 5-shot, respectively. <br>
Specifically, for 1-shot,
 ```
 python Scripts_shot1.py --feature_path your feature_path
 ```
 for 5-shot,
 ```
 python Scripts_shot5.py --feature_path your feature_path
 ```
for word embedding, please refer to my another [page](https://github.com/injadlu/few-shot-word2vec). <br>
<div align=center>
  <img src="https://github.com/injadlu/4S-MM23-/blob/main/Overview.svg">
</div>
An overview of our semantic-based selection, synthesis, and supervision method.

## contact:
**Any problems please contact me at jackie64321@gmail.com**

## References
```
@inproceedings{lu2023semantic,
  title={Semantic-based Selection, Synthesis, and Supervision for Few-shot Learning},
  author={Lu, Jinda and Wang, Shuo and Zhang, Xinyu and Hao, Yanbin and He, Xiangnan},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={3569--3578},
  year={2023}
}
```
