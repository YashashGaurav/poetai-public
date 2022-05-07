# PoetAI: Automatic Limerick Generator

> Bandish Parikh\
> Carnegie Mellon University\
> Pittsburgh, PA 15213\
> bparikh@andrew.cmu.edu

> Lokeshwar Jha\
> Carnegie Mellon University\
> Pittsburgh, PA 15213\
> lokeshwj@andrew.cmu.edu

> Sandeep Vemulapalli\
> Carnegie Mellon University\
> Pittsburgh, PA 15213\
> vvemulap@andrew.cmu.edu

> Yashash Gaurav\
> Carnegie Mellon University\
> Pittsburgh, PA 15213\
> ygaurav@andrew.cmu.edu

# Folder Structure

The folder structure at the time of publishing this repository is as follows:

```
├── LICENSE
├── README.md
├── data
│   ├── limericks_clean_with_@.csv
│   ├── limericks_clean_with_@and#.csv
│   ├── limericks_no_profanity.csv
│   ├── limericks_no_punc_digit.csv
│   └── limericks_original.csv
├── evaluations
│   └── rhyming_evaluation.ipynb
├── experiments
│   └── PoetAI_345M.ipynb
├── preprocessing
│   └── Limerick_Processing.ipynb
└── readmeassets
    └── images
        ├── context.jpeg
        ├── eval.png
        ├── gantt_chart.png
        ├── metrics_epoch_loss.png
        └── training_flow.jpeg
```

- 'data' folder contains all the variants of datasets that we used for our experiments\
- 'evaluations' folder contains our implementation for measuring our model's performances.\
- 'experiments' folder is where we have parked our experimentation notebook. It contains one file that is the core of this project\
- 'readmeassets' contains all the assets that we have references to present to you this README.md file\
- 'preprocessing' folder contains any code that we may have used for pre processing purposes.

# 1 Introduction
Poetry is a form of literary work that is often characterized by an interplay of words meant to generate an evocative response in the human mind. A limerick is a short humorous form of verse that has five lines and follows the AABBA rhyme scheme. While a lot of research has been done in the field of Natural Language Understanding, the area pertaining to generation and analysis of poetry based on its qualities remains to be explored. This is mainly because poetry is one of the oldest art forms, and the rules guiding it differ significantly depending on the language and culture. Our goal is to use models like GPT-2 which have been successful at text generation for many custom scenarios like summarising, chat bots and even in poetry generation. The clear challenge we foresee is that models must be able to understand what poetry means and what makes them different from conventional text generation tasks. For this, attributes of poetry like rhyme, rhythm, context, and other characteristics of poem must be quantified for models to learn.

# 2 Literature Review
There have been few attempts in poetry compared to other use cases where GPT-2 is used for, and each of them have employed varied techniques and approaches towards poetry generation. Some attempts with Sonnet and RNNs are also studied to understand the evolution of poetry generation over the time.

## 2.1 CMU AiBBA Limerick Generator:
Our colleagues here at CMU have already been able to do this quite successfully.[3] The team of Mitchell et. al. were able to generate Limericks using GPT-2 that uses transformer architecture. They developed and used several automatic evaluation metrics such as Rhyming coherence, subject co-reference and nonsense evaluation. They finally conducted a Turing test where they were able to fool the humans 16\% of the time. They trained their model on mostly AABBA rhyme schemes collected from Sam Ballass Datasets of 90,000 limericks [1]. For automatic evaluation, they used Rhyming Distance, Coherence and Correctness as relevant quantifiable metrics. They finally generated 8371 poems from the trained GPT-2 Model out of which a set of 1097 limericks adhered to AABBA rhyming scheme and made it to a printed book [3].

## 2.2 LimGen:
The latest results published in this domain is by Jianyou Wang et al.[5] in their GPT-2 implementation called LimGen where they use various search metrics such as the Adaptive Multi-Templated Constraint algorithm that constrains their search to the space of realistic poems, the Multi-Templated Beam Search algorithm which searches efficiently through the space, and the probabilistic storyline algorithm that provides coherent storylines related to a user-provided prompt word. Further research in this field suggests further work by Rui Yan et al. [6] using tone, rhyme and semantic coherence. They use filtering and semantic clustering of poems in a dataset to generate summarizations of required length and rhyme that are coherent. Other works are by Chen et al.[7] where they propose a semi-supervised conditional Variational Auto-Encoder model for sentiment controllable poetry generation.

## 2.3 Deep-Speare:
Deep-speare [8] is a Sonnet based model used to capture language, rhyming and meter of poetry. These models underperformed in generating human level poetry but served as good reference for rhyming capture with models. Rhyme was enforced by a cosine similarity of the last words generated by the model and a loss function was employed to penalize model when not rhyming. A rhyming dictionary was maintained to have words picked from based on the context.

## 2.4 SP-GPT 2 Vietnamese Poetry Generation:
A more recent attempt on Vietnamese poetry [9] serves as great approach for context maintained throughout a poem or limerick. Because of the structure of limericks models can lose context very easily to enforce this during the model training, the sentences of the limericks can be converted into context vectors which are passed to LSTM model to maintain necessary attention to the context which the poem is about. With this implementation the SP-GPT 2 team was able to generate poems that are more focused on the subject.

# 3 Dataset description

As ours is a unsupervised learning problem, our dataset does not have labels and consequently does not have validation and test sets. Except we need a corpus of training data to teach out GPT how to behave given prompts or without them. Our model is expected to produce limericks that conform to the AABBA rhyme scheme and so our dataset, for now, is solely based on Sam Ballas’ datasets of 90,000 limericks [2], which he collected by scraping the oedilf.com website. We are working with this dataset solely because of its size which also helps us teach our model about the varied ways that rhyming structures can work. Some samples from our dataset:

> Sample 1:\
> the ball was defended by cole\
> and over the goal-line did roll.\
> the cross to be borne: a\
> quick kick from the corner?\
> a header, a strike, it's a goal!

> Sample 2:\
> how dare you invade my sweet life,\
> you bringer of conflict and strife?\
> until you came along,\
> not a thing had gone wrong,\
> but now discord and friction are rife!

Following this, we want to further explore using Mad Kane’s repository of humourous limericks [2]. “A smile is a curve that sets everything straight.” – Phyllis Diller by teaching our model to be humourous (we hope we do) we want to help the academics who read our work laugh a little as they build on our work. Some of the sample we hope to incorporate as we go forward:


> Sample 1:\
> A strange silhouette in the sky;\
> A rustling of wings from on high.\
> Not angels divine,\
> But migrating swine –\
> Those pigs finally learned how to fly!\
> – Paul Haebig



> Sample 2:\
> There was an inventor named Knight\
> Who studied the science of flight.\
> He thought he’d be first,\
> But his efforts were cursed.\
> His designs never turned out quite Wright.\
> – Fred Bortz


We are hoping to come up with metrics that can help us understand the performance of our model. Tracking rhyme scheme, and coherence. And if possible add that as a loss that the model may try to minimize.

# 4 Baseline Model and Constraints
Poems are a product of creativity. Even after myriad amounts of research, AI/ML has faced the criticism of not being able to be creative or even generate nonsensical texts, at times. We realized that creativity to AI is nothing but a generation of well defined objective functions that a model needs to be optimized on. Conventional loss functions wouldn’t be able to capture the desired output of creativeness. While assigning objective functions to judge a particular piece of poetry/limerick, our main goal would be to answer the following question “What makes this piece of poem/limerick a good one?”. 

Apart from it being grammatically/syntactically sound, a limerick must possess one or more qualities that make it compelling to the human eye. There is no fixed definition or strategy for this but the best limericks share some aspects in common - rhyme, rhythm, flow, emotion, intellect, humor, surprise, value, novelty and many more. A poem is generally called good if it has a high quotient in at least one or more of the above qualities. Although it is a difficult task to quantify each of these qualities, using constraints for them in the objective function along with a threshold value for each would be our main aim. Recent attempts to measure some of these features give basis for a further extension of research in poetry and limericks [10].

We identified three key areas to work on- limerick generation, identification of an evaluation metric to quantify the text generated and, clubbing both the tasks together to enhance the overall quality. We picked [3] as our baseline model because it demonstrated the performance of a pre-trained GPT-2 on limerick generation after the addition of structure (Example: [’<|endoftext|>’]). However, we wanted to understand whether the learning process of the rhyming scheme of AABBA is generalized by GPT-2. And hence, kept the tokens (Example: ‘\$’, ‘@‘) between each line as a demarcation to train the model. We trained the 117M GPT-2(pre-trained on Guggenheim Poetry and Poetry Foundation corpus) model on limerick dataset for 31,600 epochs. 

# 5 Research Methods

GPT-2 uses a loss function in the form of Cross Entropy. While that is specifically designed to generate text, adding some more constraints to it would be crucial to make it generate poetry, or more specifically, limericks. The 2 main qualities we want to enforce are rhyme and context. The generated text must be penalized if the AABBA rhyme pattern is not followed, as well as the five lines generated in the form of limericks must tell the same story. 

Tuan Nguyen et al. [9] has been successful in imposing the context constraint in Vietnamese poetry. They used the output of the GPT-2 and fed it to a self-attention LSTM to generate context vectors corresponding to the two lines of poetry that they wanted to contextually synchronize. A MSE loss was calculated between the two context vectors that they wished to contextually synchronize, and the loss was added to the Cross-Entropy loss function of GPT-2. We plan to use the same methodology to generate context vectors for the 5 lines of the limericks, use them to calculate the MSE loss, and add that loss function of the GPT-2. Given below is the figure that summarizes the approach of Tuan Nguyen et al. [9] We tweak the implementation for our specific use case where we had to maintain context between five lines of limericks instead of two.

![Adding Context!](/readmeassets/images/context.jpeg "Adding Context")\
Figure 1: Adding Context

The prior work done by Mitchell B. Fogelson et al. [3] focused on evaluation of the rhyming criterion based on a metric known as rhyming score. The formula they used is:

![Evaluation Criterion- Rhyming Score!](/readmeassets/images/eval.png "Evaluation Criterion- Rhyming Score")\
Figure 2: Evaluation Criterion - Rhyming Score

This formula generates a rhyme distance between pairs of lines, taking a value between 0 (very low rhyming) and 1 (very high rhyming). We plan on altering this to generate 0 for perfect rhyming and a high number for low rhyming and add it to the loss functions. This would make sure that the limericks being generated are penalized for not following the specified rhyme pattern, and the model would adapt itself, rather than just using rhyming metric for evaluation.

The overall flow can be best demonstrated with the help of a flowchart

![Overall flow](/readmeassets/images/training_flow.jpeg "Overall flow")
Figure 3: Overall flow

# 6 Results

After training for about 10 hours and 80000 iterations, the training loss was 2.814. Following are some examples of limericks generated by the model:

> I am writing this limerick rhyme,\
> as I age I can no longer use time.\
> I’ll still be a master\
> and my verse will come faster,\
> and with each new age come new years day.

> I can’t do my job so it’s said,\
> I am simply a lazy old shred.\
> Of my old job I do best\
> as a crone and a pest,\
> to a caddy i work as a maid.

![Train Loss on Baseline Model](/readmeassets/images/metrics_epoch_loss.png "Train Loss on Baseline Model")
Figure 4: Train loss vs number of iterations


We tested the generated samples from each of the experiment setups, and the following results were obtained. We used 350 generated samples from each experiment to evaluate on the rhyme score. Rhyme score was calculated using the phonemes of the last word for lines (1,2), (1,5), (2,5), and
(2,3).

# 7 Conclusion
Our objective is to create a generative model that can produce limericks (short poems that conform to AABBA rhyme scheme) with or without initial context. Some of our predecessors at Carnegie Mellon have already attempted to solve this problem[3] and we are furthering their work as we try to push the boundaries of creative text generation to the next level. In our efforts to create a better model, our pursuit has led us to come up with and imbibe new loss functions and filtering mechanisms to make a more predictable model. We successfully moved the code base from TensorFlow v1 (TensorFlow v1 doesn’t support eager computation) to PyTorch, allowing us to add custom loss functions.

GPT-2 uses a default token max length of 1024, which is too high, making the computation
expensive. As limericks are just five lines long, the maximum length can be much smaller than
1024. We changed the maximum length to 64, which was enough to accommodate five lines, thereby
decreasing the training time per iteration, making the overall process 4x faster. This allowed the model to learn faster, as during the warm-up period it would not unnecessarily generate <pad> tokens. Making this change also allowed us to use the GPT-2 medium architecture, which required 345M parameters, and is attributed with a much better performance than the 117M model, as described in [11]. 

GPT-2 is not capable of producing rhyme on its own, as it uses a BPE (byte-pair encoding) of the tokens. No information related to phonemes/syllables is maintained in GPT-2 embeddings, which is required to maintain rhyme. An external constraint in the form of rhyme is required to ensure that GPT-2 produces rhyming limericks. We have attempted to impose a rhyme loss as well as a context loss on the GPT-2 training procedure, allowing it to produce much better results than a vanilla GPT-2 model. Further studies might include making the rhymes more robust by considering other rhyming
forms (partial rhyme, full rhyme, slant rhyme, etc.).

# References
[1] S. Ballas, “PoetRNN,” Available at https://github.com/sballas8/PoetRNN/ (2015) \
[2] Madeleine Begun Kane’s Archive for the ‘Limerick Contest’ Category
http://www.madkane.com/humor_blog/category/limerick-contest \
[3] Mitchell B. Forelson, Qifei Dong, Christopher Dare, and Xinkai Chen in “AiBBA: A GPT2 Based Limerick
Generator” at https://drive.google.com/file/d/1BORroctAKMLcZmmnmxDZjQwPFYDA8yJ2/view \
[4] Experiments by Gwern Branwen on Gwern.net \
[5] Jianyou Wang, Xiaoxuan Zhang, Yuren Zhou, Christopher Suh, Cynthia Rudin; There Once Was a Really Bad
Poet, It Was Automated but You Didn’t Know It. Transactions of the Association for Computational Linguistics
2021; 9 605–620. doi: https://doi.org/10.1162/tacl_a_00387 \
[6] Rui Yan, Han Jiang , Mirella Lapata , Shou-De Lin , Xueqiang Lv , and Xiaoming Li in i, Poet: Automatic
Chinese Poetry Composition through a Generative Summarization Framework under Constrained Optimization
at https://www.ijcai.org/Proceedings/13/Papers/324.pdf \
[7] Huimin Chen , Xiaoyuan Yi , Maosong Sun, Wenhao Li, Cheng Yang and Zhipeng Guo in Sentiment-
Controllable Chinese Poetry Generation at https://www.ijcai.org/proceedings/2019/0684.pdf \
[8] Jey Han Lau, Trevor Cohn, Timothy Baldwin, Julian Brooke, and Adam Hammond. 2018. Deep-speare:
A joint neural model of poetic language, meter and rhyme. In Proceedings of the 56th Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers), pages 1948–1958, Melbourne, Australia.
Association for Computational Linguistics. \
[9] T. Nguyen, P. Nguyen, H. Pham, T. Bui, T. Nguyen and D. Luong, "SP-GPT2: Semantics Improvement in
Vietnamese Poetry Generation," 2021 20th IEEE International Conference on Machine Learning and Applications
(ICMLA), 2021, pp. 1576-1581, doi: 10.1109/ICMLA52953.2021.00252. \
[10] Shamas, Victor. Deep creativity: Inside the creative mystery. Morgan James Publishing, 2017.
https://arxiv.org/pdf/2201.06118.pdf \
[11] Alec Radford et al. “Language models are unsupervised multitask learners”. In: OpenAI blog
1.8 (2019), p. 9.

