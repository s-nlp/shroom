# SmurfCat at SemEval-2024 Task 6: Leveraging Synthetic Data for Hallucination Detection

**Elisei Rykov<sup>1,2</sup>**, **Yana Shishkina<sup>2,3</sup>**, **Kseniia Petrushina<sup>1,4</sup>**, **Kseniia Titova<sup>1,5</sup>**, **Sergey Petrakov<sup>1</sup>**, and **Alexander Panchenko<sup>1,6</sup>**

<sup>1</sup>Skolkovo Institute of Science and Technology, <sup>2</sup>Tinkoff, <sup>3</sup>HSE University, <sup>4</sup>Moscow Institute of Physics and Technology, <sup>5</sup>MTS AI, <sup>6</sup>AIRI

{e.rykov, y.a.shishkina}@tinkoff.ai, {ksenia.petrushina, ksenia.titova, sergey.petrakov, a.panchenko}@skol.tech

In this paper, we present our novel systems developed for the SemEval-2024 hallucination detection task. Our investigation spans a range of strategies to compare model predictions with reference standards, encompassing diverse baselines, the refinement of pre-trained encoders through supervised learning, and an ensemble approaches utilizing several high-performing models. Through these explorations, we introduce three distinct methods that exhibit strong performance metrics. To amplify our training data, we generate additional training samples from unlabelled training subset. Furthermore, we provide a detailed comparative analysis of our approaches. Notably, our premier method achieved a commendable 9th place in the competition's model-agnostic track and 17th place in model-aware track, highlighting its effectiveness and potential.

https://aclanthology.org/2024.semeval-1.125/

# Synthetic data

## LLaMA2-7B adapters
For the first setup, we trained several LoRA adapters using small annotated data from the validation set. These adapters were used to generate some false paraphrases with hallucinations. We used unlabeled training samples as seeds for the inference adapter and to collect synthetic data. We also additionally filtered some samples using the Mutual Implication Score. See the paper for more details. The full data can be found here: `data/llama_synt`. Note that according to our ablation, the most powerful combination is PG with DM synthetic data.


## GPT-4
In the second setup, we generated both correct and incorrect hypotheses for paraphrases using GPT-4. This subset contains about 13k samples of both hallucinated and correct paraphrases. See the file here: `data/gpt4_synt`.

We utilized this prompt to produce hallucinations:
```
Your aim is to produce an incorrectly paraphrased sentence that contains a hallucination for the given source sentence. Hallucinations in a paraphrase can add new information that wasn't present in the source sentence, or exclude some important information, or reverse the meaning of the source sentence. Remember that reversing source sentence has the lowest level of priority, so use it only if there is no other way to make a hallucination. Usually it's much better to misrepresent some information, add new or exclude something important. If there is some quantitative information in the source, feel free to change them slightly. Complete the task using the examples below. The examples also show the correct paraphrase for the source sentences. Note that there are no hallucinations in the correct paraphrase, whereas your aim is to corrupt the source and produce a false paraphrase. 

Examples:
Source: "I have a permit."
The correct paraphrase: "Uh, I’m validated."
The incorrect paraphrase: "I have a permit to carry it."
Explanation: The incorrect paraphrase adds information that is not present in the source sentence ("to carry it")

Source: "Easy, easy."
The correct paraphrase: "Watch it now."
The incorrect paraphrase: "The process is easy."
Explanation: The incorrect paraphrase introduces additional information ("The process is") 

Source: "A five, six, seven, eight."
The correct paraphrase: "And 5, 6, 7, 8."
The incorrect paraphrase: "A number between five and eight."
Explanation: While the source sentence is a rhythmic count or sequence of specific numbers, the incorrect paraphrase generalizes it to "a number between five and eight".

Source: "A lot safer that way."
The correct paraphrase: "Because it’s safer."
The incorrect paraphrase: "That is a safer way to travel."
Explanation: The major hallucination lies in the addition of "That is," which wasn't present in the original source sentence. This introduces a new element and changes the focus from the general concept of safety to a specific way of travel

Source: "You’re a scam artist."
The correct paraphrase: "You are an imposter."
The incorrect paraphrase: "You’re not a good scam artist."
Explanation: While the source sentence simply states "You’re a scam artist," the incorrect paraphrase implies a judgment on the person's skill as a scam artist

Don't answer now, read the source and think step by step how to make a false paraphrase for the source sentence. Before answering, provide several examples with explanations and choose the best one. Answer starting with 'The incorrect paraphrase: 
```

For correct paraphrases, this prompt was used:
```
Read the source sentence and the paraphrased hypothesis and answer whether there are any hallucinations or related observable overgeneration errors for the paraphrasing task. 
Before answering, think step by step and write why you chose the answer you did. 
Answer the last string with 'The hypothesis is correct' if there are no hallucinations or misgenerations. Otherwise, answer with 'The hypothesis is false'.

Example 1:
Source sentence: "The European Parliament does not approve the budget."
Paraphrased hypothesis: "The budget cannot be adopted against the will of the European Parliament."
The hypothesis is false

Example 2:
Source sentence: "Everyone is capable of enjoying a good education in a society."
Paraphrased hypothesis: "We must create a society where everyone is able to enjoy a good education."
The hypothesis is correct
```

# Training scripts
Fine-tuning of the E5-Mistral could be fined here: `train/main.py`

# Citation
```
@inproceedings{rykov-etal-2024-smurfcat,
    title = "{S}murf{C}at at {S}em{E}val-2024 Task 6: Leveraging Synthetic Data for Hallucination Detection",
    author = "Rykov, Elisei  and
      Shishkina, Yana  and
      Petrushina, Ksenia  and
      Titova, Ksenia  and
      Petrakov, Sergey  and
      Panchenko, Alexander",
    editor = {Ojha, Atul Kr.  and
      Do{\u{g}}ru{\"o}z, A. Seza  and
      Tayyar Madabushi, Harish  and
      Da San Martino, Giovanni  and
      Rosenthal, Sara  and
      Ros{\'a}, Aiala},
    booktitle = "Proceedings of the 18th International Workshop on Semantic Evaluation (SemEval-2024)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.semeval-1.125",
    pages = "869--880",
    abstract = "In this paper, we present our novel systems developed for the SemEval-2024 hallucination detection task. Our investigation spans a range of strategies to compare model predictions with reference standards, encompassing diverse baselines, the refinement of pre-trained encoders through supervised learning, and an ensemble approaches utilizing several high-performing models. Through these explorations, we introduce three distinct methods that exhibit strong performance metrics. To amplify our training data, we generate additional training samples from unlabelled training subset. Furthermore, we provide a detailed comparative analysis of our approaches. Notably, our premier method achieved a commendable 9th place in the competition{'}s model-agnostic track and 20th place in model-aware track, highlighting its effectiveness and potential.",
}
```
