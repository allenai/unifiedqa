## IR documents used for ARC dataset 

There are two variants of these retrieved sentences. One has a retrieved set of sentences per answer choice, as used in the Aristo*BERT* submissions: 
 
 
 
You can download a zip file from [here](ARC-OBQA-RegLivEnv-IR10V2.zip), with the combined ARC/ARC-Easy/OBQA/RegLivEnv train/dev/test sets, 
along with associated retrieved contexts from the Aristo corpus.

You'll see that the question ids have a prefix for each set:

```
ARCCH_ = ARC (Challenge)
ARCEZ_ = ARC-Easy
OBQA_ = OBQA
RegLivEnv_ = Regents Living Environments
```

An example looks like

```json
{"id": "ARCCH_Mercury_SC_415702", 
 "question": {
   "stem": "George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?", 
   "choices": [
      {"label": "A", "text": "dry palms", "para": "Nay, it is enouph to rub the dry hands together to feel the heat produced by friction, and which is far greater than the heating which takes place when the hands lie gently on each other. Rub the oil in the palms of your hands to warm it up before it touches your partner's skin. Working quickly, wash \nand dry your hands and rub them with oil. Dry them with thick, warmed towels as if their entire skin surface is erotic (it is, of course). He rubbed dirt into his palms, trying to dry them. Cool for a few minutes and rub them in your palms to remove most of the skins. Palming Rub your hands together to make them warm. Rub the palms of your hands quickly together to illustrate this generation of heat. He rubbed the palm of his hand over her reddened skin, feeling the heat and the tiny welts. His skin, his body-linen, his trousers, everything is grey, greasy, bespotted, and when with a familiar gesture he rubs his palms on his behind to wipe them dry, you ask yourself which is going to dirty the other, the seat of his trousers or his hands."}, 
      {"label": "B", ...}, ...
   ]}, 
 "answerKey": "A"}
 ```

where the "para" field for each answer choice is the retrieved context, typically 10 sentences ordered such that the one with highest IR score comes last ("His skin, his body-linen, his trousers, ..." in the above example).

The other variant, used by UnifiedQA, combines these to a single context for the overall question (by first taking the top-scoring sentence for each answer choice followed by the next six highest scoring sentences). 
This can be downloaded from [here](ARC-OBQA-RegLivEnv-IR10V8.zip). 

The format is the same, except the "para" field is now at the top level, again with the 10 sentences sorted with the highest scoring ones last:

```json
{"id": "ARCCH_Mercury_SC_415702", 
 "question": {
   "stem": "George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?", 
   "choices": [{"label": "A", "text": "dry palms"}, {"label": "B", "text": "wet palms"}, {"label": "C", "text": "palms covered with oil"}, {"label": "D", "text": "palms covered with lotion"}]
  }, 
  "answerKey": "A",
  "para": "Cool for a few minutes and rub them in your palms to remove most of the skins. Gently rub both hands together and warm the oil with your body heat. Palming Rub your hands together to make them warm. Rub the palms of your hands quickly together to illustrate this generation of heat. Warm some lotion in your hands by rubbing them together in a circular motion. Rub the oil in the palms of your hands to warm it up before it touches your partner's skin. Apply a dollop of lotion to the palm of your hand, then rub them together. He rubbed the palm of his hand over her reddened skin, feeling the heat and the tiny welts. His skin, his body-linen, his trousers, everything is grey, greasy, bespotted, and when with a familiar gesture he rubs his palms on his behind to wipe them dry, you ask yourself which is going to dirty the other, the seat of his trousers or his hands. Pour a little baby oil or pure vegetable oil in your palms and rub your hands together to warm them and the oil."
  }
```

* Please direct any questions to 	Oyvind Tafjord (oyvindt [at] allenai [dot] org). 
