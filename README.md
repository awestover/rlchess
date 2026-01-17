# Thoughts on training against interp

I’m really excited about training against interp or model-internals badness detectors. However, Fabien has convinced me that it’s at least somewhat hard to do a good project on this. In this document I’ll spell out some desiderata that good projects should satisfy, and sketch Fabien’s project idea.

(Huge thanks to Fabien for generating all the good ideas in this document\!)

## General desiderata

Training against interp is basically only interesting if it’s actually the best technique. If you artificially ban techniques, that’s pretty, similar to discussion [here](https://www.lesswrong.com/posts/oNEFGrLRupzgrHcQv/methodological-considerations-in-making-malign).

IIUC [OAI’s training against CoT](https://openai.com/index/chain-of-thought-monitoring/) paper didn’t use action monitors as part of the reward signal. Fabien thinks this is super sad. I’m sympathetic to this.

Fabien doesn’t want to do this with a malign init. I’m less sold that we shouldn’t use a malign init. But I agree that, e.g., [password locked models](https://arxiv.org/abs/2405.19550) fine-tuned to imitate weaker models would be a sad malign init. 

## An experiment you could run:

Suppose you suck at Chess, but you want to train Qwen (specifically Qwen3-8B) to be good at chess.

More specifically: 

- Qwen will be given a chess state, and asked to output (Hidden reasoning, justification for the move, the move)  
- Qwen is scored on the basis of how good the move was, as computed by some measure of how good a board state is which we’ll google.

Here are some techniques that you could try: 

- **Basic strategy:**  
  - Prompt: reason really hard about what a great chess move would be and then take that.  
- **Train on hackable reward signals:**  
  - RL on proxy “does your move capture one of the opponent’s pieces”.  
    - Probably we should prompt or SDF the model with the knowledge that we are going to do this training.   
  - RL the model with self-judgements.   
    - Caveat: you’re required to feed the judge the model’s justification.  
    - We could use the model before training, or the model after training.  
- **Making the hackable reward signals hurt less:**   
  - You might employ “early stopping” (as in [reward model overoptimization](https://arxiv.org/abs/2210.10760)).  
  - Inoculation prompting. (Tell the model: I’m going to train you on this proxy reward, you should feel free to pursue this reward for now, but later I’m going to ask you to actually try to win the game).   
  - RL the CoT to look nice.  
  - Use fancy training against interp techniques.

## Derisking the chess experiment

In order for the chess experiment to be compelling we’d need the following things:

1. Qwen is poorly elicited on Chess.  
2. We can elicit Qwen to be good at chess with a moderate amount of RL on ground truth.  
3. Reasoning is important in order for Qwen to be good at chess.  
4. RLing Qwen on the proxy reward doesn’t suffice to elicit Qwen on chess, even with early stopping.  
5. Naively RLing the CoT to look nice doesn’t work.   
6. RLing with fancy training against interp techniques works. 

This is summarized in the following diagram.   

The first steps to take towards derisking this project would be to find the blue and green curves.

Finding the red curve isn’t too much additional work. 

If those curves all look good, then we’re in business and we can try all the fancy training against interp strategies. 