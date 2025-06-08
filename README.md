# Dialog Box Detection to Counter Forsaken 1x1x1x1 'Entanglement' Attack

This program was developed with Python and PyTorch as a precursor to a macro dedicated for automatically clicking the windows that appear when attacked by the boss 1x1x1x1 in the Roblox Game Forsaken.

However, this program if improved is not limited to the game. It can be used to detect window dialog boxes, particularly pop-ups and can help mitigate viruses that utilize pop-ups.

Recommendations:
- increase and uniformize data points in the dataset (increase to at least 200), and also use Windows dialog boxes to further generalize
    - when this is done, adjust the hyperparameters such that it balances fitting
- use a more recent version of You Only Look Once (YOLO)

## Pictures:

<p align="center">
  <img src="https://github.com/mortrpestl/forsaken-entanglement-cheat/blob/main/docu/1.png?raw=true" alt="1" width="45%" />
  &nbsp;&nbsp;
  <img src="https://github.com/mortrpestl/forsaken-entanglement-cheat/blob/main/docu/3.png?raw=true" alt="3" width="45%" />
</p>
This is the current output of the model with 30 images. Currently, it is the bounding boxes that the model is having difficulty placing.

<p align="center">
  <img src="https://github.com/mortrpestl/forsaken-entanglement-cheat/blob/main/docu/2.png?raw=true" alt="2" width="60%" />
</p>
This is part of the desired behavior.

Software and Paradigms Used:
- Python  
- [YOLO v1 (You Only Look Once)](https://arxiv.org/abs/1506.02640)  
- [TQDM](https://github.com/tqdm/tqdm)  
- [PyTorch](https://pytorch.org/)
- [ModifiedOpenLabeling, derived from OpenLabeling](https://github.com/ivangrov/ModifiedOpenLabelling)
