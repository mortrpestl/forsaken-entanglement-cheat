# Dialog Box Detection to Counter Forsaken 1x1x1x1 'Entanglement' Attack

__DISCLAIMER: This program is intended for educational purposes and was never meant for use beyond private servers in Roblox.__

_by @mortrpestl_

This program was developed with Python and PyTorch as a precursor to a macro dedicated for automatically clicking the windows that appear when attacked by the boss 1x1x1x1 in the Roblox Game [Forsaken](https://www.roblox.com/games/18687417158/Forsaken).

However, this program if improved is not limited to the game. It can be used to detect window dialog boxes, particularly pop-ups and can help mitigate viruses that utilize dialog boxes in phishing, obfuscating the experience of the user, among other applications.

## Future Goals:
- test speed in 24 FPS videos
- integrate to a real-time live capture for use in actual Roblox servers (with the disclaimer in mind)
- integrate to a more general use case (pop-up detection)

## Recommendations:
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
This is the desired behavior, to be achieved with an increase in data points. Feel free to send recommendations in where to receive labeled images of Windows dialog boxes (or even Forsaken 1x1x1x1 Entanglement gameplay in the perspective of the attacked player!) through my Github profile.

## Software and Paradigms Used:
- Python  
- [YOLO v1 (You Only Look Once)](https://arxiv.org/abs/1506.02640)  
- [TQDM](https://github.com/tqdm/tqdm)  
- [PyTorch](https://pytorch.org/)
- [ModifiedOpenLabeling, derived from OpenLabeling](https://github.com/ivangrov/ModifiedOpenLabelling)

All rights reserved by @mortrpestl.
