# Solution for Comma.ai's Calibration Challenge

If you want to solve it yourself, download the code for the challenge [here](https://github.com/commaai/calib_challenge/tree/main). <br/>
Full Explanation of the code in [this article](https://medium.com/@marklpd/predicting-car-movements-in-videos-with-a-neural-network-bedd2cc83497).

#### Challenge: Predict car movement in videos with a small and corrupted dataset with less than 25% error.

### How to run
1. Install PyTorch v2.5.0 or later
2. Run `python main.py` and wait for the model to finish training and predicting the unlabeled videos
3. Run `python eval.py unlabeled/` to check the accuracy (ideally, the error should be <25%)
