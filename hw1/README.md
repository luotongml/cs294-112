# CS294-112 HW 1: Imitation Learning

## Section 2. Warmup

View `pdf/learning_rates.pdf`. This file contains the learning rates of models using behavioral cloning for every environment provided. The model consists of a Neural Network with 1 hidden layer of 32 units and a ReLU activation function, followed by an output layer with a linear activation. It was trained with an RMSProp optimizer, a learning rate of 0.001, and using MSE as the loss function, during 50 epochs.

To generate this file, first run `generate_expert_rollouts.py` to get actions and observations for 50 rollouts with each expert, stored in `expert_rollouts/[envname].pkl`. This script accepts --envname (if you only want to roll on a specific task), and --num\_rollouts and --render as optional parameters which default to 50 and False, respectively.

Now run `train_model.py`, to train models for each environment and generate `pdf/learning_rates.pdf` . You can pass --envname, if you need to train a model in one specific environment, --model\_save\_suffix and --pdf\_save\_path if you want to save the model or pdf to a different location, and --units, --epochs, --hidden\_layers, --learning\_rate, which default to the model described above.

## Section 3. Behavioral Cloning

View `pdf/model_expert_comparison.pdf`. This file contains a table showing the obtained mean reward, the expert's mean reward, the obtained std of the rewards, and the expert's std for every trained agent. Here we can see that Hopper-v1 got almost 3x better results on the expert agent, that our model had a really bad performance in complex tasks such as Humanoid-v1 and Walker2d-v1, and that Ant-v1 had almost the same reward as the expert agent (sometimes even higher).

Also see `pdf/hyperparameter_tuning.pdf` and `pdf/hyperparameter_tuning_comparison.pdf` to see how changing hyperparameters such as the number of units per hidden layer (from 50 to 100) and the number of training epochs (from 50 to 100) affects performance. Here we can see significant improvements in tasks such as Humanoid-v1 and Walker2d-v1, but other tasks showed little to no upgrades in performance.

To obtain these files, begin by running `python run_model.py` followed by `python hyperparameter_tuning.py --units 100 --epochs 100`. Both scripts accept many optional parameters (--envname, --num\_rollouts, --render for `run_model.py` and --envname, --model\_save\_suffix, --pdf\_save\_path, --units, --epochs, --hidden\_layers, --learning\_rate, --num\_rollouts, --render for `hyperparameter_tuning.py`)

## Section 4. DAgger

View `pdf/dagger_results.pdf`. In this file, we can have a look at graphs showing the mean performance of each model as well as its standard deviation against the number of DAgger iterations. Again, we can see big improvements in complex tasks such as Humanoid-v1, Walker2d-v1 and even Hopper-v1, whereas HalfCheetah-v1 actually decreased in performance with the number of DAgger iterations. To generate this file run `python run_dagger.py`, script that accepts --envname, --pdf\_save\_path, --hidden\_layers, --units, --learning\_rate, --num\_episodes, --render and --num\_iterations as parameters.
