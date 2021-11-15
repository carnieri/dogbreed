# Mini ML project: Dog Breed Recognition

[Dataset](https://drive.google.com/u/0/uc?id=1Xrr_C0ho9UpOarWBluK4pTY1ps92EqxR)

Part 1: Train a classifier for 100 dog breeds. [Trained model](https://drive.google.com/uc?id=1UyLNp68kYoZglzfyDOduEPzXcS6sJwR4)

Part 2: Create a system where the user can enroll new images of existing or new dog breeds. If the user gives a new image of a known breed to be classified, the system should output the correct breed.

Part 3: Handle previously unseen breeds.

**See the [train_and_test.ipynb](https://github.com/carnieri/dogbreed/blob/master/train_and_test.ipynb) notebook for the full report, training logs, and test results.**

## Installation

`pip install -r requirements.txt`

If you're using a conda env, you may have to do `ipython kernel install --user --name=<conda_env_name>` to be able to choose your conda env from the Jupyter notebook.

## To reproduce training and testing

The recommended way to reproduce training and testing is to create a Jupyter server (`jupyter notebook`, open [train_and_test.ipynb](https://github.com/carnieri/dogbreed/blob/master/train_and_test.ipynb), and run all cells.

If you prefer to run training and testing from the command line, you can `./run.sh` instead, which will convert the Jupyter notebook to a .py script and run it. But you won't get any plots that way.

## To run the web app demos

`./run_webapp_part1.sh` 

`./run_webapp_part2.sh`
