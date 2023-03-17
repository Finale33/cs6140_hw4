# cs6140_hw4

## Collaborators

<b>Jiahui Zou</b>
- NUID: 001535167

<b>Yihan Xu</b>
- NUID: 001566238

## Structure of Project
- This project is consists of 7 python files
- task 1 to 4 are in the python file with the corresponding name.
- All constants are in const.py, and are shared among other files.
- Extension tasks are in alphabet_letters.py and ImageNet_transfer_learning.py.
- All datasets are in data folder.
- All path files are in path folder.

## Prerequisite for Running
The following dependencies were used and can be installed with `pip install <dependency>`:
- Torch
- Tensorflow
- NumPy
- Pandas
- Sklearn (installed with pip install scikit-learn)
- Matplotlib
- Seaborn

## Instructions for Running
- Before executing the tasks, please go to const.py, and modify the FILE_ROOT to the local path that this repo in cloned into.
- To run each task separately, please go to each file, and run main function.
- Please run the tasks from task 1 to task 4.
- When running task 2, it might take some time to finish all tests. One suggestion is to comment out all other tests and run only one test at a time.
- Before running task 3, please unzip the dataset files in dataset folder.
- Please find instructions on running extensions in the "Instructions on Extensions" section below.


## Instructions on Extensions
The Extensions we have covered:
- Experimented 6 dimensions instead of 3 dimensions in task 2.
- Created some additional test data for the greek letter task.
- Explored 4 different architectures for task 4.
- Followed the ImageNet transfer learning tutorial and achieved accuracy 0.97.
- Explored creating networks for an alphabet dataset.

Instructions on running extensions:
- When testing additional samples for task 3, please replace the test_loader file path in task_3.py with const.TESTING_SET_WITH_15_SAMPLES.
- To run imageNet_transfer_learning, please download the data from the tutorial website on canvas, then go to imageNet_transfer_learning.py and run main function.
- To run alphabet_letters, please go to alphabet_letters.py and run main function.

## Travel days used
No travel day is used.


## Operating System and IDE

<b>Yihan Xu</b>
- <b>OS</b>: MacOS
- <b>IDE</b>: PyCharm
- <b>Language</b>: Python 3.8

<b>Jiahui Zou</b>
- <b>OS</b>: MacOS
- <b>IDE</b>: PyCharm
- <b>Language</b>: Python 3.8
