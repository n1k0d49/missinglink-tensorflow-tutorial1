# Introduction

In this tutorial we will take the existing implementation of a deep learning algorithm and integrate it into the MissingLink system. 

We start with a [code sample](https://github.com/tensorflow/models/tree/master/tutorials/image/mnist) that trains a model that is based on the MNIST dataset using a convolutional neural network, add the MissingLink SDK and eventually run the experiment in a MissingLink controlled emulated server.

# Getting Started

## Prerequisites

To run this tutorial you will need a MissingLink account. If you don't have one, please [head to the MissingLink website and sign up](https://missinglink.ai/console/signup/userdetails).

---
**NOTE**

This tutorial assumes you’re using virtualenv to scope your working environment.
If you don't have it installed, you can follow [this guide](https://packaging.python.org/guides/installing-using-pip-and-virtualenv/) to get it set up.

---

## First things first ...

Let’s head to the MissingLink TensorFlow Tutorial 1 [Github repository](https://github.com/missinglinkai/missinglink-tensorflow-tutorial1), and examine it.

 Notice it contains the program file, `mnist_cnn.py`, and a `requirements.txt` file. This code trains a simple convnet on the MNIST dataset (borrowed from [TensorFlow tutorials](https://github.com/tensorflow/models/tree/master/tutorials/image/mnist)).  

To make changes, you will need to create a copy of the repo and fetch it to your local development environment. Please go ahead and create a fork of the [tutorial repository](https://github.com/missinglinkai/missinglink-tensorflow-tutorial1) by clicking the fork button.

![Fork on Github](../images/fork_repo.png)

After the forked repository is created, clone it locally in your workstation. Click the clone button in Github:

![Fork on Github](../images/clone_button.png)

Now copy the URL for cloning the repository:

![Copy repo url](../images/copy_repo_url_button.png)

Next, let’s open a terminal and `git clone` using the pasted URL of your forked repository:  

```bash
$ git clone git@github.com:<YOUR_GITHUB_USERNAME>/missinglink-tensorflow-tutorial1.git
```

Now that the code is on your machine, let's prepare the environment. Run the following commands:

```bash
$ python3 -m virtualenv env
$ source env/bin/activate
$ pip install -r requirements.txt
```

## Let's run it

You can try to run the example:

```bash
$ python mnist_cnn.py
```

![Experiment progress in terminal](../images/tutorials-experiment-start.gif)

As you can see, the code runs the experiment in 10 epochs.

# Integrating the MissingLink SDK

Now, let's see how, by adding a few lines of code and a few commands, we're able to follow the experiment in MissingLink's web dashboard.

## Install and initialize the MissingLink CLI

MissingLink provides a command line interface (CLI) that allows you to control everything from the terminal.

Let's go ahead and install it:

```bash
$ pip install MissingLink
```

Next, authenticate with the MissingLink backend.

---
**NOTE**

Once you run the following command, a browser window will launch and navigate to the MissingLink website.

If you're not logged in, you will be asked to log on. When the process is completed, you will get a message to go back to the terminal.


---

```bash
$ ml auth init
```

## Creating a project

MissingLink allows you to manage several projects. Let's create a new project for this tutorial:

```bash
$ ml projects create --display-name tutorials
```

---
**NOTE**

You can see a list of all your projects by running `ml projects list`, or obviously, by going to the [MissingLink web dashboard](https://missinglink.ai/console).

---

## Updating the requirements

Go ahead and open the code in your favorite IDE.
Add the MissingLink SDK as a requirement under the `requirements.txt` file:

```diff
tensorflow
+missinglink
```

Now install the new requirements:

```bash
$ pip install -r requirements.txt
```

## Create the experiment in MissingLink

Open the `mnist_cnn.py` script file and import the MissingLink SDK:
```diff
// ...
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
+import missinglink

# CVDF mirror of http://yann.lecun.com/exdb/mnist/
SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
WORK_DIRECTORY = 'data'
// ...
```

Now we need to initialize the project so that we could have TensorFlow call the MissingLink server during the different stages of the experiment.

<!--- TODO: Make sure it works without user id and project id / token) --->

```diff
// ...
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import missinglink
+
+project = missinglink.TensorFlowProject()

# CVDF mirror of http://yann.lecun.com/exdb/mnist/
SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
WORK_DIRECTORY = 'data'
// ...
```

Now let's have TensorFlow use our project. We would need to inject calls to MissingLink during the training stage.  
Let's scroll to the `run_training()` function and wrap the epoch loop with a MissingLink experiment:

```diff
// ...
        # Initialize the graph
        init = tf.global_variables_initializer()
        session = tf.Session()
        session.run(init)

+       with missinglink_project.create_experiment() as experiment:

            # Start the training loop
-           for step in range(MAX_STEPS):
+           for step in experiment.loop(max_iterations=MAX_STEPS):
                feed_dict = fill_feed_dict(data_sets.train, images_placeholder, labels_placeholder)

                _, loss_value = session.run([train_op, loss], feed_dict=feed_dict)

                # Validate the model with the validation dataset
// ...
```

Next, we need to wrap the training call with a call to the SDK. We'll `experiment.train` scope before the `session.run` which runs the optimizer to let the SDK know it should collect the metrics as training metrics.

```diff
            for step in experiment.loop(max_iterations=MAX_STEPS):
                feed_dict = fill_feed_dict(data_sets.train, images_placeholder, labels_placeholder)

-               _, loss_value = session.run([train_op, loss], feed_dict=feed_dict)
+               with experiment.train(
+                   monitored_metrics={'loss': loss, 'acc': eval_correct}):
+                   # Note that you only need to provide the optimizer op. The SDK will automatically run the metric
+                   # tensors provided in the `experiment.train` context (and `experiment` context).
+                   _, loss_value = session.run([train_op, loss], feed_dict=feed_dict)

                # Validate the model with the validation dataset
                if (step + 1) % 500 == 0 or (step + 1) == MAX_STEPS:
```

Lastly, let the MissingLink SDK know we're starting the testing stage:

```diff
// ...
                # Validate the model with the validation dataset
                if (step + 1) % 500 == 0 or (step + 1) == MAX_STEPS:
                    print('Step %d: loss = %.2f' % (step, loss_value))
                    print('Running on validation dataset...')
+                   with experiment.validation(monitored_metrics={'loss': loss, 'acc': eval_correct}):
                        do_eval(session, eval_correct, images_placeholder, labels_placeholder, data_sets.validation)

// ...
```

## Run the integrated experiment
We're all set up to run the experiment again, but this time to see it in the Missing Link dashboard.  

Go back to the terminal and run the script again:

```bash
$ python mnist_cnn.py
```

You should see the initialization and the beginning of training. Now, switch back to the MissingLink dashboard.

Open the [MissingLink dashboard](https://missinglink.ai/console) and click the projects toolbar button on the left. In this page, you should see the list of experiments that belong to your project.

![List of projects](../images/project_list_tutorials_project.png)

Choose the `tutorials` project. Your experiment appears.  

![Experiment in list](../images/tutorial_experiment.png)

Now you can click anywhere on the experiment line to show more information about the experiment's progress.

![Experiment detailed view](../images/tutorials_experiment_info.png)

---
**NOTE**

Feel free to browse through the table and the different tabs of the experiment you're running, and see how the metrics update as the experiment progresses. This tutorial does not include an explanation about these screens. 

---

## Commit the code changes

Let's commit our code to the repo. Go to your terminal and run the following commands:

```bash
$ git add .
$ git commit -m "integrate with missinglink"
$ git push
```

# Adding Resource Management

Now that we have everything working on our local workstation, let's take the integration to the next level. 
<!--- TODO: Link to a good RM explanation --->

<!--- Moshe: which page can we link to in RM docs? --->

The next step for us would be to run the experiment on a managed server. 
MissingLink can help you manage your servers, so that you don't have to worry about it.

For the sake of simplicity, we will not connect real GPU servers in this tutorial, but rather emulate a real server on our local workstation. This all should definitely give you a sense of how it would work when running on real servers.

## The missing step

The most important step for setting up Resource Management in your project would be to give us access to your training machines. To enable access, you will  need to install MissingLink on your existing machines, or give us limited access to your cloud hosting account so we can spin up machines for you. As mentioned above, we will not perform this step in this tutorial.

## Let's emulate

Now for some magic.

We'll need to run a command for launching the local server using the MissingLink CLI. Run the following in your terminal:

```bash
$ ml run local --git-repo git@github.com:<YOUR_GITHUB_USERNAME>/missinglink-tensorflow-tutorial1.git --command "python mnist_cnn.py"
```

This command takes the code you've committed to your forked repository, clones it to your local server, installs the requirements, and runs `python mnist_cnn.py`.

---
**NOTE**

The command for running the same thing on a real server is very similar.

---

## Observe the progress

If everything goes well, we can now observe the progress of our experiment, running on a managed server, right in the dashboard.

Go to https://missinglink.ai/console and click the Resource Groups toolbar button on the left. You should see a newly created resource group representing our local emulated server.

<!--- TODO: Add a screenshot of the resource group --->

---
**NOTE**
  
This resource group is temporary and will disappear from the list once the job we're running is completed.

---

Click on the line showing the emulated server. You are taken to a view of the logs of the task running in our local server.

<!--- TODO: Add a gif showing the progress of the logs --->

Let's see the actual progress of our experiment. Click the projects toolbar button on the left and choose the `tutorials` project. You should see the new experiment's progress.

<!--- Moshe: Screenshots have the name Elad Shaham. OK? --->