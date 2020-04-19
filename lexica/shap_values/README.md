I made a Google Collaboratory (Colab for short) notebook. The notebook isn't perfect, but it is a starting which is self-contained. 

The python file version of the same notebook. 

Now some less interesting, but finer details that I wanted to share with you:
- the padding is on the left, so for a concrete example if the window is of size 10 (i.e. TIMESTEP=10) then after tokenization and padding "hi how are you doing?" -> [ PAD, PAD, PAD, PAD, 'hi', 'how', 'are', 'you', 'doing', '?']  (NOTE: I had previously said that this was padded on the right)
- tf.keras and keras behavior are quite different. Particularly keras converges faster than tf.keras with the exact same parameters. For you, this doesn't matter since you'll be using PyTorch. Nonetheless, I wanted to let you know.

Other points of interest. This code does not rely on the empathy dictionary repository. 

## Running python file
Some quick instructions on how to setup and run in a virtual environment
 - setup the virtual environment `python3 -m venv ~/venvs/shap-lexica-creation`<sup>[1](#venvpath)</sup>
 - activate virtual environment  `. ~/venvs/shap-lexica-creation/bin/activate`
 - get the repository with git `git clone https://github.com/jsedoc/empathy_dictionary/`
 - switch into the branch that you are working in `git checkout shap-lexica-creation`
 - get into the correct directory `cd empathy_dictionary/lexica/shap_values`
 - install requirement `pip install -r requirements.txt`
 - now you are ready to run `python learning_word_ratings_for_empathy_from_document_level_user_responses.py`
 
 ## Model visualization
 In order to plot the model you'll [need graphviz](https://datascience.stackexchange.com/questions/37428/graphviz-not-working-when-imported-inside-pydotplus-graphvizs-executables-not).
 The set the variable VISUALIZE_MODEL to True in the python file.
 
 
 ---
 <a name="venvpath">1</a>: You may prefer to change the virtual environment directory path `~/venvs/shap-lexica-creation`
