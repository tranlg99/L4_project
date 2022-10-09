## Notes ##
I updated Paul on my work from last week: research, setting up VCS and reference manager Zotero.

I had questions about:

1. __Implementing the "particle-video" model with [PIPs](https://github.com/aharley/pips) in Colab;__ as the training time is long and would need too much memory in Colab.

    Paul clarified that no traning is needed but tha aim is to use the existing model to get position of targets across given frames:

   * First use videos they use in the paper
   * then try with some driving video dataset e.g. KITTI

    __How to implement python files into Colab__ because it uses .pyntb rather than .py files so I'd have to transcribe certain files.
Only transcribe the main code into a python notebook then have one library.py file in Google Drive to import necessary functions.

2. __His idea of the whole year plan__ 

    Would be a subject to change depending on the progress, but it has given me a better idea of the scope of the project. More info in [plan.md](https://github.com/tranlg99/L4_project/blob/main/plan.md) file.


3. __How much I should focus on the dissertation right now__

    For now it is better to focus on the implementation, but it is a good idea to keep notes from papers I read to later incorporate these into the background section of the dissertation.




## Plan ##
* Further research needed to understand main concepts (RAFT, ...)
* Make notes when reading papers
* Implement the model in Colab and feed it certain videos from their data set.
* Figure out how to keep Colab under VCS
