### Congestion-Sensitive Grid Aggregation

This repository contains the code and the data for the case study of the paper "_Congestion-Sensitive Grid Aggregation for
DC Optimal Power Flow_". 

The files have to executed in the following order:
1. The cases from the folder _data_ should be copied to a folder outside the repository (defined by the variable 'data_folder'). 
2. Secondly, the _model.py_ file has to be executed. This determines the optimal solution for the full model for the period 19 (e.g. case='IEEE_24_p19'). The results are stored in the folder 'results'.
3. Then the file _evaluation.py_ has to be executed, which determines the aggregation for the defined partitioning methods, calculates the results of the aggregated models and evaluates the error results. **NOTE:** The evaluation file should not be executed multiple times, as the results would get additionally stored. To run the evaluation file multiple times, either the options _recalculate_clusert_ & _recalculate_aggregated_models_ have to be set to _False_ or the results in the folder 'results' have to be deleted.
