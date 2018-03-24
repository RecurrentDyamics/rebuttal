# README

## imputeTS
The imputeTS.R is a general scripts used in all the experiments. It reads columns in missing.csv and imputes each column respectively.

## HealthCare Experiment
We merge the data of all patients into a single csv file ***raw.csv*** which can be found in the folder.

This folder contains the implementation of GRU_D and BRITS. To run the code, first create a folder named data under HealthCare. Then run
```
python gen_data.py --task train
python gen_data.py --task test
```
respectively.

To test the model, run
```
python main.py --model GRU_D --batch_size 64
python main.py --model BRITS --batch_size 64
```

The original paper in GRU_D uses a dropout layer on the top of the regression. However, in our code, we find such dropout layer is harmful to the final AUC. In our code, we only use recurrent dropout to prevent the overfitting.

## Air Quality
The air quality code is contained in the RecurrentDynamics folder. Make sure to put the csv files in csv folder and run generate.py script before testing the model.
