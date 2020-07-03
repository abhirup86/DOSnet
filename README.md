# DOSnet

Updated 7/2/2020 - Victor Fung

Demo code for DOSnet, based on implementation detailed in manuscript: TBD. Several demo input files containing DOS from data described in the paper is uploaded. 

1. Unppack the input file containing the DOS and energies:
  tar -xvf "file.tar.gz" 
or for the combined data:
  cat Combined_data.tar.gz.split* > Combined_data.tar.gz
  tar -xvf Combined_data.tar.gz

2. Run DOSnet for a particular adsorbate by changing the filename in line 32 of Demo.py, or specifying the code to run the combined case. 

3. Output will be written to txt files as "model_predict_test.txt" or "model_predict_train.txt" and/or "CV_predict.txt"

Additional questions please contact Victor Fung at fungv(at)ornl.gov
