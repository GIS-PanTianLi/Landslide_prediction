# Landslide_prediction

1. Installation
conda create -n convtran python=3.10.16
conda activate convtran
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
go to https://huggingface.co/madbuda/triton-windows-builds to download triton-2.1.0
pip install triton-2.1.0-cp310-cp310-win_amd64.whl
conda env update --file environment.yml
  
3. Model testing
  run_eval.py is the code for evaluating the result of model from "A novel vegetation-aware deep learning model for predicting landslides including the effect of antecedent rainfall"
  
  python run_eval.py 
  
  run_eval_other_sota.py is the code for evaluating the result of model from "AI-powered landslide susceptibility assessment in Hong Kong"
  
  python run_eval_other_sota.py

cnn_lstm_best_model_paper.pth is the weight of model from "AI-powered landslide susceptibility assessment in Hong Kong"
lai_geoenv_test_all_model_all.pth is the weight of model from "A novel vegetation-aware deep learning model for predicting landslides including the effect of antecedent rainfall"




