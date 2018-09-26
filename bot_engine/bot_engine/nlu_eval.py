from rasa_nlu import evaluate

evaluate.run_evaluation(
    'nlu/data/testing/',
    'nlu/models/default/model_20000101-000000',
    errors_filename='nlu/evaluation/errors.json',
    confmat_filename='nlu/evaluation/confmat.png',
    intent_hist_filename='nlu/evaluation/hist.png')