from azureml.core import Experiment, RunConfiguration, ScriptRunConfig, Workspace, Environment, Model
from azureml.train.dnn import TensorFlow
from azureml.core.conda_dependencies import CondaDependencies

ws = Workspace.from_config()
fra_eng_ds = ws.datasets['fra-eng-translation']

environment = Environment.get(ws, "sentiment-env")

estimator = TensorFlow(
    source_directory="translator",
    entry_script="experiment.py",
    framework_version="2.1",
    environment_definition=environment,
    compute_target="local",
    #script_params={'--data-size': 3000},
    inputs=[fra_eng_ds.as_named_input('in_data')]
)

experiment = Experiment(workspace=ws, name="translator-fr-en")
run = experiment.submit(config=estimator)

run.wait_for_completion(show_output=True)


run.register_model( model_name='translator-fr-en',
                    model_path='outputs/',
                    description='A translation model from english to french',
                    tags={'source_language': 'eng','target_language':'fr'},
                    model_framework=Model.Framework.TENSORFLOW,
                    model_framework_version='2.2.0',
                    properties={'BLEU Score': run.get_metrics()['bleu_score']})