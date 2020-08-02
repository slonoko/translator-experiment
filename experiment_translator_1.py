from azureml.core import Experiment, RunConfiguration, ScriptRunConfig, Workspace, Environment, Model
from azureml.train.dnn import TensorFlow
from azureml.core.conda_dependencies import CondaDependencies
from azureml.pipeline.steps import PythonScriptStep, EstimatorStep
from azureml.pipeline.core import Pipeline
from azureml.core.runconfig import DEFAULT_CPU_IMAGE

ws = Workspace.from_config()
fra_eng_ds = ws.datasets['fra-eng-translation']

run_config = RunConfiguration()
run_config.environment.docker.enabled = True
run_config.environment.docker.base_image = DEFAULT_CPU_IMAGE
run_config.environment.python.user_managed_dependencies = False
run_config.environment.python.conda_dependencies = CondaDependencies.create(
    pip_packages=['azureml-sdk[notebooks,automl,explain]'])

environment = Environment.get(ws, "sentiment-env")

estimator = TensorFlow(
    source_directory="translator",
    entry_script="experiment.py",
    framework_version="2.1",
    conda_packages=["python=3.7.4", "tensorflow", "tensorflow-datasets"],
    pip_packages=["azureml-sdk[notebooks,automl,explain]"],
    compute_target="archi-trainer"
)

model_step = EstimatorStep(name="training model", estimator=estimator, compute_target="archi-trainer",
                           estimator_entry_script_arguments=['--data-size', 3000], inputs=[fra_eng_ds.as_named_input('in_data')])

sentiment_pipe = Pipeline(workspace=ws, steps=[model_step])
sentiment_pipe.validate()

experiment = Experiment(workspace=ws, name="translator-fr-en")
run = experiment.submit(config=sentiment_pipe)

run.wait_for_completion(show_output=True)

ds.upload('outputs/translator_fr_en_model.h5',
          'models', overwrite=True, show_progress=True)
