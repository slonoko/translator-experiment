from azureml.core.model import InferenceConfig
from azureml.core.webservice import LocalWebservice
from azureml.core.model import Model
from azureml.core import Workspace

classifier_inference_config = InferenceConfig(runtime= "python",
                                              source_directory = 'deploy',
                                              entry_script="eval_emotion.py",
                                              conda_file="deploy-env.yml")


classifier_deploy_config = LocalWebservice.deploy_configuration(port=8080)


ws = Workspace.from_config()
model = ws.models['sentiment_model']
service = Model.deploy(workspace=ws,
                       name = 'classifier-service',
                       models = [model],
                       inference_config = classifier_inference_config,
                       deployment_config = classifier_deploy_config,
                       deployment_target = None)
service.wait_for_deployment(show_output = True)

endpoint = service.scoring_uri
print(endpoint)

