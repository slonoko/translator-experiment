from azureml.core import Dataset, Workspace
from dotnetcore2 import runtime
runtime.version = ("18", "04", "0")
runtime.dist = "ubuntu"
ws = Workspace.from_config()

default_ds = ws.get_default_datastore()

data_ref = default_ds.upload_files(['data/fra-eng.tsv'],target_path='/data/files', overwrite=True, show_progress=True)
fra_eng_ds = Dataset.Tabular.from_delimited_files(path=(default_ds,'/data/files/fra-eng.tsv'), separator='\t')
fra_eng_ds.register(workspace=ws, name='fra-eng-translation', create_new_version=True)