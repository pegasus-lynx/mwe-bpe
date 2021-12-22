## Conf Copy Util : Design Spec

#### Problem Description
In the current scenario, when we need to run the experiments, we need to create two config files : **conf.yml** ( for experiment ) and **prep.yml** ( for preparing data for exp ).
This process of creation of config files is not automated. Although the process is not tough, but it requires constant engagement from the user.

#### Desired Solution
1. User can manipulate the parameters of conf files from a command.
2. Old conf files can be reused to create new conf files.
3. Script format, so that user can put it in a loop to make the process automated.

#### Specs:

**script_name** : create_conf
**parameters** :
- base_conf : config file to use as base, we will overwrite this file with the parameters from cli.
- working_dir : folder to save the modified file in.
- kwargs : all the parameters that you want to override
- name : the name of the output file. 