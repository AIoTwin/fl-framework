This should be a *true* local-first pyhton-native federated learning framework for PyTorch.

# Setup [TODO]

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install spock-config==3.0.2 --no-deps
```
To use the .venv in pycharm
1. ctrl + alt + s (Open Project Settings)
2. Go to Project Interpreter
3. Add interpreter
4. Select existing 

# Project Structure [TODO]

Paper applications as subroots, but share .venv from repo root

- project_root/paper_root

resources folder structure..

entities are first class citizen concepts for our related line of papers

Add custom data loaders and datasets into loader


# Config Pattern [TODO]

## Creating config files

- group related experiments into a folder
- Pack files into a folder for a single experiment
    - Spock allows us to use compose and decompose config objects and files. The former is very useful when we start
      separate subprocess
    - For now, we need to create separate files. I would like to change it that we can create a single file and that
      configs are passed to subproceses asap
- Create a root config file for the experiment entry
    - Reference sub config files with TODO: Add spock link to tutorial

## Structuring Spock Classes

Either typed or untyped:

Typed:
@spock
NameParams
- param1
- param2
etc.

Untyped (params are too dynamic)
@spock
NameParams
params: Mapping[str, object]

@spock
{Name}Config

- name_type
- name_params

## Dynamically Assignining Configuration

Certain properties may not be known on startup. E.g., client id or the servers a client should connect to.
To handle this, we can use Spocks CLI override feature by passing the argument to the process pool.

For example:

```python
for client_id in range(num_clients):
    client_futures.append(
        executor.submit(run_script,
                        "fl_common/clients/client_exec.py",
                        ["-c",
                         "config/example_config/client_config.yaml",
                         "--ClientConfig.client_id",
                         f"{client_id}",
                         "--server_address",
                         f"{server_address}"])
    )
```

[More details](https://fidelity.github.io/spock/advanced_features/Command-Line-Overrides)

# Todo Infra

## Priority todos:

- Support for hierarchical clustering (for manually defined and inferred cluster configurations)
    - Note: Should support arbitrary depth from the beginning on to not break our necks later on

## Notable todos:
- Better strategy for sharing dataset references
- Support for implementing, registering and retrieving custom FL strategies
- Global device management (I have some implementation lying around, it wasn't necessary in my work,
  but it would be very useful, in local Multi GPU setups e.g., Unicorn)
    - Also, we won't have to pass device so many layers down the hierarchy anymore