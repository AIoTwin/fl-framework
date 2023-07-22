# How to run an Experiment

## Follow the tutorial:

https://docs.wandb.ai/quickstart

## Start clients and server:

Change the variables

* project_name="exp2.75"
* server_rounds=50
* client_epochs=2
* number_clients=3
* number_unreliable_clients=1
* failure_rounds=2

sh run.sh

### Run clients and server individually:

#### Start the server:

python3 server.py --project exp2_3n --rounds 50 --id 0 --num_clients 3

...where

* --project is the name of the project, has to be the same for clients and server
* --rounds number of rounds
* --id id of the client - each client needs a different one
* --num_clients is the total number of clients

#### Start a client:

python3 client.py --epochs 2 --num_clients 3 --project exp2_3n --monitor False --id 0

* --epochs number of local training epochs
* --num_clients same as num_clients of server, the reliable clients are num_clients - number_unreliable_clients
* --project same as project of server
* --monitor local monitoring, create a file with log output of the loss and accuracy
* --id id of the server
* --failure the client fails every $failure round.