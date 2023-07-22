#!/bin/bash

project_name="exp_2.75"
server_rounds=50
client_epochs=2
number_clients=3
number_failing_clients=1
failure_rounds=4

# Start server first
python3  server.py --number_clients "$number_clients" --project "$project_name" --id 0 --rounds "$server_rounds"&

#wait for server to start
sleep 10


# Start server clients
for i in $(seq 1 $((number_clients - number_failing_clients)))
do
  python3 client.py --number_clients "$number_clients" --project "$project_name" --epochs "$client_epochs"  --monitor False --id "c$i" --failure 0&
done

#additionally start clients that fail
for i in $(seq 1 $number_failing_clients)
do
  python3 client.py --number_clients "$number_clients" --project "$project_name" --epochs "$client_epochs"  --monitor False --id "f$i" --failure "$failure_rounds"&
done

# Wait for all background processes to complete
wait
