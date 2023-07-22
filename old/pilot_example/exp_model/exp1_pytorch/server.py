import time

import flwr as fl

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1,
    fraction_evaluate=1,
    min_available_clients=2,
    min_fit_clients=2,
    min_evaluate_clients=2,
    accept_failures=True,
)

start_time = time.time()
# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:5050",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)

end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

f = open("eval.csv", "a")
f.write("Total time " + str(elapsed_time) + "\n")
f.close()

print("Elapsed Time:", elapsed_time, "seconds")
