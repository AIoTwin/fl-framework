import flwr as fl

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1,
    fraction_evaluate=1,
    min_available_clients=3,
    min_fit_clients=3,
    min_evaluate_clients=3,
    accept_failures=True,
)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:5050",
    config=fl.server.ServerConfig(num_rounds=50),
    strategy=strategy,
)
