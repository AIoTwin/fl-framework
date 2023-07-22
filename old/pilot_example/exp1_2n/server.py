import flwr as fl

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1,
    fraction_evaluate=1,
    min_available_clients=2,
    min_fit_clients=2,
    min_evaluate_clients=2,
    accept_failures=True,
)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:5050",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)
