Model construction can get very complex and slow, so rather than passing model configuration and
re-constructing it everywhere, we re-initialize the trainable weights

Having clustering in mind, we encapsulate the run logic in the Server/Client classes. This lets us:

- create topologies, validate initialization before we start any needless threads
- defer server port assignment and assigning clients to servers.
    - Necessary  when we first need to create the topology before it is known where the clients can connect to
- sanity check clustering algorithms in isolation without requiring additional implementation
- Add more elaborate run logic to individual classes without creating any coupling to other server/client classes

Running the servers corresponding clients is delegated to some Orchestrator.
Note that this may not reflect a real deployment scenario, as it implies prior knowledge on participating clients.
However, that does not change anything about the validity of our system and experiments)