Most of the code is ported and adjusted from torchdistill. 
However, I'm not happy with how preparing and retrieving datasets work. 

For now, this should work for most cases,  i.e., it should  support the custom Dataset and Dataloading implementations
w.r.t, hierarchical clustering we plan on working on. 

Implementing it properly isn't as simple as it seems for a couple of reasons, such as:

- For complex tasks (e.g., Object Detection), there is a lot of pre- and post-processing required at several places.
- Evaluation can require some complex dataloading and collation logic (e.g., evaluating mAP)
- There is no uniform way of applying transformations
    - Sometimes we want transforms where we can pass the input and the targets together, sometimes they are separate,
      sometimes we have more than three inputs to a transform, sometimes they require named parameters and output
      dictionaries where we must retrieve the correct keys, etc.
- Support for multiple transformation libraries (torchvision, albumentation) and custom transformations


If time permitting, I'll get back to it at some point. Although, I'd preferif maybe Ivan, Anna or maybe some
bachelor student we supervise could work on it. 

To cut down on complexity, I will assume that we won't use distributed data parallel training, which, as far as I
understand, is completely orthogonal to Federated Learning anyway,