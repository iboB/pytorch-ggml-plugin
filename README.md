# PyTorch GGML Plugin

This is an example of how to create a [ggml](https://github.com/ggerganov/ggml) plugin for PyTorch.

Note that it relies on ggml features which are not in the main repo (yet). Most noitably:

* Instantiating a ggml cuda backend where the cuda device, stream handle and CUBLAS handle are provided externally
* Setting an external pointer to a ggml tensor, one that is not allocated and managed from a ggml buffer

These changes are required to use the ggml cuda backend and the data pointers from torch cuda tensors directly. Using the ggml cpu backend or copying the actual data between ggml and torch tensors will work with vanilla ggml.

[This is the PR](https://github.com/ggerganov/ggml/pull/570) which tracks the proposed changes from this repo.

## Structure

* `model/` - a static library which has a (trivial) ggml model
* `pytorch-plugin/` - a PyTorch plugin which exposes the module to a PyTorch app
* `pytorch-example.py` - an example of funning the model from the plugin
* `cpp-example.cpp` - an example of running the model as a standalone C++ executable using the static library

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This software is distributed under the MIT Software License.

See accompanying file LICENSE or copy [here](https://opensource.org/licenses/MIT).

Copyright &copy; 2023 [Borislav Stanimirov](http://github.com/iboB)
