# PyTorch GGML Plugin

This is an example of how to create a [ggml](https://github.com/ggerganov/ggml) plugin for PyTorch.

Note that it relies on ggml features which are not in the main repo (yet). Most noitably:

* Instantiating a ggml cuda backend where the cuda device, stream handle and CUBLAS handle are provided externally
* Setting an external pointer to a ggml tensor, one that is not allocated and managed from a ggml buffer

These changes are required to use the ggml cuda backend and the data pointers from torch cuda tensors directly. Using the ggml cpu backend or copying the actual data between ggml and torch tensors will work with vanilla ggml.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This software is distributed under the MIT Software License.

See accompanying file LICENSE or copy [here](https://opensource.org/licenses/MIT).

Copyright &copy; 2023 [Borislav Stanimirov](http://github.com/iboB)
