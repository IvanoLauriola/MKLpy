


MKLpy welcomes contributions from users, developers, and researches in the form of suggestions, bug reports, feature requests, code cleaning, novel implementations, or examples of research projects.



In case you experience issues using MKLpy, do not hesitate to submit a ticket using the [GitHub issue tracker](https://github.com/IvanoLauriola/MKLpy/issues). 
You are also welcome to post feature or pull requests.

- - -

## How to Contribute

There are multiple ways to contribute, briefly described in the following:

* **Suggestions and feature requests** - MKLpy is a novel and constantly growing package. Here, suggestions concerning the architecture or the development, or feature requests, such as popular kernels or notable algorithms are welcome.
* **Documentation** - If you find a typo in the documentation or have made improvements, for instance by creating a tutorial or a project to describe your experiments, please send us an email or a GitHub [pull request](https://github.com/IvanoLauriola/MKLpy/pulls). To this end, the sub-directory [docs/](https://github.com/IvanoLauriola/MKLpy/tree/master/docs) contains the source code of our on-line documentation.
* **Code contribution** - through GitHub [pull request](https://github.com/IvanoLauriola/MKLpy/pulls), you may submit code containing, for instance, bug fixes, efficient optimizations, or novel implementations of algorithms, metrics, and kernel functions.
* **Bug report** - the most common way to contribute is to report issues and bugs through the GitHub [issue tracker](https://github.com/IvanoLauriola/MKLpy/issues). However, the report must contain the description of the bug and the information needed to reproduce the issue.

- - -

### Documentation

If you use MKLpy for scientific or academic purposes, you may consider creating a *project* describing your method and the code to replicate the experiments. 

The *project* contains a concise description of the experiment you're running and the commented code.
An example of MKLpy *project* is [KerNET](/projects/proj_kernet/).



In order to create a new *project*, there are a few steps to be taken into account:

1. Fork the [project repository](https://github.com/IvanoLauriola/MKLpy) with your GitHub account and clone it into your local machine
1. Install the documentation dependencies:
	```sh
	pip install mkdocs mkdocs-material mkdocs-material-extensions pymdown-extensions
	```
1. Create your project file in markdown style and save it into `/docs/projects/proj_name.md` where `name` is the name of the project
1. Evaluate the implementation with [mkdocs](https://www.mkdocs.org/). Specifically,
	1. from the MKLpy directory, start the mkdocs dev-server `$ mkdocs serve`
	1. open a browser at `localhost:8000` to view your updated documentation
1. Create a [pull request](https://github.com/IvanoLauriola/MKLpy/pulls)

- - -

### Submitting a bug report or feature request

We use GitHub [issues](https://github.com/IvanoLauriola/MKLpy/issues) to track bugs and feature requests.
If you have found a bug or if you want to propose a new feature implemented please open an issue.

If you ask for a **feature request**, please add the following information in the issue:

1. A reasonable description of the feature
1. The reason why you think the feature may be helpful in MKLpy
1. The scientific paper behind the feature if any.


Differently, if you're submitting a **bug report**, please provide us:

1. A short description of the bug
1. A reproducible code snippet that we can use to easily reproduce the bug (see [this](https://stackoverflow.com/help/minimal-reproducible-example) for more details)
1. If the code you're running raises an exception, please provide us the full  traceback
1. The version of the software and the OS you're using


- - -

### Code contribution

If you extended MKLpy, for instance by introducing a novel MKL algorithm, a kernel function, or by defining different callbacks or scheduler, you may consider integrating these functionalities into MKLpy through a GitHub [pull request](https://github.com/IvanoLauriola/MKLpy/pulls) 

Please provide together with your pull request the following information:

1. A reasonable description of the implementation 
1. The reason why you think this feature may be helpful in MKLpy
1. The associated scientific citation if any.

A code contribution is not limited to novel features, but also bug fixes and generic code optimization.
