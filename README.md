# A Jupyter Notebook on DQMC

This is a naive implement of DQMC(Determinant Quantum Monte-Carlo) method.

## Usage

To use this Notebook, you should have Julia installed at least. [Visit Julia download website](https://julialang.org/downloads/) for more information to install.

After Julia installed on your system, install IJulia to enable Julia kernel in Jupyter.

```
] add IJulia
```

Start Jupyter Notebook

```
using IJulia
notebook()
```

If Jupyter notebook is not installed on your system. This step will install a miniconda environment as backend. This won't conflict with your other Python install. For more information, [visit IJulia's website](https://julialang.github.io/IJulia.jl/stable/).

If you don't prefer Jupyter Notebook, you can run it with julia REPL.

```
include("main.jl")

main()
```

## License

This project is licensed under [GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0.html).