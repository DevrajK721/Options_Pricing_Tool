# Options Pricing Tool
An Options Pricing Tool for European and American Options with additional support for exotic options. This tool will use the standard derived equation to find the option price of vanilla call or put options and use a Monte Carlo Simulation or Gaussian Quadrature Approach for other European Style Options. The American Style Options will be priced using either Binomial Trees or through a Finite Difference PDE. 

## Usage 
### Python Environment 
1. Create `conda` environment. 
```shell
conda create -n Options_Env python=3.12
conda activate Options_Env
```

2. Install required Python libraries 
```shell
conda install numpy scipy streamlit
```

### Streamlit App 
```shell
streamlit run app.py
```

## Docs 
[Introduction to Options](docs/Options.md)

[Methods for Pricing European Options](docs/European_Methods.md)

[Methods for Pricing American Options](docs/American_Methods.md)