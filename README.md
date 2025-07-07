# Options Pricing Tool
If you simply wish to use the tool, it is available as a web application at [Options Pricing Tool Application](https://dk-options-pricing-tool.streamlit.app/?), although I would recommend building using the instructions below as Streamlit Cloud has limited resources and can be slow.

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
conda install numpy scipy streamlit matplotlib
```

### Streamlit App 
```shell
streamlit run app.py
```

## Docs 
The docs were lifted from my Obsidian notes and edited so they don't render properly in Github. Cloning the repository and visualizing in your VSCode works though.
[Introduction to Options](docs/Options.md)

[Method for Pricing European Options](docs/Gauss-Hermite.md)

