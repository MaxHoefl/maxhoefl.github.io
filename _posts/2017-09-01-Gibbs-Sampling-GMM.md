---
layout: post
title: Gibbs sampling for Gaussian mixture model
---


# Gibbs sampling for Gaussian mixture model

In this article we will

*   Explain what a Gaussian mixture model is
*   Go into detail of how to fit this model using Gibbs sampling
*   Test the implementation on a little mock dataset

### Gaussian Mixture Model (GMM)

In generative models we try to explain the observed data through a density we have to specify. When there are multiple modes (peaks) in the data density and we chose a Gaussian density to characterise the data distribution then we could only capture one of the peaks and miss out on the other ones. Instead, a mixture of Gaussian densities would be better suited in this scenario. An underlying (i.e. unobserved) hidden variable $z\sim Mult(\mathbf{\pi})$ which is multinomially distributed with $P(z=k)=\pi_k$ assigns an observation $\mathbf{x}\in R^d$ to one of several Gaussians $N(\mu_k,\Sigma_k)$ each fitted to a sub-part of the data. We call those Gaussians 'components' of the mixture. That is the data likelihood becomes \begin{eqnarray*} p(\{\mathbf{x}_i\}_{i=1}^N)&=&\prod_{i=1}^Np(\mathbf{x}_i)\\ &=&\prod_{i=1}^N\sum_{k=1}^Kp(\mathbf{x}_i,z_i=k)\\ &=&\prod_{i=1}^N\sum_{k=1}^Kp(\mathbf{x}_i|\ z_i=k)p(z_i=k)\\ &=&\prod_{i=1}^N\sum_{k=1}^KN(\mathbf{\mu}_k,\Sigma_k)\pi_k \end{eqnarray*}

### Full conditionals in Bayesian framework for GMM

To arrive at a fully Bayesian framework we need to specify the distribution of the parameters in the GMM.

#### Dirichlet distribution for component membership

For the component membership parameter $\pi\sim Dir(\mathbf{\alpha})$ the Dirichlet distribution makes sense since it is a distribution on the Simplex $\{(\pi_1,\ldots,\pi_k)\in [0,1]^k|\ \sum \pi_k = 1\}$. This Dirichlet hyperparameter distribution let's us encode prior knowledge about the prevalence of particular components of the mixture. Figure 1 shows three different ways to initialize the parameter $\alpha$ of the Dirichlet distribution which can be interpreted as the pseudo counts of occurrences of the corresponding component

![Fig1\. - Weak assumption $\pi_1=\pi_2=\pi_3$, Strong assumption $\pi_1=\pi_2=\pi_3$, Strong assumption $\pi_3>\pi_2\gg\pi_1$](../images/Dirichlet.png)


#### Distribution for component Gaussian parameters

Next we need to put a prior distribution on the parameters of the component Gaussians $N(\mathbf{\mu}_k, \Sigma_k),\ k=1,\ldots,K$. For the mean parameter we choose a Gaussian $N(\mathbf{\mu}_k|\ \mathbf{m}, \mathbf{V})$ and for the covariance matrix a very common choice is the Inverse Wishart distribution, $\Sigma_k\sim IW(\mathbf{S},\nu)$ where $\nu>d-1$ and $\mathbf{S}$ must be positive definite. We will cover the Inverse Wishart in more detail in another post.

We are now able to write down the full joint distribution of observed, hidden variables plus the hyperparameters that govern the component distributions \begin{eqnarray*} p(\{\mathbf{x}\}_{i=1}^N,\mathbf{z},\{\mathbf{\mu}_k,\Sigma_k,\pi_k\}_{k=1}^K) &=& p(\mathbf{x}|\ \mathbf{z},\mathbf{\mu},\mathbf{\Sigma})p(\mathbf{z}|\ \mathbf{\pi}) \prod_{k=1}^Kp(\mathbf{\mu}_k)p(\mathbf{\Sigma}_k)\\ &=& \left(\prod_{i=1}^N\prod_{k=1}^K(\pi_kN(\mathbf{x}_i|\ \mathbf{\mu}_k,\mathbf{\Sigma}_k))^{\mathbf{1(z_i=k)}}\right)\times\\ && Dir(\mathbf{\pi}|\ \mathbf{\alpha})\prod_{k=1}^KN(\mathbf{\mu}_k|\ \mathbf{m}_0,\mathbf{V}_0)IW(\mathbf{\Sigma}_k|\ \mathbf{S}_0, \nu_0) \end{eqnarray*}

### Gibbs Sampling for GMM

We can sample from the posterior for $z_i, \mathbf{\mu}_k,\mathbf{\Sigma}_k,\pi_k$ by iteratively sampling from their full conditionals.

#### Sampling new component memberships

Given the latest samples from the Gaussian parameters of each mixture component and their current prior responsibility for generating an observation, we sample for each observation its component membership $z_i=k$ using $ p(z_i=k|\ \mathbf{x}_i,\mathbf{\mu},\mathbf{\Sigma},\mathbf{\pi})\propto \pi_kN(\mathbf{x}_i|\ \mathbf{\mu}_k,\mathbf{\Sigma}_k) $

![Sampling for each observation its component membership](../images/GMM_membership.png)


#### Sampling new component priors

Given our hyperparameter $\mathbf{\alpha}$ which represents our initial belief about the prevalence of individual mixture components and the previously sampled component memberships, we can sample new component prevalences $\pi_k$ by updating the Dirichlet distribution simply by the number of observations belonging to each component $ p(\pi_k|\ \mathbf{z})=Dir(\alpha_k+N_k) $

#### Sampling new component Gaussian density parameters

Given the previously samples memerships of each observation to their respective Gaussian, we can now sample new means and covariance matrices from the posterior (which is a bit involved) \begin{eqnarray*} p(\mathbf{\mu}_k|\ \mathbf{\Sigma}_k, \mathbf{z},\mathbf{x})&=& N(\mathbf{\mu}_k|\ \mathbf{m}_k,\mathbf{V}_k)\\ \mathbf{V}_k^{-1}&=&\mathbf{V}_0^{-1} + N_k\mathbf{\Sigma}_k^{-1}\\ \mathbf{m}_k&=& \mathbf{V}_k(\mathbf{\Sigma}_k^{-1}N_k\overline{\mathbf{x}}_k+\mathbf{V}_0^{-1}\mathbf{m}_0)\\ p(\mathbf{\Sigma}_k|\ \mathbf{\mu}_k, \mathbf{z},\mathbf{x})&=& IW(\mathbf{\Sigma}_k|\ \mathbf{S}_k,\nu_k)\\ \nu_k&=& \nu_0+N_k\\ \mathbf{S}_k&=& \mathbf{S}_0+\sum_{i=1}^N\mathbf{1}(z_i=k)(\mathbf{x}_i-\mathbf{\mu}_k)(\mathbf{x}_i-\mathbf{\mu}_k)^T \end{eqnarray*} Note that when we use an uninformative prior $\mathbf{V}_0=\infty I$, the posterior for the means reduces to $ p(\mathbf{\mu}_k|\ \mathbf{\Sigma},\mathbf{z},\mathbf{x})=N(\mathbf{\mu}_k|\ \overline{\mathbf{x}}_k,\frac{1}{N_k}\mathbf{\Sigma}_k) $ which is a bit easier to interpret: The $k$-th Gaussian is centered around the mean around all observations that are allocated to this component and the covariance diminishes as the number of those observations $N_k$ increases.

### Testing the implementation on a mock dataset

Our mock dataset contains 2-d data that is generated from three Gaussians. An oracle tells us that there are three underlying Gaussians. We let the Gibbs sampler run for 15 iterations after which we see a satisfying convergence to the three data clusters.

![Fig3\. - Plotting the joint $p(\mathbf{x}|\ \mathbf{\mu},\mathbf{\Sigma})=\sum_{k=1}^3\pi_k^jN(\mathbf{x}|\ \mathbf{\mu}_k^j,\mathbf{\Sigma}_k^j)$ after the $j$-th Gibbs iteration](../images/GMM_gif.gif)

The nice thing about solving the GMM with a Gibbs sampler rather than analytically is that stale but not optimal solutions are avoided. By stale situation I mean an initialization to the GMM which is not optimal but the Gaussian parameters don't change as the MAP cluster memberships are the same as under the initialization. In Gibbs sampling we sample from the posterior cluster membership distribution instead of using the MAP estimate, so we escape a stale situation.\\ The code can again be accessed on [github](https://github.com/MaxHoefl/Blog).
