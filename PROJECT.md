# Bayesian Reanalysis of METR AI Developer Productivity Study

## Overview

This project is a Bayesian reanalysis of "Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity" (Becker, Rush, Barnes, Rein 2025).

The original study found a surprising 19% slowdown when AI tools were allowed (contrary to predictions of 20-40% speedup). We're building a generative Bayesian model that makes explicit assumptions about the data-generating process, rather than treating forecasts as exogenous covariates as the original analysis did.

The original paper can be found in `resources/reference_paper/metr_ai_productivity.pdf`. The code and data are at https://github.com/METR/Measuring-Early-2025-AI-on-Exp-OSS-Devs.

## Statistical Philosophy and Workflow

Before building any models, consult the materials in `resources/betancourt/`. This directory contains critical reference documents from statistician Michael Betancourt covering statistical philosophy, principled model building workflow, modeling techniques, and case studies. These should serve as both inspiration and philosophical guidance throughout the project.

When writing code, structure it as similarly to Betancourt's as possible. Rely heavily on the functions in the directory `mcmc_visualization_tools`. Use base R and Stan throughout — no ggplot2, no tidyverse.

## Critical Note: Iterative Model Building

**Do not jump straight to the full model described below.** The plan is to start with a Bayesian version of the original paper's simple model (OLS with forecast as covariate), then build up iteratively — adding one piece at a time. At each step, compare results to the previous model. We may stop short of the full model if additional complexity doesn't buy us anything. This iterative approach lets us understand what each modeling choice contributes.

## Why a Generative Model?

The original paper's analysis conditions on developer forecasts as an exogenous covariate for variance reduction. This treats the forecast as a fixed input rather than as a noisy measurement of underlying latent structure.

We take a different approach: model the data-generating process explicitly. This means:

1. **Latent structure**: We posit that there are latent variables (developer ability, within-developer task variation) that *cause* what we observe (forecasts, completion times, self-reported familiarity ratings).

2. **Forecasts as outcomes, not inputs**: A developer's forecast is a noisy measurement of the latent structure — how hard they perceive the task to be given their ability. It's not a cause of completion time; both forecast and completion time are effects of the same underlying latents.

3. **Partial pooling**: With only 16 developers, hierarchical modeling lets us borrow strength across developers rather than treating each one independently or ignoring developer-level variation entirely.

4. **Proper measurement models**: The self-reported ordinal covariates (task familiarity, resource needs) should be modeled as ordinal, not continuous. They're noisy measurements of latent constructs.

5. **Principled handling of missing data**: Some covariates are missing by design (questions added partway through the study). The generative structure allows coherent imputation that respects the relationships among variables.

6. **Honest uncertainty**: Rather than relying on asymptotic approximations (CLT) with 16 developers, we get full posteriors that reflect uncertainty in the latent structure.

## Causal Assumptions

**Developer Ability**: A latent trait, one per developer, representing general competence and experience. More able developers forecast faster completion, report higher task familiarity, need fewer external resources, and complete tasks faster.

**Within-Developer Task Variation**: A latent variable, one per issue, representing how hard this task is *relative to this developer's typical task*. This is NOT inherent task difficulty — each task is done by only one developer from their own repository, so we cannot identify task difficulty independent of the developer. Higher values mean the task is harder than typical for this developer, leading to longer forecasts, higher resource needs, and longer completion times.

**Task Exposure (observed, ordinal)**: A measurement of developer ability for this type of task. Reflects the developer's history with similar work. NOT caused by task variation — it's about the developer, not this specific task.

**Resource Needs (observed, ordinal)**: Caused by both ability and task variation. Harder tasks (for this developer) require more resources. More able developers require fewer resources.

**Forecast (observed, continuous)**: Caused by both ability and task variation. More able developers forecast faster. Harder tasks (for this developer) are forecasted to take longer. The forecast does NOT cause completion time — both are downstream of the latent structure.

**Log Completion Time (observed, continuous)**: The outcome. Caused by ability, task variation, and treatment. More able developers are faster. Harder tasks take longer. Treatment (AI allowed) has some effect — the sign and magnitude are what we're trying to learn.

**Treatment (observed, binary)**: Randomized at the issue level. Only affects completion time. Does not affect forecasts, task exposure, or resource needs (these are either pre-randomization or about the developer, not the task outcome).

## Key Questions to Answer

1. How does the posterior on the ATE differ from the original paper's confidence interval (shape, width)?
2. How does the effect of AI access depend on developer ability?
