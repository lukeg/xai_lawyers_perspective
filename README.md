### Introduction

A repository of the codes for the paper Ł. Górski, S. Ramakrishna, *Explainable Artificial Intelligence, Lawyer's Perspective*, presented at ICAIL'21.

### Repository Content

This repository contains the codes used to generate the explanations described in the paper. In addition to the explanation module code, it contains parts of the Distilbert embedding generator.

The code is dependent on the number of external libraries that were either included or used in their source form and adapted for this work. In particular, a code published by Haebin Shin, [Implementation of Grad-CAM for text](https://github.com/HaebinShin/grad-cam-text "Implementation of Grad-CAM for text") was used with the author's permission. The file `dataset.py` contains solely our additions to the original code.

For SHAP and LIME explanations generation, the following libraries were used: [SHAP](https://github.com/slundberg/shap "SHAP"), [LIME](https://github.com/marcotcr/lime "LIME").