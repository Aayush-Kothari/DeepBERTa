# DeepBERTa Benchmarking and Dataset Conversion

This repository contains scripts and resources for benchmarking molecular datasets using DeepBERTa and ChemBERTa models. It includes tools for converting SMILES representations to DeepSMILES format and fine-tuning models on MoleculeNet tasks.

## 🧪 Project Overview
- Train a DeepBERTa--Deepsmiles variant of ChemBERTa-- model on a 1 billion molecule dataset  
- Convert benchmark datasets from SMILES to DeepSMILES
- Fine-tune ChemBERTa models on classification and regression tasks
- Evaluate model performance across multiple datasets

## Project Background / Inspiration
Project Name: Deep Learning for Molecule Understanding in Cheminformatics
Cheminformatics is a field where computer science techniques are applied to solve chemical problems, such as predicting molecular properties or accelerating drug discovery. In recent years, deep learning models—including transformers, graph neural networks (GNNs), and recurrent neural networks (RNNs)—have shown strong performance in many chemical applications. These models typically rely on large molecular datasets (hundreds of millions to billions of compounds) and require a suitable molecular representation to process the input effectively. One of the most widely used representations is SMILES, a linear string notation for molecules. However, a newer variant called DeepSMILES simplifies the syntax and has been shown in recent studies to improve performance in some tasks. Despite the field's shift toward large-scale deep learning, little work has explored training transformer models directly on DeepSMILES. In this project, we introduce DeepBERTa, a DeepSMILES-based transformer built on the established ChemBERTa architecture, and train it on a billion-scale dataset to evaluate whether DeepSMILES can outperform SMILES in both prediction and generative tasks.
