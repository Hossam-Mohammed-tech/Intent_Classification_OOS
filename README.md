# Improved Out-of-Scope Intent Classification with Dual Encoding and Threshold-based Re-Classification

## Description
Code for Dual Encoder for Threshold-Based Re-Classification (DETER) to address these challenges. This end-to-end framework efficiently detects out-of-scope intents without requiring assumptions on data distributions 
or additional post-processing steps. The core of DETER utilizes dual text encoders, the Universal Sentence Encoder (USE) and the Transformer-based Denoising AutoEncoder (TSDAE), to generate user utterance embeddings, 
which are classified through a branched neural architecture. Further, DETER generates synthetic outliers using self-supervision and incorporates out-of-scope phrases from open-domain datasets. This approach ensures a 
comprehensive training set for out-of-scope detection. Additionally, a threshold-based re-classification mechanism refines the model's initial predictions.

## License
This source code is licensed under the MIT license and can be found in the LICENSE file in the root directory of this source tree.

## Reference
If you find the source code useful in your research, we ask that you cite our paper:

### Hossam M. Zawbaa, Wael Rashwan, Sourav Dutta, Haytham Assem, "Improved Out-of-Scope Intent Classification with Dual Encoding and Threshold-based Re-Classification", Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING), pp. 8708-8718, May 2024.
