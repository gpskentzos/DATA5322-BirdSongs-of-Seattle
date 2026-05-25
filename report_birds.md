# Identifying Pacific Northwest Birds from Acoustic Recordings: A Deep Learning Approach

**Course:** DATA 5322, Spring 2026
**Author:** Paul Skentzos
**Date:** May 17, 2026 (updated May 24, 2026)

---

::: {.revision-notes}
**Revision Notes (May 2026 resubmission)**

The following sections were revised in response to instructor feedback. No results, figures, or code were changed.

- **Section 2 — Theoretical Background:** Added a new paragraph providing broad background on artificial neural networks. Specifically, I have included what they are, how they learn hierarchical representations, and why they are the appropriate approach for this task.
- **Section 3 — Methodology:** Has been rewritten from a bold-label outline style into full flowing paragraphs with sufficient procedural detail to reproduce the experiment, addressing the graders comments.
- **Section 5 — Discussion:** Added a new opening paragraph added that directly summarizes the best-performing architectures and the key reasons for their superiority before moving into subsections.
- **All figures:** Removed the duplicate short alt-text captions that appeared above each figure, replacing the text so that each figure now carries a single descriptive caption below it.
:::

## Abstract

This report presents a convolutional neural network (CNN) pipeline for classifying 12 Pacific Northwest bird species from mel spectrogram representations of 3-second audio clips. A total of 1,981 recordings from the Xeno-Canto archive are stored as 128×517 mel spectrograms. Two binary CNN architectures distinguish Bewick's Wren from Dark-eyed Junco: Architecture A achieves 61.0% test accuracy with a strong Wren bias, while Architecture B with Dropout achieves 73.2% with more balanced recall. Three 12-class CNNs trained from scratch perform near the 8.3% chance level (best: 9.4%), failing to learn species-specific features from the limited dataset. An EfficientNet-B0 backbone pretrained on ImageNet and fine-tuned in two phases achieves 67.2% test accuracy with macro F1 of 0.53, confirming that transfer learning is essential at this dataset scale. Two of three unlabeled test clips are flagged as containing multiple species.

---

## 1. Introduction

Automated species identification from acoustic field recordings is a valuable application of machine learning in ecology and conservation biology. Manual review of large audio archives requires specialist expertise and is prohibitively time-consuming at scale, making computational approaches essential for programs such as BirdNET (Kahl et al., 2021).

Bird vocalizations are well suited to image-based deep learning because they can be represented as mel spectrograms: 2D time-frequency images where pixel intensity encodes acoustic energy on a perceptually motivated frequency scale. CNNs exploit the local spatial structure of these images to learn call-specific feature detectors, which is the same inductive bias that makes them effective for natural image classification (James et al., 2021).

This report covers a binary classification task (Bewick's Wren vs. Dark-eyed Junco), a 12-class from-scratch CNN comparison, EfficientNet-B0 transfer learning, and sliding-window analysis of three unlabeled test clips. Code and figures are at: https://github.com/gpskentzos/DATA5322-BirdSongs-of-Seattle

---

## 2. Background

Mel spectrograms are computed by applying a Short-Time Fourier Transform to overlapping audio frames, mapping the power spectrum through a mel filterbank (approximately linear below 1 kHz, logarithmic above), and converting to decibels. All recordings use librosa defaults: 22,050 Hz sample rate, 128 mel bins, hop size 128, window 512, yielding 128×517 arrays in the range [−80, 0] dB (McFee et al., 2015). Values are normalized to [0, 1] via $(x + 80) / 80$ before training.

Artificial neural networks are composed of layers of interconnected units, each of which applies a learned linear transformation to its inputs followed by a non-linear activation function (such as ReLU). By stacking many such layers, a network builds a hierarchy of representations: early layers detect simple patterns such as edges, tones, and onset transients, while deeper layers combine these into progressively abstract, class-specific features. This capacity for automatic, hierarchical feature learning is what makes neural networks fundamentally more powerful than classical approaches such as SVMs or Random Forests trained on hand-crafted features -- the network discovers what to look for directly from labeled examples, without requiring the analyst to specify relevant signal characteristics in advance (James et al., 2021). For acoustic classification, where the discriminative patterns span multiple frequency scales and time intervals and vary across species in ways that are difficult to articulate as rules, this learned representation is essential.

Convolutional Neural Networks (CNNs) extend this idea to 2D structured inputs by replacing full matrix multiplication with local convolution operations. CNNs learn local feature detectors that scan the full spectrogram via weight sharing, drastically reducing parameters relative to a fully connected approach on the 66,176-dimensional input. Global Average Pooling (GAP) aggregates the final feature maps to a fixed-length vector without flattening, further controlling model size. Dropout (Srivastava et al., 2014) and Batch Normalization (Ioffe and Szegedy, 2015) are evaluated as regularization strategies. All models are trained with the Adam optimizer (Kingma and Ba, 2015), EarlyStopping (patience = 12), and ReduceLROnPlateau (factor = 0.5, patience = 5).

The dataset spans a 17:1 imbalance from House Sparrow (630 recordings) to Northern Flicker (37). Inverse-frequency class weighting is applied during multi-class training:

$$w_c = \frac{N}{C \cdot N_c}$$

where $N$ is total training samples, $C = 12$ classes, and $N_c$ is the count for class $c$. Weights range from 0.261 (House Sparrow) to 4.610 (Northern Flicker).

---

## 3. Methodology

The dataset consists of 1,981 audio recordings spanning 12 Pacific Northwest bird species, sourced from the Xeno-Canto archive (Vellinga and Planque, 2015) and filtered by the instructor for quality and duration. Each recording was preprocessed using librosa to produce a 128×517 mel spectrogram (22,050 Hz sample rate, 128 mel bins, hop size 512, window size 2048), converted to decibels relative to peak power, and stored in HDF5 format as floating-point arrays. Before training, all spectrogram values were normalized to [0, 1] via the transformation $(x + 80) / 80$. Table 1 lists the 12 species and their recording counts; Figure 1 shows the resulting class distribution, which spans a 17:1 ratio from House Sparrow (630 recordings) to Northern Flicker (37).

**Table 1: Species and recording counts**

| Code | Common Name | N | Code | Common Name | N |
|------|-------------|---|------|-------------|---|
| amecro | American Crow | 66 | houspa | House Sparrow | 630 |
| amerob | American Robin | 172 | norfli | Northern Flicker | 37 |
| bewwre | Bewick's Wren | 144 | rewbla | Red-winged Blackbird | 187 |
| bkcchi | Black-capped Chickadee | 45 | sonspa | Song Sparrow | 263 |
| daejun | Dark-eyed Junco | 125 | spotow | Spotted Towhee | 137 |
| houfin | House Finch | 84 | whcspa | White-crowned Sparrow | 91 |

![](fig_class_distribution.png)

*Figure 1: Recording counts per species. The 17:1 imbalance ratio between House Sparrow and Northern Flicker requires class weighting during training.*

All experiments used a stratified 70/15/15 split, yielding 1,383 training, 299 validation, and 299 test samples, with class proportions preserved across all three splits. For the binary classification task, only recordings from Bewick's Wren (144 samples) and Dark-eyed Junco (125 samples) were retained, with the same stratified split applied. Two architectures were compared: Architecture A consists of three convolutional blocks (Conv2D + MaxPool), a Global Average Pooling layer, and a Dense(64) head with sigmoid output, totaling 100,993 parameters and no explicit regularization. Architecture B adds Dropout(0.25) after each convolutional block and Dropout(0.40) before the output layer, totaling 109,313 parameters. Both were compiled with binary cross-entropy loss and the Adam optimizer, and trained with EarlyStopping (patience = 12) and ReduceLROnPlateau (factor = 0.5, patience = 5).

For the 12-class task, inverse-frequency class weights were applied during training to mitigate the dataset imbalance. Three architectures sharing the same three-convolution-block backbone were compared: (1) a Baseline CNN with no regularization, (2) CNN + Dropout, and (3) CNN + Batch Normalization. All used sparse categorical cross-entropy loss and the same Adam optimizer settings described above. To assess whether transfer learning could overcome the dataset size limitation, EfficientNet-B0 (Tan and Le, 2019) pretrained on ImageNet was fine-tuned in two phases. Because EfficientNet expects three-channel RGB input, single-channel spectrograms were replicated across all three channels and rescaled from [0, 1] to [0, 255] to match the ImageNet preprocessing convention. In Phase 1, the entire EfficientNet backbone was frozen and only the classification head was trained at a learning rate of 1e-3 for up to 30 epochs. In Phase 2, the top 40 of the 238 backbone layers were unfrozen and the full network was fine-tuned at a reduced learning rate of 1e-4, again with EarlyStopping and ReduceLROnPlateau.

Finally, three unlabeled MP3 field recordings (test1.mp3: 23.3 s, test2.mp3: 5.3 s, test3.mp3: 15.9 s) were analyzed using a sliding-window approach. Each clip was segmented into 3-second windows with a 1-second stride; each window was preprocessed identically to the training data and passed through the best-performing model (EfficientNet-B0) to obtain a 12-class probability distribution. The dominant species was identified by majority vote across windows, and a second species was flagged as present if its mean window probability exceeded 0.20.

---

## 4. Results

### 4.1 Binary Classification

**Table 2: Binary architecture comparison**

| Architecture | Params | Epochs | Time | Val Acc | Test Acc |
|---|---|---|---|---|---|
| A: Simple CNN | 100,993 | 19 | 11 s | 0.585 | 0.610 |
| B: CNN + Dropout | 109,313 | 13 | 8 s | 0.610 | **0.732** |

Architecture A achieves 61.0% but exhibits a strong Wren bias (recall = 1.00 for Wren, 0.16 for Junco) -- it defaults to Bewick's Wren when uncertain. Architecture B achieves 73.2% with more balanced performance (recall = 0.91 for Wren, 0.53 for Junco); Dropout regularization prevented the model from collapsing to the majority class.

![](fig_binary_training_curves.png)

*Figure 2: Training and validation accuracy for both binary architectures. Architecture B achieves higher validation accuracy (0.610) and converges more stably.*

![](fig_binary_confusion.png)

*Figure 3: Confusion matrices (41 test samples). Architecture A classifies all 22 Wren recordings correctly but misses 16 of 19 Junco. Architecture B correctly classifies 20 of 22 Wren and 10 of 19 Junco.*

### 4.2 Multi-class Classification

**Table 3: Multi-class architecture comparison**

| Architecture | Params | Epochs | Time | Val Acc | Test Acc |
|---|---|---|---|---|---|
| 1: Baseline CNN | 110,732 | 14 | 56 s | 0.094 | 0.094 |
| 2: CNN + Dropout | 128,780 | 29 | 122 s | 0.117 | **0.090** |
| 3: CNN + BatchNorm | 129,676 | 12 | 69 s | 0.087 | 0.054 |

All three architectures perform near the 8.3% chance level. CNN+Dropout (best by validation accuracy) shows slight bias toward Dark-eyed Junco (recall = 0.68) while 8 of 12 species have zero recall. The class weighting reduced majority-class collapse but was insufficient to produce generalizable features from approximately 115 training samples per class.

![](fig_mc_training_curves.png)

*Figure 4: Validation accuracy for three multi-class architectures. All plateau below 12%, indicating failure to learn meaningful species representations from scratch.*

![](fig_mc_confusion.png)

*Figure 5: Confusion matrix for the best from-scratch model (CNN+Dropout, 9.0% test accuracy). Most predictions land on Dark-eyed Junco regardless of true species.*

### 4.3 Transfer Learning: EfficientNet-B0

**Table 4: Full model comparison (12-class test set)**

| Model | Params | Test Acc | Macro F1 | Phase 1 | Phase 2 |
|---|---|---|---|---|---|
| MC Baseline CNN | 110,732 | 0.094 | 0.014 | -- | -- |
| MC Dropout CNN | 128,780 | 0.090 | 0.051 | -- | -- |
| MC BatchNorm CNN | 129,676 | 0.054 | 0.027 | -- | -- |
| EfficientNet-B0 | 4,380,591 | **0.672** | **0.533** | 67.1 s | 247.3 s |

EfficientNet achieves 67.2% test accuracy and macro F1 of 0.53 -- roughly 10× the macro F1 of the best from-scratch model. Phase 2 fine-tuning showed rapid overfitting (training accuracy ~93%, validation ~67%), consistent with the small dataset limiting further backbone adaptation. Non-zero recall is achieved for 11 of 12 species; the sole failure is Black-capped Chickadee (31 training, 7 test samples), where support is too small for learning or reliable evaluation.

![](fig_efficientnet_training.png)

*Figure 7: EfficientNet-B0 training curves. Phase 1 (blue) shows steady head adaptation. Phase 2 (orange) shows rapid training accuracy gain with limited validation improvement, indicating overfitting.*

![](fig_efficientnet_confusion.png)

*Figure 8: EfficientNet-B0 confusion matrix. Unlike the from-scratch models, predictions are distributed across all 12 species with a bright diagonal, indicating genuine multi-class discrimination.*

### 4.4 External Test-Clip Predictions

![](fig_test_preprocessing_validation.png)

*Figure 9: Preprocessing validation. Training spectrograms (top row) and first windows from each test clip (bottom row) share identical shape and dB range, confirming the test pipeline is correct.*

![](fig_test_spectrograms.png)

*Figure 10: Full mel spectrograms for the three test clips. Test1.mp3 and test3.mp3 show multiple distinct call events across their duration, consistent with varied vocalizations or multiple birds.*

**Table 5: Test-clip predictions (EfficientNet-B0, 67.2% test accuracy)**

| Clip | Duration | Windows | Top species | Confidence | 2nd species | 2nd conf. | Multiple birds? |
|---|---|---|---|---|---|---|---|
| test1.mp3 | 23.3 s | 21 | Red-winged Blackbird | 0.340 | House Sparrow | 0.212 | **Yes** |
| test2.mp3 | 5.3 s | 3 | Northern Flicker | 0.664 | Red-winged Blackbird | 0.151 | No |
| test3.mp3 | 15.9 s | 13 | White-crowned Sparrow | 0.343 | Song Sparrow | 0.331 | **Yes** |

Test2.mp3 produces the most confident prediction (Northern Flicker, 66.4%). Test1.mp3 and test3.mp3 both trigger the multi-bird flag: House Sparrow (0.212) accompanies Red-winged Blackbird in test1, and White-crowned Sparrow (0.343) and Song Sparrow (0.331) are nearly tied in test3, meaning either two sparrow species overlap or a single sparrow produces calls with features of both.

![](fig_test_heatmap.png)

*Figure 11: Per-window probability heatmap. Northern Flicker dominates test2.mp3 across all windows. Test3.mp3 shows White-crowned Sparrow and Song Sparrow sharing probability mass across time.*

---

## 5. Discussion

The results establish a clear hierarchy among the architectures tested. For the 12-class problem, EfficientNet-B0 fine-tuned from ImageNet weights is unambiguously the best model, achieving 67.2% test accuracy and macro F1 of 0.53, which is a ten-fold improvement in macro F1 over the best from-scratch model (CNN + Dropout, macro F1 = 0.051). The reason is straightforward: EfficientNet arrives with convolutional feature detectors already trained on millions of images, so it needs only a small number of examples to adapt those detectors to spectrogram patterns rather than learning features from noise. Among the from-scratch multi-class models, CNN + Dropout performed best by validation accuracy (11.7%), though all three performed near the 8.3% chance level on the test set, confirming that the dataset is simply too small to support learning from scratch. For the binary task, Architecture B (CNN + Dropout) outperforms Architecture A (73.2% vs. 61.0%) for a clear mechanistic reason: Architecture A lacks regularization and collapses to predicting the majority class (Bewick's Wren) under uncertainty, achieving perfect Wren recall but only 16% Junco recall. The Dropout layers in Architecture B penalize over-reliance on any single activation pathway, forcing the model to distribute its evidence and producing much more balanced recall across both classes.

### 5.1 Limitations

The primary limitation is dataset size relative to input dimensionality. With 1,383 training samples across 12 classes (average 115 per class, minimum 25 for Northern Flicker) and a 66,176-dimensional spectrogram input, from-scratch CNNs cannot learn generalizable features. A bird call typically occupies 0.5--1 second of a 3-second window, leaving the majority of each spectrogram as background noise at the −80 dB floor, which further dilutes the gradient signal per sample. EfficientNet succeeds where the custom CNNs fail precisely because its pretrained weights provide strong feature detectors that require only a few examples to redirect toward spectrogram patterns, rather than learning from noise.

Total training time for all six models was 581 seconds (9.7 minutes) on an Apple M-series GPU via `tensorflow-metal`: the five from-scratch models completed in 267 seconds combined; EfficientNet required 314 seconds (Phase 1: 67 s, Phase 2: 247 s). The full notebook ran in 598 seconds (10 minutes).

### 5.2 Species Difficulty and Acoustic Characteristics

EfficientNet recall patterns map directly onto acoustic structure. Species with compact, distinctive call shapes, such as the American Crow (sparse low-frequency bursts below 4 kHz), Red-winged Blackbird (narrow ascending whistle from 1--4 kHz), Northern Flicker (evenly spaced low-frequency pulses), and Spotted Towhee (buzzy trills with visible harmonic stack), achieve F1 scores of 0.55--0.63 because their spectrogram signatures are visually isolated from competing classes. The sparrow group (Song Sparrow F1 0.67, White-crowned Sparrow 0.48, House Sparrow 0.86) all vocalize densely in the 2--6 kHz band with similar melodic phrasing; their spectral overlap drives the test3.mp3 confusion (White-crowned and Song Sparrow nearly tied at 0.343 vs. 0.331). Black-capped Chickadee (recall 0.00) has an audibly distinctive whistle, but 31 training examples and 7 test samples are insufficient to learn or evaluate its signature reliably.

### 5.3 Alternative Models and Why Neural Networks

Several non-neural alternatives could address this classification task. SVMs with MFCC features are the traditional baseline for audio classification: 13--40 Mel Frequency Cepstral Coefficient values per frame are summarized by their mean and variance across the clip, producing a compact fixed-length feature vector for an RBF kernel SVM. This approach was state of the art for bird classification before deep learning (Briggs et al., 2012) but requires hand-crafted features that encode domain assumptions and cannot adapt to the full richness of the spectrogram. Random Forests on MFCC features offer similar interpretability with multi-class handling and feature importance diagnostics, but share the same limitation of fixed, handcrafted representations. k-Nearest Neighbors on raw or PCA-compressed spectrograms requires no training but is prohibitively slow at inference and sensitive to irrelevant background noise in the majority of each spectrogram frame.

Neural networks are the appropriate choice for this task for three reasons. First, the input is a 2D structured image with spatially local, repeating patterns. Convolutional weight sharing is specifically designed to exploit this structure. Second, learned feature hierarchies can adapt to the specific acoustic patterns of each species without domain expertise: shallow filters detect harmonic partials and onset transients; deeper filters combine these into species-specific motifs that no hand-crafted MFCC can represent. Third, and most importantly demonstrated here, CNNs support transfer learning: the 10× improvement in macro F1 from EfficientNet (0.53 vs. 0.05) is possible only because convolutional features learned from millions of images transfer to spectrogram classification, enabling strong performance from 115 training samples per class.

---

## 6. Conclusions

A from-scratch binary CNN achieves 73.2% accuracy distinguishing Bewick's Wren from Dark-eyed Junco, confirming the preprocessing pipeline and architecture are sound. Three 12-class CNNs trained from scratch fail at near-chance accuracy due to the fundamental mismatch between dataset size (115 samples per class average) and input dimensionality (66,176 features). EfficientNet-B0 fine-tuned from ImageNet weights achieves 67.2% test accuracy and macro F1 of 0.53, which is a 10× improvement over the best from-scratch model, establishing that transfer learning is essential for effective bird call classification at this dataset scale. The most impactful future directions are SpecAugment data augmentation (Park et al., 2019) and domain-adapted pretrained models such as BirdNET (Kahl et al., 2021).

---

## References

Briggs, F., Fern, X. Z., and Raich, R. (2012). Acoustic classification of multiple simultaneous bird species: A multi-instance multi-label approach. *Proceedings of the 18th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1374-1382. https://doi.org/10.1145/2339530.2339737

Ioffe, S. and Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *Proceedings of the 32nd International Conference on Machine Learning (ICML)*, 448-456.

James, G., Witten, D., Hastie, T., and Tibshirani, R. (2021). *An Introduction to Statistical Learning with Applications in R* (2nd ed.). Springer. https://doi.org/10.1007/978-1-0716-1418-1

Kahl, S., Wood, C. M., Eibl, M., and Klinck, H. (2021). BirdNET: A deep learning solution for avian diversity monitoring. *Ecological Informatics*, 61, 101236. https://doi.org/10.1016/j.ecoinf.2021.101236

Kingma, D. P. and Ba, J. (2015). Adam: A method for stochastic optimization. *Proceedings of the 3rd International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1412.6980

McFee, B., Raffel, C., Liang, D., Ellis, D., McVicar, M., Battenberg, E., and Nieto, O. (2015). librosa: Audio and music signal analysis in Python. *Proceedings of the 14th Python in Science Conference (SciPy)*, 18-25. https://doi.org/10.25080/Majora-7b98e3ed-003

Park, D. S., Chan, W., Zhang, Y., Chiu, C.-C., Zoph, B., Cubuk, E. D., and Le, V. Q. (2019). SpecAugment: A simple data augmentation method for automatic speech recognition. *Proceedings of Interspeech 2019*, 2613-2617. https://doi.org/10.21437/Interspeech.2019-2680

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 15(1), 1929-1958.

Tan, M. and Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *Proceedings of the 36th International Conference on Machine Learning (ICML 2019)*, 6105-6114. https://arxiv.org/abs/1905.11946

Vellinga, W.-P. and Planque, R. (2015). The Xeno-Canto collection and its relation to sound recognition and classification. *CLEF 2015 Working Notes*, 1391.
