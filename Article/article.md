# Automated Best Image Selection and Multitask Deep Learning for Retinopathy of Prematurity Screening from Fundus Videos: A Multicenter Study

---

## 2. Materials and Methods

### 2.1 Study Design

This was a multicenter retrospective study conducted across five institutions in [Country — placeholder]. The study comprised two components: (1) development and validation of an automated best image selection algorithm using data from two institutions, and (2) development and evaluation of a multitask deep learning model for ROP classification using data from all five institutions. The study protocol was approved by the institutional review board of [IRB institution — placeholder] (approval number: [placeholder]). Informed consent was obtained [or waived — placeholder] in accordance with institutional guidelines.

### 2.2 Datasets

#### 2.2.1 Best Image Selection Development Dataset

For the development and validation of the automated best image selection algorithm, fundus examination videos were collected from 22 eyes of 22 patients at two institutions ([Institution names — placeholder]). A total of 6,689 individual frames were extracted from these videos. Two experienced ophthalmologists (Y.F. and H.K.) independently reviewed each video and selected the best-quality frames suitable for clinical assessment, yielding 138 consensus best images across all cases (mean 6.3 images per video). These human-selected images served as the reference standard for evaluating the automated selection algorithm.

#### 2.2.2 Multicenter ROP Classification Dataset

For the ROP classification model, fundus examination videos were collected from 300 patients (348 videos) across five institutions: Institution A (n = 110 videos), Institution B (n = 90), Institution C (n = 64), Institution D (n = 43), and Institution E (n = 41). [OWCH/FU/AMU/KCMC/YCH — formal names to be inserted]. Multiple videos per patient were included when separate examination sessions were available. Approximately 8,800 frames were extracted from these videos.

#### 2.2.3 Image Quality Grading

All extracted frames from the multicenter dataset were independently graded for image quality by an experienced ophthalmologist (K.K.) into three categories: Good (suitable for diagnosis), Fair (suboptimal quality but interpretable), and Bad (unsuitable for diagnosis). Only images graded as Good or Fair were included in the final dataset for model training and evaluation, yielding 6,491 images.

#### 2.2.4 Clinical Labels

Each video was annotated by expert graders for the following clinical variables:

- **Zone** (3 classes): Zone I, Zone II, or Zone III, indicating the extent of retinal vascularization
- **Stage** (4 classes): Stage 0 (no ROP), Stage 1, Stage 2, or Stage 3, representing the severity of retinal changes at the boundary of vascularized and avascular retina
- **Plus disease** (3 classes): Normal, Pre-plus, or Plus, characterizing the degree of posterior pole vascular dilation and tortuosity
- **Aggressive ROP (A-ROP)**: A rapidly progressive form of ROP requiring urgent intervention, regardless of zone or stage
- **Treatment-requiring ROP**: Clinical judgment of whether treatment intervention was indicated

All images from a given video shared the same clinical labels. The class distributions across the 6,491 images are shown in **Table 2**.

#### 2.2.5 Referral-Warranted ROP Definition

Referral-warranted ROP (RW-ROP) was defined as the presence of any of the following: Zone I disease, Stage 3 or higher, or Plus disease. This composite endpoint was derived from the individual task predictions rather than labeled independently, following established screening criteria.

### 2.3 Automated Image Processing Pipeline

The image processing pipeline consisted of two sequential stages: lens detection and fundus structure segmentation (Figure 1).

#### 2.3.1 Lens Detection

A Real-Time Detection Transformer (RT-DETR-L) model was trained to detect the circular eyepiece lens region in raw fundus video frames. The model was trained on 4,880 images with 1,221 validation images at an input resolution of 640 × 640 pixels, using the Ultralytics framework with random flipping augmentation (horizontal and vertical, p = 0.5). Early stopping was applied with a patience of 100 epochs out of a maximum of 1,000 epochs. The detected lens bounding box was used to crop each frame, and a circular mask was applied to exclude regions outside the lens aperture, with peripheral areas filled with a neutral gray value (RGB: 114, 114, 114).

#### 2.3.2 Fundus Structure Segmentation

An RF-DETR-Nano model, a lightweight object detection and segmentation architecture based on a DINOv2 Vision Transformer backbone, was trained to segment three anatomical structures within the lens-cropped images: fundus (retina), optic disc, and macula. The model was trained on 16,479 images with 4,023 validation images at an input resolution of 448 × 448 pixels, using a batch size of 4 with gradient accumulation over 4 steps. Training was performed for up to 100 epochs with early stopping (patience = 20). Learning rates were set to 1 × 10⁻⁴ for the main network and 1.5 × 10⁻⁴ for the encoder, with a weight decay of 1 × 10⁻⁴. Exponential moving average (EMA) was applied for model weight stabilization.

During preliminary evaluation, macula detection showed insufficient reliability (confidence consistently below 0.70), and was therefore excluded from downstream analyses. Only fundus (retina) and optic disc segmentation masks were used for subsequent feature extraction and image quality assessment.

### 2.4 Best Image Selection Algorithm

#### 2.4.1 Quality Feature Extraction

For each frame that passed lens detection (i.e., lens detected and retina ratio > 0), the following image quality features were computed from the segmentation masks:

1. **Retina ratio**: The proportion of detected retinal area relative to the total lens area (retina_area / lens_area × 100), reflecting the extent of visible fundus.

2. **Modified Blur Sensitivity Score (MBSS)**: A composite sharpness metric computed within the retinal mask region, defined as a weighted combination of four sub-components:
   - Multi-scale Laplacian variance (L_multi; weight 0.35): measures image sharpness across spatial frequencies
   - FFT high-frequency energy ratio (HF_ratio; weight 0.25): quantifies fine detail preservation
   - Spectral centroid (Spec_centroid; weight 0.20): characterizes the frequency distribution of image content
   - Gradient 90th percentile (Grad_p90; weight 0.20): captures edge strength and focus quality

   Each sub-component was z-score normalized within each video before weighted summation.

3. **Disc ring sharpness score**: The Laplacian variance computed in an annular region surrounding the optic disc, z-score normalized within each video, reflecting the local image quality near the clinically important disc region.

4. **Mean saturation (S_mean)**: The mean saturation value in HSV color space within the retinal mask, serving as an indicator of image contrast and color fidelity. Lower values indicate washed-out or desaturated images.

5. **Disc edge coverage ratio**: The proportion of the optic disc boundary (edge pixels) that is covered by the surrounding retinal mask. This metric was designed to identify images where the optic disc is well-centered within the visible fundus area, as images with partially obscured disc edges tend to have suboptimal positioning for clinical assessment.

#### 2.4.2 Algorithm Development (22-Case Dataset)

The best image selection algorithm was iteratively developed using the 22-case dataset with human expert selections as the reference standard.

**Initial approach**: A rank-based scoring method was first evaluated, combining within-video ranks of retina area, MBSS, disc ring sharpness, and mean saturation with weights assigned proportional to assumed clinical relevance (retina area weighted highest). This initial approach achieved an image concordance rate of approximately 44%.

**Failure analysis and algorithm refinement**: Analysis of videos with zero concordance between AI and human selections revealed that experienced clinicians consistently preferred images where the optic disc was fully surrounded by visible retina, even when alternative frames had larger retina area or higher sharpness scores. Human-selected best images had substantially higher disc edge coverage (mean 0.955) compared to all candidate images (mean 0.89). Based on this finding, the algorithm was redesigned to incorporate disc edge coverage ratio as a primary filtering criterion.

**Final algorithm**: The refined approach consisted of two stages:

1. **Filtering stage**: Candidate images were required to have disc_edge_coverage_ratio ≥ 0.80. The threshold was systematically optimized by evaluating concordance rates across a range of cutoff values (0.75–0.98) on the 22-case dataset; 0.80 yielded the highest image concordance. If fewer than 10 images passed this threshold, a fallback strategy selected images based on retina ratio alone.

2. **Scoring stage**: Passing candidates were scored using a normalized weighted combination:

$$\text{score} = 0.4 \times \hat{r}_{\text{retina}} + 0.4 \times \hat{g}_{\text{Grad\_p90}} + 0.2 \times \hat{m}_{\text{MBSS}}$$

where $\hat{r}$, $\hat{g}$, and $\hat{m}$ denote min-max normalized values of retina ratio, gradient 90th percentile, and MBSS composite score within each video, respectively. Scoring weights were assigned based on the relative importance observed during development: retina coverage and image sharpness (Grad_p90) contributed equally as the primary quality indicators, with the composite MBSS score serving as a secondary tiebreaker.

This final algorithm was validated on the 22-case dataset (Section 3.2) and subsequently applied without modification to the multicenter dataset, where the top-10 and top-5 images per video were selected for downstream classification experiments.

### 2.5 ROP Classification Model

#### 2.5.1 Model Architecture

The classification model was based on a multitask learning framework that jointly predicted five clinical endpoints from a single shared feature representation (Figure 1). The architecture comprised three components:

**Image encoder**: An EfficientNet-B0 backbone pretrained on ImageNet was used to extract 1,280-dimensional feature vectors from fundus images resized to 512 × 512 pixels. Global average pooling was applied to the final convolutional feature maps.

**Clinical data encoder**: A multilayer perceptron (MLP) processed four clinical variables — gestational age (GA, weeks), birth weight (BW, grams), postmenstrual age at examination (PMA, weeks), and sex — into a 32-dimensional embedding. The MLP consisted of two fully connected layers (4 → 64 → 32) with ReLU activation, batch normalization, and dropout (p = 0.3). Continuous variables (GA, BW, PMA) were z-score normalized using training fold statistics; sex was encoded as binary (0/1), with missing values imputed as 0.5.

**Task-specific heads**: The image features (1,280-dim) and clinical embeddings (32-dim) were concatenated into a 1,312-dimensional fused representation, which was shared across five parallel classification heads. Each head consisted of two fully connected layers (1,312 → 256 → *n_classes*) with ReLU activation and dropout (p = 0.5 and 0.25 for the first and second layers, respectively). The five heads predicted: Zone (3 classes), Stage (4 classes), Plus disease (3 classes), A-ROP (2 classes), and treatment indication (2 classes).

#### 2.5.2 Training Procedure

**Loss function**: A class-balanced multitask loss was used, combining weighted cross-entropy loss with focal loss (γ = 2.0). For each task, the final loss was computed as the average of the weighted cross-entropy and focal loss terms, with label smoothing (ε = 0.1) applied to all tasks. Class weights were computed inversely proportional to class frequency within each training fold. Task-level weights were set to 1.0 for Zone, Stage, and Plus, and 1.5 for A-ROP and Treatment, reflecting the clinical importance of detecting severe disease.

**Data augmentation**: Training images were augmented with random rotation (±180°, p = 0.9), horizontal and vertical flipping (p = 0.5 each), color jitter (brightness, contrast, saturation, hue; p = 0.7), CLAHE (p = 0.3), Gaussian or motion blur (p = 0.3), Gaussian noise (variance 10–50, p = 0.3), and random resized cropping (scale 0.85–1.0, p = 0.4). MixUp augmentation (α = 0.2, p = 0.5) was also applied, blending both images and their corresponding clinical feature vectors.

**Optimization**: The model was trained using AdamW optimizer with a learning rate of 1 × 10⁻⁴, weight decay of 1 × 10⁻³, and OneCycleLR scheduling. Training proceeded for up to 200 epochs with a batch size of 16, with early stopping based on validation loss (patience = 15 epochs). All images were normalized using ImageNet statistics (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]).

**Two-phase training**: To maximize the use of available training data while maintaining a separate held-out test set for unbiased evaluation, a two-phase training procedure was employed. In Phase 1, the model was trained on the training set (60% of data) with early stopping based on validation loss (patience = 15 epochs), and the best epoch *N* was recorded. In Phase 2, the model was reinitialized and retrained from scratch on the combined training and validation sets (80% of data) for exactly *N* epochs using OneCycleLR scheduling. The retrained model was then evaluated on the held-out test set (20%).

#### 2.5.3 Referral-Warranted ROP Inference

RW-ROP probability was derived from the individual task outputs using the inclusion-exclusion principle for the union of independent events:

$$P(\text{RW-ROP}) = 1 - (1 - P(\text{Zone I})) \times (1 - P(\text{Stage 3})) \times (1 - P(\text{Plus}))$$

where $P(\text{Zone I})$, $P(\text{Stage 3})$, and $P(\text{Plus})$ denote the softmax probabilities for Zone I, Stage 3, and Plus disease, respectively. A threshold of 0.5 was applied by default for binary classification.

### 2.6 Video-Level Aggregation

Since clinical decisions are made at the video (examination) level rather than the individual frame level, per-image predictions were aggregated to produce a single classification per video. Two aggregation strategies were evaluated:

- **Soft voting**: Per-image class probabilities were averaged across selected frames, and the final prediction was determined by applying argmax (for multiclass tasks) or a probability threshold (for binary tasks) to the averaged probabilities.

- **Hard voting**: Per-image hard predictions (argmax of softmax outputs) were tallied, and the final prediction was assigned to the majority class.

Three image selection conditions were compared:
1. **All images**: All quality-filtered images from each video
2. **Top-10**: The 10 highest-scoring images per the best image selection algorithm
3. **Top-5**: The 5 highest-scoring images per the best image selection algorithm

### 2.7 Cross-Validation and Evaluation Metrics

#### 2.7.1 Cross-Validation Strategy

Model performance was evaluated using 5-fold stratified group cross-validation (StratifiedGroupKFold) with a train–validation–test split. Stratification was performed on the joint distribution of Zone, Stage, and Treatment labels to maintain class balance, including rare treatment-requiring cases, across folds. Grouping was applied at the video level to ensure that all images from the same video were assigned to the same fold, preventing data leakage. In each iteration, one fold served as the test set, an adjacent fold as the validation set, and the remaining three folds as the training set (approximately 60%, 20%, and 20% of the data, respectively). Clinical feature normalization statistics and class weights were recomputed from the training folds in each iteration. The two-phase training procedure (Section 2.5.2) was applied, with Phase 1 using the training and validation sets and Phase 2 using the combined training–validation set for final model training.

#### 2.7.2 Per-Image Evaluation

For multiclass tasks (Zone, Stage, Plus), performance was assessed using overall accuracy, quadratic weighted kappa (QWK), and macro-averaged F1 score. For binary tasks (A-ROP, Treatment) and the derived RW-ROP endpoint, sensitivity, specificity, positive predictive value (PPV), negative predictive value (NPV), and area under the receiver operating characteristic curve (AUC) were reported. Per-image results are reported as mean ± standard deviation across the 5 folds.

#### 2.7.3 Video-Level Evaluation

For video-level evaluation, predictions from all five folds were pooled (each image was predicted exactly once, by the model trained without that image's fold), and video-level aggregation was applied to the pooled predictions. This approach ensures that every video received predictions from a model that did not see any of its images during training.

#### 2.7.4 Best Image Selection Evaluation

The automated best image selection was evaluated against human expert selections using two concordance metrics:

- **Image concordance rate**: The proportion of AI-selected top-5 images that matched human-selected best images, computed as the total number of concordant images divided by the total number of AI-selected images across all videos (5 × number of videos).
- **Video concordance rate**: The proportion of videos in which at least one AI-selected top-5 image matched a human-selected best image.

---

## 3. Results

### 3.1 Dataset Characteristics

The class distributions of the multicenter dataset (n = 6,491 images from 348 videos) are summarized in **Table 2**. Zone II was the most prevalent zone class (61.6%), Stage 0 was the most common stage (43.5%), and normal vascular appearance predominated in Plus classification (83.5%). A-ROP was present in 228 images (3.5%) from 9 videos, and treatment was indicated for 627 images (9.7%) from 33 videos. RW-ROP was present in 1,417 images (21.8%) from 84 videos.

### 3.2 Best Image Selection Validation (22 Cases)

The automated best image selection algorithm was validated against consensus selections by two ophthalmologists across 22 examination videos containing 6,689 frames. Each video yielded 5 AI-selected images, for a total of 110 evaluated image–human pairs.

The initial rank-based algorithm (without disc edge coverage filtering) achieved an image concordance rate of 43.6% (48/110) and a video concordance rate of 81.8% (18/22). After incorporation of the disc edge coverage filter (≥ 0.80), the final algorithm improved to an image concordance rate of **54.5%** (60/110) and a video concordance rate of **86.4%** (19/22), representing an improvement of 10.9 and 4.5 percentage points, respectively.

Among the three videos with zero concordance in the final algorithm, one case was attributable to differing temporal preferences between AI and human raters (AI favored later frames with higher retina ratio, while humans preferred earlier frames with better disc positioning despite comparable disc edge coverage), and another was due to the human-selected frames being absent from the extracted dataset as a result of the frame sampling interval.

### 3.3 Per-Image ROP Classification Performance

The per-image classification performance across 5-fold cross-validation is presented in **Table 3**.

**Table 3. Per-image classification performance (5-fold cross-validation with two-phase training, n = 6,491)**

| Task | Accuracy | QWK | Macro F1 |
|------|----------|-----|----------|
| Zone | 80.3% ± 1.9% | 0.722 ± 0.029 | 0.769 ± 0.021 |
| Stage | 76.3% ± 3.2% | 0.814 ± 0.045 | 0.737 ± 0.045 |
| Plus | 90.8% ± 2.0% | 0.775 ± 0.063 | 0.696 ± 0.112 |

| Task | Sensitivity | Specificity | PPV | NPV | AUC |
|------|-------------|-------------|-----|-----|-----|
| A-ROP | 75.4% ± 38.2% | — | — | — | 0.774 ± 0.389 |
| Treatment | 70.7% ± 18.2% | 97.3% ± 1.3% | 77.1% ± 7.5% | 97.2% ± 1.2% | 0.922 ± 0.033 |
| RW-ROP | 86.4% ± 3.4% | 87.7% ± 3.1% | 66.0% ± 9.3% | 96.0% ± 0.7% | 0.932 ± 0.029 |

Zone accuracy reached 80.3% (QWK 0.722) and Stage classification achieved the highest ordinal agreement (QWK 0.814). Plus disease classification had the highest overall accuracy (90.8%), partly attributable to the predominance of the normal class. For binary tasks, the model achieved the highest discrimination for treatment-requiring ROP (AUC 0.922) and RW-ROP (AUC 0.932). A-ROP detection showed the widest performance variability across folds (sensitivity 75.4% ± 38.2%, AUC 0.774 ± 0.389), consistent with the extreme class imbalance (3.5% prevalence) and the small number of positive cases in individual folds.

### 3.4 Video-Level Classification with Best Image Selection

Video-level classification results using soft voting across different image selection strategies are presented in **Table 4**. The per-image baseline represents pooled predictions across all folds before video-level aggregation.

**Table 4. Video-level classification performance with soft voting (n = 348 videos)**

*Multiclass tasks:*

| Task | Per-Image (pooled) | All (soft) | Top-10 (soft) | Top-5 (soft) |
|------|-------------------|------------|---------------|--------------|
| Zone Acc | 80.3% | 81.9% | **83.3%** | 83.6% |
| Zone QWK | 0.724 | 0.676 | **0.700** | 0.699 |
| Stage Acc | 76.1% | 77.9% | **79.0%** | 78.7% |
| Stage QWK | 0.800 | 0.686 | **0.702** | 0.699 |
| Plus Acc | 90.6% | 91.4% | 91.4% | **91.7%** |
| Plus QWK | 0.742 | 0.707 | 0.705 | **0.715** |

*Binary tasks:*

| Task | Metric | Per-Image | All (soft) | Top-10 (soft) | Top-5 (soft) |
|------|--------|-----------|------------|---------------|--------------|
| Treatment | Sens | 72.7% | 78.8% | **81.8%** | **81.8%** |
| Treatment | Spec | 97.4% | 96.8% | **96.8%** | 96.5% |
| Treatment | AUC | 0.924 | **0.981** | 0.979 | 0.978 |
| A-ROP | Sens | 96.5% | **100%** | **100%** | **100%** |
| A-ROP | AUC | 0.964 | 0.984 | **0.986** | **0.987** |
| RW-ROP | Sens | 87.2% | 82.1% | **84.5%** | **84.5%** |
| RW-ROP | Spec | 87.8% | **93.6%** | 93.2% | 92.8% |
| RW-ROP | AUC | 0.935 | 0.944 | **0.947** | 0.944 |

Video-level aggregation with soft voting improved performance for most classification tasks compared to per-image prediction. A-ROP achieved **100% sensitivity** across all voting conditions (All, Top-10, and Top-5), correctly identifying all 9 A-ROP cases at the video level. Treatment sensitivity improved from 72.7% (per-image) to 81.8% with Top-10 and Top-5 soft voting. Zone classification showed the largest improvement among multiclass tasks, with accuracy increasing from 80.3% (per-image) to 83.3% (Top-10 soft) and 83.6% (Top-5 soft).

For RW-ROP, video-level specificity improved substantially (87.8% per-image → 93.6% All-soft, 93.2% Top-10-soft) while sensitivity showed a modest trade-off (87.2% → 84.5% with Top-10/Top-5). The increase in AUC from 0.935 (per-image) to 0.947 (Top-10 soft) indicates that video-level aggregation improved overall discrimination.

Best image selection (Top-10 or Top-5) generally outperformed using all images for multiclass tasks and maintained or improved binary task AUC, suggesting that the quality-filtering algorithm successfully excluded low-quality frames. The benefit was most pronounced for Zone classification (accuracy 81.9% → 83.3%) and A-ROP detection (AUC 0.984 → 0.987). Video-level threshold optimization to recover treatment sensitivity is presented in Section 3.5.

Hard voting results showed similar trends and are presented in **Supplementary Table S1**.

### 3.5 Threshold Optimization

Post hoc threshold optimization was performed for treatment-requiring ROP and RW-ROP at both the per-image level (Table 5) and the video level with Top-5 soft voting (Table 6).

**Table 5. Per-image threshold optimization (pooled 5-fold cross-validation, n = 6,491)**

| Task | Threshold Method | Threshold | Sensitivity | Specificity |
|------|-----------------|-----------|-------------|-------------|
| Treatment | Default (0.50) | 0.500 | 72.7% | 97.4% |
| Treatment | Youden index | 0.346 | 85.3% | 92.8% |
| Treatment | Sens ≥ 95% | 0.300 | 95.1% | 43.7% |
| RW-ROP | Default (0.50) | 0.500 | 87.2% | 87.8% |
| RW-ROP | Youden index | 0.529 | 85.4% | 89.9% |
| RW-ROP | Sens ≥ 95% | 0.391 | 95.1% | 59.4% |

At the per-image level, the Youden-optimal threshold for treatment-requiring ROP (0.346) improved sensitivity from 72.7% to 85.3% with a specificity of 92.8%. However, achieving ≥95% sensitivity at the per-image level required a substantial specificity trade-off (43.7% for Treatment, 59.4% for RW-ROP).

**Table 6. Video-level threshold optimization (Top-5 soft voting, n = 348 videos)**

| Task | Threshold Method | Threshold | Sensitivity | Specificity |
|------|-----------------|-----------|-------------|-------------|
| Treatment | Default (0.50) | 0.500 | 81.8% | 96.5% |
| Treatment | Youden index | 0.359 | 97.0% | 93.0% |
| Treatment | Sens ≥ 95% | 0.359 | 97.0% | 93.0% |
| RW-ROP | Default (0.50) | 0.500 | 94.1% | 87.5% |
| RW-ROP | Youden index | 0.489 | 95.2% | 87.1% |
| RW-ROP | Sens ≥ 95% | 0.489 | 95.2% | 87.1% |

Video-level aggregation with threshold optimization substantially mitigated the specificity trade-off observed at the per-image level. With Top-5 soft voting and the Youden-optimal threshold, treatment-requiring ROP achieved **97.0% sensitivity** and **93.0% specificity**, identifying 32 of 33 treatment-requiring videos with only 22 false positives among 315 non-treatment videos. For RW-ROP, the Youden-optimal threshold achieved 95.2% sensitivity and 87.1% specificity. These results demonstrate that video-level probability aggregation, by smoothing per-image prediction noise, enables high sensitivity with clinically acceptable specificity.

---

## Tables and Figures

### Table 1. Dataset characteristics

| Characteristic | Best Image Selection Dataset | Multicenter Classification Dataset |
|---------------|-----------------------------|------------------------------------|
| Institutions | 2 ([placeholder]) | 5 ([placeholder]) |
| Patients | 22 | 300 |
| Videos | 22 | 348 |
| Total extracted frames | 6,689 | ~8,800 |
| Quality-filtered images | — | 6,491 (Good + Fair) |
| Human-selected best images | 138 | — |

### Table 2. Class distribution of the multicenter dataset (n = 6,491 images)

| Classification | Class | n (%) |
|---------------|-------|-------|
| **Zone** | Zone I | 750 (11.6%) |
| | Zone II | 3,999 (61.6%) |
| | Zone III | 1,742 (26.8%) |
| **Stage** | Stage 0 | 2,825 (43.5%) |
| | Stage 1 | 1,566 (24.1%) |
| | Stage 2 | 1,238 (19.1%) |
| | Stage 3 | 840 (12.9%) |
| **Plus** | Normal | 5,422 (83.5%) |
| | Pre-plus | 639 (9.8%) |
| | Plus | 408 (6.3%) |
| **A-ROP** | No | 6,263 (96.5%) |
| | Yes | 228 (3.5%) |
| **Treatment** | Not indicated | 5,864 (90.3%) |
| | Indicated | 627 (9.7%) |
| **RW-ROP** | No | 5,074 (78.2%) |
| | Yes | 1,417 (21.8%) |

*Note: 22 images with unknown Stage/Plus labels were excluded from those task-specific analyses.*

### Figure 1 (placeholder)

**System overview.** Schematic diagram illustrating the end-to-end pipeline from video input through lens detection (RT-DETR-L), fundus segmentation (RF-DETR-Nano), automated best image selection (disc edge coverage filtering + quality scoring), multitask classification (EfficientNet-B0 + clinical data), and video-level aggregation (soft voting).

### Figure 2 (placeholder)

**Confusion matrices** for Zone, Stage, and Plus classification tasks (pooled 5-fold cross-validation predictions, n = 6,491 images).

### Figure 3 (placeholder)

**Receiver operating characteristic (ROC) curves** for Treatment-requiring ROP, Aggressive ROP, and Referral-warranted ROP at the per-image level (pooled 5-fold cross-validation) and video level (Top-10 soft voting, n = 348).

### Figure 4 (placeholder)

**Comparison of per-image versus video-level performance.** Bar chart showing sensitivity, specificity, and AUC for Treatment, A-ROP, and RW-ROP across per-image, All-soft, Top-10-soft, and Top-5-soft conditions.
