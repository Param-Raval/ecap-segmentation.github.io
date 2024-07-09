---
feature_text: |
  ## Automatic Fiber Type Segmentation in Exogenous Peripheral Neural Signal
feature_image: "https://param-raval.github.io/ecap-segmentation.github.io/assets/img/bios_bg.png"
---

# Abstract
Vagus nerve stimulation (VNS) is a promising innovation in the treatment of chronic conditions already treating a large population in epilepsy, depression, and obesity. In its current clinical applications, patients receive a static dose uptitrated individually during clinic visits. It has been suggested that patient-responsive closed-loop therapies would lead to better outcomes and this area has seen a lot of research recently. Development of a closed-loop neuromodulation system
is challenging with the large amounts of data being transferred between the brain and the organs. Identifying different fibre types directly from nerve recordings is an important stepping stone towards closed-loop therapies. In the present work, we use machine learning to identify what fibres have been recruited in the vagus nerve following electrical stimulation. We propose novel ANN architectures trained on data from 6 subjects. The results are encouraging but we conclude that more data is needed to lower the error rate. Subsequently, we propose an annotation tool to visualise and annotate the segmented responses to create a richer dataset.


- [Abstract](#abstract)
- [Neural Data and Challenges](#neural-data-and-challenges)
    - [Peripheral nervous system and vagus nerve stimulation](#peripheral-nervous-system-and-vagus-nerve-stimulation)
  - [Background on eCAP Segmentation](#background-on-ecap-segmentation)
    - [Dataset](#dataset)
  - [Methods](#methods)
    - [Baseline](#baseline)
    - [Proposed Approaches in eCAP Segmentation](#proposed-approaches-in-ecap-segmentation)
    - [BiLSTMs+Attention](#bilstmsattention)
    - [LSTM-ED](#lstm-ed)
    - [Conv-ED](#conv-ed)
  - [Experiment Setting](#experiment-setting)
  - [Results](#results)
  - [Challenges Identified](#challenges-identified)
  - [Bootstrapping Analysis](#bootstrapping-analysis)
    - [Results from this analysis](#results-from-this-analysis)
- [eCAP Annotation Tool](#ecap-annotation-tool)
- [Retraining and Steps Towards Live Integration](#retraining-and-steps-towards-live-integration)
- [Conclusion](#conclusion)
- [Appendix A](#appendix-a)

# Neural Data and Challenges


### Peripheral nervous system and vagus nerve stimulation
The peripheral nervous system (PNS) serves as the conduit between the central nervous
system (CNS), comprising the brain and spinal cord, and the rest of the body. It encompasses
a vast network of nerves that extend throughout the body, connecting organs, muscles, and
tissues to the CNS. The PNS plays a crucial role in transmitting sensory information from
the environment to the brain, as well as coordinating motor responses that govern movement
and bodily functions.

One of the key components of the PNS is the vagus nerve, the longest cranial nerve in the body, which extends from the brainstem to various organs in the chest and abdomen. The vagus nerve is a major player in the parasympathetic nervous system (PSNS), regulating essential bodily functions such as heart rate, digestion, and respiratory rate. 

<img src="{{ '/assets/img/vns.png' | prepend: site.url}}" alt="image" />

_Fig. 1. Illustration of a traditional setup for vagus nerve stimulation consisting of a stimulator that passes electric stimulation through the leads and cuffs to the attached vagus nerve._


Vagus nerve stimulation (VNS) is a therapeutic technique that involves the delivery of electrical impulses to the vagus nerve. This process typically involves the surgical implantation of a small device, similar to a pacemaker, which is connected to the vagus nerve. The device generates controlled electrical pulses that travel along the vagus nerve, modulating its activity and influencing neural pathways involved in various physiological processes. By modulating the activity of the vagus nerve, VNS has been shown to influence neural pathways involved in mood regulation, seizure control, inflammation modulation, and autonomic function.

VNS has been utilised in clinical settings as an adjunctive therapy for various treatment resistant
neuropsychiatric and neurological conditions. For instance, VNS has been used as an adjunctive treatment for individuals with epilepsy with studies demonstrating reductions in seizure frequency and severity in patients receiving VNS therapy. Among other neurological disorders, VNS has emerged as a promising intervention for individuals with Treatment-Resistant Depression (TRD), a form of depression that does not respond to standard antidepressant medications or psychotherapy.

## Background on eCAP Segmentation

__eCAPs__: At the core of VNS efficacy are the evoked compound action potentials (eCAPs), which provide crucial insights into the complex neurophysiological responses triggered by neural stimulation. The activations of different fibres in a nerve are evoked in the form of action potentials. The sum of synchronised action potentials from distinct axons is called a compound action potential (CAP). The neural response following VNS is thus characterised by eCAPs.

eCAPs represent the nerve responses to electrical stimulation including the activation of individual fibres that constitute the nerve. Understanding these eCAPs helps decipher the reaction of the nervous system to specific stimuli. This guides the parameter optimisation of the neuromodulation therapy to target specific physiological effects. To determine which vagal fibres produce the intended physiological effect, upon stimulation, different eCAPs must be elicited such that presence and absence of different fibre types can be controlled to examine the effects of activation pattern on physiology, see [Berthon (2023)](https://www.biorxiv.org/content/10.1101/2023.08.30.555487v1.full). 

__Different fibre types__: Fibre types in these nerves are classified based on the Erlanger-Gasser Classification into A, B and C types, with A having the lowest threshold for evoking eCAPs, followed by B and C. Further classification along with diameters and conduction velocities is shown in Table 1.

| Erlanger-Gasser Classification | Diameter   | Conduction velocity	 |
|--------------------------------|------------|---------------------|
| A-alpha                        | 13–20 μm   | 80–120 m/s          |
| A-beta                         | 6–12 μm    | 33–75 m/s           |
| A-gamma                        | 5–8 μm     | 4–24 m/s            |
| A-delta                        | 1–5 μm     | 3–30 m/s            |
| B                              | < 2 μm     | 3-14 m/s            |
| C                              | 0.2-1.5 μm | 0.5–2.0 m/s         |
__Table 1: Erlanger-Gasser Classification of nerve fibres._

Firing of different types of fibres is linked to different physiological responses including laryngeal muscle, breathing, and heart rate responses. It is therefore highly desirable to be able to identify the recruited fibres following VNS. For example, the relationships between (i) changes in heart rate and B-fibre activations, as well as (ii) laryngeal muscle contractions (common VNS side-effect) and A-fibre activations have been studied in the literature.

__Modeling eCAPs__: There are no known methods of automatically segmenting eCAPs following VNS and demonstrations in literature have relied on manual segmentation. This is a tedious process but more importantly the lack of automation precludes the use of segmented eCAPs in closed-loop therapy development (for example in optimising stimulation parameters for cardiac fibre recruitment). Moreover, with changing stimulation parameters such as current and pulse width, eCAP responses also change in their location of activation and shapes in the recordings. This becomes increasingly difficult to track manually. 

Automated eCAP segmentation involves partitioning the neural response obtained from neural recordings as a time-series into distinct segments, each corresponding to specific fibre types activated by the stimulus. The neurogram samples are 1-channel 1D signals. Thus, in machine learning, this problem can be formulated as a 1D time-series segmentation problem. 

Moreover, the stimulation experiment parameters have considerable effect on fibre activations. Across subjects it is seen that the B and A-delta fibres are activated more during higher current stimuli and activated earlier in stimuli with higher pulse widths. Additionally, the polarity of the stimulus and location of the cuff also affects the position of the activation. We design our models such that the features of current intensity (mA), stimulation pulse width (μs), and stimulation polarity (anodic/cathodic) can be fed in.

### Dataset
__Neural interface__: The vagus nerve is multi-fascicle in nature which poses a challenge to activate specific fibres and record their activation. The recorded intensity of a fibre activation varies around the nerve, therefore in this work we use proprietary, spatially-selective cuff electrodes that capture the responses from various positions on the nerve (see Figure 2). 
We collect neural responses to typical VNS waveforms: current (30-2500μA), frequency (1-20 Hz), pulse width (130-1000μs), and train duration (1-10s). 

The responses were obtained from 6 porcine subjects. Typical neural responses to 17 different stimulation intensities are shown in Figure 3. Fibres of interest are labelled.

Every neurogram sample is a time-series of between 250 and 290 data points representing nerve activation in μV. To train sequential models, all samples were padded to 290 with trailing zeroes. The manually labelled bounds marking the start and end of different fibre activations were converted to a continuous label mask of the same length as the sample. Every data point corresponds to an integer label denoting the fibre type it belongs to. Points not belonging to one of the 4 fibre types of interest were encoded to a generic label “other”. The dataset contains 129,768 samples.


__Manual labelling__: The data used in the current experiments was labelled manually by experts who observe the trends of activation of various fibres using a stacked plot similar to Figure 3. The subplots are arranged in increasing order of stimulation current intensity. Prior knowledge about the distance from the stimulation site to the recording site, the order in which various fibres are recruited and noting the gradual intensification of the activation of certain fibres help the expert in identifying them.

For instance, activations for fibres A-beta and A-gamma appear early at lower stimulation
currents than B-fibre. These are followed by a trailing laryngeal muscle artefact which
appears without conduction delay unlike other eCAPs. Across subjects, however, activations
of fibres differ in temporal location of activation as well as the intensity and shapes of the
activation spikes.


<img src="{{ '/assets/img/cuff.png' | prepend: site.url}}" alt="image" />

_Fig. 2. Illustration of the three cuff electrode interface placed on the vagus nerve. Sourced from BIOS Health._

<div style="display: flex; justify-content: center;">
<img src="{{ '/assets/img/stacked1.png' | prepend: site.url}}" alt="image" />
</div>

_Fig. 3. Typical neural response after a stimulation (St, not shown) of pulse width 260μs and varying current. Note how activations become more prominent with increasing current, and how fibres like B and A-delta are activated only at higher currents._

## Methods

### Baseline

The simplest baseline averages the manually labelled start and end coordinates of the bounds for every fibre type across training subjects. All the test samples were assigned the same bounds for each fibre type and evaluated using F1-scores. This establishes a reasonable data-based baseline that achieves an average F1 score of x. However, as shown in Figure 5, we note that there is huge variability among subjects as well as different fibre types.

### Proposed Approaches in eCAP Segmentation

There exists no prior ANN work on eCAP segmentation, so we develop and compare 3 different models. Noting the success of image segmentation models using encoder-decoder architectures, we formulate our learning problem similarly where the model trains to convert a given sequence into another sequence of predicted labels for every point. 

LSTM-based models have been successfully applied to sequence prediction and other time-series related tasks. As a starting point we take inspiration from existing work in ECG segmentation and begin with BiLSTM-based architectures. The BiLSTM architecture extends the capabilities of traditional LSTM networks by processing input sequences in both directions. Later, we move on to encoder-decoder based sequence-to-sequence architectures that are better suited for our problem. 

Including multi–headed attention layers in these models allows them to focus on different parts of the input sequence simultaneously, enhancing its ability to capture relevant information.  This is particularly helpful in combining the additional numerical features.


### BiLSTMs+Attention

<img src="{{ '/assets/img/segnet2.0.png' | prepend: site.url}}" alt="image" />

_Fig. 4. Architecture of the BiLSTMs model with an attention layer to weigh the numerical features. In order, there are 2 BiLSTM layers, 2 linear layers (32x1) with ReLU and 0.5 dropout, 1 linear layer for numerical features, self-attention with 8 heads, 0.1 dropout, and an output linear layer to predict the mask. “fc” represents fully-connected layers._

Together, BiLSTM layers enable the model to capture bidirectional context of the order
of occurrence of eCAPs, while attention mechanisms allow it to focus on relevant parts of
the input sequence and identify the fibre types.

### LSTM-ED

<img src="{{ '/assets/img/lstm_ae_model.png' | prepend: site.url}}" alt="image" />

_Fig. 5.  An encoder-decoder architecture with an added attention layer (4 attention heads with a dropout rate of 0.1)  to weigh the encoded representation with encoded numerical features. The encoder and decoder can be a series of BiLSTM layers (3 layers of BiLSTMs with a hidden size of 32 in LSTM-ED) or convolutional layers (Conv-ED). “fc” represents fully-connected layers._

The LSTM-ED model uses an encoder-decoder style architecture which is commonly used in image segmentation problems in computer vision.

__Encoder__:  The encoder is similar to the BiLSTM+Attention model except being directly attached to the attention layer. The BiLSTMs in the encoder process the input eCAP data and extract high-level representations that capture temporal dynamics and patterns such as locations of fibre activations. The encoded output is as a condensed representation of the input which is weighted with the attention weights learnt from the encoded hidden state and encoded vector of numerical features.

__Decoder__: The decoder attached immediately after the latter. This also has 3 layers of BiLSTMs with a hidden size of 32. The decoder is followed by a fully connected layer to produce the prediction mask. The decoder uses this learned representation of the eCAP data and transforms it to the corresponding output sequence of labels.

### Conv-ED

Conv-ED follows the same architecture as shown in Figure 5 but with 1D convolutional layers in place of BiLSTM layers. In the encoder, there are 4 1D convolutional layers with 3x3 kernels each followed by a ReLU activation. In some layers, we also add dropout and 1D max pooling. The decoder has 4 transposed 1D convolutional layers with square kernels each followed by a ReLU and dropout except for the final convolutional layer which connects to a classification head. 

Without the sequential and memory concepts of LSTM, the convolution layers learn local interactions between neighbouring data points in a time series. Convolutional layers automatically learn hierarchical features from the input and can capture patterns at different levels of abstraction. Moreover, the translation invariance of convolutions makes the model immune to time delays across samples, meaning they can detect patterns regardless of their position in the input sequence.

Table 2 notes the number of trainable parameters of each of these models.

| Model        | # of trainable parameters |
|--------------|---------------------------|
| BiLSTMs+Attn | 80,229                    |
| LSTM-ED      | 38,949                    |
| Conv-ED      | 63,333                    |
__Table 2. Models and the number of parameters. Conv-ED is heavier given the multiple layers of convolutional layers in both encoder and decoder._

## Experiment Setting

__Normalisation__: Since inter-subject variability is a well-recorded issue with eCAP data, a different strategy was used to normalise the training data. The normalisation parameters, mean and standard deviation, were computed for every subject instead for the entire training set. This makes sure for each subject that the individual data distributions and activation shapes are preserved and not biased by other subjects.


__Training split__: Given the redundancy in a large proportion of samples within a subject, a random training, testing, and validation split is likely to cause samples similar to those “seen” in training to appear in testing. This does not simulate real-life settings where the model is expected to be utilised on a completely new subject. The splits were designed such that data from one subject remains wholly independent as a test set, one as validation, and the rest of the subjects form the training set.

__Training setting__: The models were trained with a batch size of 128, for 40 epochs without early stopping using cross entropy loss, the Adam optimizer, and a learning rate of 1e-4.

<img src="{{ '/assets/img/val_f1.png' | prepend: site.url}}" alt="image" />

_Fig. 6: Validation performance plateaus for the major fibre types around epoch 40 across subjects. Training beyond this either brings no change or reduces performance._

## Results


<img src="{{ '/assets/img/good_example.png' | prepend: site.url}}" alt="image" />

_Fig. 7: A typical “good” example of eCAP segmentation from the test set with A-beta, A-gamma, and B fibre activations correctly captured (top: predicted segments, bottom: ground truth)._

<img src="{{ '/assets/img/bad_example.png' | prepend: site.url}}" alt="image" />

_Fig. 8:  A typical “bad” example of eCAP segmentation from the test set. B fibre activations are missed along with A-delta and the A-beta prediction overlaps A-gamma activations (top: predicted segments, bottom: ground truth). Mistakes of these types are commonly found in the incorrect predictions._

Table 3 and Table 4 give two levels of summarised results with Appendix A giving more details per subject. Table 3 shows the weighted averages of the F1 scores and the average macro  F1 scores of all fibre types (except “other”) considered over six test subjects. Given the class imbalance issues found in the dataset, these scores give a good overview to rank the models. Table 4 shows the average of F1 scores of 4 fibre types. We omit the “other” fibre type in the computation of these scores because it is the majority class in all the samples and the models can easily score well on it. Moreover, it is irrelevant to the downstream task of detecting various fibre types and given the good performance of the models on this class (check Appendix A), it can be safely ignored.

From Table 4, we can see that A-beta and A-gamma are easier to predict relative to other fibres. This can be explained from their consistent activations, in position and shape, in the majority of samples. On the contrary, the lack of consistency in B-fibre activations across subjects, brings the average down considerably. The poor performance in A-delta and B fibres can be explained by their muted activations and lack of proper representation in the data (discussed further later).

Overall, BiLSTM+Attn performs better than the rest and beats the baseline by a slim margin. Conv-ED fails to perform well whereas LSTM-ED does not give good enough improvement over the baseline.

| Model           | Macro F1 score | Weighted Average F1 |
|-----------------|----------------|---------------------|
| Baseline        | 0.37           | 0.40                |
| **BiLSTM+Attn** | **0.485**      | **0.498**           |
| LSTM-ED         | 0.405          | 0.485               |
| Conv-ED         | 0.331          | 0.391               |
_Table 3: Macro and Weighted Average Macro F1 scores per model_


| Model       | A-beta | A-gamma | B     | A-delta |
|-------------|--------|---------|-------|---------|
| Baseline    | 0.58   | 0.53    | 0.36  | 0.03    |
| BiLSTM+Attn | 0.635  | 0.590   | 0.442 | 0.123   |
| LSTM-ED     | 0.601  | 0.541   | 0.433 | 0.032   |
| Conv-ED     | 0.520  | 0.501   | 0.297 | 0.023   |
_Table 4: Average F1 scores per model per fibre type_

## Challenges Identified

Comparing the four methods reported here, the performance improvements after using ML are not significant. However, the ML model offers adaptability to unknown subjects and a more reliable, data-driven approach to addressing the task. Yet, there are a number of challenges that restrain performance and generalisation across subjects.

| class        |    point count      | point dist | sample count | sample dist |
|--------------------------------------------|----------| -------|---------|-------|
| other                                      | 29672849 | 0.788 | 129768 | 1.000 |
| B                                          | 3761043  | 0.100 | 103415 | 0.797 |
| A-gamma                                    | 2144551  | 0.057 | 118816 | 0.916 |
| A-beta                                     | 1780301  | 0.047 | 129301 | 0.996 |
| A-delta                                    | 273976   | 0.007 | 16805  | 0.130 |
| | | | | |

_Table 5. A frequency distribution and statistics table highlighting the representation of different fibre types in the data. Point count and distribution show the number of data points that are labelled as the corresponding fibre type and their fraction relative to the rest of the points respectively. Similarly, sample count and distribution show the same statistics but on a sample level with 129,768 being the total number of samples in the dataset._


- Class imbalance: Table 5 shows various statistics about the frequency distribution of the classes in the dataset. It is evident from the point distribution column the diminished representation of fibre types of interest when compared to other fibres which take up the most space in the dataset. This explains the high performance of the latter class. However, A-gamma and A-beta have sufficient representation in the number of samples that contain them. This is not the case for A-delta which is the minority class on both fronts. Because of this, the performance for A-delta is egregiously low in all the experiments.

This problem could not be resolved sufficiently well with weighted loss functions (like Focal loss or Dice loss), weighted resampling, data augmentation by rolling the time series, jittering, and linear modulation. Compared to our initial datasets (around 2,500 samples) where the samples contained pulses from only one recording channel (recording electrode) and the pulse trains were averaged, using entire pulse trains from all channels has boosted performance.


- Inter-subject variability: As seen in Figures in Appendix A, the inter quartile range (IQR) for every fibre type is widened by large variations in performance among different test subjects. Certain subjects like _IT6-A5_ and _IT7-A1_ consistently perform poorly across fibres and models. Inspecting visually, the shapes of the eCAPs in _IT7-A1_ are quite different from the rest of the subjects. In _IT6-A5_, the samples are visually similar except for a muted peak in A-beta and the absence of a valley-like shape towards the beginning which is present in other subjects (see Appendix A). 

Such ambiguous differences among subjects have been difficult to explain and hinder model generalisation. Moreover, this also makes effective data augmentation challenging since the number of samples are large, uniformly varied representation of the classes is insufficient. 


- Class representation: Table 5 shows how A-delta is the least prominent fibre within a sample and the number of samples that contain A-delta is just 13% of the entire set. In the presence of other prominent fibres, getting the model to get better at A-delta has been challenging.

For B-fibre, despite having decent representation in the dataset, models fail to perform beyond a certain point. Subjects in both LSTM-ED and Conv-ED struggle to cross the 0.7 score while some remain around 0.1. Given the relatively mutated activations in B-fibre peaks, there are not sufficiently varied samples in quality and quantity in the dataset that the model can strongly fit to. Subjects in which B-fibre performs well have consistent and prominent activations that resemble other subjects. To replicate this in other subjects, a larger number of varied samples is necessary.

## Bootstrapping Analysis


With the data challenges identified, acquisition of more and better data is required to drive this problem towards a solution. A bootstrapping analysis was conducted to confirm this intuition.

<img src="{{ '/assets/img/bootstrapping.png' | prepend: site.url}}" alt="image" />

_Fig. 9. Diagrammatic explanation of the bootstrapping process._

* The aim was to validate that with greater number of subjects, better generalisation can be achieved.
* The setup involved the following procedure: 
  * ∀N ∈ 1,2,3,4, N randomly sampled subjects formed the training set, one subject from the rest for validation, and one for testing. 
  * The selected model was trained and the metrics logged. 
  * This process was repeated T = 3 times for every N. 
* LSTM-ED was used with 8 attention heads with the same cross-validated split strategy and training hyperparameters as described earlier. 
* A larger model was chosen in order to minimise variability in results due to model selection and to retain focus on the trends instead of individual scores.

### Results from this analysis

Figure 10 shows the results for A-beta, A-gamma, and B fibres from the bootstrapping
analysis. They can be summarised as follows:

* Scores for A-beta and A-gamma show a clear upward trend as the number of training subjects increases. The IQR reduces greatly, signifying the increase in prediction confidence across test subjects. 
* This shows that adding more varied subjects helps models to capture general trends better. 
* The last chart shows the results for the B-fibre where the upward trend of improvement
is still visible. 
* More interesting is the performance stagnation in a few test subjects
where despite more data the model is not able to get equivalent gains. 
* This further validates the B-fibre challenges shared earlier where the higher variety B-fibre activations in a small sample space obstructs performance. 

<img src="{{ '/assests/img/btsp_abeta.png' | prepend: site.baseurl | prepend: site.url}}" alt="image" />
<img src="{{ '/assests/img/btsp_agamma.png' | prepend: site.baseurl | prepend: site.url}}" alt="image" />
<img src="{{ '/assests/img/btsp_b.png' | prepend: site.baseurl | prepend: site.url}}" alt="image" />
_Fig. 10. Results from bootstrapping analysis of A-beta, A-gamma, and B-fibres._


# eCAP Annotation Tool
With the conclusion that more data is required to improve the performance of the models, a web-based annotation tool was developed to facilitate the process of labelling eCAPs. The tool allows users to visualise the segmented eCAPs along with the bounds predicted from the segmentation model and edit these bounds for every sample. The annotated data can then be used to retrain the models and improve their performance.

Existing method of creating manual labels for eCAP neurograms involves looking at multiple ramp up plots for a subject and creating a fixed set of bounds for different fibre types. Issues with this include: 

* Only fixed constraints and conditions can be added manually for stim current, polarity, and pulse duration, and recording cuff pair among others. 
* This process leads to creation of labels which are not precise enough for training a machine learning model.
* Labelling includes several erroneous labels for fibre types which are either not present or not recognized if present.

However, with the eCAP annotation tool,

* Annotators can visualise, create, and edit bounds for individual samples.
* Use an UI which enables interaction with the bounds with functionality to create, delete, and edit bound boxes on top of the visualised neurogram. This improves precision and correctness of the labels.
* With an integrated predictive model, the predicted bounds can be displayed reducing the annotation task to only correction.
* A new training set with precise labels can be used to retrain the predictive model to improve performance.
* The tool can exist offline independently and integration with a pipeline to annotate samples during a live trial can use retraining to improve model predictions during the session.

Figure 11 shows a brief demonstration of the tool showing how a user can create, edit, and delete individual fibre bounds for a sample.

<img src="{{ '/assets/vid/tool_demo.gif' | prepend: site.url}}" alt="gif" />

_Fig. 11. Demo of the annotation tool with the user editing, adding, and deleting bounds for a sample. Meta data for the sample is useful for the team to get better context of the sample while annotating._

# Retraining and Steps Towards Live Integration

The proposed models and annotation tool can be seen integrated with existing frameworks and trial workflows to give 1) live predictions during a trial to allow real-time optimisation of stimulation parameters, and 2) update training data with new predictions verified via the tool and retrain models to improve performance. Retraining can be a straightforward process where predictions from new subjects are edited and verified from the tool and added to the training set, which triggers retraining of the models. The new version of the model can be evaluated with a series of unit tests and deployed to replace the existing version.

# Conclusion

The proposed approaches demonstrate the effectiveness of machine learning models towards modelling and segmenting eCAPs towards the goal of developing closed-loop neuromodulation therapies. With our results and analysis, we show how more trials and data collection efforts can improve the performance and enable deployed, practical use of this system. With the proposed annotation tool and retraining workflow, we provide a framework for this purpose.


# Appendix A


<p style="text-align: center;">Fixed bounds baseline (test F1 score)</p>
<img src="{{ '/assests/img/baseline.png' | prepend: site.baseurl | prepend: site.url}}" alt="image" />

_Fig. 12. Baseline F1 scores on the test sets__

<p style="text-align: center;">BiLSTM+Attention (test F1 score)</p>

<img src="{{ '/assests/img/segnetresult.png' | prepend: site.baseurl | prepend: site.url}}" alt="image" />

_Fig. 13. F1 scores from BiLSTM+Attention on the test sets__

<p style="text-align: center;">Conv-ED (test F1 score)</p>

<img src="{{ '/assests/img/convaeresult.png' | prepend: site.baseurl | prepend: site.url}}" alt="image" />

_Fig. 14. F1 scores from the convolutional encoder-decoder on the test sets__

<p style="text-align: center;">LSTM-ED (test F1 score)</p>

<img src="{{ '/assests/img/lstmaeresult.png' | prepend: site.baseurl | prepend: site.url}}" alt="image" />

_Fig.15. F1 scores from the LSTM encoder-decoder on the test sets._


__Baseline__: Figure 12 shows that the consistency of A-beta and A-gamma in their frequency and location of appearance gives them fairly decent scores when predicted blindly. A-delta and B fibres leave significant room for improvement. It is also worth noting that the interquartile range (IQR) is decently sized despite significant inter-subject variability. The non-fibre label (“other”), being the majority class, is the easiest to predict even using a data agnostic method.

__BiLSTM+Attention__: Figure 13 shows that the model does not perform well enough. While producing gains in A-delta and B fibres for some subjects, overall many others have taken a hit. The IQR has also widened in 3 of the 5 labels with four subjects being out of the range in A-beta and B.

__Conv-ED__: Figure 14 shows while the improvements over BiLSTM+Attention are modest in A-beta and A-gamma, there is a notable jump in performance in the B-fibre compared to the baseline. Three of the six subjects perform over 0.4 while two have over 0.6 in their F1-scores. In A-beta and A-gamma, the averages are still close to the baseline but the IQR has reduced slightly.

__LSTM-ED__: Figure 15 reports that the average performance for A-beta, A-gamma, and B fibres has improved over the baseline, with a notable advance in the B-fibre. Four subjects give a score of 0.4 and above in the B-fibre while the IQR in other fibres has narrowed further. 
