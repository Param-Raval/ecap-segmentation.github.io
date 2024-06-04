---
feature_text: |
  ## Automatic Fiber Type Segmentation in Exogenous Peripheral Neural Signal
feature_image: "/assests/img/bios_bg.png"
---

# Abstract
Vagus nerve stimulation (VNS) is a promising innovation in the treatment of chronic conditions
through neuromodulation. Development of a closed-loop neuromodulation interface
is a challenging problem to solve given the complexity and large amounts of data transferred
as exogenous neural signals between the brain and the organs. Machine learning has
the required capabilities to process and analyse this data and develop innovative solutions.
Nerve fibres that activated upon electrical stimulation of the vagus nerve fire up in the form of evoked compound action
potentials (eCAPs). Identifying various fibre types from these responses aide in linking them
to their physiological role in the body and thus can be modulated to develop personalised
therapies with minimal side-effects. This work proposes an approach to automatically
segment eCAPs into activations of various fibre types. We report that of the four fibre types, two are reliably segmented by the two proposed models while some challenges are identified that limit improvement in the others. Finally, we propose a web-based annotation tool to visualise and annotate the segmented eCAPs to create a richer dataset that, with more data, will improve the performance of the models.


# Neural Data and Challenges


### Peripheral nervous system and vagus nerve stimulation
The peripheral nervous system (PNS) serves as the conduit between the central nervous
system (CNS), comprising the brain and spinal cord, and the rest of the body. It encompasses
a vast network of nerves that extend throughout the body, connecting organs, muscles, and
tissues to the CNS. The PNS plays a crucial role in transmitting sensory information from
the environment to the brain, as well as coordinating motor responses that govern movement
and bodily functions.

One of the key components of the PNS is the vagus nerve, the longest
cranial nerve in the body, which extends from the brainstem to various organs in the chest
and abdomen. The vagus nerve is a major player in the parasympathetic nervous system (PSNS), regulating essential bodily
functions such as heart rate, digestion, and respiratory rate. 

Vagus nerve stimulation (VNS) is a therapeutic technique that involves the delivery of electrical impulses to the vagus nerve. This process typically involves the surgical implantation of a small device, similar to a pacemaker, which is connected to the vagus nerve. The
device generates controlled electrical pulses that travel along the vagus nerve, modulating
its activity and influencing neural pathways involved in various physiological processes. By
modulating the activity of the vagus nerve, VNS has been shown to influence neural pathways
involved in mood regulation, seizure control, inflammation modulation, and autonomic function.

### Dataset

__Neural interface__: The vagus nerve is multi-fascicle in nature which poses a challenge to activate specific fibres and record their activation. Since the recorded intensity of a fibre activation varies around the nerve, a unique electrode arrangement in the neural interface helps capture the responses from various positions on the nerve. The neural interface can be described as follows:
* The interface involves reading electrical impulses using electrodes following an electrical stimulation. 
* The electrodes, known as cuff electrodes, wrap around the nerve externally to avoid damage to the nerve.
  * These cuff electrodes have a lower signal-to-noise ratio than more penetrative electrodes available but they are the least invasive type of electrodes that are more suited in clinical settings. 
  * Each multi-contact cuff contain eight electrodes arranged in longitudinal bipolar pairs. 
* This system collect responses by applying stimulation pulses with varying current (10-
2500μA), frequency (1-1500 Hz), pulse width (130-1000μs), and train duration (1-10s). 

The graphic in Figure 1 stages the cuff electrode placement and shows the arrangement of the
eight electrodes in an unrolled cuff. 


The datasets were obtained from the three different trials performed by BIOS Health. The trials were acute trials each conducted on five porcine subjects with the neural interface attached surgically to record neural activity
and other vital signs and physiological parameters. A typical neural response following such experiments is shown in Figure 2 where some of the fibres of interest are labelled.

__Manual labelling__: The data used in the current experiments was labelled manually by experts who observe the trends of activation of various fibres using a stacked plot similar to Figure 2. The subplots are arranged in increasing order of stimulation current intensity. Prior knowledge about the order in which various fibres are fired and noting the gradual intensification of the activation of certain fibres help the expert in identifying them.

For instance, activations for fibres A-beta and A-gamma appear early at lower stimulation
currents than B-fibre. These are followed by a trailing laryngeal muscle artefact which
appears without conduction delay unlike other eCAPs. Across subjects, however, activations
of fibres differ in temporal location of activation as well as the intensity and shapes of the
activation spikes.


<img src="{{ '/assests/img/cuff.png' | prepend: site.baseurl | prepend: site.url}}" alt="image" />

__Fig. 1. Illustration of the three cuff electrode interface placed on the vagus nerve. Sourced from BIOS Health.__

<div style="display: flex; justify-content: center;">
<img src="{{ '/assests/img/stacked1.png' | prepend: site.baseurl | prepend: site.url}}" alt="image" />
</div>

__Fig. 2. Typical neural response after a stimulation (St, not shown) of pulse width 260μs and varying current.__


## Background on eCAP Segmentation

__eCAPs__: The effect of VNS on the physiology is influenced by vagal nerve fibre recruitment. To determine which vagal fibers produce the intended physiological effect, upon stimulation, different nerve activation patterns must be elicited such that presence and absence of different fiber types can be controlled to examine the effects of activation pattern on physiology. The activations of different fibres in a nerve evoke in the form of action potentials. 

The cumulative response of all action potentials from different fibre types in the said nerve is called a compound action potential (CAP). The neural response following VNS is thus characterized by evoked compound action potentials (eCAPs).

__Different fibre types__: Fibre types in peripheral nerves are classified into A, B and C types, with A having the lowest threshold for evoking eCAPs, followed by B and C. A-type fibres, being the largest in diameter (1–20 μm), are further divided into A-alpha, A-beta, A-gamma, and A-delta in decreasing order of diameters. 

Firing of different types of fibres is linked to different physiological responses including laryngeal muscle, breathing, and heart rate responses. The relationships between changes in heart rate and B-fibre activations post VNS have been closely studied. On the other hand, laryngeal muscles contractions are unintended side effects of A-fibre activations. 

Thus, using eCAPs recorded with a range of stimulation parameters, the presence of activations from different types and their intensities can be studied.

__Modeling eCAPs__: Usually in a VNS therapy, eCAPs are modelled manually to filter out specific fibre types to predict the physiological response. This is a tedious and error-prone process, and due to the absence of opportunities to optimise stimulation parameters, may not lead to the most effective therapeutic results. Moreover, with changing stimulation parameters such as current and pulse width, eCAP responses also change in their location of activation and shapes in the recordings. This becomes increasingly difficult to track manually. 

Automated eCAP segmentation involves partitioning the neural response obtained from neural recordings as a time-series into distinct segments, each corresponding to specific fiber types activated by the stimulus. In machine learning, this problem can be formulated as a time-series segmentation problem.

## Methods

### Baseline

The baselines were computed in a data agnostic fashion by simply
averaging the manually labelled start and end coordinates of the bounds for every fibre type
across training subjects. All the test samples were assigned the same bounds for each fibre
type and evaluated using F1-scores. Figure 6 shows the results of the baseline with the
X-axis showing different fibre types and the Y-axis their F1-score for a given test subject.
### Proposed Approaches in eCAP Segmentation

### BiLSTMs+Attention

<img src="{{ '/assests/img/segnet2.0.png' | prepend: site.baseurl | prepend: site.url}}" alt="image" />

__Fig. 3. Architecture of the BiLSTMs model with an attention layer to weigh the numerical features.__

- 2 Bidirectional LSTM layers
- 2 linear layers to encode the neurogram into a vector of size 32×1. Linear layers use ReLU activation and two dropout layers of rate 0.5.
- Another fully connected layer to encode the numerical features into the same hidden size, 32. 
- A multi headed self-attention layer (8 heads, 0.1 dropout) that uses the concatenated feature vector. 
- A fully connected layer to produce a prediction mask of the same size as the input.

### LSTM-ED

<img src="{{ '/assests/img/lstm_ae_model.png' | prepend: site.baseurl | prepend: site.url}}" alt="image" />

__Fig. 4. An encoder-decoder architecture with an added attention layer to weigh the encoded representation with encoded numerical features. The encoder and decoder can be a series of BiLSTM layers (LSTM-ED) or convolutional layers (Conv-ED).__

The LSTM-ED model uses an encoder-decoder style architecture which is commonly used in image segmentation problems in computer vision.

__Encoder__:  The encoder is similar to the BiLSTM+Attention model except being directly attached to the attention layer. It has 3 layers of BiLSTMs with a hidden size of 32. The attention layer contains 8 attention heads with a dropout rate of 0.1.

__Decoder__: The decoder attached immediately after the latter. This also has 3 layers of BiLSTMs with a hidden size of 32. The decoder is followed by a fully connected layer to produce the prediction mask.


### Conv-ED
Conv-ED follows the same architecture as shown in Figure 5 but with 1D convolutional layers in place of BiLSTM layers. 

__Encoder__: There are 4 convolutional layers with 3x3 kernels and 16, 16, 32, and 64 filters respectively, each followed by a ReLU activation. Dropout with a rate of 0.1 is performed after the 2nd, 3rd and 4th convolutional layer. 1D max pooling with size 2 and stride 2 is done after the 2nd and 3rd convolutional layers.

__Decoder__: The decoder has 4 transposed convolutional layers with the square kernel size and number of filters respectively being (2, 128), (3, 64), (2, 32), (3, 16). These layers are followed by a ReLU and dropout except for the final convolutional layer which connects to a classification head. 


Without the sequential and memory concepts of LSTM, the convolution layers learn local interactions between neighbouring data points in a time series. Convolutional layers automatically learn hierarchical features from the input and can capture patterns at different levels of abstraction. Moreover, the translation invariance of convolutions makes the model immune to time delays across samples, meaning they can detect patterns regardless of their position in the input sequence. 


## Results

<p style="text-align: center;">Fixed bounds baseline (test F1 score)</p>
<img src="{{ '/assests/img/baseline.png' | prepend: site.baseurl | prepend: site.url}}" alt="image" />

__Fig. 5. Baseline F1 scores on the test sets__

<p style="text-align: center;">BiLSTM+Attention (test F1 score)</p>

<img src="{{ '/assests/img/segnetresult.png' | prepend: site.baseurl | prepend: site.url}}" alt="image" />

__Fig. 6. F1 scores from BiLSTM+Attention on the test sets__

<p style="text-align: center;">Conv-ED (test F1 score)</p>

<img src="{{ '/assests/img/convaeresult.png' | prepend: site.baseurl | prepend: site.url}}" alt="image" />

__Fig. 7. F1 scores from the convolutional encoder-decoder on the test sets__

<p style="text-align: center;">LSTM-ED (test F1 score)</p>

<img src="{{ '/assests/img/lstmaeresult.png' | prepend: site.baseurl | prepend: site.url}}" alt="image" />

__Fig. 8. F1 scores from the LSTM encoder-decoder on the test sets.__


__Baseline__: Figure 5 shows that the consistency of A-beta and A-gamma in their frequency and location of appearance gives them fairly decent scores when predicted blindly. A-delta and B fibres leave significant room for improvement. It is also worth noting that the interquartile range (IQR) is decently sized despite significant inter-subject variability. The non-fibre label (“other”), being the majority class, is the easiest to predict even using a data agnostic method.

__BiLSTM+Attention__: Figure 6 shows that the model does not perform well enough. While producing gains in A-delta and B fibres for some subjects, overall many others have taken a hit. The IQR has also widen in 3 of the 5 labels with four subjects being out of the range in A-beta and B.

__Conv-ED__: Figure 7 shows while the improvements over BiLSTM+Attention are modest in A-beta and A-gamma, there is a notable jump in performance in the B-fibre compared to the baseline. Three of the six subjects perform over 0.4 while two have over 0.6 in their F1-scores. In A-beta and A-gamma, the averages are still close to the baseline but the IQR has reduced slightly.

__LSTM-ED__: Figure 8 reports that the average performance for A-beta, A-gamma, and B fibres has improved over the baseline, with a notable advance in the B-fibre. Four subjects give a score of 0.4 and above in the B-fibre while the IQR in other fibres has narrowed further. 


## Bootstrapping Analysis


With the data challenges identified, acquisition of more and better data is required to drive this problem towards a solution. A bootstrapping analysis was conducted to confirm this intuition.

<img src="{{ '/assests/img/bootstrapping.png' | prepend: site.baseurl | prepend: site.url}}" alt="image" />

__Fig. 9. Diagrammatic explanation of the bootstrapping process.__

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
__Fig. 10. Results from bootstrapping analysis of A-beta, A-gamma, and B-fibres.__


Existing studies have linked B-fibres to heart functionalities and targeting on improving
performance would be an important milestone for this project. With visual evidence,
performance metrics, and the bootstrapping analysis, a convincing argument can be made
to acquire data with better B-fibre activations in more number of trials.

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

# Retraining and Steps Towards Live Integration

