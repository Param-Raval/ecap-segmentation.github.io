---
feature_text: |
  ## Automatic Fiber Type Segmentation in Exogenous Peripheral Neural Signal
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


## Peripheral nervous system and vagus nerve stimulation
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

## Dataset

The datasets were obtained from the three different trials performed
by BIOS. The neural interface, developed to read and write signals from and to the nerve, involves
reading electrical impulses following an electrical stimulation. The electrodes, known as cuff
electrodes, used for this wrap around the nerve externally to avoid damage to the nerve.
These cuff electrodes have a lower signal-to-noise ratio than more penetrative electrodes
available but they are the least invasive type of electrodes that are more suited in clinical
settings. 

Each multi-contact cuff contained eight electrodes arranged in longitudinal bipolar
pairs. This system collect responses by applying stimulation pulses with varying current (10-
2500μA), frequency (1-1500 Hz), pulse width (130-1000μs), and train duration (1-10s). The
graphic in Figure 1 stages the cuff electrode placement and shows the arrangement of the
eight electrodes in an unrolled cuff. The vagus nerve is multi-fascicle in nature which poses a
challenge to activate specific fibres and record their activation. Since the recorded intensity
of a fibre activation varies around the nerve, this electrode arrangement helps capture the
responses from various positions on the nerve.

The trials were acute trials each conducted on
five porcine subjects with the neural interface attached surgically to record neural activity
and other vital signs and physiological parameters. A typical neural response following such experiments is shown in Figure 2 where some of the fibres of interest are labelled. The data used in the current experiments was labelled manually by experts who observe the trends of activation of various fibres using a stacked plot similar to Figure 2. The subplots are arranged in increasing order of stimulation current intensity. Prior knowledge about the order in which various fibres are fired and noting the gradual intensification of the activation of certain fibres help the expert in identifying them.
For instance, activations for fibres A-beta and A-gamma appear early at lower stimulation
currents than B-fibre. These are followed by a trailing laryngeal muscle artefact which
appears without conduction delay unlike other eCAPs. Across subjects, however, activations
of fibres differ in temporal location of activation as well as the intensity and shapes of the
activation spikes.

__Fig. 1. Illustration of the three cuff electrode interface placed on the vagus nerve. Sourced
from BIOS Health.__


# Background on eCAP Segmentation

The effect of VNS on the physiology is influenced by vagal nerve fibre recruitment. To determine which vagal fibers produce the intended physiological effect, upon stimulation, different nerve activation patterns must be elicited such that presence and absence of different fiber types can be controlled to examine the effects of activation pattern on physiology [22]. The activations of different fibres in a nerve evoke in the form of action potentials. The cumulative response of all action potentials from different fibre types in the said nerve is called a compound action potential (CAP). The neural response following VNS is thus characterized by evoked compound action potentials (eCAPs).

Fibre types in peripheral nerves are classified into A, B and C types [18], with A having the lowest threshold for evoking eCAPs, followed by B and C. A-type fibres, being the largest in diameter (1–20 μm), are further divided into A-alpha, A-beta, A-gamma, and Adelta in decreasing order of diameters. Firing of different types of fibres is linked to different physiological responses including laryngeal muscle, breathing, and heart rate responses [11]. The relationships between changes in heart rate and B-fibre activations post VNS have been closely studied. On the other hand, laryngeal muscles contractions are unintended side effects of A-fibre activations [17]. Thus, using eCAPs recorded with a range of stimulation parameters, the presence of activations from different types and their intensities can be studied.

Usually in a VNS therapy, eCAPs are modelled manually to filter out specific fibre types to predict the physiological response. This is a tedious and error-prone process, and due to the absence of opportunities to optimise stimulation parameters, may not lead to the most effective therapeutic results. Moreover, with changing stimulation parameters such as current and pulse width, eCAP responses also change in their location of activation and shapes in the recordings. This becomes increasingly difficult to track manually. Automated eCAP segmentation involves partitioning the neural response obtained from neural recordings as a time-series into distinct segments, each corresponding to specific fiber types activated by the stimulus. In machine learning, this problem can be formulated as a time-series segmentation problem.


# Proposed Approach in eCAP Segmentation

### BiLSTMs+Attention

As shown in the Figure 4 above, two Bidirectional LSTM layers followed by two linear layers encode the neurogram sample into a hidden vector of size 32×1. The linear layers use the rectified linear unit (ReLU) activation function and have two 12 dropout layers of rate 0.5. Another fully connected layer also encodes the numerical features into a hidden feature vector of the same hidden size, 32. A multi headed self-attention[25] layer computes the attention weights using the concatenated feature vector. There are 8 attention heads used with a dropout rate of 0.1. Finally, a fully connected layer produces a prediction mask of the same size as the input.

### LSTM-ED
This model uses an encoder-decoder style architecture which is commonly used in image segmentation problems in computer vision. Image segmentation involves identifying and segmenting various objects in an image and producing an image mask where every pixel is assigned an object label. The LSTM-ED model, as in Figure 5, uses this concept in the time-series domain.  The encoder portion of LSTM-ED is similar to the BiLSTM+Attention model apart from the encoder directly being attached to the attention layer and the decoder attached immediately after the latter. 3 layers of BiLSTMs are present in both encoder and decoder components with a hidden size of 32. The attention layer contains 4 attention heads with a dropout rate of 0.5.


### Conv-ED
Conv-ED follows the same architecture as shown in Figure 5 but with 1D convolutional layers in place of BiLSTM layers. Without the sequential and memory concepts of LSTM, the convolution layers learn local interactions between neighbouring data points in a time series. Convolutional layers automatically learn hierarchical features from the input and can capture patterns at different levels of abstraction. Moreover, the translation invariance of convolutions makes the model immune to time delays across samples, meaning they can detect patterns regardless of their position in the input sequence. 

There are 4 convolutional layers in the encoder with 3x3 kernels and 16, 16, 32, and 64 filters respectively, each followed by a ReLU activation. Dropout with a rate of 0.1 is performed after the 2nd, 3rd and 4th convolutional layer. 1D max pooling with size 2 and stride 2 is done after the 2nd and 3rd convolutional layers. The decoder has 4 transposed convolutional layers with the kernel size and number of filters respectively being (2, 128), (3, 64), (2, 32), (3, 16). All kernels are square in shape. These layers are followed by a ReLU and dropout except for the final convolutional layer which connects to a classification head. 


# Results

From the baseline results in Figure 6 it can be seen that the non-fibre label (“other”), being the majority class, is the easiest to predict even using a data agnostic method. The consistency of A-beta and A-gamma in their frequency and location of appearance gives them fairly decent scores when predicted blindly. A-delta and B fibres leave significant room for improvement. It is also worth noting that the interquartile range (IQR) is decently sized despite significant inter-subject variability. 
Compared to the baseline, the BiLSTM+Attention model does not perform sufficiently better (shown in Figure 7). While producing gains in A-delta and B fibres for some subjects, overall many others have taken a hit. Moreover, the IQR has widen in 3 of the 5 labels with four subjects being out of the range in A-beta and B. Overall, this model has not performed
well enough.

Figure 8 shows the results for the Conv-ED model. While the improvements over the BiLSTM+Attention are modest in A-beta and A-gamma, there is a notable jump in performance in the B-fibre compared to the baseline. Three of the six subjects perform over 0.4 while two have over 0.6 in their F1-scores. In A-beta and A-gamma, the averages are still close to the baseline but the IQR has reduced slightly.

Finally, Figure 9 reports the performance of the LSTM-ED model. The average performance for A-beta, A-gamma, and B fibres has improved over the baseline, with a notable advance in the B-fibre. Four subjects give a score of 0.4 and above in the B-fibre while the IQR in other fibres has narrowed further. 

# eCAP Annotation Tool

# Retraining and Steps Towards Live Integration