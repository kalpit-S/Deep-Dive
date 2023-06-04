import streamlit as st
import requests
import anthropic
import os
import json
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile







attention_is_all_you_need = r"""\title{
Attention Is All You Need
}

\author{
Ashish Vaswani* \\ Noam Shazeer* Niki Parmar* \\ Google Brain \\ avaswani@google.com \\ Google Research \\ noam@google.com \\ nikip@google.com

Llion Jones*
Google Research
llion@google.com \\ Aidan N. Gomez* † \\ University of Toronto \\ Lukasz Kaiser \\ aidan@cs.toronto.edu \\ Google Brain \\ lukaszkaiser@google.com \\ Illia Polosukhin* $\ddagger$ \\ illia.polosukhin@gmail.com

\begin{abstract}
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.
\end{abstract}

Jakob Uszkoreit

Google Research

usz@google.com

\section{Introduction}

Recurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and

${ }^{*}$ Equal contribution. Listing order is random. Jakob proposed replacing RNNs with self-attention and started the effort to evaluate this idea. Ashish, with Illia, designed and implemented the first Transformer models and has been crucially involved in every aspect of this work. Noam proposed scaled dot-product attention, multi-head attention and the parameter-free position representation and became the other person involved in nearly every detail. Niki designed, implemented, tuned and evaluated countless model variants in our original codebase and tensor2tensor. Llion also experimented with novel model variants, was responsible for our initial codebase, and efficient inference and visualizations. Lukasz and Aidan spent countless long days designing various parts of and implementing tensor2tensor, replacing our earlier codebase, greatly improving results and massively accelerating our research.

${ }^{\dagger}$ Work performed while at Google Brain.

${ }^{\ddagger}$ Work performed while at Google Research.

31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA. transduction problems such as language modeling and machine translation [35, 2, 5]. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [38, 24, 15].

Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states $h_{t}$, as a function of the previous hidden state $h_{t-1}$ and the input for position $t$. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. Recent work has achieved significant improvements in computational efficiency through factorization tricks [21] and conditional computation [32], while also improving model performance in case of the latter. The fundamental constraint of sequential computation, however, remains.

Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2] 19]. In all but a few cases [27], however, such attention mechanisms are used in conjunction with a recurrent network.

In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

\section{Background}

The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [16], ByteNet [18] and ConvS2S [9], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions [12]. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2

Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 27, 28, 22].

End-to-end memory networks are based on a recurrent attention mechanism instead of sequencealigned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [34].

To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequencealigned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [17, 18] and [9].

\section{Model Architecture}

Most competitive neural sequence transduction models have an encoder-decoder structure [5, 2, 35]. Here, the encoder maps an input sequence of symbol representations $\left(x_{1}, \ldots, x_{n}\right)$ to a sequence of continuous representations $\mathbf{z}=\left(z_{1}, \ldots, z_{n}\right)$. Given $\mathbf{z}$, the decoder then generates an output sequence $\left(y_{1}, \ldots, y_{m}\right)$ of symbols one element at a time. At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next.

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1 respectively. 

![](https://cdn.mathpix.com/cropped/2023_05_26_3628174813d05169d045g-03.jpg?height=1141&width=794&top_left_y=237&top_left_x=663)

Figure 1: The Transformer - model architecture.

\subsection{Encoder and Decoder Stacks}

Encoder: The encoder is composed of a stack of $N=6$ identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network. We employ a residual connection [11] around each of the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is LayerNorm $(x+\operatorname{Sublayer}(x))$, where Sublayer $(x)$ is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_{\text {model }}=512$.

Decoder: The decoder is also composed of a stack of $N=6$ identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$.

\subsection{Attention}

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key. 

![](https://cdn.mathpix.com/cropped/2023_05_26_3628174813d05169d045g-04.jpg?height=594&width=420&top_left_y=237&top_left_x=1205)

Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.

\subsubsection{Scaled Dot-Product Attention}

We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of dimension $d_{k}$, and values of dimension $d_{v}$. We compute the dot products of the query with all keys, divide each by $\sqrt{d_{k}}$, and apply a softmax function to obtain the weights on the values.

In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix $Q$. The keys and values are also packed together into matrices $K$ and $V$. We compute the matrix of outputs as:

$$
\operatorname{Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right) V
$$

The two most commonly used attention functions are additive attention [2], and dot-product (multiplicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of $\frac{1}{\sqrt{d_{k}}}$. Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

While for small values of $d_{k}$ the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of $d_{k}[3]$. We suspect that for large values of $d_{k}$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients ${ }^{4}$ To counteract this effect, we scale the dot products by $\frac{1}{\sqrt{d_{k}}}$.

\subsubsection{Multi-Head Attention}

Instead of performing a single attention function with $d_{\text {model-dimensional keys, values and queries, }}$, we found it beneficial to linearly project the queries, keys and values $h$ times with different, learned linear projections to $d_{k}, d_{k}$ and $d_{v}$ dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding $d_{v}$-dimensional output values. These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2

${ }^{4}$ To illustrate why the dot products get large, assume that the components of $q$ and $k$ are independent random variables with mean 0 and variance 1 . Then their dot product, $q \cdot k=\sum_{i=1}^{d_{k}} q_{i} k_{i}$, has mean 0 and variance $d_{k}$. Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

$$
\begin{aligned}
\operatorname{MultiHead}(Q, K, V) & =\operatorname{Concat}\left(\operatorname{head}_{1}, \ldots, \operatorname{head}_{\mathrm{h}}\right) W^{O} \\
\text { where } \operatorname{head}_{\mathrm{i}} & =\operatorname{Attention}\left(Q W_{i}^{Q}, K W_{i}^{K}, V W_{i}^{V}\right)
\end{aligned}
$$

Where the projections are parameter matrices $W_{i}^{Q} \in \mathbb{R}^{d_{\text {model }} \times d_{k}}, W_{i}^{K} \in \mathbb{R}^{d_{\text {model }} \times d_{k}}, W_{i}^{V} \in \mathbb{R}^{d_{\text {model }} \times d_{v}}$ and $W^{O} \in \mathbb{R}^{h d_{v} \times d_{\text {model }} \text {. }}$

In this work we employ $h=8$ parallel attention layers, or heads. For each of these we use $d_{k}=d_{v}=d_{\text {model }} / h=64$. Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.

\subsubsection{Applications of Attention in our Model}

The Transformer uses multi-head attention in three different ways:

- In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [38, 2, 9].

- The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder.

- Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to $-\infty$ ) all values in the input of the softmax which correspond to illegal connections. See Figure 2

\subsection{Position-wise Feed-Forward Networks}

In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.

$$
\operatorname{FFN}(x)=\max \left(0, x W_{1}+b_{1}\right) W_{2}+b_{2}
$$

While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1 . The dimensionality of input and output is $d_{\text {model }}=512$, and the inner-layer has dimensionality $d_{f f}=2048$.

\subsection{Embeddings and Softmax}

Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension $d_{\text {model }}$. We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities. In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [30]. In the embedding layers, we multiply those weights by $\sqrt{d_{\text {model }}}$.

\subsection{Positional Encoding}

Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. $n$ is the sequence length, $d$ is the representation dimension, $k$ is the kernel size of convolutions and $r$ the size of the neighborhood in restricted self-attention.

\begin{tabular}{lccc}
\hline Layer Type & Complexity per Layer & $\begin{array}{c}\text { Sequential } \\
\text { Operations }\end{array}$ & Maximum Path Length \\
\hline Self-Attention & $O\left(n^{2} \cdot d\right)$ & $O(1)$ & $O(1)$ \\
Recurrent & $O\left(n \cdot d^{2}\right)$ & $O(n)$ & $O(n)$ \\
Convolutional & $O\left(k \cdot n \cdot d^{2}\right)$ & $O(1)$ & $O\left(\log _{k}(n)\right)$ \\
Self-Attention (restricted) & $O(r \cdot n \cdot d)$ & $O(1)$ & $O(n / r)$ \\
\hline
\end{tabular}

tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension $d_{\text {model }}$ as the embeddings, so that the two can be summed. There are many choices of positional encodings, learned and fixed [9].

In this work, we use sine and cosine functions of different frequencies:

$$
\begin{aligned}
P E_{(p o s, 2 i)} & =\sin \left(p o s / 10000^{2 i / d_{\text {model }}}\right) \\
P E_{(p o s, 2 i+1)} & =\cos \left(p o s / 10000^{2 i / d_{\text {model }}}\right)
\end{aligned}
$$

where pos is the position and $i$ is the dimension. That is, each dimension of the positional encoding corresponds to a sinusoid. The wavelengths form a geometric progression from $2 \pi$ to $10000 \cdot 2 \pi$. We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset $k, P E_{\text {pos }+k}$ can be represented as a linear function of $P E_{\text {pos. }}$.

We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.

\section{Why Self-Attention}

In this section we compare various aspects of self-attention layers to the recurrent and convolutional layers commonly used for mapping one variable-length sequence of symbol representations $\left(x_{1}, \ldots, x_{n}\right)$ to another sequence of equal length $\left(z_{1}, \ldots, z_{n}\right)$, with $x_{i}, z_{i} \in \mathbb{R}^{d}$, such as a hidden layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we consider three desiderata.

One is the total computational complexity per layer. Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required.

The third is the path length between long-range dependencies in the network. Learning long-range dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies [12]. Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer types.

As noted in Table1] a self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer requires $O(n)$ sequential operations. In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence length $n$ is smaller than the representation dimensionality $d$, which is most often the case with sentence representations used by state-of-the-art models in machine translations, such as word-piece [38] and byte-pair [31] representations. To improve computational performance for tasks involving very long sequences, self-attention could be restricted to considering only a neighborhood of size $r$ in the input sequence centered around the respective output position. This would increase the maximum path length to $O(n / r)$. We plan to investigate this approach further in future work.

A single convolutional layer with kernel width $k<n$ does not connect all pairs of input and output positions. Doing so requires a stack of $O(n / k)$ convolutional layers in the case of contiguous kernels, or $O\left(\log _{k}(n)\right)$ in the case of dilated convolutions [18], increasing the length of the longest paths between any two positions in the network. Convolutional layers are generally more expensive than recurrent layers, by a factor of $k$. Separable convolutions [6], however, decrease the complexity considerably, to $O\left(k \cdot n \cdot d+n \cdot d^{2}\right)$. Even with $k=n$, however, the complexity of a separable convolution is equal to the combination of a self-attention layer and a point-wise feed-forward layer, the approach we take in our model.

As side benefit, self-attention could yield more interpretable models. We inspect attention distributions from our models and present and discuss examples in the appendix. Not only do individual attention heads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic and semantic structure of the sentences.

\section{Training}

This section describes the training regime for our models.

\subsection{Training Data and Batching}

We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. Sentences were encoded using byte-pair encoding [3], which has a shared sourcetarget vocabulary of about 37000 tokens. For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary [38]. Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.

\subsection{Hardware and Schedule}

We trained our models on one machine with 8 NVIDIA P100 GPUs. For our base models using the hyperparameters described throughout the paper, each training step took about 0.4 seconds. We trained the base models for a total of 100,000 steps or 12 hours. For our big models,(described on the bottom line of table 3p, step time was 1.0 seconds. The big models were trained for 300,000 steps (3.5 days).

\subsection{Optimizer}

We used the Adam optimizer [20] with $\beta_{1}=0.9, \beta_{2}=0.98$ and $\epsilon=10^{-9}$. We varied the learning rate over the course of training, according to the formula:

$$
\text { lrate }=d_{\text {model }}^{-0.5} \cdot \min \left(s t e p \_n u m^{-0.5}, \text { step_num } \cdot \text { warmup_steps }{ }^{-1.5}\right)
$$

This corresponds to increasing the learning rate linearly for the first warmup_steps training steps, and decreasing it thereafter proportionally to the inverse square root of the step number. We used warmup_steps $=4000$.

\subsection{Regularization}

We employ three types of regularization during training:

Residual Dropout We apply dropout [33] to the output of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of $P_{d r o p}=0.1$ Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost.

\begin{tabular}{lccccc}
\hline \multirow{2}{*}{ Model } & \multicolumn{2}{c}{ BLEU } & & \multicolumn{2}{c}{ Training Cost (FLOPs) } \\
\cline { 2 - 3 } \cline { 6 - 7 } & EN-DE & EN-FR & & EN-DE & EN-FR \\
\hline ByteNet [18] & 23.75 & & & & \\
Deep-Att + PosUnk [39] & & 39.2 & & $1.0 \cdot 10^{20}$ \\
GNMT + RL [38] & 24.6 & 39.92 & & $2.3 \cdot 10^{19}$ & $1.4 \cdot 10^{20}$ \\
ConvS2S [9] & 25.16 & 40.46 & & $9.6 \cdot 10^{18}$ & $1.5 \cdot 10^{20}$ \\
MoE [32] & 26.03 & 40.56 & & $2.0 \cdot 10^{19}$ & $1.2 \cdot 10^{20}$ \\
\hline Deep-Att + PosUnk Ensemble [39] & & 40.4 & & & $8.0 \cdot 10^{20}$ \\
GNMT + RL Ensemble [38] & 26.30 & 41.16 & & $1.8 \cdot 10^{20}$ & $1.1 \cdot 10^{21}$ \\
ConvS2S Ensemble [9] & 26.36 & $\mathbf{4 1 . 2 9}$ & & $7.7 \cdot 10^{19}$ & $1.2 \cdot 10^{21}$ \\
\hline Transformer (base model) & 27.3 & 38.1 & & $\mathbf{3 . 3} \cdot \mathbf{1 0}^{\mathbf{1 8}}$ \\
Transformer (big) & $\mathbf{2 8 . 4}$ & $\mathbf{4 1 . 8}$ & & $2.3 \cdot 10^{19}$ \\
\hline & & & & & \\
\hline
\end{tabular}

Label Smoothing During training, we employed label smoothing of value $\epsilon_{l s}=0.1$ [36]. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.

\section{Results}

\subsection{Machine Translation}

On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in Table 2 outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4. The configuration of this model is listed in the bottom line of Table 3. Training took 3.5 days on 8 P100 GPUs. Even our base model surpasses all previously published models and ensembles, at a fraction of the training cost of any of the competitive models.

On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all of the previously published single models, at less than $1 / 4$ the training cost of the previous state-of-the-art model. The Transformer (big) model trained for English-to-French used dropout rate $P_{d r o p}=0.1$, instead of 0.3 .

For the base models, we used a single model obtained by averaging the last 5 checkpoints, which were written at 10 -minute intervals. For the big models, we averaged the last 20 checkpoints. We used beam search with a beam size of 4 and length penalty $\alpha=0.6$ [38]. These hyperparameters were chosen after experimentation on the development set. We set the maximum output length during inference to input length +50 , but terminate early when possible [38].

Table 2 summarizes our results and compares our translation quality and training costs to other model architectures from the literature. We estimate the number of floating point operations used to train a model by multiplying the training time, the number of GPUs used, and an estimate of the sustained single-precision floating-point capacity of each GPU ${ }^{5}$

\subsection{Model Variations}

To evaluate the importance of different components of the Transformer, we varied our base model in different ways, measuring the change in performance on English-to-German translation on the development set, newstest2013. We used beam search as described in the previous section, but no checkpoint averaging. We present these results in Table 3

In Table 3 rows (A), we vary the number of attention heads and the attention key and value dimensions, keeping the amount of computation constant, as described in Section 3.2.2. While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads.

${ }^{5}$ We used values of $2.8,3.7,6.0$ and 9.5 TFLOPS for K80, K40, M40 and P100, respectively. Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013. Listed perplexities are per-wordpiece, according to our byte-pair encoding, and should not be compared to per-word perplexities.

\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|}
\hline & $N$ & $d_{\text {model }}$ & $d_{\mathrm{ff}}$ & $h$ & $d_{k}$ & $d_{v}$ & $P_{\text {drop }}$ & $\epsilon_{l s}$ & $\begin{array}{l}\text { train } \\
\text { steps }\end{array}$ & $\begin{array}{l}\text { PPL } \\
\text { (dev) }\end{array}$ & $\begin{array}{c}\text { BLEU } \\
\text { (dev) }\end{array}$ & $\begin{array}{c}\text { params } \\
\times 10^{6}\end{array}$ \\
\hline base & 6 & 512 & 2048 & 8 & 64 & 64 & 0.1 & 0.1 & $100 \mathrm{~K}$ & 4.92 & 25.8 & 65 \\
\hline \multirow{4}{*}{ (A) } & $\overline{1}$ & . & . & 1 & 512 & 512 & & $\bar{F}$ & - & 5.29 & 24.9 & \\
\hline & & & & 4 & 128 & 128 & & & & 5.00 & 25.5 & \\
\hline & & & & 16 & 32 & 32 & & & & 4.91 & 25.8 & \\
\hline & & & & 32 & 16 & 16 & & & & 5.01 & 25.4 & \\
\hline \multirow{2}{*}{ (B) } & & & & $\overline{-}$ & 16 & . & & & & 5.16 & 25.1 & 58 \\
\hline & & & & & 32 & & & & & 5.01 & 25.4 & 60 \\
\hline \multirow{7}{*}{ (C) } & 2 & & & & . & & & & & 6.11 & 23.7 & 36 \\
\hline & 4 & & & & & & & & & 5.19 & 25.3 & 50 \\
\hline & 8 & & & & & & & & & 4.88 & 25.5 & 80 \\
\hline & & 256 & & & 32 & 32 & & & & 5.75 & 24.5 & 28 \\
\hline & & 1024 & & & 128 & 128 & & & & 4.66 & 26.0 & 168 \\
\hline & & & 1024 & & & & & & & 5.12 & 25.4 & 53 \\
\hline & & & 4096 & & & & & & & 4.75 & 26.2 & 90 \\
\hline \multirow{4}{*}{ (D) } & & & . & & & & 0.0 & & & 5.77 & 24.6 & . \\
\hline & & & & & & & 0.2 & & & 4.95 & 25.5 & \\
\hline & & & & & & & & 0.0 & & 4.67 & 25.3 & \\
\hline & & & & & & & & 0.2 & & 5.47 & 25.7 & \\
\hline (E) & \multicolumn{9}{|c|}{ positional embedding instead of sinusoids } & 4.92 & 25.7 & \\
\hline big & 6 & 1024 & 4096 & 16 & & - & 0.3 & - & $300 \mathrm{~K}$ & 4.33 & 26.4 & 213 \\
\hline
\end{tabular}

Table 4: The Transformer generalizes well to English constituency parsing (Results are on Section 23 of WSJ)

\begin{tabular}{c|c|c}
\hline Parser & Training & WSJ 23 F1 \\
\hline Vinyals \& Kaiser el al. (2014) [37] & WSJ only, discriminative & 88.3 \\
Petrov et al. (2006) [29] & WSJ only, discriminative & 90.4 \\
Zhu et al. (2013) [40] & WSJ only, discriminative & 90.4 \\
Dyer et al. (2016) [8] & WSJ only, discriminative & 91.7 \\
\hline Transformer (4 layers) & WSJ only, discriminative & 91.3 \\
\hline Zhu et al. (2013) [40] & semi-supervised & 91.3 \\
Huang \& Harper (2009) [14] & semi-supervised & 91.3 \\
McClosky et al. (2006) [26] & semi-supervised & 92.1 \\
Vinyals \& Kaiser el al. (2014) [37] & semi-supervised & 92.1 \\
\hline Transformer (4 layers) & semi-supervised & 92.7 \\
\hline Luong et al. (2015) [23] & multi-task & 93.0 \\
Dyer et al. (2016) [8] & generative & 93.3 \\
\hline
\end{tabular}

In Table 3 rows (B), we observe that reducing the attention key size $d_{k}$ hurts model quality. This suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D) that, as expected, bigger models are better, and dropout is very helpful in avoiding over-fitting. In row (E) we replace our sinusoidal positional encoding with learned positional embeddings [9], and observe nearly identical results to the base model.

\subsection{English Constituency Parsing}

To evaluate if the Transformer can generalize to other tasks we performed experiments on English constituency parsing. This task presents specific challenges: the output is subject to strong structural constraints and is significantly longer than the input. Furthermore, RNN sequence-to-sequence models have not been able to attain state-of-the-art results in small-data regimes [37].

We trained a 4-layer transformer with $d_{\text {model }}=1024$ on the Wall Street Journal (WSJ) portion of the Penn Treebank [25], about 40K training sentences. We also trained it in a semi-supervised setting, using the larger high-confidence and BerkleyParser corpora from with approximately $17 \mathrm{M}$ sentences [37]. We used a vocabulary of $16 \mathrm{~K}$ tokens for the WSJ only setting and a vocabulary of $32 \mathrm{~K}$ tokens for the semi-supervised setting.

We performed only a small number of experiments to select the dropout, both attention and residual (section 5.4p, learning rates and beam size on the Section 22 development set, all other parameters remained unchanged from the English-to-German base translation model. During inference, we increased the maximum output length to input length +300 . We used a beam size of 21 and $\alpha=0.3$ for both WSJ only and the semi-supervised setting.

Our results in Table 4 show that despite the lack of task-specific tuning our model performs surprisingly well, yielding better results than all previously reported models with the exception of the Recurrent Neural Network Grammar [8].

In contrast to RNN sequence-to-sequence models [37], the Transformer outperforms the BerkeleyParser [29] even when training only on the WSJ training set of $40 \mathrm{~K}$ sentences.

\section{Conclusion}

In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.

For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles.

We are excited about the future of attention-based models and plan to apply them to other tasks. We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video. Making generation less sequential is another research goals of ours.

The code we used to train and evaluate our models is available at https://github.com/ tensorflow/tensor2tensor

Acknowledgements We are grateful to Nal Kalchbrenner and Stephan Gouws for their fruitful comments, corrections and inspiration.

\section{References}

[1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. CoRR, abs/1409.0473, 2014.

[3] Denny Britz, Anna Goldie, Minh-Thang Luong, and Quoc V. Le. Massive exploration of neural machine translation architectures. CoRR, abs/1703.03906, 2017.

[4] Jianpeng Cheng, Li Dong, and Mirella Lapata. Long short-term memory-networks for machine reading. arXiv preprint arXiv:1601.06733, 2016.

[5] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. CoRR, abs/1406.1078, 2014.

[6] Francois Chollet. Xception: Deep learning with depthwise separable convolutions. arXiv preprint arXiv:1610.02357, 2016. [7] Junyoung Chung, Çaglar Gülçehre, Kyunghyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. CoRR, abs/1412.3555, 2014.

[8] Chris Dyer, Adhiguna Kuncoro, Miguel Ballesteros, and Noah A. Smith. Recurrent neural network grammars. In Proc. of NAACL, 2016.

[9] Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolutional sequence to sequence learning. arXiv preprint arXiv:1705.03122v2, 2017.

[10] Alex Graves. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850, 2013.

[11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770-778, 2016.

[12] Sepp Hochreiter, Yoshua Bengio, Paolo Frasconi, and Jürgen Schmidhuber. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies, 2001.

[13] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735-1780, 1997.

[14] Zhongqiang Huang and Mary Harper. Self-training PCFG grammars with latent annotations across languages. In Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing, pages 832-841. ACL, August 2009.

[15] Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. Exploring the limits of language modeling. arXiv preprint arXiv:1602.02410, 2016.

[16] Łukasz Kaiser and Samy Bengio. Can active memory replace attention? In Advances in Neural Information Processing Systems, (NIPS), 2016.

[17] Łukasz Kaiser and Ilya Sutskever. Neural GPUs learn algorithms. In International Conference on Learning Representations (ICLR), 2016.

[18] Nal Kalchbrenner, Lasse Espeholt, Karen Simonyan, Aaron van den Oord, Alex Graves, and Koray Kavukcuoglu. Neural machine translation in linear time. arXiv preprint arXiv:1610.10099v2, 2017.

[19] Yoon Kim, Carl Denton, Luong Hoang, and Alexander M. Rush. Structured attention networks. In International Conference on Learning Representations, 2017.

[20] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.

[21] Oleksii Kuchaiev and Boris Ginsburg. Factorization tricks for LSTM networks. arXiv preprint arXiv:1703.10722, 2017.

[22] Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua Bengio. A structured self-attentive sentence embedding. arXiv preprint arXiv:1703.03130, 2017.

[23] Minh-Thang Luong, Quoc V. Le, Ilya Sutskever, Oriol Vinyals, and Lukasz Kaiser. Multi-task sequence to sequence learning. arXiv preprint arXiv:1511.06114, 2015.

[24] Minh-Thang Luong, Hieu Pham, and Christopher D Manning. Effective approaches to attentionbased neural machine translation. arXiv preprint arXiv:1508.04025. 2015.

[25] Mitchell P Marcus, Mary Ann Marcinkiewicz, and Beatrice Santorini. Building a large annotated corpus of english: The penn treebank. Computational linguistics, 19(2):313-330, 1993.

[26] David McClosky, Eugene Charniak, and Mark Johnson. Effective self-training for parsing. In Proceedings of the Human Language Technology Conference of the NAACL, Main Conference, pages 152-159. ACL, June 2006. [27] Ankur Parikh, Oscar Täckström, Dipanjan Das, and Jakob Uszkoreit. A decomposable attention model. In Empirical Methods in Natural Language Processing, 2016.

[28] Romain Paulus, Caiming Xiong, and Richard Socher. A deep reinforced model for abstractive summarization. arXiv preprint arXiv:1705.04304, 2017.

[29] Slav Petrov, Leon Barrett, Romain Thibaux, and Dan Klein. Learning accurate, compact, and interpretable tree annotation. In Proceedings of the 21st International Conference on Computational Linguistics and 44th Annual Meeting of the ACL, pages 433-440. ACL, July 2006.

[30] Ofir Press and Lior Wolf. Using the output embedding to improve language models. arXiv preprint arXiv:1608.05859, 2016.

[31] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. arXiv preprint arXiv:1508.07909, 2015.

[32] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.

[33] Nitish Srivastava, Geoffrey E Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(1):1929-1958, 2014.

[34] Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus. End-to-end memory networks. In C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neural Information Processing Systems 28, pages 2440-2448. Curran Associates, Inc., 2015.

[35] Ilya Sutskever, Oriol Vinyals, and Quoc VV Le. Sequence to sequence learning with neural networks. In Advances in Neural Information Processing Systems, pages 3104-3112, 2014.

[36] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. CoRR, abs/1512.00567, 2015.

[37] Vinyals \& Kaiser, Koo, Petrov, Sutskever, and Hinton. Grammar as a foreign language. In Advances in Neural Information Processing Systems, 2015.

[38] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google's neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144, 2016.

[39] Jie Zhou, Ying Cao, Xuguang Wang, Peng Li, and Wei Xu. Deep recurrent models with fast-forward connections for neural machine translation. CoRR, abs/1606.04199, 2016.

[40] Muhua Zhu, Yue Zhang, Wenliang Chen, Min Zhang, and Jingbo Zhu. Fast and accurate shift-reduce constituent parsing. In Proceedings of the 51st Annual Meeting of the ACL (Volume 1: Long Papers), pages 434-443. ACL, August 2013. 

\section{Attention Visualizations}

![](https://cdn.mathpix.com/cropped/2023_05_26_3628174813d05169d045g-13.jpg?height=721&width=1353&top_left_y=331&top_left_x=405)

Figure 3: An example of the attention mechanism following long-distance dependencies in the encoder self-attention in layer 5 of 6. Many of the attention heads attend to a distant dependency of the verb 'making', completing the phrase 'making...more difficult'. Attentions here shown only for the word 'making'. Different colors represent different heads. Best viewed in color. 
![](https://cdn.mathpix.com/cropped/2023_05_26_3628174813d05169d045g-14.jpg?height=1524&width=1346&top_left_y=580&top_left_x=408)

Figure 4: Two attention heads, also in layer 5 of 6, apparently involved in anaphora resolution. Top: Full attentions for head 5. Bottom: Isolated attentions from just the word 'its' for attention heads 5 and 6. Note that the attentions are very sharp for this word. 
![](https://cdn.mathpix.com/cropped/2023_05_26_3628174813d05169d045g-15.jpg?height=1436&width=1336&top_left_y=625&top_left_x=405)

Figure 5: Many of the attention heads exhibit behaviour that seems related to the structure of the sentence. We give two such examples above, from two different heads from the encoder self-attention at layer 5 of 6 . The heads clearly learned to perform different tasks."""

crispr_cas9_systems = r"""\title{
Genome engineering using the CRISPR-Cas9 system
}

\author{
F Ann Ran ${ }^{1-5,8}$, Patrick D Hsu ${ }^{1-5,8}$, Jason Wright ${ }^{1}$, Vineeta Agarwala ${ }^{1,6,7}$, David A Scott ${ }^{1-4}$ \& Feng Zhang Zh $^{1-4}$
}

${ }^{1}$ Broad Institute of Massachusetts Institute of Technology (MIT) and Harvard, Cambridge, Massachusetts, USA. ${ }^{2}$ McGovern Institute for Brain Research, Cambridge, Massachusetts, USA. ${ }^{3}$ Department of Brain and Cognitive Sciences, MIT, Cambridge, Massachusetts, USA. ${ }^{4}$ Department of Biological Engineering, MIT, Cambridge, Massachusetts, USA. ${ }^{5}$ Department of Molecular and Cellular Biology, Harvard University, Cambridge, Massachusetts, USA. ${ }^{6}$ Program in Biophysics, Harvard University, MIT, Cambridge, Massachusetts, USA. ${ }^{7}$ Harvard-MIT Division of Health Sciences and Technology, MIT, Cambridge, Massachusetts, USA. ${ }^{8}$ These authors contributed equally to this work. Correspondence should be addressed to F.Z. (zhang@broadinstitute.org).

Published online 24 0ctober 2013; doi:10.1038/nprot.2013.143

Targeted nucleases are powerful tools for mediating genome alteration with high precision. The RNA-guided Cas9 nuclease from the microbial clustered regularly interspaced short palindromic repeats (CRISPR) adaptive immune system can be used to facilitate efficient genome engineering in eukaryotic cells by simply specifying a 20-nt targeting sequence within its guide RNA. Here we describe a set of tools for Cas9-mediated genome editing via nonhomologous end joining (NHEJ) or homology-directed repair (HDR) in mammalian cells, as well as generation of modified cell lines for downstream functional studies. To minimize off-target cleavage, we further describe a double-nicking strategy using the Cas 9 nickase mutant with paired guide RNAs. This protocol provides experimentally derived guidelines for the selection of target sites, evaluation of cleavage efficiency and analysis of off-target activity. Beginning with target design, gene modifications can be achieved within as little as 1-2 weeks, and modified clonal cell lines can be derived within 2-3 weeks.

\section*{INTRODUCTION}

The ability to engineer biological systems and organisms holds enormous potential for applications across basic science, medicine and biotechnology. Programmable sequence-specific endonucleases that facilitate precise editing of endogenous genomic loci are now enabling systematic interrogation of genetic elements and causal genetic variations ${ }^{1,2}$ in a broad range of species, including those that have not previously been genetically tractable ${ }^{3-6}$. A number of genome editing technologies have emerged in recent years, including zinc-finger nucleases $(\mathrm{ZFNs})^{7-10}$, transcription activator-like effector nucleases (TALENs) ${ }^{10-17}$ and the RNA-guided CRISPR-Cas nuclease system ${ }^{18-25}$. The first two technologies use a strategy of tethering endonuclease catalytic domains to modular DNA-binding proteins for inducing targeted DNA double-stranded breaks (DSBs) at specific genomic loci. By contrast, Cas9 is a nuclease guided by small RNAs through Watson-Crick base pairing with target DNA ${ }^{26-28}$ (Fig. 1), representing a system that is markedly easier to design, highly specific, efficient and well-suited for highthroughput and multiplexed gene editing for a variety of cell types and organisms.

\section*{Precise genome editing using engineered nucleases}

Similarly to ZFNs and TALENs, Cas9 promotes genome editing by stimulating a DSB at a target genomic locus ${ }^{29,30}$. Upon cleavage by Cas9, the target locus typically undergoes one of two major pathways for DNA damage repair (Fig. 2): the error-prone NHEJ or the high-fidelity HDR pathway, both of which can be used to achieve a desired editing outcome. In the absence of a repair template, DSBs are re-ligated through the NHEJ process, which leaves scars in the form of insertion/deletion (indel) mutations. NHEJ can be harnessed to mediate gene knockouts, as indels occurring within a coding exon can lead to frameshift mutations and premature stop codons ${ }^{31}$. Multiple DSBs can additionally be exploited to mediate larger deletions in the genome ${ }^{22,32}$.

HDR is an alternative major DNA repair pathway. Although HDR typically occurs at lower and substantially more variable frequencies than NHEJ, it can be leveraged to generate precise, defined modifications at a target locus in the presence of an exogenously introduced repair template. The repair template can either be in the form of conventional double-stranded DNA targeting constructs with homology arms flanking the insertion sequence, or single-stranded DNA oligonucleotides (ssODNs). The latter provides an effective and simple method for making small edits in the genome, such as the introduction of singlenucleotide mutations for probing causal genetic variations ${ }^{32}$. Unlike NHEJ, HDR is generally active only in dividing cells, and its efficiency can vary widely depending on the cell type and state, as well as the genomic locus and repair template ${ }^{33}$.

Cas9: an RNA-guided nuclease for genome editing CRISPR-Cas is a microbial adaptive immune system that uses RNA-guided nucleases to cleave foreign genetic elements ${ }^{18-21,26}$. Three types (I-III) of CRISPR systems have been identified across a wide range of bacterial and archaeal hosts, wherein each system comprises a cluster of CRISPR-associated (Cas) genes, noncoding RNAs and a distinctive array of repetitive elements (direct repeats). These repeats are interspaced by short variable sequences ${ }^{20}$ derived from exogenous DNA targets known as protospacers, and together they constitute the CRISPR RNA (crRNA) array. Within the DNA target, each protospacer is always associated with a protospacer adjacent motif (PAM), which can vary depending on the specific CRISPR system ${ }^{34-36}$.

The Type II CRISPR system is one of the best characterized $26-28,37,38$, consisting of the nuclease Cas9, the crRNA array that encodes the guide RNAs and a required auxiliary trans-activating crRNA (tracrRNA) that facilitates the processing of the crRNA array into discrete units ${ }^{26,28}$. Each crRNA unit then contains a 20-nt guide sequence and a partial direct repeat, where the former directs Cas9 to a 20-bp DNA target via Watson-Crick base pairing (Fig. 1). In the CRISPR-Cas system derived from Streptococcus pyogenes (which is the system used in this protocol), the target DNA must immediately precede a $5^{\prime}$-NGG $\mathrm{PAM}^{27}$, whereas 

![](https://cdn.mathpix.com/cropped/2023_06_04_a1a4274c0576b994cbb5g-02.jpg?height=485&width=852&top_left_y=185&top_left_x=145)

Figure 1 | Schematic of the RNA-guided Cas9 nuclease. The Cas9 nuclease from S. pyogenes (in yellow) is targeted to genomic DNA (shown for example is the human EMX1 locus) by an sgRNA consisting of a 20-nt guide sequence (blue) and a scaffold (red). The guide sequence pairs with the DNA target (blue bar on top strand), directly upstream of a requisite 5'-NGG adjacent motif (PAM; pink). Cas9 mediates a DSB 3 bp upstream of the PAM (red triangle).

other Cas9 orthologs may have different PAM requirements, such as those of $S$. thermophilus (5'-NNAGAA ${ }^{22,26}$ for CRISPR1 and $5^{\prime}$-NGGNG ${ }^{28,37}$ for CRISPR3) and Neisseria meningiditis $\left(5^{\prime}-\right.$ NNNNGATT ${ }^{39}$.

The RNA-guided nuclease function of CRISPR-Cas is reconstituted in mammalian cells through the heterologous expression of human codon-optimized Cas9 and the requisite RNA component $\mathrm{s}^{22-25}$. Furthermore, the crRNA and tracrRNA can be fused together to create a chimeric, single-guide RNA (sgRNA $)^{27}$ (Fig. 1). Cas9 can thus be re-directed toward almost any target of interest in immediate vicinity of the PAM sequence by altering the 20-nt guide sequence within the sgRNA.

Given its ease of implementation and multiplexing capacity, Cas9 has been used to generate engineered eukaryotic cells carrying specific mutations via both NHEJ and $\mathrm{HDR}^{22-25,40}$. Direct injection of sgRNA and mRNA encoding Cas9 into embryos has enabled the rapid generation of transgenic mice with multiple modified alleles ${ }^{41,42}$. These results hold enormous promise for editing organisms that are otherwise genetically intractable.

Cas9 nucleases carry out strand-specific cleavage by using the conserved $\mathrm{HNH}$ and RuvC nuclease domains, which can be mutated and exploited for additional function ${ }^{37}$. An aspartate-to-alanine (D10A) mutation in the RuvC catalytic domain 27,28 allows the Cas9 nickase mutant (Cas9n) to nick rather than cleave DNA to yield single-stranded breaks, and the subsequent preferential repair through $\mathrm{HDR}^{22}$ can potentially decrease the frequency of unwanted indel mutations from off-target DSBs. Appropriately offset sgRNA pairs can guide Cas9n to simultaneously nick both strands of the target locus to mediate a DSB, thus effectively increasing the specificity of target recognition ${ }^{43}$. In addition, a Cas9 mutant with both DNAcleaving catalytic residues mutated has been adapted to enable transcriptional regulation in Escherichia coli44, demonstrating the potential of functionalizing Cas9 for diverse applications, such as recruitment of fluorescent protein labels or chromatinmodifying enzymes to specific genomic loci for reporting or modulating gene function. Here we explain in detail how to use a human codonoptimized, nuclear localization sequence-flanked wild-type (WT) Cas9 nuclease or mutant Cas9 nickase to facilitate eukaryotic gene editing. We describe considerations for designing the 20-nt guide sequence, protocols for rapid construction and functional validation of sgRNAs and finally the use of the Cas9 nuclease to mediate both NHEJ- and HDR-based genome modifications in human embryonic kidney (HEK 293FT) and human stem cell (HUES9) lines (Fig. 3). The Cas9 system can similarly be applied to other cell types and organisms, including humans ${ }^{22,23,25}$ mice $^{22,41,45}$, zebrafish ${ }^{45}$, Drosophila ${ }^{46}$ and Caenorhabditis elegans ${ }^{47}$.

\section*{Comparison with other genome editing technologies}

As with other designer nuclease technologies such as ZFNs and TALENs, Cas 9 can facilitate targeted DNA DSBs at specific loci of interest in the mammalian genome and stimulate genome editing via NHEJ or HDR. Cas9 offers several potential advantages over ZFNs and TALENs, including the ease of customization, higher targeting efficiency and the ability to facilitate multiplex genome editing. As custom ZFNs are often difficult to engineer, we will primarily compare Cas9 with TALEN.

- Ease of customization. Cas9 can be easily retargeted to new DNA sequences by simply purchasing a pair of oligos encoding the 20 -nt guide sequence. In contrast, retargeting of TALEN for a new DNA sequence requires the construction of two new TALEN genes. Although a variety of protocols exist for TALEN construction ${ }^{14,17,48,49}$, it takes substantially more hands-on time to construct a new pair of TALENs.

- Cleavage pattern. WT S. pyogenes Cas9 (SpCas9) is known to make a blunt cut between the 17th and 18th bases in the target sequence ( $3 \mathrm{bp} 5^{\prime}$ of the PAM $)^{27}$. Mutating catalytic residues in either the RuvC or the HNH nuclease domain of SpCas9 converts the enzyme into a DNA nicking enzyme ${ }^{22,27}$. In contrast, TALENs cleave nonspecifically in the 12-24-bp linker between the pair of TALEN monomer-binding sites ${ }^{50}$.

![](https://cdn.mathpix.com/cropped/2023_06_04_a1a4274c0576b994cbb5g-02.jpg?height=382&width=860&top_left_y=1774&top_left_x=1054)

Figure 2 | DSB repair promotes gene editing. DSBs induced by Cas9 (yellow) can be repaired in one of two ways. In the error-prone NHEJ pathway, the ends of a DSB are processed by endogenous DNA repair machinery and rejoined, which can result in random indel mutations at the site of junction. Indel mutations occurring within the coding region of a gene can result in frameshifts and the creation of a premature stop codon, resulting in gene knockout. Alternatively, a repair template in the form of a plasmid or sSODN can be supplied to leverage the HDR pathway, which allows high fidelity and precise editing. Single-stranded nicks to the DNA can also induce HDR. 

![](https://cdn.mathpix.com/cropped/2023_06_04_a1a4274c0576b994cbb5g-03.jpg?height=1145&width=836&top_left_y=187&top_left_x=145)

-Editing efficiency. SpCas9 and TALENs have both been shown to facilitate efficient genome editing in a variety of cell types and organisms. However, owing to the ease of targeting, Cas9 can be used to target multiple genomic loci simultaneously, by co-delivering a combination of sgRNAs to the cells of interest.

\section*{Limitations of the Cas 9 system}

Cas9 can be targeted to specific genomic loci via a 20-nt guide sequence on the sgRNA. The only requirement for the selection of Cas9 target sites is the presence of a PAM sequence directly $3^{\prime}$ of the 20-bp target sequence. Each Cas9 ortholog has a unique PAM sequence; for example, SpCas9 requires a 5'-NGG PAM sequence. This PAM requirement does not severely limit the targeting range of SpCas9-in the human genome, such target sites can be found on average every 8-12 bp (refs. 22,51). In addition to the targeting range, another possible limitation is the potential for off-target mutagenesis; please see Boxes 1 and 2 for details and strategies on minimizing offtarget modifications.

\section*{Experimental design}

Target selection for sgRNA. The specificity of the Cas9 nuclease is determined by the 20 -nt guide sequence within the sgRNA. For the S. pyogenes system, the target sequence (e.g., 5'-GTC ACCTCCAATGACTAGGG- $3^{\prime}$ ) must immediately precede (i.e., be $5^{\prime}$ to) a $5^{\prime}$-NGG PAM, and the 20 -nt guide sequence base pairs with the opposite strand to mediate Cas9 cleavage at $\sim 3 \mathrm{bp}$ upstream of the PAM (Figs. 1 and 4a top strand example). Note Figure 3 | Timeline and overview of experiments. Steps for reagent design, construction, validation and cell line expansion are depicted. Custom sgRNAs (light blue bars) for each target, as well as genotyping primers, are designed in silico via the CRISPR Design Tool (http://tools.genomeengineering.org). sgRNA guide sequences can be cloned into an expression plasmid bearing both sgRNA scaffold backbone (BB) and Cas9, pSpCas9(BB). The resulting plasmid is annotated as pSpCas9(sgRNA). Completed and sequence-verified pSpCas9(sgRNA) plasmids and optional repair templates for facilitating HDR are then transfected into cells and assayed for their ability to mediate targeted cleavage. Finally, transfected cells can be clonally expanded to derive isogenic cell lines with defined mutations.

that the PAM sequence is required to immediately follow the target DNA locus, but that it is not a part of the 20 -nt guide sequence within the sgRNA.

Thus, there are two main considerations in the selection of the 20-nt guide sequence for gene targeting: (i) the $5^{\prime}$-NGG PAM for S. pyogenes Cas9 and (ii) the minimization of off-target activity ${ }^{51,52}$. We provide an online CRISPR Design Tool (http:// tools.genome-engineering.org) that takes a genomic sequence of interest and identifies suitable target sites. To experimentally assess off-target genomic modifications for each sgRNA, we also provide computationally predicted off-target sites (for a detailed discussion, see Box 1) for each intended target, ranked according to our quantitative specificity analysis on the effects of base-pairing mismatch identity, position and distribution. For increased targeting specificity, an alternative strategy using the D10A nickase mutant of Cas9 (Cas9n) along with a pair of sgRNAs may be used. The design criteria for orientation and spacing of such sgRNA pairs are described in Box 2 .

The CRISPR Design Tool provides the sequences for all oligos and primers necessary for (i) preparing the sgRNA constructs, (ii) assaying target modification efficiency and (iii) assessing cleavage at potential off-target sites. It is worth noting that because the U6 RNA polymerase III promoter used to express the sgRNA prefers a guanine (G) nucleotide as the first base of its transcript ${ }^{59}$, an extra $G$ is appended at the $5^{\prime}$ of the sgRNA where the 20-nt guide sequence does not begin with G (Fig. 4b,c). On rare occasions, certain sgRNAs may not work for reasons yet unknown; therefore, we recommend designing at least two sgRNAs for each locus and testing their efficiencies in the intended cell type.

Approaches for sgRNA construction and delivery. Depending on the desired application, sgRNAs can be delivered as either PCR amplicons containing an expression cassette (Fig. $4 \mathbf{b}$ ) or sgRNA-expressing plasmids (Fig. 4c). PCR-based sgRNA delivery appends the custom sgRNA sequence onto the reverse PCR primer used to amplify a U6 promoter template (Fig. $\mathbf{4 b}$ ). The resulting amplicon could be co-transfected with a Cas9 expression plasmid pSpCas9. This method is optimal for rapid screening of multiple candidate sgRNAs, as cell transfections for functional testing can be performed shortly after obtaining the sgRNA-encoding primers. Because this simple method obviates the need for plasmid-based cloning and sequence verification, it is well suited for testing or co-transfecting a large number of sgRNAs for generating large knockout libraries or other scalesensitive applications. Note that the sgRNA-encoding primers are 

\section*{Box 1 | Considerations for off-target cleavage activities}

Similarly to other nucleases, Cas 9 can cleave off-target DNA targets in the genome at reduced frequencies $51,52,61$. The extent to which a given guide sequence exhibits off-target activity depends on a combination of factors including enzyme concentration and the abundance of similar sequences in the target genome. For routine application of Cas9, it is important to consider ways to minimize the degree of off-target cleavage and also to be able to detect the presence of off-target cleavage $51,52,61$.

Minimizing off-target activity. For application in cell lines, we recommend following two steps to reduce the degree of off-target genome modification. First, by using our online CRISPR Design Tool, it is possible to computationally assess the likelihood of a given guide sequence to have off-target sites. These analyses are performed through an exhaustive search in the genome for off-target sequences that are similar to the guide sequence. Comprehensive experimental investigation of the effect of mismatching bases between the sgRNA and its target DNA revealed that mismatch tolerance is (i) position dependent: the 8-14 bp on the $3^{\prime}$ end of the guide sequence is less tolerant of mismatches than the $5^{\prime}$ bases; (ii) quantity dependent: in general, more than three mismatches are not tolerated; (iii) guide sequence dependent: some guide sequences are less tolerant of mismatches than others; and (iv) concentration dependent: off-target cleavage is highly sensitive to the transfected amounts, as well as relative ratios of Cas9 and sgRNA 51 . As shown in the illustration (adapted with permission from ref. 51; error bars show s.e.m. $(n=3)$ ), Cas9 can exhibit off-target cleavage in the genome, which may be minimized by carefully titrating the amount of pSpCas9 DNA transfected. The CRISPR Design Tool integrates these criteria to provide predictions for likely off-target sites in the target genome. We also recommend titrating the amount of Cas9 and sgRNA expression plasmid to minimize off-target activity.

Detection of off-target activities. We have found experimentally that Cas9 can cleave at genomic off-target sites with either 5'-NGG or 5'-NAG PAMs. By using our CRISPR-targeting web tool, it is possible to generate a list of the most likely off-target sites, as well as primers for performing SURVEYOR or sequencing analysis of those sites. For isogenic clones generated using Cas9, we strongly recommend sequencing candidate off-target sites to check for any undesired mutations. It is worth noting that there may be off-target modifications in sites that are not included in the predicted candidate list, and full genome sequencing should be performed to completely verify the absence of off-target sites.

![](https://cdn.mathpix.com/cropped/2023_06_04_a1a4274c0576b994cbb5g-04.jpg?height=378&width=855&top_left_y=982&top_left_x=1057)
Furthermore, in multiplex assays where several DSBs are induced within the same genome, there may be low rates of translocation events and they can be evaluated by using a variety of techniques such as deep sequencing 6 .

over 100 bp long, compared with the 20-bp-long oligos required for plasmid-based sgRNA delivery.

Construction of an expression plasmid for sgRNA is also simple and rapid, involving a single cloning step with a pair of partially complementary oligonucleotides. The oligo pairs encoding the 20-nt guide sequences are annealed and ligated into a plasmid (pSpCas9(BB), Fig. 4c) bearing both Cas9 and the remainder of the sgRNA as an invariant scaffold immediately following the oligo cloning site. The transfection plasmids can also be modified to enable virus production for in vivo delivery. For these approaches, the following plasmids are used within this protocol: Cas9 alone (pSpCas9) or Cas9 with an invariant sgRNA scaffold and cloning sites for inserting a guide sequence (pSpCas9 $(\mathrm{BB})$ ). For the backbone cloning construct, we have also fused 2A-GFP or $2 \mathrm{~A}$-Puro to Cas 9 to allow screening or selection of transfected cells (pSpCas9(BB)-2AGFP or pSpCas9(BB)-2A-Puro, respectively). Finally, we provide pSpCas9n(BB), a D10A nickase mutant of Cas9 for HDR and for double-nicking applications (Box 2), along with the 2AGFP and 2A-Puro fusion constructs (pSpCas9n(BB)-2A-GFP, pSpCas9n(BB)-2A-Puro). In addition to PCR and plasmid-based delivery methods, Cas9 and sgRNAs can be introduced into cells as mRNA and RNA, respectively.

Design of repair template. Traditionally, targeted DNA modifications have required the use of plasmid-based donor repair templates that contain homology arms flanking the site of alteration ${ }^{54,55}$ (Fig. 2). The homology arms on each side can vary in length, but are typically longer than $500 \mathrm{bp}$ (refs. 55,56). This method can be used to generate large modifications, including insertion of reporter genes such as fluorescent proteins or antibiotic resistance markers. The design and construction of targeting plasmids has been described elsewhere ${ }^{57}$.

More recently, ssODNs have been used in place of targeting plasmids for short modifications within a defined locus without cloning ${ }^{32}$. To achieve high HDR efficiencies, ssODNs contain flanking sequences of at least $40 \mathrm{bp}$ on each side that are homologous to the target region, and they can be oriented in either the sense or antisense direction relative to the target locus. It is worth noting that targeting efficiencies can vary widely 

\section*{Box 2 | Double-nicking strategy for minimizing off-target mutagenesis}

To minimize off-target activity, a double nicking strategy can be used to introduce DSBs at the target site ${ }^{43}$. Whereas the WT Cas 9 nuclease is guided by an sgRNA to mediate a DSB at the target locus, the D10A mutant Cas 9 nickase (Cas9n) can be specified by a pair of appropriately spaced and oriented sgRNAs to simultaneously introduce single-stranded nicks on both strands of the target DNA. The DSBs from double nicking are then repaired via NHEJ and result in indel formation with similar levels of efficiency to that of WT Cas9. As single-stranded nicks are repaired without indel formation, DSBs would only occur if both sgRNAs are able to locate target sequences within a defined space. Thus, this strategy effectively doubles the number of bases that need to be specifically recognized at the target site and significantly increases the specificity of genome editing.

To facilitate efficient double nicking, the pair of sgRNAs must be designed such that 5' overhangs are generated upon nicking. The target loci for the sgRNA pairs must also be offset with an optimal gap of 0-20 bp (see illustration: target DNA loci, blue N's; PAM, pink; predicted cleavage sites on each strand, red triangles) ${ }^{43}$. The expression constructs for sgRNA pairs can be prepared by the PCR-based method as described for sgRNAs (Step $5 A$ ). The sgRNAs can then be combined at $1: 1$ ratio and introduced along with Cas9n by using identical procedures as for WT Cas9 and sgRNAs (Step 9, $20 \mathrm{ng}$ for each sgRNA). Editing achieved by using this double-nicking strategy can be similarly detected using SURVEYOR assay or DNA sequencing. In addition to facilitating DSB- and NHEJ-mediated mutagenesis, double

![](https://cdn.mathpix.com/cropped/2023_06_04_a1a4274c0576b994cbb5g-05.jpg?height=483&width=857&top_left_y=611&top_left_x=1053)
nicking can also be used to promote HDR with comparable efficiency as WT Cas9.

A web tool to help with the identification of suitable sgRNA pairs for double nicking can be accessed at http://tools.genomeengineering.org.

depending on cell type, target locus, type of repair donor and location of modification relative to the DSB site. As a rule of thumb, single-base correction rates drop approximately fourfold at $100 \mathrm{bp}$ away from the DSB site, and beyond $200 \mathrm{bp}$ away drug selection markers may be required ${ }^{58}$.

Clonal isolation of cell lines. Isolation of clonal cell lines with specific modifications is often desired. This can be achieved after transfection by isolating single cells through either FACS (Steps 54-65) or serial dilutions (Steps 66-70), followed by an expansion period to establish a new clonal cell line. It is worth noting that cell types can vary substantially in their responses to single-cell isolation, and literature specific to the cell type of interest should be consulted.

Functional testing. SURVEYOR nuclease assay. In cells cotransfected with a pair of sgRNAs to mediate a genomic (micro)deletion or inversion, indel mutations can be detected either by the SURVEYOR nuclease assay ${ }^{59}$ or by sequencing (Fig. 5a). Our online CRISPR Design Tool provides recommended primers for both approaches. However, SURVEYOR or sequencing primers can also be designed manually to amplify the region of interest from genomic DNA. Custom primers are chosen using the National Center for Biotechnology Information (NCBI) PrimerBLAST in order to avoid nonspecific amplification. SURVEYOR primers should be designed to amplify 200-400 bp on either side of the Cas 9 target (for a total amplicon $400-800 \mathrm{bp}$ long) to allow clear visualization of cleavage bands by gel electrophoresis (Fig. 5b). To prevent excessive primer dimer formation, SURVEYOR primers should be designed to be typically 18 to $25 \mathrm{nt}$ long with melting temperatures of $\sim 60^{\circ} \mathrm{C}$. For SURVEYOR assay or sequencing analysis, we recommend testing that each pair of candidate primers produces a single PCR product, as well as testing for the absence of nonspecific cleavage during the SURVEYOR nuclease digestion process (Fig. 5).

Plasmid- or ssODN-mediated HDR. HDR can be detected via PCR amplification, followed by either sequencing of the modified region or restriction-fragment length polymorphism (RFLP) analysis. PCR primers for these purposes should anneal outside the region spanned by the homology arms to avoid false detection of residual repair template (primers HDR-Fwd and HDR-Rev; Table 1 and Fig. 6a). For ssODN-mediated HDR, SURVEYOR PCR primers may be used. Either the WT Cas 9 nuclease or mutant Cas9 nickase can be used to mediate HDR, although the efficiency of the latter can vary widely by cell type.

Detection of indels or HDR by sequencing. Targeted genome modifications can also be detected by either Sanger or deep sequencing. For the former, genomic DNA from the modified region can be amplified with either SURVEYOR or HDR primers. Amplicons should be subcloned into a plasmid such as pUC19 for transformation, and individual colonies should be sequenced to reveal the clonal genotype.

Alternatively, deep sequencing is suitable for sampling a large number of samples or target sites. NGS primers are designed for shorter amplicons, typically in the 100-200-bp size range. a Human
![](https://cdn.mathpix.com/cropped/2023_06_04_a1a4274c0576b994cbb5g-06.jpg?height=270&width=1706&top_left_y=207&top_left_x=202)

b

5'- $\stackrel{\text { U6-FWGGGCCTATTTCCCATGATTCC }-3 '}{\longrightarrow}$

C

$5^{\prime}$ - GAGGGCCTATTTCCCATGATTCC. . . CTTGTGGAAGGGAGGAACACC - $3^{\prime}$

(bottom)

![](https://cdn.mathpix.com/cropped/2023_06_04_a1a4274c0576b994cbb5g-06.jpg?height=77&width=377&top_left_y=546&top_left_x=1232)

3'- CTCCCGGATAAAGGGTACTAAGG... GAACACCTTTCCTGCTTTGTGG $-5^{\prime}$
U6 Promoter
U6-Rev

3'- CCAGTGGAGGTTACTGATCCCCAAA $-5^{\prime}$

20-nt guide (rev. comp.) sgRNA scaffold

3'- GAACACCTTTCCTGCTTTGTGGCAGTGGGGTTACTGATCCCCAA. . CGAAAAAA -5
![](https://cdn.mathpix.com/cropped/2023_06_04_a1a4274c0576b994cbb5g-06.jpg?height=284&width=1736&top_left_y=648&top_left_x=149)

$3^{\prime}-$. ITGTGGCCCAGAAGCTCTTCTGGACAAAATCTCGATCTTTATCGTTCAATTT . . - 5'

![](https://cdn.mathpix.com/cropped/2023_06_04_a1a4274c0576b994cbb5g-06.jpg?height=202&width=871&top_left_y=905&top_left_x=998)

Figure 4 | Target selection and reagent preparation. (a) For S. pyogenes Cas9, 20-bp targets (highlighted in blue) must be followed at their 3 'ends by 5'-NGG, which can occur in either the top or the bottom strand of genomic DNA, as in the example from the human EMX1 gene. We recommend using the CRISPR Design Tool (http://tools.genome-engineering.org) to facilitate target selection. (b) Schematic for co-transfection of the Cas9 expression plasmid (pSpCas9) and a PCR-amplified U6-driven sgRNA expression cassette. By using a U6 promoter-containing PCR template and a fixed forward primer (U6-Fwd), sgRNAencoding DNA can be appended onto the U6 reverse primer (U6-Rev) and synthesized as an extended DNA oligo (Ultramer oligos from IDT). Note that the guide sequence in the U6-Rev primer, designed against an example target from the top strand (blue), is the reverse complement of the 20 -bp target sequence preceding the $5^{\prime}-N G G$ PAM. An additional cytosine (' $C$ ' in gray rectangle) is appended in the reverse primer directly 3 ' to the target sequence to allow guanine as the first base of the U6 transcript. (c) Schematic for scarless cloning of the guide sequence oligos into a plasmid containing Cas9 and the sgRNA scaffold (pSpCas9(BB)). The guide oligos for the top strand example (blue) contain overhangs for ligation into the pair of BbsI sites in pSpCas9(BB), with the top and bottom strand orientations matching those of the genomic target (i.e., the top oligo is the 20-bp sequence preceding $5^{\prime}$-NGG in genomic DNA). Digestion of pSpCas9(BB) with BbsI allows the replacement of the Type II restriction sites (blue outline) with direct insertion of annealed oligos. Likewise, a G-C base pair (gray rectangle) is added at the $5^{\prime}$ end of the guide sequence for U6 transcription, which does not adversely affect targeting efficiency. Alternate versions of pSpCas9(BB) also contain markers such as GFP or a puromycin resistance gene to aid the selection of transfected cells.

For the detection of NHEJ mutations, it is important to design primers situated at least $50 \mathrm{bp}$ from the Cas9 target site to allow for the detection of longer indels. For larger deletions mediated by multiple sgRNAs, priming sites should be designed outside the deleted region. We provide guidelines for a two-step PCR fusion method to attach bar-coded sequencing adaptors for multiplex deep sequencing. We recommend the Illumina platform for its generally low levels of false positive indel detection. By comparison, Ion Torrent is less suitable for indel analysis owing to high sequencing error rate with homo-polymers ${ }^{60}$. Detailed descriptions of NGS optimization and troubleshooting can be found in the Illumina user manual. Off-target indel analysis (Box 1) can then be performed through read-alignment programs such as ClustalW, Geneious or simple custom sequence analysis scripts.

\section*{MATERIALS}

REAGENTS

sgRNA preparation

- Plasmids: pSpCas9 (Addgene plasmid ID: 48137), pSpCas9(BB) (formerly pX330; Addgene plasmid ID: 42230), pSpCas9(BB)-2A-GFP (Addgene plasmid ID: 48138), pSpCas9(BB)-2A-Puro (Addgene plasmid ID: 48139), pSpCas9n(BB) (Addgene plasmid ID: 48873), pSpCas9n(BB)-2A-GFP (Addgene plasmid ID: 48140), pSpCas9n(BB)-2A-Puro

(Addgene plasmid ID: 48141). Annotated GenBank files for the plasmids are available through Addgene and http://www.genome-engineering.org/

- pUC19 (Invitrogen, cat. no. 15364-011) or any preferred cloning plasmid - PCR primers or oligos for sgRNA construction are listed in Table $\mathbf{1}$ and in Supplementary Data 1. Primers longer than $60 \mathrm{bp}$ can be ordered as 4-nmol ultramers (Integrated DNA Technologies)

- UltraPure DNase/RNase-free distilled water (Life Technologies, cat. no. 10977-023)

- Herculase II fusion polymerase with $5 \times$ reaction buffer (Agilent Technologies, cat. no. 600679) $\triangle$ CRITICAL To minimize error in amplifying sgRNAs, it is important to use a high-fidelity polymerase. Other high-fidelity polymerases, such as PfuUltra (Agilent) or Kapa HiFi (Kapa Biosystems), may be used as substitutes. Figure 5 | Anticipated results for multiplex-sgRNA-targeted NHEJ. (a) Schematic of the SURVEYOR assay used to determine the indel percentage. First, genomic DNA from the heterogeneous population of Cas9-targeted cells is amplified by PCR. Amplicons are then reannealed slowly to generate heteroduplexes. The reannealed heteroduplexes are cleaved by SURVEYOR nuclease, whereas homoduplexes are left intact. Cas9-mediated cleavage efficiency (percentage indel) is calculated on the basis of the fraction of cleaved DNA, as determined by integrated intensity of gel bands. (b) Two sgRNAs (orange and dark blue bars) are designed to target the human GRIN2B and DYRK1A loci. SURVEYOR gel shows modification at both loci in transfected cells. Colored arrowheads indicate expected fragment sizes for each locus. (c) Paired sgRNAs (light blue and green bars) are designed to excise an exon (dark blue) in the human EMX1 locus. Target sequences and PAMs (pink) are shown in respective colors, and sites of cleavage by Cas 9 are indicated by red triangles. A predicted junction is shown below. Individual clones isolated from cell populations transfected with sgRNA 3,4 or both are assayed by PCR (using the 0ut-Fwd and Out-Rev primers), reflecting a deletion of $\sim 270 \mathrm{bp}$ long. Representative clones with no modification (12/23), mono-allelic modification (10/23) and bi-allelic (1/23) modification are shown. (d) Quantification of clonal lines with EMX1 exon deletions. Two pairs of sgRNAs (3.1 and 3.2, left-flanking sgRNAs; 4.1 and 4.2, right flanking sgRNAs) are used to mediate deletions of various sizes around one EMX1 exon. Transfected cells are clonally isolated and expanded for genotyping analysis of deletions and inversion events. Of the 105 clones screened, $51(49 \%)$ and $12(11 \%)$ are carrying heterozygous and homozygous deletions, respectively. Only approximate deletion sizes are given, as deletion junctions may be variable.

- Taq DNA polymerase with standard Taq buffer (NEB, cat. no. M0273S)

- dNTP solution mix, $25 \mathrm{mM}$ each (Enzymatics, cat. no. N205L)

- $\mathrm{MgCl}_{2}, 25 \mathrm{mM}$ (Thermo Scientific, cat. no. R0971)

- QIAquick gel extraction kit (Qiagen, cat. no. 28704)

- QIAprep spin miniprep kit (Qiagen, cat. no. 27106)

- UltraPure TBE buffer, 10× (Life Technologies, cat. no. 15581-028)

- SeaKem LE agarose (Lonza, cat. no. 50004)

- SYBR Safe DNA stain, 10,000× (Life Technologies, cat. no. S33102)

- 1-kb Plus DNA ladder (Life Technologies, cat. no. 10787-018)

- TrackIt CyanOrange loading buffer (Life Technologies, cat. no. 10482-028)

- FastDigest BbsI (BpiI) (Fermentas/Thermo Scientific, cat. no. FD1014)

- Fermentas Tango buffer (Fermentas/Thermo Scientific, cat. no. BY5)

- DTT (Fermentas/Thermo Scientific, cat. no. R0862)

- T7 DNA ligase with $2 \times$ rapid ligation buffer (Enzymatics, cat. no. L602L). Alternative ligases, such as T4 DNA ligase, can also be used. If you are using other ligases, substitute with the compatible buffer

- T4 polynucleotide kinase (New England BioLabs, cat. no. M0201S)

- T4 DNA ligase reaction buffer, 10× (New England BioLabs, cat. no. B0202S)

- Adenosine 5'-triphosphate, $10 \mathrm{mM}$ (New England BioLabs, cat. no. P0756S)

- PlasmidSafe ATP-dependent DNase (Epicentre, cat. no. E3101K)

- One Shot Stbl3 chemically competent E. coli (Life Technologies, cat. no. C7373-03)

- SOC medium (New England BioLabs, cat. no. B9020S)

- LB medium (Sigma, cat. no. L3022)

- LB agar medium (Sigma, cat. no. L2897)

- Ampicillin, $100 \mathrm{mg} \mathrm{ml}^{-1}$, sterile filtered (Sigma, cat. no. A5354)

Mammalian cell culture

- HEK 293FT cells (Life Technologies, cat. no. R700-07)

- HUES 9 cell line (Harvard Stem Cell Science)

- DMEM, high glucose (Life Technologies, cat. no. 10313-039)

- DMEM, high glucose, no phenol red (Life Technologies, cat. no. 31053-028)

- Dulbecco's PBS (DPBS; Life Technologies, cat. no. 14190-250)

- FBS, qualified and heat inactivated (Life Technologies, cat. no. 10438-034)

- Opti-MEM I reduced-serum medium (Life Technologies,

cat. no. 11058-021)

- Penicillin-streptomycin, 100× (Life Technologies, cat. no. 15140-163)

- Puromycin dihydrochloride (Life Technologies, cat. no. A11138-03)

- TrypLE Express, no phenol red (Life Technologies, cat. no. 12604-013) a
![](https://cdn.mathpix.com/cropped/2023_06_04_a1a4274c0576b994cbb5g-07.jpg?height=712&width=324&top_left_y=196&top_left_x=1097)

b
![](https://cdn.mathpix.com/cropped/2023_06_04_a1a4274c0576b994cbb5g-07.jpg?height=710&width=470&top_left_y=199&top_left_x=1442)

C

![](https://cdn.mathpix.com/cropped/2023_06_04_a1a4274c0576b994cbb5g-07.jpg?height=111&width=760&top_left_y=967&top_left_x=1083)

5'- . GTAGCCTCAGTCTTCCCATCAGGCTCTC. . AGGGTGGGCAACCACAAACCCACGAGGGGGGCAGAGT. . -3

5' - . TGGGGCCCCTAACCCTATGTAGCCTCAGTCTTCCC'CGAGGCAGAGTGCTGCTTGCTGCTGGCCAGG . - -3 $\triangle 283 \mathrm{bp}$

\begin{tabular}{rllll} 
sgRNA: & - & 3.1 & 4.1 & \multicolumn{2}{c}{$3.1+4.1$} \\
Clone: & & & 2 & 3
\end{tabular}

$\longrightarrow=4$ WT allele

$=4$ Deletion allele

Number of modified alleles:

d

\begin{tabular}{cc|ccccc}
\hline \multicolumn{3}{|c|}{ Approx. deletion } & \multicolumn{5}{|c}{ Number of clones } \\
sgRNAs & size (bp) & $+/+$ & $\Delta /+$ & $\Delta / \Delta$ & inversion & Total \\
\hline $3.1+4.1$ & 282 & 12 & 10 & 1 & 0 & 23 \\
$3.2+4.1$ & 237 & 20 & 16 & 2 & 0 & 38 \\
$3.1+4.2$ & 425 & 4 & 14 & 5 & 0 & 23 \\
$3.2+4.2$ & 259 & 6 & 11 & 4 & 0 & 21 \\
\hline
\end{tabular}

- Lipofectamine 2000 transfection reagent (Life Technologies, cat. no. 11668027)

- Amaxa SF cell line 4D-Nucleofector X kit S, 32 RCT (Lonza, cat. no. V4XC-2032)

- Geltrex LDEV-free reduced growth factor basement membrane matrix

(Life Technologies, cat. no. A1413201)

- mTeSR1 medium (Stemcell Technologies, cat. no. 05850)

- Normocin (InvivoGen, cat. no. ant-nr-1)

- Accutase cell detachment solution (Stemcell Technologies, cat. no. 07920)

- Rho-associated protein kinase (ROCK) inhibitor (Y-27632; Millipore, cat. no. SCM075)

- Amaxa P3 primary cell 4D-Nucleofector X kit S, 32 RCT (Lonza, cat. no. V4XP-3032)

Genotyping analysis

- PCR primers for SURVEYOR, RFLP analysis or sequencing; see Table $\mathbf{1}$,

Supplementary Data 1 (alternatively, they can be custom made)

- QuickExtract DNA extraction solution (Epicentre, cat. no. QE09050)

- SURVEYOR mutation detection kit for standard gel electrophoresis

(Transgenomic, cat. no. 706025) TABLE 1 | Primer sequences for sgRNA cloning and validation.

\begin{tabular}{|c|c|c|c|}
\hline Step & Primer & Sequence $\left(5^{\prime}-3^{\prime}\right)$ & Purpose \\
\hline $5 \mathrm{~A}(\mathrm{iii})$ & U6-Fwd & GAGGGCCTATTTCCCATGATTCC & Amplify any U6-sgRNA \\
\hline $5 \mathrm{~A}($ iii) & U6-Rev & $\begin{array}{l}\text { AAAAAAAGCACCGACTCGGTGCCACTTTTTCAAGTTGAT } \\
\text { AACGGACTAGCCTTATTTTAACTTGCTATTTCTAGCTCTAA } \\
\text { AACNNNNNNNNNNNNNNNNNNNCCGGTGTTTCGTCC } \\
\text { TTTCCACAAG }\end{array}$ & $\begin{array}{l}\text { Amplify specifically designed U6-sgRNA; } \mathrm{N} \text { is } \\
\text { the reverse complement of target; appended } \\
\text { cytosine (complementary to appended } \\
\text { guanine) in lowercase }\end{array}$ \\
\hline $5 \mathrm{~B}(\mathrm{i})$ & sgRNA-top & CACCgNNNNNNNNNNNNNNNNNNN & $\begin{array}{l}\text { Clone sgRNA into pSpCas9(BB); appended } \\
\text { guanine in lowercase }\end{array}$ \\
\hline $5 B(i)$ & sgRNA-bottom & AAACNNNNNNNNNNNNNNNNNNNC & $\begin{array}{l}\text { Clone sgRNA into pSpCas9(BB); appended } \\
\text { cytosine (complementary to appended } \\
\text { guanine) in lowercase }\end{array}$ \\
\hline 117 & pUC-Fwd (M13 -20 primer) & GTAAAACGACGGCCAGT & $\begin{array}{l}\text { Sanger sequencing of modified genomic } \\
\text { regions cloned into pUC19 }\end{array}$ \\
\hline 117 & pUC-Rev (M13 -26 primer) & CAGGAAACAGCTGTAAC & $\begin{array}{l}\text { Sanger sequencing of modified genomic } \\
\text { regions cloned into pUC19 }\end{array}$ \\
\hline
\end{tabular}

- TBE Gels, 4-20\%, 1.0 mm, 15 well (Life Technologies, cat. no. C62255BOX)

- Novex Hi-Density TBE sample buffer, $5 \times$ (Life Technologies, cat. no. LC6678)

- SYBR Gold nucleic acid gel stain, 10,000× (Life Technologies, cat. no. S-11494)

- FastDigest HindIII (Fermentas/Thermo Scientific, cat. no. FD0504)

- FastDigest buffer, $10 \times$ (Fermentas/Thermo Scientific, supplied with FastDigest HindIII)

- FastAP Antarctic phosphatase (Fermentas/Thermo Scientific, cat. no. EF0654)

a

![](https://cdn.mathpix.com/cropped/2023_06_04_a1a4274c0576b994cbb5g-08.jpg?height=313&width=1072&top_left_y=1609&top_left_x=181)

b Genomic locus $\quad$ 5' - . CAGAAGAGGAGGGC ...CCAATGGGGAGGACATCGATGTCACCTCCAATGACTAGGGTGGTGGCAAC . . CTCTGGCCACTCCCT . . - 3 $3^{\prime}$ - . . GTCTTCTTCTTCCCG . . GGTTACCCCTCCTGTAGCTACAGTGGAGGTTACTGATCCCACCACCCGTTG . . GAGACCGGTGAGGGA . . - 5' $\operatorname{ssODN}$

(s) 5'- CAGAAGAGGAGGGC... ACATCGATGTCACCTCCAATGACAAGCTTGCTAGCGGTGGGCAACCACAAAC . . CTCTGGCCACTCCCT repair template

(a) 3'- GTCTTCTTCTTCCCG... TGTAGCTACAGTGGAGGTTACTGTTGGAAGATCGCCACCCGTTGGTGTTTG . . GAGACCGGTGAGGGA -5

![](https://cdn.mathpix.com/cropped/2023_06_04_a1a4274c0576b994cbb5g-08.jpg?height=104&width=1136&top_left_y=2225&top_left_x=156)

![](https://cdn.mathpix.com/cropped/2023_06_04_a1a4274c0576b994cbb5g-08.jpg?height=66&width=446&top_left_y=2325&top_left_x=319)

![](https://cdn.mathpix.com/cropped/2023_06_04_a1a4274c0576b994cbb5g-08.jpg?height=58&width=422&top_left_y=2334&top_left_x=851)

$:==$

![](https://cdn.mathpix.com/cropped/2023_06_04_a1a4274c0576b994cbb5g-08.jpg?height=34&width=489&top_left_y=2523&top_left_x=236)

- Nextera XT index kit (Illumina, cat. no. FC-131-1001)

UIPMENT

$1.5 \mathrm{ml}$ (Eppendorf, cat. no. 0030125.150

with cell strainer cap, $5 \mathrm{ml}$ (BD Falcon, cat. no. 352235

Petri dishes, $60 \mathrm{~mm} \times 15 \mathrm{~mm}$ (BD Biosciences, cat. no. 351007)

- Tissue culture plate, 24 wells (BD Falcon, cat. no. 353047$)$

- Tissue culture plate, 96 wells flat bottom

(BD Falcon, cat. no. 353075)

- Tissue culture dish, $100 \mathrm{~mm}$ (BD Falcon, cat. no. 353003 )

Figure 6 Anticipated results for HDR in HEK and HUES9 cells. (a) Either a targeting plasmid or an SSODN (sense or antisense) with homology arms can be used to edit the sequence at a target genomic locus cleaved by Cas9 (red triangle). To assay the efficiency of HDR, we introduced a HindIII site (red bar) into the target locus, which was PCR-amplified with primers that anneal outside of the region of homology. Digestion of the PCR product with HindIII reveals the occurrence of HDR events. (b) ssODNs, oriented in either the sense or the antisense (s or a) direction relative to the locus of interest, can be used in combination with Cas9 to achieve efficient HDR-mediated editing at the target locus. A minimal homology region of $40 \mathrm{bp}$, and preferably $90 \mathrm{bp}$, is recommended on either side of the modification (red bar). (c) Example of the effect of sSODNs on HDR in the EMX1 locus is shown using both wild-type Cas9 and Cas9 nickase (D10A). Each ss0DN contains homology arms of $90 \mathrm{bp}$ flanking a 12-bp insertion of two restriction sites. - Nunc EasYFlask $225 \mathrm{~cm}^{2}$ (T225 flask), filter cap, 70-ml working volume (Thermo Scientific, cat. no. 159934)

- Nunc EasYFlask $75 \mathrm{~cm}^{2}$ (T75 flask), filter cap, 25-ml working volume (Thermo Scientific, cat. no. 156499)

- INCYTO C-Chip disposable hemocytometer (VWR, cat. no. 82030-468)

- Steriflip-GP Filter Unit, 0.22 $\mu \mathrm{M}$ (Millipore, cat. no. SCGP00525)

- Thermocycler with programmable temperature stepping functionality, 96 well (Applied Biosystems Veriti, cat. no. 4375786)

- Desktop microcentrifuges (e.g., Eppendorf, cat. nos. 5424 and 5804)

- Gel electrophoresis system (PowerPac basic power supply, Bio-Rad, cat. no. 164-5050), and Sub-Cell GT System gel tray (Bio-Rad, cat. no. 170-4401)

- Novex XCell SureLock mini-cell (Life Technologies, cat. no. EI0001)

- Digital gel imaging system (GelDoc EZ, Bio-Rad, cat. no. 170-8270), and blue sample tray (Bio-Rad, cat. no. 170-8273)

- Blue-light transilluminator and orange filter goggles (SafeImager 2.0; Invitrogen, cat. no. G6600)

- Gel quantification software (Bio-Rad, ImageLab or open-source ImageJ from the National Institutes of Health (NIH), USA, available at http://rsbweb.nih.gov/ij/)

- UV spectrophotometer (NanoDrop 2000c, Thermo Scientific)

\section*{REAGENT SETUP}

TBE electrophoresis solution Dilute TBE buffer in distilled water to a $1 \times$ working solution, and store it at room temperature $\left(18-22^{\circ} \mathrm{C}\right)$ for up to 6 months.

ATP, $10 \mathrm{mM}$ Divide the solution into aliquots, and store them at $-20^{\circ} \mathrm{C}$ for up to 1 year; avoid repeated freeze-thaw cycles.

DTT, $10 \mathrm{mM}$ Prepare the solution in $\mathrm{ddH}_{2} \mathrm{O}$, divide it into aliquots and store them at $-70^{\circ} \mathrm{C}$ for up to 2 years. Use a new aliquot for each reaction, as DTT is easily oxidized.

D10 culture medium For culture of HEK 293FT cells, prepare D10 medium by supplementing DMEM with GlutaMAX and $10 \%$ (vol/vol) FBS.

For routine cell line culture and maintenance, D10 medium can be further supplemented with $1 \times$ penicillin-streptomycin. Store the medium at $4{ }^{\circ} \mathrm{C}$ for up to 1 month.

mTeSR1 culture medium For culture of human embryonic stem cells (hESCs), prepare mTeSR1 medium by supplementing it with the supplement solution supplied with the medium and $100 \mu \mathrm{g} \mathrm{ml}^{-1}$ Normocin. Prepared medium can be stored at $4^{\circ} \mathrm{C}$ for up to 2 months.

\section*{PROCEDURE}

\section*{Design of targeting components and the use of the CRISPR Design Tool TIMING $1 \mathbf{d}$}

1| Input target genomic DNA sequence. We provide an online CRISPR Design Tool (http://tools.genome-engineering.org) that takes an input sequence (for example, a 1-kb genomic fragment from the region of interest), identifies and ranks suitable target sites and computationally predicts off-target sites for each intended target. Alternatively, one can manually select guide sequences by identifying the 20 -bp sequence directly upstream of any $5^{\prime}$-NGG.

2| Order necessary oligos and primers as specified by the online tool. If the cleavage site is chosen manually, the oligos or primers should be designed as described in Figure $\mathbf{4 b , c}$.

Design of the ssODN template (optional) TIMING $1 \mathrm{~h}$

3| Design and order custom ssODN. Purchase either the sense or antisense ssODN directly from IDT or the preferred supplier. We recommend manually designing homology arms of at least 40 nt on either side and preferably 90 nt for optimal HDR efficiency. It is not necessary to PAGE-purify the ssODN.

4| Resuspend and dilute ssODN ultramers to a final concentration of $10 \mu \mathrm{M}$. Do not combine or anneal the sense and antisense ssODNs. Store them at $-20^{\circ} \mathrm{C}$.

\section*{Preparation of sgRNA expression construct}

5| To generate the sgRNA expression construct, use either the PCR expression cassette (option A) or the plasmid-based procedure (option B).

(A) Generation of the sgRNA expression construct by PCR amplification TIMING $2 \mathbf{h}$

(i) Preparation of diluted U6 PCR template. We recommend using pSpCas9(BB) or pSpCas9n(BB) (Supplementary Data 2) as a PCR template, but any U6-containing plasmid can be used. Dilute the template with $\mathrm{ddH}_{2} \mathrm{O}$ to a concentration of $10 \mathrm{ng} \mu \mathrm{L}^{-1}$. Note that if a plasmid or cassette already containing a U6-driven sgRNA is used as a template, a gel extraction will need to be performed after PCR (Step 5A(iv)), using the QIAquick gel extraction kit according to the manufacturer's instructions, to ensure that the product contains only the intended sgRNA and no trace of sgRNA carryover from the template.

(ii) Preparation of diluted PCR primers. Dilute the U6-Fwd and U6-Rev (designed either using the CRISPR Design Tool or by hand and unique for each sgRNA, Step 1) primers (Table 1) to a final concentration of $10 \mu \mathrm{M}$ in $\mathrm{ddH}_{2} \mathrm{O}$ by adding $10 \mu \mathrm{l}$ of the $100 \mu \mathrm{M}$ primer stock to $90 \mu \mathrm{l}$ of $\mathrm{ddH}_{2} \mathrm{O}$. (iii) U6-sgRNA PCR. Set up the following reaction for each U6-Rev primer as follows:

\begin{tabular}{lcc} 
Component & Amount $(\mu \mathrm{l})$ & Final concentration \\
\hline Herculase II PCR buffer, 5x & 10 & $1 \times$ \\
dNTP, $100 \mathrm{mM}$ (25 mM each) & 0.5 & $1 \mathrm{mM}$ \\
U6 PCR template (pSpCas9(BB)) & 1 & $0.2 \mathrm{ng}^{-1}$ \\
U6-Fwd primer (universal) & 1 & $0.2 \mu \mathrm{M}$ \\
U6-Rev primer (sgRNA specific) & 1 & $0.2 \mu \mathrm{M}$ \\
Herculase II fusion polymerase & 0.5 & \\
Distilled water & 36 & \\
Total & 50 & \\
\hline
\end{tabular}

CRITICAL STEP To minimize error in amplifying sgRNAs, it is important to use a high-fidelity polymerase. 0ther high-fidelity polymerases, such as PfuUltra II (Agilent) or Kapa HiFi (Kapa Biosystems), may be used as a substitute.

(iv) Perform a PCR by using the following cycling conditions:

\begin{tabular}{lllc}
\hline Cycle number & Denature & Anneal & Extend \\
\hline 1 & $95^{\circ} \mathrm{C}, 2 \mathrm{~m}$ & & \\
$2-31$ & $95^{\circ} \mathrm{C}, 20 \mathrm{~s}$ & $60^{\circ} \mathrm{C}, 20 \mathrm{~s}$ & $72^{\circ} \mathrm{C}, 20 \mathrm{~s}$ \\
32 & & & $72^{\circ} \mathrm{C}, 3 \mathrm{~min}$ \\
\hline
\end{tabular}

(v) After the reaction is complete, run a sample of the product on a gel to verify successful amplification: cast a $2 \%$ (wt/vol) agarose gel in TBE buffer with SYBR Safe dye. Run $5 \mu \mathrm{l}$ of the PCR product in the gel at $15 \mathrm{~V} \mathrm{~cm}^{-1}$ for $30 \mathrm{~min}$. Successful reactions should yield a single 370 -bp-long product, and the template should be invisible.

? TROUBLESHOOTING

(vi) Purify the PCR product by using the QIAquick PCR purification kit according to the manufacturer's directions. Elute the DNA in $35 \mu \mathrm{l}$ of EB buffer (part of the kit) or water.

PAUSE POINT Purified PCR products can be stored at $-20^{\circ} \mathrm{C}$ for up to several months.

(B) Cloning sgRNA into the pSpCas9(BB) vector for co-expression with Cas9 TIMING $3 \mathrm{~d}$

(i) Preparation of the sgRNA oligos inserts. Resuspend the top and bottom strands of oligos for each sgRNA design (Step 1) to a final concentration of $100 \mu \mathrm{M}$. Prepare the following mixture for phosphorylating and annealing the sgRNA oligos (top and bottom strands):

\begin{tabular}{lc}
\hline Component & Amount $(\mu \mathrm{l})$ \\
\hline sgRNA top $(100 \mu \mathrm{M})$ & 1 \\
sgRNA bottom $(100 \mu \mathrm{M})$ & 1 \\
T4 ligation buffer, 10x & 1 \\
T4 PNK & 1 \\
ddH ${ }_{2} 0$ & 6 \\
Total & 10 \\
\hline
\end{tabular}

(ii) Phosphorylate and anneal the oligos in a thermocycler by using the following parameters: $37^{\circ} \mathrm{C}$ for $30 \mathrm{~min}$; $95^{\circ} \mathrm{C}$ for $5 \mathrm{~min}$; ramp down to $25^{\circ} \mathrm{C}$ at $5^{\circ} \mathrm{C} \mathrm{min}^{-1}$. (iii) Dilute phosphorylated and annealed oligos 1:200 by adding $1 \mu \mathrm{l}$ of oligo to $199 \mu \mathrm{l}$ of room temperature ddH $\mathrm{H}_{2} 0$.

(iv) Cloning the sgRNA oligos into pSpCas9(BB). Set up a ligation reaction for each sgRNA, as described below. We recommend also setting up a no-insert, pSpCas9(BB)-only negative control for ligation. Note: if you are using the Cas9 D10A nickase mutant for subsequent applications, substitute pSpCas9(BB) with pSpCas9n(BB). Alternatively, if fluorescence-based screening or selection is needed, substitute with pSpCas9(BB)-2A-GFP, pSpCas9(BB)-2A-Puro, pSpCas9n(BB)-2A-GFP or pSpCas9n(BB)-2A-Puro. The following steps use pSpCas9(BB) as an example:

\begin{tabular}{lc} 
Components & Amount $(\mu \mathrm{l})$ \\
\hline pSpCas9(BB), $100 \mathrm{ng}$ & $\times$ \\
Diluted oligo duplex from Step 5B(iii) & 2 \\
Tango buffer, 10x & 2 \\
DTT, $10 \mathrm{mM}$ & 1 \\
ATP, $10 \mathrm{mM}$ & 1 \\
FastDigest BbsI & 1 \\
T7 ligase & 0.5 \\
ddH ${ }_{2} 0$ & to 20 \\
Total & 20 \\
\hline
\end{tabular}

(v) Incubate the ligation reaction for a total of $1 \mathrm{~h}$.

\begin{tabular}{lc}
\hline Cycle number & Condition \\
\hline $1-6$ & $37^{\circ} \mathrm{C}$ for $5 \mathrm{~min}, 21^{\circ} \mathrm{C}$ for $5 \mathrm{~min}$ \\
\hline
\end{tabular}

(vi) Treat the ligation reaction with PlasmidSafe exonuclease to digest any residual linearized DNA. This step is optional but highly recommended.

\begin{tabular}{lc}
\hline Component & Amount $(\mu \mathrm{l})$ \\
\hline Ligation reaction from Step 5B(v) & 11 \\
PlasmidSafe buffer, 10x & 1.5 \\
ATP, $10 \mathrm{mM}$ & 1.5 \\
PlasmidSafe exonuclease & 1 \\
Total & 15 \\
\hline
\end{tabular}

(vii) Incubate the PlasmidSafe reaction at $37^{\circ} \mathrm{C}$ for $30 \mathrm{~min}$, followed by $70^{\circ} \mathrm{C}$ for $30 \mathrm{~min}$.

D PAUSE POINT After PlasmidSafe treatment, the reaction can be stored at $-20^{\circ} \mathrm{C}$ for at least 1 week.

(viii) Transformation. Transform the PlasmidSafe-treated plasmid into a competent $E$. coli strain, according to the protocol supplied with the cells. We recommend the Stbl3 strain for quick transformation. Briefly, add $2 \mu$ of the product from Step $5 B$ (vii) into $20 \mu \mathrm{l}$ of ice-cold chemically competent Stbl3 cells, incubate the mixture on ice for $10 \mathrm{~min}$, heat-shock it at $42^{\circ} \mathrm{C}$ for $30 \mathrm{~s}$ and return it immediately to ice for $2 \mathrm{~min}$. Add $100 \mu \mathrm{L}$ of SOC medium and plate it

![](https://cdn.mathpix.com/cropped/2023_06_04_a1a4274c0576b994cbb5g-11.jpg?height=50&width=1681&top_left_y=2253&top_left_x=232)
incubate competent cells for the outgrowth period after heat shock when you are transforming ampicillinresistant plasmids. (ix) Day 2: inspect the plates for colony growth. Typically, there are no colonies on the negative control plates (ligation of BbsI-digested pSpCas9(BB) alone without annealed sgRNA oligo insert), and there are tens to hundreds of colonies on the pSpCas9(sgRNA) (sgRNA inserted into pSpCas9(BB)) cloning plates. ? TROUBLESHOOTING

(x) From each plate, pick two or three colonies to check for the correct insertion of sgRNA. Use a sterile pipette tip to inoculate a single colony into a 3-ml culture of LB medium with $100 \mu \mathrm{g} \mathrm{ml}^{-1}$ ampicillin. Incubate the culture and shake it at $37^{\circ} \mathrm{C}$ overnight.

(xi) Day 3: isolate the plasmid DNA from cultures by using a QIAprep spin miniprep kit according to the manufacturer's instructions.

(xii) Sequence validation of CRISPR plasmid. Verify the sequence of each colony by sequencing from the U6 promoter using the U6-Fwd primer. Optional: sequence the Cas9 gene by using the Cbh-Fwd and SXRP002-007 primers listed in Supplementary Data 1. Reference the sequencing results against the pSpCas9(BB) cloning vector sequence to check that the 20-nt guide sequence is inserted between the U6 promoter and the remainder of the sgRNA scaffold (Fig. 4c). Details and sequence of the pSpCas9(BB) map in GenBank vector map format ( ${ }^{*}$.gb file) can be found at http://crispr. genome-engineering.org/.

? TROUBLESHOOTING

\section*{Functional validation of sgRNAs: HEK 293FT cell culture and transfections TIMING 3-4 d}

A CRITICAL The CRISPR-Cas system has been used in a number of mammalian cell lines. Conditions may vary for each cell line. Below we detail transfection conditions for HEK 293FT cells. Note that ssODN-mediated HDR transfections, performed with Amaxa SF cell line Nucleofector kit, are described in Steps 14-29. For hESC (HUES9) culturing and transfection, follow Steps 30-53.

6| HEK 293FT maintenance. Cells are maintained according to the manufacturer's recommendations. Cells are cultured in D10 medium supplemented with $10 \%$ (vol/vol) FBS at $37^{\circ} \mathrm{C}$ and $5 \% \mathrm{CO}_{2}$.

7| To passage, remove the medium and rinse the cells once by gently adding DPBS to the side of the vessel, so as not to dislodge the cells. Add $2 \mathrm{ml}$ of TrypLE to a T75 flask, and incubate the mixture for $5 \mathrm{~min}$ at $37^{\circ} \mathrm{C}$. Add $10 \mathrm{ml}$ of warm D10 medium to inactivate the trypsin, and transfer the cells to a 50-ml Falcon tube. Dissociate the cells by pipetting them up and down gently, and then reseed them into new flasks as necessary.

A CRITICAL STEP We typically passage cells every 2-3 $d$ at a split ratio of 1:4 or $1: 8$, never allowing cells to reach more than $70 \%$ confluency. Cells are discarded upon reaching passage number 15.

8| Preparation of cells for transfection. Plate the well-dissociated cells onto 24-well plates in D10 medium without antibiotics 16-24 h before transfection. Seed the cells at a density of $1.3 \times 10^{5}$ cells per well in a total volume of $500 \mu$. Scale up or down according to the cell line supplier's manual as needed.

$\triangle$ CRITICAL STEP Do not plate more cells than the recommended density, as doing so may reduce transfection efficiency.

9| On the day of transfection, cells are optimal at 70-90\% confluency. Cells can be transfected with Lipofectamine 2000 or the Amaxa SF cell line 4D-Nucleofector X kit according to the manufacturers' instructions. Transfections should be performed as follows: for sgRNAs cloned into pSpCas9(BB), transfect $500 \mathrm{ng}$ of sequence-verified CRISPR plasmid (pSpCas9(sgRNA)); if you are transfecting more than one plasmid (Box 2), mix them at equimolar ratios and use no more than $500 \mathrm{ng}$ of total DNA. For sgRNA amplified by PCR, mix the following:

pSpCas9 (Cas9 only)

sgRNA amplicon from Step 5A (each)

pUC19 (carrier DNA) $400 \mathrm{ng}$

$20 \mathrm{ng}$

Fill up total DNA to $500 \mathrm{ng}$

CRITICAL STEP We recommend transfecting in technical triplicates for reliable quantification, and including transfection controls (e.g., GFP plasmid) to monitor transfection efficiency. pSpCas9(sgRNA)-2A-GFP or pSpCas9(sgRNA)$2 A$-Puro may be used in place of pSpCas9 if fluorescence sorting or drug selection, respectively, is desired. In addition, the pSpCas9(BB) cloning plasmid and/or the sgRNA amplicon may be transfected alone as a negative control for downstream functional assays. 10| Add Lipofectamine complex to the cells gently, as HEK 293FT cells can detach easily from the plate, which will result in a lower transfection efficiency.

11| Check cells after $24 \mathrm{~h}$ for transfection efficiency. The percentage of fluorescent cells in the transfection control (e.g., GFP) can be estimated by using a fluorescence microscope. Typically, more than $70 \%$ of cells are transfected. ? TROUBLESHOOTING

12| Supplement the culture medium with an additional $500 \mu \mathrm{l}$ of warm D10 medium.

\ CRITICAL STEP Add D10 very slowly to the side of the well, and do not use cold medium, as cells can detach easily. Puromycin selection can be applied at a concentration of $1-3 \mu \mathrm{g} \mathrm{ml}^{-1}$ for HEK 293FT cells (may vary depending on the cell line).

13| Incubate the cells for a total of 48-72 $\mathrm{h}$ after transfection before passaging them for downstream applications or harvesting for indel analysis.

Co-transfection of CRISPR plasmids and HDR templates into HEK 293FT cells (optional) TIMING 3-4 d

14| Linearize 1-2 $\mu \mathrm{g}$ of targeting vector if possible by cutting once at a restriction site in the vector backbone near one of the homology arms or at the distal end of either homology arm.

Alternatively, if you are using ssODNs, simply resuspend them to a final concentration of $10 \mu \mathrm{M}$ (see Step 4) and skip Steps 15 and 16.

15| Run a small amount of the linearized plasmid alongside uncut plasmid on a $0.8-1 \%$ (wt/vol) agarose gel to check for successful linearization. Linearized plasmids should run above the supercoiled plasmid.

16| Purify the linearized plasmid with the QIAQuick PCR Purification kit, and elute in $35 \mu \mathrm{L}$ of EB buffer.

17| Preparation of cells for transfection. Culture HEK 293FT in T75 or T225 flasks. Plan ahead to have sufficient cells for the day of transfection $\left(2 \times 10^{5}\right.$ cells per transfection if you are using the Amaxa SF cell line 4D-Nucleofector X kit S).

18| Prewarming plates for transfection. Add $1 \mathrm{ml}$ of warm D10 medium into each well of a 12-well plate. Place the plates in the incubator to keep the medium warm.

19| Use option A in the table below for preparing the co-transfection of the HDR targeting plasmid with the Cas9 plasmid or option B for the co-transfection of ssODN with the Cas9 plasmid. To prepare transfection controls, see Step 9. If an sgRNA is cloned into pSpCas9(BB)-2A-GFP, cells may also be sorted by fluorescence. If you are using Cas9 nickase to mediate HDR,

![](https://cdn.mathpix.com/cropped/2023_06_04_a1a4274c0576b994cbb5g-13.jpg?height=45&width=961&top_left_y=1719&top_left_x=141)

$\triangle$ CRITICAL STEP For HDR applications, we recommend cloning sgRNA guides into one of the sgRNA expression plasmids described in Step 5B, rather than using the PCR-based expression approach.

(A) For the cotransfection of the HDR-targeting plasmid with the Cas9 plasmid:

(i) Pre-mix the following DNA in PCR tubes:
Cas9 plasmid (pSpCas9(sgRNA))
$500 \mathrm{ng}$
Linearized targeting plasmid
$500 \mathrm{ng}$

(B) For the cotransfection of ssODN and with the Cas9 plasmid:

(i) Pre-mix the following DNA in PCR tubes:

Cas9 plasmid (pSpCas9(sgRNA)) $500 \mathrm{ng}$

ss0DN template $(10 \mu \mathrm{M})$

$1 \mu \mathrm{l}$ 20| Dissociation of cells for transfection. Remove the medium and rinse the cells once gently with DPBS, taking care not to dislodge cells. Add $2 \mathrm{ml}$ of TrypLE to a T75 flask and incubate it for $5 \mathrm{~min}$ at $37^{\circ} \mathrm{C}$, and then add $10 \mathrm{ml}$ of warm D10 medium and triturate gently in a $50-\mathrm{ml}$ Falcon tube.

A CRITICAL STEP Ensure that the cells are triturated gently and dissociated to single cells. Large clumps will reduce transfection efficiency.

21| Take a 10- $\mu \mathrm{l}$ aliquot from the cell suspension and dilute it into $90 \mu \mathrm{L}$ of $D 10$ medium for counting. Count the cells and calculate the number of cells and the volume of suspension needed for transfection. We typically transfect $2 \times 10^{5}$ cells per condition with the Amaxa SF cell line 4D-Nucleofector X kit S, and we recommend calculating for $20 \%$ more cells than required to adjust for volume loss in subsequent pipetting steps. Transfer the volume needed $(20 \mu \mathrm{l}$ per transfection plus waste volume) into a new Falcon tube.

22| Spin down the cells from Step 21 at $200 \mathrm{~g}$ for $5 \mathrm{~min}$ at room temperature.

23| Prepare the transfection solution by mixing the SF solution and S1 supplement supplied in the Amaxa SF cell line 4D-Nucleofector X kit S; a total of $20 \mu \mathrm{l}$ of supplemented SF solution is used per transfection. Likewise, we recommend calculating for $20 \%$ more volume than required.

24 Remove the medium completely from the pelleted cells from Step 22, and gently resuspend the cells in an appropriate volume (20 $\mu \mathrm{l}$ per $2 \times 10^{5}$ cells) of S1-supplemented SF solution. Do not leave the cells in SF solution for extended periods of time.

25| Pipette $20 \mu \mathrm{l}$ of resuspended cells into each DNA premix from Step 19. Pipette gently to mix and transfer to a Nucleocuvette strip chamber. Repeat this step for each transfection condition.

26| Electroporate the cells by using the Nucleofector $4 D$ program recommended by Amaxa, CM-130.

27| Gently and slowly pipette $100 \mu \mathrm{L}$ of warm D10 medium into each Nucleocuvette strip chamber, and transfer all the volume into a well with the prewarmed medium from Step 18.

A CRITICAL STEP Cells are very fragile at this stage, and harsh pipetting can cause cell death.

28| Incubate the mixture for $24 \mathrm{~h}$. At this point, transfection efficiency can be estimated from the fraction of fluorescent cells in the positive transfection control. Nucleofection typically results in $>70-80 \%$ transfection efficiency.

? TROUBLESHOOTING

29| Slowly add $1 \mathrm{ml}$ of warm D10 medium to each well without dislodging the cells. Puromycin selection can be applied at a concentration of 1-3 $\mathrm{g} \mathrm{ml}^{-1}$ for HEK 293FT cells (may vary depending on the cell line). Incubate the cells with puromycin for at least $72 \mathrm{~h}$. Cells can then be cultured in regular medium for downstream experiments or harvested for genotyping.

hESC (HUES 9) culture and transfection TIMING 3-4 d

\ CRITICAL hESCs and human induced pluripotent stem cells can vary widely in their transfection efficiency, tolerance of single-cell dissociation and maintenance conditions. For a given cell line of interest, relevant literature or the distributor should be consulted.

30| Maintaining HUES9 cells. We routinely maintain HUES9 cells (a hESC cell line) in feeder-free conditions with mTesR1 medium. Prepare mTeSR1 medium by adding the $5 \times$ supplement included with the basal medium and $100 \mu \mathrm{g} \mathrm{ml}^{-1} \mathrm{Normocin}^{\text {. }}$

31| Prepare a 10-ml aliquot of mTeSR1 medium supplemented further with $10 \mu \mathrm{M}$ ROCK inhibitor.

32| Coating a tissue culture plate. Dilute cold GelTrex 1:100 in cold DMEM and coat the entire surface of a 100-mm tissue culture plate.

33| Place the plate in an incubator for at least $30 \mathrm{~min}$ at $37^{\circ} \mathrm{C}$.

34 Thaw a vial of cells at $37^{\circ} \mathrm{C}$, transfer the cells to a 15-ml Falcon tube, add $5 \mathrm{ml}$ of mTeSR1 medium and pellet at $200 \mathrm{~g}$ for 5 min at room temperature. 35| Aspirate the GelTrex coating (Step 32) and seed $\sim 1 \times 10^{6}$ cells with $10 \mathrm{ml}$ of mTeSR1 medium containing ROCK inhibitor from Step 31.

36| Replace with mTeSR1 medium without ROCK inhibitor after $24 \mathrm{~h}$ and refeed daily.

37| Passaging cells. Passage the cells before they reach $70 \%$ confluency.

38| Aspirate the mTeSR1 medium and wash the cells once with DPBS.

39| Dissociate the cells by adding $2 \mathrm{ml}$ of Accutase and incubating them at $37^{\circ} \mathrm{C}$ for $3-5 \mathrm{~min}$.

40| Add $10 \mathrm{ml}$ of mTeSR1 medium to the detached cells, transfer the mixture to a 15 -ml Falcon tube and resuspend gently.

41| Replate the cells onto GelTrex-coated plates in mTeSR1 medium with $10 \mu \mathrm{M}$ ROCK inhibitor.

42| Replace with normal mTeSR1 medium $24 \mathrm{~h}$ after plating.

43| Transfection. We recommend culturing cells for at least 1 week after thawing and before transfecting by using the Amaxa P3 primary cell 4D Nucleofector kit.

44| Refeed log-phase growing cells (50-70\% confluency) with fresh medium $2 \mathrm{~h}$ before transfection.

45| Dissociate to single cells or small clusters of no more than ten cells (as viewed under the microscope) with Accutase and gentle resuspension.

46| Count the number of cells needed for nucleofection (200,000 cells per transfection) and spin down at $200 \mathrm{~g}$ for $5 \mathrm{~min}$ at room temperature.

47| Remove the medium completely and resuspend it in $20 \mu \mathrm{l}$ of S1-supplemented P3 nucleofection solution, per $2 \times 10^{5}$ cells.

48| Pipette the resuspended cells with added DNA (Steps 9 and 19) into electroporation cuvettes and electroporate according to the suggested program. For $2 \times 10^{5}$ cells, we typically use $1 \mu \mathrm{g}$ of total DNA.

49| Gently plate the electroporated cells onto coated 100-mm plates supplemented with $10 \mu \mathrm{M}$ ROCK inhibitor.

50| Check transfection success (Steps 11 and 28) and refeed the cells daily with regular mTeSR1 medium beginning $24 \mathrm{~h}$ after nucleofection. Puromycin selection can be applied at a concentration of $0.5 \mu \mathrm{g} \mathrm{ml}^{-1}$ (may vary depending on the cell line). Typically, we observe $>70 \%$ transfection efficiency with Amaxa nucleofection.

? TROUBLESHOOTING

51| At 48-72 h post transfection, dissociate the cells with Accutase and resuspend them gently in a $5 \times$ volume of mTeSR1. Reserve a fraction of the resuspended cells at this stage for replating (Steps 41 and 42; make sure to add ROCK inhibitor for each passaging), for downstream applications or clonal isolation (Steps 54-70), and use the remaining cells for genotyping (Steps 71-126).

$\triangle$ CRITICAL STEP Do not dissociate the cells mechanically without Accutase.

52| Spin the cells down at $200 \mathrm{~g}$ for 5 min at room temperature.

$\triangle$ CRITICAL STEP Do not spin the cells without inactivating the Accutase first, or above the recommended speed; doing so may cause cells to lyse.

53| Process pelleted cells directly for DNA extraction with the QuickExtract solution (Steps 71-74).

Isolation of clonal cell lines by FACS TIMING 2-3 $\mathrm{h}$ hands-on; 2-3 weeks expansion

A CRITICAL Isolation of clonal cell populations from the transfected cells (Step 51) can be performed $24 \mathrm{~h}$ after transfection by FACS (Steps 54-65) or by serial dilutions (Steps 66-70). Given that cell types can vary greatly in their response to FACS, clonal-density dilution or other isolation procedures, literature specific to the cell type of interest should be consulted. 54| Preparation of FACS media. Cells can be sorted in regular D10 medium supplemented with penicillin-streptomycin.

![](https://cdn.mathpix.com/cropped/2023_06_04_a1a4274c0576b994cbb5g-16.jpg?height=48&width=1596&top_left_y=255&top_left_x=142)
also required, phenol red-free DMEM or DPBS is substituted for normal DMEM.

55| To 96-well plates, add $100 \mu$ of D10 medium supplemented with penicillin-streptomycin per well.

![](https://cdn.mathpix.com/cropped/2023_06_04_a1a4274c0576b994cbb5g-16.jpg?height=42&width=1732&top_left_y=425&top_left_x=143)
plates prepared and cells sorted may need to be adjusted to ensure an adequate number of clonal lines derived.

56| Preparation of cells for FACS. Dissociate the cells (from Steps 11 or 28) by aspirating the medium completely and adding enough TrypLE to thinly cover the adherent layer of transfected cells. Incubate the mixture for 5 min at $37^{\circ} \mathrm{C}$ and add $400 \mu \mathrm{l}$ of warm D10 medium.

57| Transfer the resuspended cells into a $15-\mathrm{ml}$ Falcon tube and gently triturate 20 times.

![](https://cdn.mathpix.com/cropped/2023_06_04_a1a4274c0576b994cbb5g-16.jpg?height=43&width=1190&top_left_y=759&top_left_x=143)

58| Spin down the cells at $200 \mathrm{~g}$ for 5 min at room temperature.

59| Aspirate the medium, and resuspend it in $200 \mu \mathrm{L}$ of FACS medium.

60| Filter the cells into the cell strainer tube through its mesh cap. Place the cells on ice until sorting.

61| Sort single cells into the 96-well plates prepared from Step 55. If sgRNA is cloned into pSpCas9(BB)-2A-GFP, fluorescence may be used to enrich for transfected cells. After sorting, examine the plate under a microscope and determine the presence of a single cell in most of the wells on the plate.

? TROUBLESHOOTING

62| Return the cells to the incubator and allow them to expand for 2-3 weeks. Add $100 \mu \mathrm{l}$ of warm D10 medium $5 \mathrm{~d}$ after sorting. Change $100 \mu \mathrm{l}$ of the medium every $3-5 \mathrm{~d}$ as necessary.

63| Inspect the colonies for "clonal" appearance 1 week after sorting: rounded colonies radiating from a central point. Mark off the wells that are empty or that may have been seeded more than a single cell.

64| When the cells are more than 60\% confluent, prepare replica plates for passaging (one well for each clone) by adding $100 \mu \mathrm{l}$ of D10 medium to each well in the replica plates. Dissociate the cells directly by pipetting up and down vigorously 20 times, and plate $20 \%$ of each of the resuspended volumes into the replica wells to keep the clonal lines. Change the medium every 2-3 d thereafter and passage accordingly.

65| Use the remaining $80 \%$ of cells for DNA isolation and genotyping (Steps 71-74).

Isolation of clonal cell lines by dilution TIMING 2-3 h hands-on; 2-3 weeks expansion

\ CRITICAL As cell types can vary greatly in their response to FACS, clonal-density dilution or other isolation procedures, literature specific to the cell type of interest should be consulted.

66| Dissociate the cells from the transfected wells (Steps 11 or 28) $48 \mathrm{~h}$ after transfection. Take care to dissociate to single cells. A cell strainer (Step 60) can be used to prevent clumping of cells.

67| Count the number of cells from each 24-well plate, and serially dilute them in D10 medium to a final concentration of 0.5 cells per $100 \mu \mathrm{l}$ to reduce the likelihood of having multiple cells per well. We recommend using 60 cells in $12 \mathrm{ml}$ of D10 medium for each 96-well plate, and plating at least two 96-well plates for each transfected population.

$\triangle$ CRITICAL STEP Single-cell dissociation and accurate count of cell number are critical for clonal dilution. We recommend examining the dissociated cells under a microscope to ensure successful dissociation and recounting cells at an intermediate serial dilution stage to ensure accuracy.

? TROUBLESHOOTING 68| Multichannel-pipette $100 \mu \mathrm{l}$ of diluted cells to each well of a 96-well plate. The remaining cell suspension can be kept and used for genotyping at the population level to gauge overall modification efficiency.

69| Inspect the colonies for a clonal appearance 1 week after plating (rounded colonies radiating from a central point). Mark off the wells that may have been seeded with multiple cells.

70| Return the cells to the incubator and allow them to expand for 2-3 weeks. Refeed and replica-plate the cells as needed and as detailed in Steps 64 and 65.

Functional testing: detection of indel mutations by the SURVEYOR nuclease assay TIMING 5-6 $\mathrm{h}$

$\triangle$ CRITICAL Before assaying the cleavage efficiency of transfected cells, we recommend testing each new SURVEYOR primer on negative (untransfected) control samples for the intended cell type by SURVEYOR nuclease digestion.

71| Harvesting cells for DNA extraction. Dissociate all transfected cells (from Steps 13, 29, 53, 65 or 70) and spin them down at $200 \mathrm{~g}$ for $5 \mathrm{~min}$ at room temperature. Keep the replica plates as needed to maintain transfected cell lines in culture.

72| Aspirate the medium completely.

73| For DNA extraction, use the QuickExtract solution according to the manufacturer's instructions. We typically use $50 \mu \mathrm{l}$ or $10 \mu \mathrm{l}$ of the solution for each well of a 24 -well or 96 -well plate, respectively.

74| Normalize the extracted DNA to a final concentration of $100-200 \mathrm{ng} \mu \mathrm{l}^{-1}$ with $\mathrm{ddH}_{2} 0$.

$\square$ PAUSE POINT Extracted DNA can be stored at $-20^{\circ} \mathrm{C}$ for several months.

75| Setting up the SURVEYOR PCR. Master-mix the following using the SURVEYOR primers provided by the CRISPR Design Tool (Step 1):

\begin{tabular}{lcc} 
Component & Amount $(\mu \mathrm{l})$ & Final concentration \\
\hline Herculase II PCR buffer, 5x & 10 & $1 \times$ \\
dNTP, $100 \mathrm{mM}(25 \mathrm{mM}$ each) & 1 & $2 \mathrm{mM}$ \\
SURVEYOR-Fwd primer, $10 \mu \mathrm{M}$ & 1 & $0.2 \mu \mathrm{M}$ \\
SURVEYOR-Rev primer, $10 \mu \mathrm{M}$ & 1 & $0.2 \mu \mathrm{M}$ \\
Herculase II fusion polymerase & 1 & \\
MgCl$_{2}$, $25 \mathrm{mM}$ & 2 & $1 \mathrm{mM}$ \\
DNA template & 1 & $2 \mathrm{ng} \mu \mathrm{l}^{-1}$ \\
ddH 20 & 33 & \\
Total & 50 & \\
\hline
\end{tabular}

CRITICAL STEP SURVEYOR assays rely on the detection of single-base mismatches; therefore, it is crucial to use a high-fidelity polymerase. Other high-fidelity polymerases, such as PfuUltra (Agilent) or Kapa HiFi (Kapa Biosystems), may be used as a substitute. In addition, because SURVEYOR cleavage results can detect naturally occurring single-nucleotide polymorphisms, it is important to run negative control samples of untransfected or otherwise unmodified cells.

76| Perform a PCR with the following cycling conditions, for no more than 30 amplification cycles:

\begin{tabular}{lccc} 
Cycle number & Denature & Anneal & Extend \\
\hline 1 & $95^{\circ} \mathrm{C}, 2 \mathrm{~min}$ & \\
$2-31$ & $95^{\circ} \mathrm{C}, 20 \mathrm{~s}$ & $60^{\circ} \mathrm{C}, 20 \mathrm{~s}$ & $72{ }^{\circ} \mathrm{C}, 30 \mathrm{~s}$ \\
32 & & & $72^{\circ} \mathrm{C}, 3 \mathrm{~min}$ \\
\hline
\end{tabular}

77| Run $2-5 \mu \mathrm{l}$ of the PCR products on a $1 \%$ (wt/vol) agarose gel to check for single-band products. Although these PCR conditions are designed to work with most pairs of SURVEYOR primers, some primers may need additional optimization by adjusting the template concentration, $\mathrm{MgCl}_{2}$ concentration and/or the annealing temperature.

? TROUBLESHOOTING

![](https://cdn.mathpix.com/cropped/2023_06_04_a1a4274c0576b994cbb5g-18.jpg?height=42&width=1526&top_left_y=425&top_left_x=145)

■ PAUSE POINT Purified PCR products can be stored at $-20^{\circ} \mathrm{C}$ for up to several months.

79| DNA heteroduplex formation. Set up the annealing reaction as follows:

\begin{tabular}{lc}
\hline Component & Amount $(\mu \mathrm{l})$ \\
\hline Taq PCR buffer, 10x & 2 \\
Normalized PCR product, $20 \mathrm{ng} \mu \mathrm{l}^{-1}$ & 18 \\
Total volume & 20 \\
\hline
\end{tabular}

80| Anneal the reaction by using the following conditions:

\begin{tabular}{ll}
\hline Cycle number & Condition \\
\hline 1 & $95^{\circ} \mathrm{C}, 10 \mathrm{~min}$ \\
2 & $95-85^{\circ} \mathrm{C},-2{ }^{\circ} \mathrm{C} \mathrm{s}^{-1}$ \\
3 & $85^{\circ} \mathrm{C}, 1 \mathrm{~min}$ \\
4 & $85-75{ }^{\circ} \mathrm{C},-0.3{ }^{\circ} \mathrm{C} \mathrm{s}^{-1}$ \\
5 & $75^{\circ} \mathrm{C}, 1 \mathrm{~min}$ \\
6 & $75-65^{\circ} \mathrm{C},-0.3{ }^{\circ} \mathrm{C} \mathrm{s}^{-1}$ \\
7 & $65^{\circ} \mathrm{C}, 1 \mathrm{~min}$ \\
8 & $65-55^{\circ} \mathrm{C},-0.3{ }^{\circ} \mathrm{C} \mathrm{s}^{-1}$ \\
9 & $55^{\circ} \mathrm{C}, 1 \mathrm{~min}$ \\
10 & $55-45^{\circ} \mathrm{C},-0.3{ }^{\circ} \mathrm{C} \mathrm{s}^{-1}$ \\
11 & $45^{\circ} \mathrm{C}, 1 \mathrm{~min}$ \\
12 & $45-35^{\circ} \mathrm{C},-0.3{ }^{\circ} \mathrm{C} \mathrm{s}^{-1}$ \\
13 & $35^{\circ} \mathrm{C}, 1 \mathrm{~min}$ \\
14 & $35-25^{\circ} \mathrm{C},-0.3{ }^{\circ} \mathrm{C} \mathrm{s}^{-1}$ \\
15 & $25^{\circ} \mathrm{C}, 1 \mathrm{~min}$ \\
16 & $25-4{ }^{\circ} \mathrm{C},-0.3{ }^{\circ} \mathrm{C} \mathrm{s}^{-1}$ \\
17 & $4{ }^{\circ} \mathrm{C}, \mathrm{hold}$
\end{tabular}

81| SURVEYOR nuclease S digestion. Master-mix and add the following components on ice to the annealed heteroduplexes from Step 80, to a final volume of $25 \mu \mathrm{l}$ :

\begin{tabular}{lcc}
\hline Component & Amount $(\mu \mathrm{l})$ & Final concentration \\
\hline Annealed heteroduplex & 20 & \\
$\mathrm{MgCl}_{2}$ stock solution supplied with kit, $0.15 \mathrm{M}$ & 2.5 & $15 \mathrm{mM}$ \\
$\mathrm{ddH}_{2} \mathrm{O}$ & 0.5 & \\
SURVEYOR nuclease S & 1 & $1 \times$ \\
SURVEYOR enhancer S & 1 & $1 \times$ \\
Total & 25 & \\
\hline
\end{tabular}

CRITICAL STEP Note that the $\mathrm{MgCl}_{2}$ used for SURVEYOR nuclease digestion (included in the SURVEYOR mutation detection kit) is a higher concentration from that used for SURVEYOR PCR.

82| Vortex the mixture well and spin it down briefly. Incubate the reaction at $42^{\circ} \mathrm{C}$ for $30 \mathrm{~min}$.

83| (Optional) Add $2 \mu \mathrm{l}$ of the Stop Solution from the SURVEYOR kit if you do not intend to visualize the reaction products (next step) immediately.

PAUSE POINT The digested products with Stop Solution can be stored at $-20^{\circ} \mathrm{C}$ for at least $2 \mathrm{~d}$ at this point for later analysis.

84|Visualizing the SURVEYOR reaction. SURVEYOR nuclease digestion products can be visualized on a $2 \%$ (wt/vol) agarose gel. For better resolution, products can be run on a 4-20\% gradient polyacrylamide TBE gel. Load $10 \mu \mathrm{l}$ of the product with the recommended loading buffer and run the gel according to the manufacturer's instructions. Typically, we run the gel until the bromophenol blue dye has migrated to the bottom of the gel. Include the DNA ladder and negative (untransfected) controls on the same gel.

85| Stain the gel with SYBR Gold dye diluted 1:10,000 in TBE $(20 \mu \mathrm{l}$ of stock in $200 \mathrm{ml}$ of TBE). Rock the gel gently for $15 \mathrm{~min}$. Be sure to shield the staining solution from light to avoid photobleaching of the dye.

86| Image the gel by using a quantitative imaging system without overexposing the bands. The negative controls should have only one band corresponding to the size of the PCR product, but they may have occasional nonspecific cleavage bands of other sizes. These will not interfere with analysis if they are distinct in size from the target cleavage bands. The sum of target cleavage band sizes, provided by the CRISPR Design Tool, should be equal to the size of the PCR product.

? TROUBLESHOOTING

87| Estimation of the cleavage intensity. Measure the integrated intensity of the PCR amplicon and cleaved bands by using ImageLab, ImageJ or other gel quantification software.

88| For each lane, calculate the fraction of the PCR product cleaved $\left(f_{\text {cut }}\right)$ by using the following formula: $f_{c u t}=(b+c) /(a+b+c)$, where $a$ is the integrated intensity of the undigested PCR product and $b$ and $c$ are the integrated intensities of each cleavage product. A sample gel is shown in Figure 6.

89| Indel occurrence can be estimated with the following formula, based on the binomial probability distribution of duplex formation:

$$
\text { indel }(\%)=100 \times\left(1-\sqrt{\left.\left(1-f_{\text {cut }}\right)\right)}\right.
$$

Functional testing: detection of genomic microdeletions by PCR TIMING 3-4 $\mathrm{h}$ hands-on; 2-3 weeks expansion 90| Transfect the cells as described in Steps 8-13 or Steps 43-51 with a pair of sgRNAs flanking the region to be deleted.

91| At $24 \mathrm{~h}$ after transfection, isolate the clones by FACS or by serial dilution as described above (Steps 54-70).

92| Expand the cells for 2-3 weeks.

93| Extract the DNA from clonal lines as described above (Steps 71-74) by using $10 \mu \mathrm{l}$ of QuickExtract solution, and normalize the genomic DNA with $\mathrm{ddH}_{2} \mathrm{O}$ to a final concentration of $100 \mathrm{ng} \mu \mathrm{l}^{-1}$.

94| $P C R$ amplification and analysis of the modified region. For analysis of (micro)deletions, follow option A; for analysis of inversions, follow option B.

\section*{(A) Deletion or microdeletion analysis}

(i) For the analysis of microdeletions, use the Out-Fwd and Out-Rev primers, both of which are designed to anneal outside of the deleted region, to verify the successful deletion by product size analysis. If the deletion size is more than $1 \mathrm{~kb}$, set up a parallel set of PCRs with In-Fwd and In-Rev primers to screen for the presence of the WT allele (Fig. 5c). As with SURVEYOR assays, include a negative (untransfected sample) control. Set up the PCR as follows:

\begin{tabular}{|c|c|c|}
\hline Component & Amount ( $\mu l)$ & Final concentration \\
\hline Herculase II PCR buffer, $5 \times$ & 10 & $1 x$ \\
\hline $\mathrm{dNTP}, 100 \mathrm{mM}$ (25 mM each) & 1 & $2 \mathrm{mM}$ \\
\hline Out-Fwd primer, $10 \mu \mathrm{M}$ & 1 & $0.2 \mu \mathrm{M}$ \\
\hline Out-Rev primer, $10 \mu \mathrm{M}$ & 1 & $0.2 \mu \mathrm{M}$ \\
\hline Herculase II fusion polymerase & 1 & \\
\hline $\mathrm{MgCl}_{2}, 25 \mathrm{mM}$ & 2 & $1 \mathrm{mM}$ \\
\hline DNA template & 1 & $2 \mathrm{ng} \mu \mathrm{l}^{-1}$ \\
\hline $\mathrm{ddH}_{2} \mathrm{O}$ & 33 & \\
\hline Total & 50 & \\
\hline
\end{tabular}

\section*{(B) Inversion analysis}

(i) To screen for inversions, set up the PCR (Fig. 5c) as described below. Note that primers are paired either as 0ut-Fwd + In-Fwd or Out-Rev + In-Rev. Include a negative control.

\begin{tabular}{lcc}
\hline Component & Amount $(\mu \mathrm{l})$ & Final concentration \\
\hline Herculase II PCR buffer, 5x & 10 & $1 \times$ \\
dNTP, $100 \mathrm{mM}$ (25 mM each) & 1 & $2 \mathrm{mM}$ \\
Out-Fwd or Out-Rev primer, $10 \mu \mathrm{M}$ & 1 & $0.2 \mu \mathrm{M}$ \\
In-Fwd or In-Rev primer, $10 \mu \mathrm{M}$ & 1 & $0.2 \mu \mathrm{M}$ \\
Herculase II fusion polymerase & 1 & \\
MgCl$_{2}, 25 \mathrm{mM}$ & 2 & $1 \mathrm{mM}$ \\
DNA template & 1 & $2 \mathrm{ng} \mu \mathrm{l}^{-1}$ \\
ddH ${ }_{2} 0$ & 33 & \\
Total & 50 & \\
\hline
\end{tabular}

95| Perform a PCR by using the following cycling conditions:

\begin{tabular}{lccc}
\hline Cycle number & Denature & Anneal & Extend \\
\hline 1 & $95^{\circ} \mathrm{C}, 2 \mathrm{~min}$ & & \\
$2-31$ & $95^{\circ} \mathrm{C}, 20 \mathrm{~s}$ & $60^{\circ} \mathrm{C}, 20 \mathrm{~s}$ & $72^{\circ} \mathrm{C}, 30 \mathrm{~s}$ \\
32 & & & $72^{\circ} \mathrm{C}, 3 \mathrm{~min}$ \\
\hline
\end{tabular}

96 Run 2-5 $\mu \mathrm{l}$ of PCR product on a 1-2\% (wt/vol) agarose gel to check for size of the products in the case of deletions, or for the presence or absence of PCR products in the case of inversions. Although these PCR conditions are designed to work with most primers, some primers may need additional optimization by adjusting the template concentration, $\mathrm{MgCl}_{2}$ concentration and/or the annealing temperature.

? TROUBLESHOOTING

Functional testing: genotyping of HDR-mediated targeted modifications by RFLP analysis TIMING 3-4 h

97| Extract the DNA as described in Steps 71-74 by using the QuickExtract solution, and normalize the genomic DNA with water to a final concentration of $100-200 \mathrm{ng} \mu \mathrm{l}^{-1}$. 98| PCR amplification of the modified region. The HDR-Fwd and HDR-Rev primers are designed to anneal outside of the region of homology between the ssODN and targeted genomic region, to verify successful sequence conversion. Include a negative (untransfected) control sample. Set up the PCR as follows:

\begin{tabular}{lcc} 
Component & Amount $(\mu \mathrm{l})$ & Final concentration \\
\hline Herculase II PCR buffer, $5 \times$ & 10 & $1 \times$ \\
dNTP, $100 \mathrm{mM}(25 \mathrm{mM}$ each) & 1 & $2 \mathrm{mM}$ \\
HDR-Fwd primer, $10 \mu \mathrm{M}$ & 1 & $0.2 \mu \mathrm{M}$ \\
HDR-Rev primer, $10 \mu \mathrm{M}$ & 1 & $0.2 \mu \mathrm{M}$ \\
Herculase II fusion polymerase & 1 & \\
MgCl $25 \mathrm{mM}$ & 2 & $1 \mathrm{mM}$ \\
DNA template & 1 & $2 \mathrm{ng} \mu \mathrm{l}^{-1}$ \\
dd $\mathrm{H}_{2} 0$ & 33 & \\
Total & 50 & \\
\hline
\end{tabular}

99| Run the following program:

\begin{tabular}{|c|c|c|c|}
\hline Cycle number & Denature & Anneal & Extend \\
\hline 1 & $95^{\circ} \mathrm{C}, 2 \mathrm{~min}$ & & \\
\hline $2-31$ & $95^{\circ} \mathrm{C}, 20 \mathrm{~s}$ & $60^{\circ} \mathrm{C}, 20 \mathrm{~s}$ & $72^{\circ} \mathrm{C}, 30-60 \mathrm{~s}$ per kb \\
\hline 32 & & & $72^{\circ} \mathrm{C}, 3 \mathrm{~min}$ \\
\hline
\end{tabular}

100| Run $5 \mu \mathrm{l}$ of PCR product on a 0.8-1\% (wt/vol) agarose gel to check for a single band of product. Primers may need additional optimization by adjusting the template concentration, $\mathrm{MgCl}_{2}$ concentration and/or the annealing temperature. ? TROUBLESHOOTING

101| Purify the PCRs by using the QIAQuick PCR purification kit.

102| In the HDR example provided in Figure 6, a HindIII restriction site was inserted into the EMX1 gene. These are detected by an RFLP analysis of the PCR amplicon:

\begin{tabular}{lc} 
Component & Amount $(\boldsymbol{\mu l})$ \\
\hline Purified PCR amplicon & $x(200-300 \mathrm{ng})$ \\
FastDigest buffer & 1 \\
HindIII (or other enzyme as necessary) & 0.5 \\
$\mathrm{ddH}_{2} \mathrm{O}$ & Up to 10 \\
Total & 10 \\
\hline
\end{tabular}

103| Digest the DNA for 10 min at $37^{\circ} \mathrm{C}$. 104| Run $10 \mu \mathrm{l}$ of the digested product with loading dye on a 4-20\% gradient polyacrylamide TBE gel until the xylene cyanol band has migrated to the bottom of the gel.

105| Stain the gel with SYBR Gold dye while rocking for $15 \mathrm{~min}$. Be sure to shield the staining solution from light to avoid photobleaching of the dye.

106| Image and quantify the cleavage products as described above for the SURVEYOR assay section (Steps 86-89).

107| HDR efficiency is estimated by using the following formula: $(b+c) /(a+b+c)$, where $a$ is the integrated intensity for the undigested HDR PCR product, and $b$ and $c$ are the integrated intensities for the HindIII-cut fragments.

108| Alternatively, clone the genotype-purified PCR amplicons from Step 101 via Sanger sequencing (Steps 109-117) or deep sequencing (Steps 118-126).

Assessment of Cas9 cleavage or HDR-mediated target modification efficiency by Sanger sequencing TIMING 3 d $\triangle$ CRITICAL Instead of the SURVEYOR or RFLP assays, genomic amplicons of the target region (produced in Step 78 or 101) from transfected cells (Steps 8-13, and 14-28 for HEK 293FT cells, or steps 43-51 for HUES9 cells) can be cloned into a plasmid, and a set of clones can be prepared for Sanger sequencing to assess Cas9 cleavage or HDR-mediated target modification efficiency. SURVEYOR or HDR primers can be used for Sanger sequencing if appropriate restriction sites are appended to the forward and reverse primers. For cloning into the recommended pUC19 backbone, EcoRI can be used for the Fwd primer and HindIII for the Rev primer.

109| Target-site amplicon digestion. Set up the digestion reaction as follows:

\begin{tabular}{|c|c|}
\hline Component & Amount ( $\mu \mathrm{l}$ \\
\hline FastDigest buffer, 10x & 3 \\
\hline FastDigest EcoRI & 1 \\
\hline FastDigest HindIII & 1 \\
\hline $\begin{array}{l}\text { Purified PCR product, } 20 \mathrm{ng} \mu \mathrm{l}^{-1} \\
\text { (Step } 78 \text { or 101) }\end{array}$ & 10 \\
\hline $\mathrm{ddH}_{2} \mathrm{O}$ & 15 \\
\hline Total volume & 30 \\
\hline
\end{tabular}

110| pUC19 backbone digestion. Set up the digestion reaction as follows and incubate it at $37^{\circ} \mathrm{C}$ for $15 \mathrm{~min}$ :

\begin{tabular}{|c|c|}
\hline Component & Amount ( $\mu \mathrm{l})$ \\
\hline FastDigest buffer, 10x & 3 \\
\hline FastDigest EcoRI & 1 \\
\hline FastDigest HindIII & 1 \\
\hline FastAP alkaline phosphatase & 1 \\
\hline pUC19 vector (200 $\left.\mathrm{ng} \mu \mathrm{l}^{-1}\right)$ & 5 \\
\hline $\mathrm{ddH}_{2} \mathrm{O}$ & 20 \\
\hline Total volume & 30 \\
\hline
\end{tabular}

111 Purify the digestion reactions with the QIAQuick PCR purification kit.

■ PAUSE POINT Purified PCR product can be stored at $-20^{\circ} \mathrm{C}$ overnight. 112| Ligate the digested pUC19 backbone and PCR product at a 1:3 vector:insert ratio and incubate it at room temperature for $15 \mathrm{~min}$. As always, it is important to include a vector-only ligation control.

\begin{tabular}{lc} 
Component & Amount $(\boldsymbol{\mu l})$ \\
\hline Digested pUC19 & $x(50 \mathrm{ng})$ \\
Digested PCR product (insert) & $x(1: 3$ vector to insert molar ratio) \\
T7 ligase & 1 \\
Rapid ligation buffer, $2 x$ & 10 \\
ddH 20 & Up to 20 \\
Total volume & 20 \\
\hline
\end{tabular}

113| Treat the ligation reaction with PlasmidSafe exonuclease to digest any residual linearized DNA. This step is optional but highly recommended.

\begin{tabular}{lc}
\hline Component & Amount $(\boldsymbol{\mu l})$ \\
\hline Ligation reaction from Step 112 & 11 \\
PlasmidSafe buffer, 10× & 1.5 \\
ATP, $10 \mathrm{mM}$ & 1.5 \\
PlasmidSafe exonuclease & 1 \\
Total & 15 \\
\hline
\end{tabular}

114| Transformation of bacteria. Transform the PlasmidSafe-treated plasmid into a competent $E$. coli strain, according to the protocol supplied with the cells. We recommend Stbl3 for quick transformation. Briefly, add $5 \mu$ of the product from Step 113 into $20 \mu \mathrm{l}$ of ice-cold chemically competent Stbl3 cells; incubate the mixture on ice for $10 \mathrm{~min}$, heat-shock it at $42{ }^{\circ} \mathrm{C}$ for $30 \mathrm{~s}$, return it immediately to ice for $2 \mathrm{~min}$, add $100 \mu \mathrm{L}$ of SOC medium and plate it onto LB plates containing $100 \mu \mathrm{mg}^{-1}$ ampicillin. Incubate the mixture overnight at $37^{\circ} \mathrm{C}$.

115| Day 2: inspect the plates for colony growth. Typically, there are no colonies on the negative control plates (ligation of vector only, with no Sanger amplicon insert), and tens to hundreds of colonies on the experimental plates. Pick a minimum of 48 clones to inoculate in $3 \mathrm{ml}$ of LB-ampicillin culture.

? TROUBLESHOOTING

116| Day 3: isolate the plasmid DNA from overnight cultures by using a QIAprep spin miniprep kit.

117| Sanger sequencing. Verify the sequence of each colony by sequencing from the pUC19 backbone using the pUC19-Fwd or pUC19-Rev primer. Reference the sequencing results against the expected genomic sequence to check for the presence of Cas9-induced NHEJ or HDR modifications. Calculate the percentage of editing efficiency as (no. of modified clones)/

(no. of total clones). It is important to pick a reasonable number of clones $(>24)$ to generate an accurate approximation of modification efficiencies.

Deep sequencing and off-target analysis TIMING 2-3 d

118| Designing deep-sequencing primers. Primers for deep sequencing are designed to produce short PCR amplicons, typically in the 100- to 200-bp size range. You can manually design primers by using the NCBI Primer-Blast or generate them with the CRISPR Design Tool (http://tools.genome-engineering.org).

119| Extract genomic DNA from Cas9-targeted cells (Steps 71-74). Normalize QuickExtract genomic DNA to 100-200 ng $\mathrm{nl}^{-1}$ with $\mathrm{ddH}_{2} \mathrm{O}$. 120| Initial library preparation-PCR. By using the primers from Step 118, prepare the initial library preparation PCR (include untransfected sample as negative control):

\begin{tabular}{|c|c|c|}
\hline Component & Amount ( $\mu l)$ & Final concentration \\
\hline Herculase II PCR buffer, $5 x$ & 10 & $1 x$ \\
\hline dNTP, $100 \mathrm{mM}$ (25 mM each) & 1 & $2 \mathrm{mM}$ \\
\hline Fwd primer $(10 \mu \mathrm{M})$ & 1 & $0.2 \mu \mathrm{M}$ \\
\hline Rev primer $(10 \mu \mathrm{M})$ & 1 & $0.2 \mu \mathrm{M}$ \\
\hline Herculase II fusion polymerase & 1 & \\
\hline $\mathrm{MgCl}_{2}(25 \mathrm{mM})$ & 2 & $1 \mathrm{mM}$ \\
\hline DNA template & 1 & $2 \mathrm{ng} \mu \mathrm{l}^{-1}$ \\
\hline $\mathrm{ddH}_{2} \mathrm{O}$ & 33 & \\
\hline Total & 50 & \\
\hline
\end{tabular}

121| Perform the PCR with the following cycling conditions, for no more than 20 amplification cycles:

\begin{tabular}{lccc}
\hline Cycle number & Denature & Anneal & Extend \\
\hline 1 & $95^{\circ} \mathrm{C}, 2 \mathrm{~min}$ & \\
$2-21$ & $95^{\circ} \mathrm{C}, 20 \mathrm{~s}$ & $60^{\circ} \mathrm{C}, 20 \mathrm{~s}$ & $72{ }^{\circ} \mathrm{C}, 15 \mathrm{~s}$ \\
22 & & $72{ }^{\circ} \mathrm{C}, 3 \mathrm{~min}$ \\
\hline
\end{tabular}

122| Run $2-5 \mu \mathrm{l}$ of PCR product on a $1 \%$ (wt/vol) agarose gel to check for single-band product. As with all genomic DNA PCRs, the primers may require additional optimization by adjusting the template concentration, $\mathrm{MgCl}_{2}$ concentration and/or the annealing temperature.

? TROUBLESHOOTING

123| Purify the PCRs by using the QIAQuick PCR purification kit and normalize the eluants to $20 \mathrm{ng}_{\mathrm{Ll}^{-1}}$.

■AUSE POINT Purified PCR product can be stored at $-20^{\circ} \mathrm{C}$ overnight or longer.

124 Nextera XT DNA sample preparation kit. According to the manufacturer's protocol, generate Miseq sequencing-ready libraries with unique bar codes for each sample.

125| Sequence the samples prepared in Step 124 on the Illumina Miseq according to the Illumina user manual.

126| Analyze sequencing data. By using the expected reference genome sequence, perform indel analysis with read alignment programs such as ClustalW (http://www.clustal.org/), Geneious (http://www.geneious.com/) or by simple sequence analysis scripts.

\section*{? TROUBLESHOOTING}

Troubleshooting advice can be found in Table 2. TABLE 2 | Troubleshooting table.

\begin{tabular}{|c|c|c|c|}
\hline Step & Problem & Possible reason & Possible solution \\
\hline $5 A(v)$ & No amplification of sgRNA & $\begin{array}{l}\text { Incorrect template or primer. } \\
\text { Incorrect template or primer } \\
\text { concentration }\end{array}$ & $\begin{array}{l}\text { Titrate U6-template concentration to } 10-50 \mathrm{ng} \text { for a } 50-\mu \\
\text { reaction. Titrate primer concentration to a final } \\
\text { concentration of } 0.1-0.5 \mu \mathrm{M}\end{array}$ \\
\hline $5 B(i x), 115$ & $\begin{array}{l}\text { Colonies growing on } \\
\text { negative control plate }\end{array}$ & $\begin{array}{l}\text { Incomplete digestion of } \\
\text { pSpCas9(BB) or pUC19 plasmid }\end{array}$ & $\begin{array}{l}\text { Increase the amount of restriction enzymes; add } \\
\text { phosphatase treatment to the plasmid digestions to } \\
\text { reduce self-ligation of empty vector }\end{array}$ \\
\hline 5B(xii) & $\begin{array}{l}\text { No sgRNA sequences or } \\
\text { wrong sequences }\end{array}$ & $\begin{array}{l}\text { Ligation failure, incomplete } \\
\text { digestion of cloning plasmid }\end{array}$ & $\begin{array}{l}\text { Screen additional colonies; reanneal sgRNA oligos; titrate } \\
\text { sgRNA oligo concentration during ligation; redigest } \\
\text { pSpCas9(BB) or pUC19 }\end{array}$ \\
\hline 11 & $\begin{array}{l}\text { Low Lipofectamine } \\
\text { transfection efficiency }\end{array}$ & $\begin{array}{l}\text { Incorrect amount or poor } \\
\text { quality of DNA used for } \\
\text { transfection; poorly or unevenly } \\
\text { seeded cells }\end{array}$ & $\begin{array}{l}\text { Use low-passage-number cells (passage number }<20 \text { ) and } \\
\text { do not let cells reach }>90 \% \text { confluence; titrate DNA } \\
\text { (200 to } 500 \mathrm{ng} \text { for } 200,000 \text { cells); add GFP transfection } \\
\text { control; reseed cells evenly at recommended density; } \\
\text { prepare new DNA for transfection }\end{array}$ \\
\hline 28,50 & $\begin{array}{l}\text { Low nucleofection } \\
\text { transfection efficiency }\end{array}$ & $\begin{array}{l}\text { Incorrect amount or poor } \\
\text { quality of DNA used for } \\
\text { transfection; clumpy cells }\end{array}$ & $\begin{array}{l}\text { Use low-passage-number cells (passage number }<20 \text { ); } \\
\text { check cell density; titrate DNA; gently dissociate cells to } \\
\text { single-cell suspension }\end{array}$ \\
\hline 61 & $\begin{array}{l}\text { Clumps or no cells after } \\
\text { FACS }\end{array}$ & $\begin{array}{l}\text { Cells not dissociated well in } \\
\text { Step } 56\end{array}$ & $\begin{array}{l}\text { Filter cells before FACS; dissociate to single cells; } \\
\text { re-suspend in no more than } 10^{7} \text { cells per } \mathrm{ml}\end{array}$ \\
\hline 67 & $\begin{array}{l}\text { Clumps or no cells in } \\
\text { serial dilution }\end{array}$ & $\begin{array}{l}\text { Cells not dissociated well; } \\
\text { incorrect cell number or } \\
\text { concentration }\end{array}$ & $\begin{array}{l}\text { Dissociate to single cells and filter through strainer; } \\
\text { recount cells; check serial dilution to make sure cell } \\
\text { density is at an average of } 0.5 \text { cells per well }\end{array}$ \\
\hline $\begin{array}{l}77,96 \\
100,122\end{array}$ & $\begin{array}{l}\text { Multiple bands or } \\
\text { no amplification in } \\
\text { SURVEYOR, microdeletion, } \\
\text { RFLP analysis, or } \\
\text { deep-sequencing PCR }\end{array}$ & $\begin{array}{l}\text { Inefficient or nonspecific priming; } \\
\text { incorrect template concentration }\end{array}$ & $\begin{array}{l}\text { Titrate } \mathrm{MgCl}_{2}(0-1 \mathrm{mM} \text { final concentration }) \text {; normalize } \\
\text { and titrate template concentration }(50-200 \mathrm{ng} \text { total }) \text {; } \\
\text { titrate annealing temperature }\left(58-62{ }^{\circ} \mathrm{C}\right) \text {; re-design } \\
\text { primers to minimize primer dimer formation or nonspecific } \\
\text { amplification }\end{array}$ \\
\hline 86 & $\begin{array}{l}\text { High SURVEYOR } \\
\text { background in } \\
\text { negative sample; } \\
\text { nonspecific or smeared } \\
\text { bands on SURVEYOR gel; } \\
\text { no SURVEYOR cleavage }\end{array}$ & $\begin{array}{l}\text { Polymorphism within the } \\
\text { examined region or co- } \\
\text { amplification of highly } \\
\text { homologous region; impure input } \\
\text { DNA; input DNA too low or too } \\
\text { high; improper annealing; } \\
\text { overdigestion by SURVEYOR } \\
\text { nuclease; sgRNA for a particular } \\
\text { locus does not mediate cleavage }\end{array}$ & $\begin{array}{l}\text { Redesign primers to different locations in the locus } \\
\text { examined; purify PCR product; normalize total input } \\
\text { DNA to 200-400 ng; reanneal heteroduplex (Step 80); } \\
\text { incubate SURVEYOR digestion at } 42{ }^{\circ} \mathrm{C} \text { for }<60 \text { min; } \\
\text { re-design sgRNAs; sequence verify sgRNA or Cas9 on } \\
\text { pSpCas9(BB) backbone }\end{array}$ \\
\hline
\end{tabular}

\section*{TIMING}

Steps 1-4, design of targeting components (sgRNA and sSODN) and use of the CRISPR Design Tool: $1 \mathrm{~d}$

Step 5A, PCR-based generation of sgRNA expression cassette: $2 \mathrm{~h}$

Step $5 B$, cloning of sgRNA expression vector: $3 \mathrm{~d}$

Steps 6-13, functional validation of sgRNAs: HEK 293FT cell culture and transfections: $3-4 \mathrm{~d}$

Steps 14-29, co-transfection of CRISPR plasmids and HDR templates into HEK 293FT cells (optional): 3-4 d

Steps 30-53, hESC (HUES 9) culture and transfection: 3-4 d

Steps 54-65, isolation of clonal cell lines by FACS: 2-3 h hands-on; 2-3 weeks expansion

Steps 66-70, isolation of clonal cell lines by dilution: 2-3 h hands-on; 2-3 weeks expansion

Steps 71-89, SURVEYOR assay for the assessment of CRISPR cleavage efficiency: 5-6 $\mathrm{h}$

Steps 90-96, detection of genomic microdeletion by PCR: 3-4 h hands-on; 2-3 weeks expansion Steps 97-108, genotyping of HDR-mediated targeted modifications by RFLP analysis: $3-4 \mathrm{~h}$

Steps 109-117, genotyping by Sanger sequencing: 3 d

Steps 118-126, genotyping by next-generation deep sequencing: 2-3 d

\section*{ANTICIPATED RESULTS}

We have compiled a list of most-frequently asked questions from our web-based CRISPR discussion forum (discuss.genomeengineering.org) to clarify points of confusion when applying the CRISPR system (Box 3). CRISPR-Cas can be easily multiplexed to facilitate high-efficiency genome editing in mammalian cells: by using two sgRNAs, we were able to demonstrate simultaneous targeting of the human DYRK1A and GRIN2B loci at efficiencies of 65-68\% for each locus in HEK 293FT cells (Fig. 5b). Likewise, a pair of sgRNAs can be used to mediate microdeletions, such as excision of EMX1 exon 3, which we genotyped by PCR at a clonal level (Fig. 5c). Note that the precise location of exon junctions can vary. We also demonstrate here the use of ssODNs and targeting vector to mediate HDR (Fig. 6a,b) with both WT and the D10A nickase mutant of Cas9 in HEK 293FT and HUES9 cells (Fig. 6c). Note that we have not been able to detect HDR in HUES9 cells by using Cas9n with a sgRNA, which may be due to low efficiency or a potential difference in repair activities in HUES9 cells. The state of ploidy of the cell type of interest may also affect modification efficiency. However, Cas9n may be paired with two sgRNAs to mediate a DSB and stimulate HDR in HUES9 cells.

\section*{Box 3 | Frequently asked questions from the CRISPR Discussion Forum}

The following questions are selected from the CRISPR Discussion Forum (discuss.genome-engineering.org).

Q1: Should I use crRNA + tracrRNA or sgRNA?

Although we initially reported that sgRNAs with a truncated tracrRNA $3^{\prime}$ tail, sgRNA(+48), are less effective for gene targeting in mammalian cells than using crRNA and tracrRNA pairs22, we have optimized the sgRNA architecture by extending its tracrRNA tail. sgRNA $(+85)$, used in pSpCas9 vectors, is able to mediate the highest efficiency of genome modification ${ }^{57}$.

Q2: My 20-bp target site does not start with G; can I replace the first base with G?

For efficient U6 transcription of sgRNA, a G is preferred at the $5^{\prime}$ position, which corresponds with the first base of the 20-bp guide sequence. For guide sequences that do not begin with a $\mathrm{G}$, we recommend adding an additional $\mathrm{G}$ to the $5^{\prime}$ of the guide sequence, resulting in a 21-bp guide sequence (5'-GNNNNNNNNNNNNNNNNNNNN-3', where the $20 \mathrm{Ns}$ correspond to the 20-bp genomic target sequence immediately upstream of the $5^{\prime}$-NGG PAM). The addition of a $5^{\prime} G$ does not alter the specificity of the sgRNA or affect the efficiency of Cas9 cleavage.

Q3: Should I include the PAM sequence in my sgRNA?

No, the PAM sequence should not be included as a part of the sgRNA. The PAM sequence is located in the genomic target site, immediately following the 20-bp target sequence, but it is not a part of the sgRNA. Only the 20-bp target sequence is used to construct the sgRNA. For S. pyogenes Cas9, the PAM sequence is 5'-NGG.

Q4: Which one is more efficient for mediating HDR, WT Cas9 or Cas9 nickase?

WT Cas9 is more efficient at mediating homologous recombination than Cas9 nickase. The efficiency of HDR mediated by Cas9 nickase is highly dependent on the cell type. Although Cas9 nickase can reduce the risk for off-target activity, we suggest testing both WT Cas9 and Cas9 nickase when performing gene targeting experiments.

Q5: Does the CBh promoter in pSpCas9 work in my cell line?

The CBh promoter is an altered version of the CAG promoter ${ }^{69}$, and we have validated it in the following cell lines: HEK 293FT, human ESCS, mouse ESCs, HepG2, HeLa and Neuro 2a. For other cell lines, we recommend testing Cas9 expression by using pSpCas9(BB)-2AGFP and checking for green fluorescence, or staining for Cas9 expression using antibodies against the $3 \times$ FLAG epitope that has been fused to the N-term of Cas9.

Q6: Which method of sgRNA expression is more effective, PCR-generated U6-sgRNA expression cassette or pSpCas9 plasmid carrying the appropriate sgRNA?

We have found that the use of pSpCas9 plasmid often results in higher cleavage efficiency. The use of PCR-generated U6-sgRNA expression cassettes allows easy and rapid comparison of sgRNA efficiencies so that the most optimal sgRNA, in terms of both efficiency and specificity, can be identified before subsequent cloning into pSpCas9.

Q7: Can the HDR template (plasmid or ssODN) contain the sgRNA target site?

We have found that it is possible to achieve HDR by using both plasmids and ssODNs that contain the target site. However, single bases can be silently mutated to prevent cleavage of the templates. Note: Any Supplementary Information and Source Data files are available in the online version of the paper.

ACKNOWLEDGMENTS We thank B. Holmes for help with computational tools. P.D.H. is a James Mills Pierce Fellow and D.A.S. is a National Science Foundation (NSF) pre-doctoral fellow. V.A. is supported by NIH Training Grants T32GM007753 and T32GM008313. This work was supported by an $\mathrm{NIH}$ Director's Pioneer Award (1DP1-MH100706); an NIH Transformative R01 grant (1R01-DK097768); the Keck, McKnight, Damon Runyon, Searle Scholars, Vallee, Merkin, Klingenstein and Simons Foundations; Bob Metcalfe; and Jane Pauley. Reagents are available to the academic community through Addgene and associated protocols; support forums and computational tools are available via the Zhang lab website (http://www.genome-engineering.org/).

AUTHOR CONTRIBUTIONS F.A.R., P.D.H., J.W., D.A.S. and F.Z. designed and performed the experiments. V.A. contributed to the online tool. F.A.R., P.D.H. and F.Z. wrote the manuscript with help from all authors.

COMPETING FINANCIAL INTERESTS The authors declare competing financial interests: details are available in the online version of the paper.

Reprints and permissions information is available online at http://www.nature. com/reprints/index.html.

1. Ding, Q. et al. A TALEN genome-editing system for generating human stem cell-based disease models. Cell Stem Cell 12, 238-251 (2013).

2. Soldner, F. et al. Generation of isogenic pluripotent stem cells differing exclusively at two early onset Parkinson point mutations. Cell 146, 318-331 (2011).

3. Carlson, D.F. et al. Efficient TALEN-mediated gene knockout in livestock. Proc. Natl. Acad. Sci. USA 109, 17382-17387 (2012).

4. Geurts, A.M. et al. Knockout rats via embryo microinjection of zinc-finger nucleases. Science 325, 433-433 (2009).

5. Takasu, Y. et al. Targeted mutagenesis in the silkworm Bombyx mori using zinc finger nuclease mRNA injection. Insect Biochem. Molec. 40, 759-765 $(2010)$

6. Watanabe, T. et al. Non-transgenic genome modifications in a hemimetabolous insect using zinc-finger and TAL effector nucleases. Nat. Commun. 3, 1017 (2012).

7. Porteus, M.H. \& Baltimore, D. Chimeric nucleases stimulate gene targeting in human cells. Science 300, 763 (2003).

8. Miller, J.C. et al. An improved zinc-finger nuclease architecture for highly specific genome editing. Nat. Biotechnol. 25, 778-785 (2007)

9. Sander, J.D. et al. Selection-free zinc-finger-nuclease engineering by context-dependent assembly (CoDA). Nat. Methods 8, 67-69 (2011).

10. Wood, A.J. et al. Targeted genome editing across species using ZFNs and TALENs. Science 333, 307 (2011).

11. Christian, M. et al. Targeting DNA double-strand breaks with TAL effector nucleases. Genetics 186, 757-761 (2010).

12. Zhang, F. et al. Efficient construction of sequence-specific TAL effectors for modulating mammalian transcription. Nat. Biotechnol. 29, 149-153 $(2011)$.

13. Hockemeyer, D. et al. Genetic engineering of human pluripotent cells using TALE nucleases. Nat. Biotechnol. 29, 731-734 (2011).

14. Reyon, D. et al. FLASH assembly of TALENs for high-throughput genome editing. Nat. Biotechnol. 30, 460-465 (2012).

15. Boch, J. et al. Breaking the code of DNA binding specificity of TAL-type III effectors. Science 326, 1509-1512 (2009).

16. Moscou, M.J. \& Bogdanove, A.J. A simple cipher governs DNA recognition by TAL effectors. Science 326, 1501 (2009).

17. Sanjana, N.E. et al. A transcription activator-like effector toolbox for genome engineering. Nat. Protoc. 7, 171-192 (2012).

18. Deveau, H., Garneau, J.E. \& Moineau, S. CRISPR/Cas system and its role in phage-bacteria interactions. Annu. Rev. Microbiol. 64, 475-493 (2010).

19. Horvath, P. \& Barrangou, R. CRISPR/Cas, the immune system of bacteria and archaea. Science 327, 167-170 (2010).

20. Makarova, K.S. et al. Evolution and classification of the CRISPR-Cas systems. Nat. Rev. Microbiol. 9, 467-477 (2011).

21. Bhaya, D., Davison, M. \& Barrangou, R. CRISPR-Cas systems in bacteria and archaea: versatile small RNAs for adaptive defense and regulation. Annu. Rev. Genet. 45, 273-297 (2011).

22. Cong, L. et al. Multiplex genome engineering using CRISPR/Cas systems. Science 339, 819-823 (2013).

23. Mali, P. et al. RNA-guided human genome engineering via Cas9. Science 339, 823-826 (2013). 24. Jinek, M. et al. RNA-programmed genome editing in human cells. eLife 2, e00471 (2013).

25. Cho, S.W., Kim, S., Kim, J.M. \& Kim, J.S. Targeted genome engineering in human cells with the Cas9 RNA-guided endonuclease. Nat. Biotechnol. 31, 230-232(2013).

26. Garneau, J.E. et al. The CRISPR/Cas bacterial immune system cleaves bacteriophage and plasmid DNA. Nature 468, 67-71 (2010).

27. Jinek, M. et al. A programmable dual-RNA-guided DNA endonuclease in adaptive bacterial immunity. Science 337, 816-821 (2012).

28. Gasiunas, G., Barrangou, R., Horvath, P. \& Siksnys, V. Cas9-crRNA ribonucleoprotein complex mediates specific DNA cleavage for adaptive immunity in bacteria. Proc. Natl. Acad. Sci. USA 109, E2579-E2586 $(2012)$

29. Urnov, F.D., Rebar, E.J., Holmes, M.C., Zhang, H.S. \& Gregory, P.D. Genome editing with engineered zinc-finger nucleases. Nat. Rev. Genet. 11, $636-646(2010)$.

30. Hsu, P.D. \& Zhang, F. Dissecting neural function using targeted genome engineering technologies. ACS Chem. Neurosci. 3, 603-610 (2012).

31. Perez, E.E. et al. Establishment of HIV-1 resistance in $\mathrm{CD4}^{+} \mathrm{T}$ cells by genome editing using zinc-finger nucleases. Nat. Biotechnol. 26, 808-816 $(2008)$

32. Chen, F. et al. High-frequency genome editing using SsDNA oligonucleotides with zinc-finger nucleases. Nat. Methods 8, 753-755 (2011).

33. Saleh-Gohari, N. \& Helleday, T. Conservative homologous recombination preferentially repairs DNA double-strand breaks in the $S$ phase of the cell cycle in human cells. Nucleic Acids Res. 32, 3683-3688 (2004).

34. Marraffini, L.A. \& Sontheimer, E.J. CRISPR interference limits horizontal gene transfer in staphylococci by targeting DNA. Science 322, 1843-1845 $(2008)$.

35. Brouns, S.J. et al. Small CRISPR RNAs guide antiviral defense in prokaryotes. Science 321, 960-964 (2008).

36. Barrangou, R. et al. CRISPR provides acquired resistance against viruses in prokaryotes. Science 315, 1709-1712 (2007).

37. Sapranauskas, R. et al. The Streptococcus thermophilus CRISPR/Cas system provides immunity in Escherichia coli. Nucleic Acids Res. 39, 9275-9282 $(2011)$

38. Magadan, A.H., Dupuis, M.E., Villion, M. \& Moineau, S. Cleavage of phage DNA by the Streptococcus thermophilus CRISPR3-Cas system. PLoS ONE 7, e40913 (2012).

39. Zhang, Y. et al. Processing-Independent CRISPR RNAs limit natural transformation in Neisseria meningitidis. Mol. Cell 50, 488-503 $(2013)$

40. Hwang, W.Y. et al. Efficient genome editing in zebrafish using a CRISPR-Cas system. Nat. Biotechnol. 31, 227-229 (2013).

41. Wang, H. et al. One-step generation of mice carrying mutations in multiple genes by CRISPR/Cas-mediated genome engineering. Cell 153, 910-918 (2013).

42. Shen, B. et al. Generation of gene-modified mice via Cas9/RNA-mediated gene targeting. Cell Res. 23, 720-723 (2013).

43. Ran, F.A. et al. Double nicking by RNA-guided CRISPR Cas 9 for enhanced genome editing specificity. Cell 154, 1380-1389 (2013).

44. Qi, L.S. et al. Repurposing CRISPR as an RNA-guided platform for sequence-specific control of gene expression. Cell 152, 1173-1183 (2013).

45. Chang, N. et al. Genome editing with RNA-guided Cas 9 nuclease in zebrafish embryos. Cell Res. 23, 465-472 (2013).

46. Gratz, S.J. et al. Genome engineering of Drosophila with the CRISPR RNA-guided Cas9 nuclease. Genetics 4, 1029-1035 (2013).

47. Friedland, A.E. et al. Heritable genome editing in C. elegans via a CRISPRCas9 system. Nat. Methods 10, 741-743 (2013).

48. Cermak, T. et al. Efficient design and assembly of custom TALEN and other TAL effector-based constructs for DNA targeting. Nucleic Acids Res. 39, e82 (2011).

49. Schmid-Burgk, J.L., Schmidt, T., Kaiser, V., Honing, K. \& Hornung, V. A ligation-independent cloning technique for high-throughput assembly of transcription activator-like effector genes. Nat. Biotechnol. 31, 76-81 $(2013)$

50. Miller, J.C. et al. A TALE nuclease architecture for efficient genome editing. Nat. Biotechnol. 29, 143-148 (2011).

51. Hsu, P.D. et al. DNA targeting specificity of RNA-guided Cas9 nucleases. Nat. Biotechnol. 31, 827-832 (2013).

52. Fu, Y. et al. High-frequency off-target mutagenesis induced by CRISPR-CaS nucleases in human cells. Nature Biotechnol. 31, 822-826 (2013).

53. Tuschl, T. Expanding small RNA interference. Nat. Biotechnol. 20, 446-448 (2002). 54. Smithies, 0., Gregg, R.G., Boggs, S.S., Koralewski, M.A. \& Kucherlapati, R.S. Insertion of DNA sequences into the human chromosomal -globin locus by homologous recombination. Nature 317, 230-234 (1985).

55. Thomas, K.R., Folger, K.R. \& Capecchi, M.R. High frequency targeting of genes to specific sites in the mammalian genome. Cell 44, 419-428 (1986).

56. Hasty, P., Rivera-Perez, J. \& Bradley, A. The length of homology required for gene targeting in embryonic stem cells. Mol. Cell Biol. 11, 5586-5591 $(1991)$

57. Wu, S., Ying, G.X., Wu, Q. \& Capecchi, M.R. A protocol for constructing gene targeting vectors: generating knockout mice for the cadherin family and beyond. Nat. Protoc. 3, 1056-1076 (2008).

58. Elliott, B., Richardson, C., Winderbaum, J., Nickoloff, J.A. \& Jasin, M. Gene conversion tracts from double-strand break repair in mammalian cells. Mol. Cellular Biol. 18, 93-101 (1998). 59. Guschin, D.Y. et al. A rapid and general assay for monitoring endogenous gene modification. Methods Mol. Biol. 649, 247-256 (2010).

60. Loman, N.J. et al. Performance comparison of benchtop high-throughput sequencing platforms. Nat. Biotechnol. 30, 434-439 (2012).

61. Jiang, W., Bikard, D., Cox, D., Zhang, F. \& Marraffini, L.A. RNA-guided editing of bacterial genomes using CRISPR-Cas systems. Nat. Biotechnol. 31, 233-239 (2013).

62. Oliveira, T.Y. et al. Translocation capture sequencing: a method for high throughput mapping of chromosomal rearrangements. J. Immunol. Methods 375, 176-181 (2012).

63. Gray, S.J. et al. Optimizing promoters for recombinant adeno-associated virus-mediated gene expression in the peripheral and central nervous system using self-complementary vectors. Human Gene Ther. 22, $1143-1153(2011)$."""

rwkv = """\title{
RWKV: Reinventing RNNs for the Transformer Era
}

\author{
Bo Peng ${ }^{1 *}$ Eric Alcaide ${ }^{2,3,4 *}$ Quentin Anthony ${ }^{2,5 *}$ \\ Alon Albalak ${ }^{2,6}$ Samuel Arcadinho $^{2,7}$ Huanqi Cao $^{8}$ Xin Cheng $^{9}$ Michael Chung $^{10}$ \\ Matteo Grella ${ }^{11}$ Kranthi Kiran GV ${ }^{12}$ Xuzheng $\mathrm{He}^{2}$ Haowen Hou ${ }^{13}$ Przemysaw Kazienko ${ }^{14}$ \\ Jan Kocoń ${ }^{14}$ Jiaming Kong ${ }^{15}$ Bartłomiej Koptyra ${ }^{14}$ Hayden Lau $^{2}$ Krishna Sri Ipsit Mantri ${ }^{16}$ \\ Ferdinand Mom ${ }^{17,18}$ Atsushi Saito ${ }^{2,19}$ Xiangru Tang ${ }^{20}$ Bolun Wang ${ }^{27}$ Johan S. Wind ${ }^{21}$ Stanisław Woźniak $^{14}$ \\ Ruichong Zhang ${ }^{8}$ Zhenyuan Zhang ${ }^{2}$ Qihang Zhao ${ }^{22,23}$ Peng Zhou ${ }^{27}$ Jian Zhu ${ }^{24}$ Rui-Jie Zhu ${ }^{25,26}$ \\ ${ }^{1}$ RWKV Foundation ${ }^{2}$ EleutherAI ${ }^{3}$ University of Barcelona ${ }^{4}$ Charm Therapeutics ${ }^{5}$ Ohio State University \\ ${ }^{6}$ University of California, Santa Barbara ${ }^{7}$ Zendesk ${ }^{8}$ Tsinghua University ${ }^{9}$ Peking University \\ ${ }^{10}$ Storyteller.io ${ }^{11}$ Crisis $24{ }^{12}$ New York University ${ }^{13}$ National University of Singapore \\ ${ }^{14}$ Wroclaw University of Science and Technology ${ }^{15}$ Databaker Technology Co. Ltd ${ }^{16}$ Purdue University \\ ${ }^{17}$ Criteo AI Lab ${ }^{18}$ Epita ${ }^{19}$ Nextremer Co. Ltd. ${ }^{20}$ Yale University ${ }^{21}$ University of Oslo \\ ${ }^{22}$ University of Science and Technology of China ${ }^{23}$ Kuaishou Technology Co. Ltd \\ ${ }^{24}$ University of British Columbia ${ }^{25}$ University of California, Santa Cruz \\ ${ }^{26}$ University of Electronic Science and Technology of China ${ }^{27}$ RuoxinTech
}

\begin{abstract}
Transformers have revolutionized almost all natural language processing (NLP) tasks but suffer from memory and computational complexity that scales quadratically with sequence length. In contrast, recurrent neural networks (RNNs) exhibit linear scaling in memory and computational requirements but struggle to match the same performance as Transformers due to limitations in parallelization and scalability. We propose a novel model architecture, Receptance Weighted Key Value (RWKV), that combines the efficient parallelizable training of Transformers with the efficient inference of RNNs. Our approach leverages a linear attention mechanism and allows us to formulate the model as either a Transformer or an $\mathrm{RNN}$, which parallelizes computations during training and maintains constant computational and memory complexity during inference, leading to the first non-transformer architecture to be scaled to tens of billions of parameters. Our experiments reveal that RWKV performs on par with similarly sized Transformers, suggesting that future work can leverage this architecture to create more efficient models. This work presents a significant step towards reconciling the trade-offs between computational efficiency and model performance in sequence processing tasks. ${ }^{1}$
\end{abstract}

\section{Introduction}

Deep learning techniques have made significant strides in artificial intelligence, playing a pivotal

* Equal first authorship. Others listed alphabetically.

${ }^{1}$ Code at: https://github.com/BlinkDL/RWKV-LM role in various scientific and industrial applications. These applications often involve complex sequential data processing tasks that include natural language understanding, conversational AI, time-series analysis, and even indirect modalities that can be reframed as sequences, such as images and graphs (Brown et al., 2020; Ismail Fawaz et al., 2019; Wu et al., 2020; Albalak et al., 2022). Predominant among these techniques are RNNs, convolutional neural networks (CNNs), and the Transformer models (Vaswani et al., 2017).

Each of these has distinct drawbacks that restrict their efficiency in certain scenarios. RNNs suffer from the vanishing gradient problem, making them difficult to train for long sequences. Additionally, they cannot be parallelized in the time dimension during training, which restricts their scalability (Hochreiter, 1998; Le and Zuidema, 2016). CNNs, on the other hand, are only adept at capturing local patterns, which limits their capacity to deal with long-range dependencies, crucial to many sequence processing tasks (Bai et al., 2018).

Transformer models emerged as a powerful alternative due to their ability to handle both local and long-range dependencies and their capability for parallelized training (Tay et al., 2022). Recent models such as GPT-3 (Brown et al., 2020), ChatGPT (OpenAI, 2022; Kocoń et al., 2023), GPT-4 (OpenAI, 2023), LLaMA (Touvron et al., 2023), and Chinchilla (Hoffmann et al., 2022) exemplify the capability of this architecture, pushing the frontiers of what's possible in NLP. Despite these significant advancements, the self-attention mechanism inherent to Transformers poses unique challenges, 

\begin{tabular}{lcc}
\hline Model & Time & Space \\
\hline Transformer & $O\left(T^{2} d\right)$ & $O\left(T^{2}+T d\right)$ \\
Reformer & $O(T \log T d)$ & $O(T \log T+T d)$ \\
Linear Transformers & $O\left(T d^{2}\right)$ & $O\left(T d+d^{2}\right)$ \\
Performer & $O\left(T d^{2} \log d\right)$ & $O\left(T d \log d+d^{2} \log d\right)$ \\
AFT-full & $O\left(T^{2} d\right)$ & $O(T d)$ \\
MEGA & $O(c T d)$ & $O(c T d)$ \\
\hline RWKV (ours) & $O(\mathbf{T d})$ & $O(\mathbf{d})$ \\
\hline
\end{tabular}

Table 1: Complexity comparison with different Transformers: Reformer (Kitaev et al., 2020), Linear Transformer (Katharopoulos et al., 2020), Performer (Choromanski et al., 2020), AFT (Zhai et al., 2021), MEGA (Ma et al., 2023). Here $T$ denotes the sequence length, $d$ the feature dimension, and $c$ is MEGA's chunk size of quadratic attention.

primarily due to its quadratic complexity. This complexity renders the architecture computationally expensive and memory-intensive for tasks involving long input sequences or in resource-constrained situations. These limitations have spurred a wealth of research aiming to improve the scaling properties of Transformers, often at the expense of some of the properties that make it so effective (Wang et al., 2020; Zaheer et al., 2020; Dao et al., 2022a).

To tackle these challenges, we introduce the Receptance Weighted Key Value (RWKV) model, a novel architecture that effectively combines the strengths of RNNs and Transformers while circumventing key drawbacks. RWKV is carefully designed to alleviate the memory bottleneck and quadratic scaling associated with Transformers (Katharopoulos et al., 2020) with a more efficient linear scaling, while still preserving the rich, expressive properties that make the Transformer a dominant architecture in the field.

One of the defining characteristics of RWKV is its ability to offer parallelized training and robust scalability, similar to Transformers. Moreover, we have reformulated the attention mechanism in RWKV to introduce a variant of linear attention, eschewing the traditional dot-product token interaction in favor of more effective channeldirected attention. This approach contrasts significantly with the traditional Transformer architecture, where specific token interactions predominantly drive attention. The implementation of linear attention in RWKV is carried out without approximation, which offers a considerable improvement in efficiency and enhances the scalability, see Table 1.

The overarching motivation behind developing RWKV is to bridge the gap between computational efficiency and expressive capacity in neural net- work architectures. It offers a promising and viable solution for handling tasks involving large-scale models with billions of parameters, exhibiting competitive performance at a fraction of the computational cost. Our experimental results suggest that RWKV could be a valuable tool for addressing the ongoing challenges in scaling and deploying AI models across various domains, particularly those involving sequential data processing. Thus, RWKV paves the way for the next generation of more sustainable and computationally efficient AI models for sequence processing tasks.

Our contributions in this paper are as follows:

- We introduce the RWKV network architecture, which combines the advantages of RNNs and Transformers while mitigating their known limitations.

- We propose a new attention mechanism reformulation that results in linear attention, eschewing the quadratic complexity associated with standard Transformer models.

- We conduct a comprehensive series of experiments on benchmark datasets to showcase the performance, efficiency and scaling of RWKV in managing tasks involving large-scale models and long-range dependencies.

- We release pretrained model ranging in size from 169 million to 14 billion parameters trained on the Pile (Gao et al., 2020). ${ }^{2}$

\section{Related Work}

Recently, a number of techniques have been proposed to address the limitations of transformers.

Optimizing Attention Mechanism Many transformer variants ("x-formers") have been introduced to reduce the complexity of transformers (Tay et al., 2022), including sparse attention (Beltagy et al., 2020; Kitaev et al., 2020; Guo et al., 2022), approximating the full attention matrix (Wang et al., 2020; Ma et al., 2021; Choromanski et al., 2020), combining chunked attention with gating (Ma et al., 2023) and other efficient methods (Katharopoulos et al., 2020; Jaegle et al., 2021).

Some recent works like FlashAttention (Dao et al., 2022a) and others (Rabe and Staats, 2022; Jang et al., 2019) share similarities with RWKV's chunked computation scheme. Despite being memory-efficient, their time complexity remains quadratic or contains chunk size as a hidden factor. In contrast, RWKV achieves better space and

${ }^{2} \mathrm{https}: / /$ huggingface.co/RWKV time complexity during inference by formulating a linear attention as an $\mathrm{RNN}$.

Attention Free Models Another line of research replaces the attention mechanism with other modules to scale to long sequences. MLP-Mixer and others (Tolstikhin et al., 2021; Liu et al., 2021) proposed the replacement of attention by MultiLayer Perceptrons (MLPs) in computer vision tasks. The Attention Free Transformer (AFT) (Zhai et al., 2021) replaces dot-product self-attention with a computationally efficient alternative which can be seen as a multi-head attention where each feature dimension corresponds to a head. Inspired by AFT, RWKV takes a similar approach but modifies the interaction weights for simplicity such that it can be transformed into an RNN. In parallel, RNNstyle (Hochreiter and Schmidhuber, 1997; Chung et al., 2014) recursive components have also been modified to increase context length, such as the Recurrent Memory Transformer (Bulatov et al., 2022, 2023) and Linear Recurrent Units (Orvieto et al., 2023). State space models (SSM) like S4 (Gu et al., 2022) and its variants (Dao et al., 2022b; Poli et al., 2023) are also proposed.

Notably, Quasi-Recurrent neural network (QRNN) (Bradbury et al., 2017) uses both convolutional layers and recurrent pooling functions across timesteps and channels. While QRNN utilizes convolutional filters with fixed sizes, RWKV employs a time-mixing module as an attention mechanism with time-decaying factors. Different from the element-wise pooling in QRNN, RWKV includes a parametrized channel-mixing module (see the green blocks in Fig.1c) that is parallelizable.

\section{Background}

Here we briefly review the fundamentals of RNNs and Transformers.

\subsection{Recurrent Neural Networks (RNNs)}

Popular RNN architectures such as LSTM (Hochreiter and Schmidhuber, 1997) and GRU (Chung et al., 2014) are characterized by the following formulation (shown for LSTM, others can be reasoned similarly):

$$
\begin{aligned}
f_{t} & =\sigma_{g}\left(W_{f} x_{t}+U_{f} h_{t-1}+b_{f}\right), \\
i_{t} & =\sigma_{g}\left(W_{i} x_{t}+U_{i} h_{t-1}+b_{i}\right), \\
o_{t} & =\sigma_{g}\left(W_{o} x_{t}+U_{o} h_{t-1}+b_{o}\right), \\
\tilde{c}_{t} & =\sigma_{c}\left(W_{c} x_{t}+U_{c} h_{t-1}+b_{c}\right), \\
c_{t} & =f_{t} \odot c_{t-1}+i_{t} \odot \tilde{c}_{t}, \\
h_{t} & =o_{t} \odot \sigma_{h}\left(c_{t}\right) .
\end{aligned}
$$

The data flow of RNNs is shown in Fig. 1a. Although RNNs can be factored into two linear blocks ( $W$ and $U$ ) and an RNN-specific block (1)-(6), as noted by Bradbury et al. (2017), the data dependency relying on previous time steps prohibits parallelizing these typical RNNs.

\subsection{Transformers and AFT}

Introduced by Vaswani et al. (2017), Transformers are a class of neural networks that have become the dominant architecture for several NLP tasks. Instead of operating on sequences step-by-step like RNNs, Transformers rely on attention mechanisms to capture relationships between all input and all output tokens:

$$
\operatorname{Attn}(Q, K, V)=\operatorname{softmax}\left(Q K^{\top}\right) V
$$

where the multi-headness and scaling factor $\frac{1}{\sqrt{d_{k}}}$ is omitted for convenience. The core $Q K^{\top}$ multiplication is an ensemble of pairwise attention scores between each token in a sequence, which can be decomposed as vector operations:

$$
\operatorname{Attn}(Q, K, V)_{t}=\frac{\sum_{i=1}^{T} e^{q_{t}^{\top} k_{i}} v_{i}}{\sum_{i=1}^{T} e^{q_{t}^{\top} k_{i}}}
$$

In AFT (Zhai et al., 2021), this is alternately formulated as

$$
\operatorname{Attn}^{+}(W, K, V)_{t}=\frac{\sum_{i=1}^{t} e^{w_{t, i}+k_{i}} v_{i}}{\sum_{i=1}^{t} e^{w_{t, i}+k_{i}}}
$$

where $\left\{w_{t, i}\right\} \in R^{T \times T}$ is the learned pair-wise position biases, and each $w_{t, i}$ is a scalar.

Inspired by AFT, we let each $w_{t, i}$ in RWKV be a channel-wise time decay vector multiplied by the relative position, traced backwards from current time as it decays:

$$
w_{t, i}=-(t-i) w
$$

where $w \in\left(R_{\geq 0}\right)^{d}$, with $d$ the number of channels. We require $w$ to be non-negative to ensure that $e^{w_{t, i}} \leq 1$ and the per-channel weights decay backwards in time. 

![](https://cdn.mathpix.com/cropped/2023_06_04_9aef8213db464f91daabg-04.jpg?height=229&width=551&top_left_y=271&top_left_x=227)

(a) RNN

![](https://cdn.mathpix.com/cropped/2023_06_04_9aef8213db464f91daabg-04.jpg?height=231&width=485&top_left_y=273&top_left_x=797)

(b) QuasiRNN (Bradbury et al., 2017)

![](https://cdn.mathpix.com/cropped/2023_06_04_9aef8213db464f91daabg-04.jpg?height=229&width=528&top_left_y=271&top_left_x=1301)

(c) $\mathrm{RWKV}$

Figure 1: Computation structure of the RWKV in comparison to QRNN and RNN (Vanilla, LSTM, GRU, etc) architectures. Color codes: orange indicates time-mixing, convolutions or matrix multiplications, and the continuous block indicates that these computations can proceed simultaneously; blue signifies parameterless functions that operate concurrently along the channel or feature dimension (element-wise). Green indicates channel-mixing.

\section{The Receptance Weighted Key Value (RWKV) Model}

The RWKV architecture derives its name from the four primary model elements used in the timemixing and channel-mixing blocks:

- $R$ : Receptance vector acting as the acceptance of past information.

- $W$ : Weight is the positional weight decay vector. A trainable model parameter.

- $K$ : Key is a vector analogous to $K$ in traditional attention.

- $V$ : Value is a vector analogous to $V$ in traditional attention.

Interactions between the main elements for every timestep are multiplicative, as illustrated in Fig. 2

\subsection{High-Level Summary}

The RWKV architecture is comprised of a series of stacked residual blocks, each formed by a timemixing and a channel-mixing sub-blocks with recurrent structures.

The recurrence is formulated both as a linear interpolation between the current input and the input at the previous time step (a technique we refer to as time-shift mixing or token shift, indicated by the diagonal lines in Fig. 3), which can be adjusted independently for every linear projection of the input embedding (e.g., $R, K, V$ in time-mixing, and $R$, $K$ in channel-mixing), and as the time-dependent update of the $W K V$ which is formalized in equation 14. The $W K V$ computation is similar to AFT (Zhai et al., 2021), but $W$ is now a channel-wise vector multiplied by relative position rather than a pairwise matrix in AFT. We also introduce a vector $U$ for separately attending to the current token in order to compensate for potential degeneration of $W$ (see Appendix G for more details).

![](https://cdn.mathpix.com/cropped/2023_06_04_9aef8213db464f91daabg-04.jpg?height=839&width=694&top_left_y=800&top_left_x=1089)

Figure 2: RWKV block elements (left) and RWKV residual block with a final head for language modeling (right) architectures.

![](https://cdn.mathpix.com/cropped/2023_06_04_9aef8213db464f91daabg-04.jpg?height=782&width=762&top_left_y=1796&top_left_x=1064)

Figure 3: RWKV architecture for language modelling. The time-mixing block is given by:

$$
\begin{aligned}
r_{t} & =W_{r} \cdot\left(\mu_{r} x_{t}+\left(1-\mu_{r}\right) x_{t-1}\right) \\
k_{t} & =W_{k} \cdot\left(\mu_{k} x_{t}+\left(1-\mu_{k}\right) x_{t-1}\right) \\
v_{t} & =W_{v} \cdot\left(\mu_{v} x_{t}+\left(1-\mu_{v}\right) x_{t-1}\right) \\
w k v_{t} & =\frac{\sum_{i=1}^{t-1} e^{-(t-1-i) w+k_{i}} v_{i}+e^{u+k_{t}} v_{t}}{\sum_{i=1}^{t-1} e^{-(t-1-i) w+k_{i}}+e^{u+k_{t}}} \\
o_{t} & =W_{o} \cdot\left(\sigma\left(r_{t}\right) \odot w k v_{t}\right)
\end{aligned}
$$

where the $W K V$ computation, $w k v_{t}$, plays the role of $\operatorname{Attn}(Q, K, V)$ in Transformers without incurring a quadratic cost as interactions are between scalars. Intuitively, as time $t$ increases, the vector $o_{t}$ is dependent on a long history, represented by the summation of an increasing number of terms. For the target position $t$, RWKV performs a weighted summation in the positional interval of $[1, t]$, and then multiplies with the receptance $\sigma(r)$. Therefore, interactions are multiplicative inside a given timestep and summed over different timesteps.

Further, the channel-mixing block is given by:

$$
\begin{aligned}
r_{t} & =W_{r} \cdot\left(\mu_{r} x_{t}+\left(1-\mu_{r}\right) x_{t-1}\right), \\
k_{t} & =W_{k} \cdot\left(\mu_{k} x_{t}+\left(1-\mu_{k}\right) x_{t-1}\right), \\
o_{t} & =\sigma\left(r_{t}\right) \odot\left(W_{v} \cdot \max \left(k_{t}, 0\right)^{2}\right),
\end{aligned}
$$

where we adopt squared ReLU activation (So et al., 2021). Note that in both time-mixing and channelmixing, by taking the sigmoid of the receptance, we're intuitively using it as a "forget gate" to eliminate unnecessary historical information.

\subsection{Transformer-like Parallelization}

RWKV can be efficiently parallelized in what we call a time-parallel mode, reminiscent of Transformers. The time complexity of processing a batch of sequences in a single layer is $O\left(B T d^{2}\right)$, which mainly consists of matrix multiplications $W_{\square}, \square \in\{r, k, v, o\}$ (assuming $B$ sequences, $T$ maximum tokens and $d$ channels). Meanwhile, updating attention scores $w k v_{t}$ requires a serial scan (see Appendix B for more detail) and has complexity $O(B T d)$.

The matrix multiplications can be parallelized akin to $W^{\square}, \square \in\{Q, K, V, O\}$ in typical Transformers. The element-wise $W K V$ computation is time-dependent, but can be readily parallelized along the other two dimensions (Lei et al., 2018) ${ }^{3}$.

${ }^{3}$ If the sequence is very long, more sophisticated methods such as Martin and Cundy (2017) that parallelize over sequence length could be used. Additionally, token shift is implemented as a simple offset in the temporal dimension at each block using PyTorch (Paszke et al., 2019) library as $\mathrm{nn}$. $\operatorname{ZeroPad} 2 \mathrm{~d}((\theta, \theta, 1,-1))$.

\subsection{RNN-like Sequential Decoding}

It is common in recurrent networks to use output at state $t$ as input at state $t+1$. This is especially evident in the autoregressive decoding inference of a language model, requiring each token to be computed before fed into the next step, making it possible for RWKV to take advantage of its RNNlike structure, referred to as time-sequential mode. In such circumstances, RWKV can be conveniently formulated recursively for decoding during inference, as shown in Appendix B, which leverages the advantage that each output token is dependent only on the latest state, which is of constant size, irrespective of the sequence length.

It then behaves as an RNN decoder, yielding constant speed and memory footprint with respect to the sequence length, enabling the processing of longer sequences more efficiently. In contrast, selfattention typically requires a $\mathrm{KV}$ cache growing linearly with respect to the sequence length, resulting in degraded efficiency and increasing memory footprint and time as the sequence grows longer.

\subsection{Software Implementation}

RWKV is originally implemented using the Pytorch Deep Learning Library (Paszke et al., 2019) and a custom CUDA kernel for the $W K V$ computation explained in 4.7. Although RWKV is a general recurrent network, its current implementation focuses in the task of language modeling (RWKV-LM). The model architecture is comprised of an embedding layer, for which we follow the setup described in Section 4.7 and several identical residual blocks applied sequentially as seen in Fig. 2 and 3 following the principles outlined in Section 4.6. After the last block, a simple output projection head composed by a LayerNorm (Ba et al., 2016) and a linear projection is used to obtain the logits to be used in the next-token prediction task and calculate the cross entropy loss during training. Both the embeddings generated after the last residual block and the logits could also be used later for downstream NLP tasks. Training is performed in time-parallel mode (Section 4.2) while autoregressive inference and a potential chat interface ${ }^{4}$

${ }^{4} \mathrm{https}: / /$ github.com/BlinkDL/ChatRWKV leverage the time-sequential mode (Section 4.3).

\subsection{Gradient Stability and Layer Stacking}

The RWKV architecture has been designed as a fusion of both Transformers and RNNs, offering the advantage of stable gradients and deeper architectures of Transformers compared to traditional RNNs while being efficient in inference.

Previous work has sought to tackle the problem of gradient stability in RNNs with a variety of techniques including using non-saturated activation functions (Chandar et al., 2019), gating mechanism (Gu et al., 2019), gradient clipping (Pascanu et al., 2012), and adding constraints (Kanai et al., 2017; Miller and Hardt, 2018). While these techniques have seen little success, RWKV avoids the problem inherently by utilizing softmax in conjunction with RNN-style updates.

The RWKV model features a single-step process for updating attention-like scores, which includes a time-dependent softmax operation that helps numerical stability and guards against vanishing gradients (for rigorous proof, see Appendix F). Intuitively, this operation ensures the gradient is propagated along the most relevant path. Layer normalization (Ba et al., 2016) is another key aspect of the architecture which enhances the training dynamics of deep neural networks by stabilizing gradients, addressing both vanishing and exploding gradient issues.

These design elements not only contribute to the RWKV architecture's stability and learning capabilities but enable the stacking of multiple layers in a manner that surpasses the capabilities of any existing RNN. In doing so, the model is able to capture more complex patterns across various levels of abstraction (see also Appendix G).

\subsection{Harnessing Temporal Structure for Sequential Data Processing}

RWKV captures and propagates sequential information through the combination of three mechanisms: recurrence, time decay and token shift.

The recurrence in the time-mixing block of RWKV is the basis for the model's capacity to capture intricate relationships between sequence elements and to propagate locality information through time.

The time decay mechanism $\left(e^{-w}\right.$ and $e^{u}$ in equation 14), maintains sensitivity to the positional relationship between sequence elements. By gradually diminishing the influence of past information over time, the model preserves a sense of temporal locality and progression, which is essential for sequential processing. This treatment of positional information in sequential data exhibits similarities to the Attention with Linear Biases (ALiBi) model (Press et al., 2022), where the linear biases facilitate input length extrapolation. In this context, the RWKV architecture can be perceived as a trainable version of $\mathrm{ALiBi}$, seamlessly incorporating positional information without the necessity for explicit encoding. It can also be seen as an extension of the gated convolution introduced in Zhai et al. (2021) to the full sequence length until a given step.

The token shift or time-shift mixing, or (diagonal arrows in Figure 3), also contributes to the model's adaptation to sequential data. By linearly interpolating between the current input and the previous time step input, the model naturally aggregates and gates information in the input channels. The overall structure of time-shift mixing bears resemblance to the causal convolution with no dilations in WaveNet (van den Oord et al., 2016), which is a classical architecture used for forecasting time series data.

\subsection{Additional Optimizations}

Custom Kernels To address inefficiencies in the $W K V$ computation due to the sequential nature of the task when using standard deep learning frameworks, we implement a custom CUDA kernel so as to launch a single compute kernel in training accelerators. All other parts of the model are matrix multiplications and point-wise operations that can already be efficiently parallelized.

FFN with R gate Prior research (Tolstikhin et al., 2021; Liu et al., 2021; Yu et al., 2022) suggests that self-attention may not be as essential in Transformer-based vision tasks as previously thought. Although it provided us with some insights, replacing self-attention entirely in natural language tasks could be too drastic. In our study, we partially dismantle the attention mechanism by replacing the fixed $Q K V$ formula with $K V$ and introducing a new time-decaying factor $W$. This approach enables us to incorporate token and channelmixing components akin to MLP-mixer (Tolstikhin et al., 2021) and a gating unit $R$ similar to gMLP (Liu et al., 2021), which enhance the performance of our RWKV model.

Small Init Embedding During the initial stage of training a transformer model (Vaswani et al., 2017), we observe that the embedding matrix undergoes slow changes, which pose a challenge for the model to deviate from its initial noisy embedding state. To mitigate this issue, we propose an approach that involves initializing the embedding matrix with small values and subsequently applying an additional LayerNorm operation. By implementing this technique, we accelerate and stabilize the training process, enabling the training of deep architectures with post-LN components. The effectiveness of this approach is demonstrated in Figure 8 , where it is shown to facilitate improved convergence by allowing the model to quickly transition away from the initially small embedding. This is achieved through small changes following a single step, which in turn lead to substantial alterations in directions and subsequently significant changes after the LayerNorm operation.

Custom Initialization Building on principles from previous works (He et al., 2016; Jumper et al., 2021), we initialize parameters to values as similar as possible to an identity mapping while breaking symmetry so there is a clean information path. Most weights are initialized to zero. No biases are used for linear layers. Specific formulas are given in Appendix D. We find the choice of initialization to be significant in convergence speed and quality (see Appendix E).

\section{Evaluations}

In this section, we focus on evaluating to answer the following questions:

- RQ1: Is RWKV competitive against quadratic transformer architectures with equal number of parameters and training tokens?

- RQ2: When increasing the number of parameters, does RWKV remain competitive against quadratic transformer architectures?

- RQ3: Does increasing parameters of RWKV yield better language modeling loss, when RWKV models are trained for context lengths that most open-sourced quadratic transformers cannot efficiently process?

Addressing RQ1 and RQ2, from Fig. 4, we can see that RWKV is very competitive on six benchmarks (Winogrande, PIQA, ARC-C, ARC-E, LAMBADA, and SciQ) against major open source quadratic complexity transformer models: Pythia (Biderman et al., 2023), OPT (Zhang et al., 2022) and BLOOM (Scao et al., 2022). RWKV even outperforms Pythia and GPT-Neo (Black et al., 2022) in four tasks: PIQA, OBQA, ARC-E, and COPA (See details in Appendix H). For RQ3, Fig. 5 shows that increasing context length leads to lower test loss on the Pile, an indication that RWKV can make effective use of long contextual information.

\section{Inference Experiments}

We benchmark inference requirements according to size and family. Specifically, we evaluate text generation speed and memory requirements on a typical compute platforms including CPU (x86) and GPU (NVIDIA A100 80GB). For all our experiments we use float 32 precision. We include all model parameters in parameter count, including both embedding and non-embedding layers. Performance under different quantization setups is left to further work. See Appendix I for more results.

![](https://cdn.mathpix.com/cropped/2023_06_04_9aef8213db464f91daabg-07.jpg?height=588&width=780&top_left_y=1211&top_left_x=1049)

Figure 6: Cumulative time during text generation for different LLMs.

Additionally, we carried out comparative studies on RWKV-4 and ChatGPT / GPT-4, see Appendix J. They revealed that RWKV-4 is very sensitive to prompt engineering. When the prompts were adjusted from the ones used for GPT to more suitable for RWKV, the F1-measure performance increased even from $44.2 \%$ to $74.8 \%$.

\section{Future Work}

There are several promising directions for future work on the RWKV architecture:

- Increasing model expressivity with enhanced time-decay formulations and exploring initial model states while maintaining efficiency. 

![](https://cdn.mathpix.com/cropped/2023_06_04_9aef8213db464f91daabg-08.jpg?height=314&width=437&top_left_y=317&top_left_x=318)

(a) Winogrande

![](https://cdn.mathpix.com/cropped/2023_06_04_9aef8213db464f91daabg-08.jpg?height=317&width=437&top_left_y=750&top_left_x=313)

(d) ARC-Easy

![](https://cdn.mathpix.com/cropped/2023_06_04_9aef8213db464f91daabg-08.jpg?height=314&width=437&top_left_y=317&top_left_x=792)

(b) PIQA

![](https://cdn.mathpix.com/cropped/2023_06_04_9aef8213db464f91daabg-08.jpg?height=317&width=437&top_left_y=750&top_left_x=792)

(e) LAMBADA

![](https://cdn.mathpix.com/cropped/2023_06_04_9aef8213db464f91daabg-08.jpg?height=312&width=434&top_left_y=318&top_left_x=1276)

(c) ARC-Challenge

![](https://cdn.mathpix.com/cropped/2023_06_04_9aef8213db464f91daabg-08.jpg?height=320&width=437&top_left_y=751&top_left_x=1272)

(f) SciQ

Figure 4: Zero-Shot Performance: The horizontal axis is a number of parameters and the vertical axis is accuracy.

![](https://cdn.mathpix.com/cropped/2023_06_04_9aef8213db464f91daabg-08.jpg?height=343&width=740&top_left_y=1276&top_left_x=247)

Figure 5: Increasing context length contributes to lower test loss on the Pile (Gao et al., 2020).

- Further improving RWKV computational efficiency by applying parallel scan in the $w k v_{t}$ step to reduce the computational cost to $O(B \log (T) d)$.

- Investigating the application of RWKV to encoder-decoder architectures and potential replacement of cross-attention mechanism. This could have applicability seq 2 seq or multimodal settings, enhancing efficiency both in training and inference.

- Leveraging RWKV's state (or context) for interpretability, predictability in sequence data and safety. Manipulating the hidden state could also guide behavior and allow greater customizability through prompt tuning.

- Exploring fine-tuned models in specific settings for enhanced interaction with humans (Ouyang et al., 2022). Particularly interest- ing would be the performance under different datasets and specific use cases.

- Adapting parameter-efficient fine-tuning methods such as LoRA (Hu et al., 2022) and characterizing behavior under different quantization schemes for the proposed architecture

\section{Conclusions}

We introduced RWKV, a new approach to RNN models exploiting the potential of time-based mixing components. RWKV introduces several key strategies which allow it to capture locality and long-range dependencies, while addressing limitations of current architectures by: (1) replacing the quadratic $\mathrm{QK}$ attention by a scalar formulation with linear cost, (2) reformulating recurrence and sequential inductive biases to unlock efficient training parallelization and efficient inference, and (3) enhancing training dynamics using custom initializations.

We benchmark the proposed architecture in a wide variety of NLP tasks and show comparable performance to SoTA with reduced cost. Further experiments on expressivity, interpretability, and scaling showcase the model capabilities and draw parallels in behavior between RWKV and other LLMs.

RWKV opens a new door to scalable and efficient architectures to model complex relation- ships in sequential data. While many alternatives to Transformers have been proposed with similar claims, ours is the first to back up those claims with pretrained models with tens of billions of parameters.

\section{Limitations}

While our proposed RWKV model has demonstrated promising results regarding training and memory efficiency during inference, some limitations should be acknowledged and addressed in future work. First, the linear attention of RWKV leads to significant efficiency gains but still, it may also limit the model's performance on tasks that require recalling minutiae information over very long contexts. This is due to the funneling of information through a single vector representation over many time steps, compared with the full information maintained by the quadratic attention of standard Transformers. In other words, the model's recurrent architecture inherently limits its ability to "look back" at previous tokens, as opposed to traditional self-attention mechanisms. While learned time decay helps prevent the loss of information, it is mechanistically limited compared to full selfattention.

Another limitation of this work is the increased importance of prompt engineering in comparison to standard Transformer models. The linear attention mechanism used in RWKV limits the information from the prompt that will be carried over to the model's continuation. As a result, carefully designed prompts may be even more crucial for the model to perform well on tasks.

\section{Acknowledgements}

We acknowledge EleutherAI and StabilityAI for compute access and technical support in development of RWKV. We also acknowledge the members of the RWKV Discord server for their help and work on further extending the applicability of RWKV to different domains. Finally, we thank Stella Biderman for feedback on the paper.

\section{References}

Alon Albalak, Yi-Lin Tuan, Pegah Jandaghi, Connor Pryor, Luke Yoffe, Deepak Ramachandran, Lise Getoor, Jay Pujara, and William Yang Wang. 2022 FETA: A benchmark for few-sample task transfer in open-domain dialogue. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 10936-10953, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. 2016. Layer normalization.

Shaojie Bai, J. Zico Kolter, and Vladlen Koltun. 2018. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling.

Francesco Barbieri, Jose Camacho-Collados, Luis Espinosa Anke, and Leonardo Neves. 2020. TweetEval: Unified benchmark and comparative evaluation for tweet classification. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 1644-1650, Online. Association for Computational Linguistics.

Iz Beltagy, Matthew E. Peters, and Arman Cohan. 2020. Longformer: The long-document transformer. $\operatorname{arXiv}: 2004.05150$.

Stella Biderman, Hailey Schoelkopf, Quentin Anthony, Herbie Bradley, Kyle O'Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. 2023. Pythia: A suite for analyzing large language models across training and scaling. arXiv preprint $\operatorname{arXiv} 22304.01373$.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. 2020. Piqa: Reasoning about physical commonsense in natural language. In Thirty-Fourth AAAI Conference on Artificial Intelligence.

Sid Black, Leo Gao, Phil Wang, Connor Leahy, and Stella Biderman. 2022. Gpt-neo: Large scale autoregressive language modeling with mesh-tensorflow, 2021. URL: https://doi. org/l0.5281/zenodo, 5297715.

James Bradbury, Stephen Merity, Caiming Xiong, and Richard Socher. 2017. Quasi-recurrent neural networks. In ICLR.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877-1901.

Aydar Bulatov, Yuri Kuratov, and Mikhail S. Burtsev. 2023. Scaling transformer to $1 \mathrm{~m}$ tokens and beyond with rmt.

Aydar Bulatov, Yury Kuratov, and Mikhail Burtsev. 2022. Recurrent memory transformer. Advances in Neural Information Processing Systems, 35:1107911091. A. P. Sarath Chandar, Chinnadhurai Sankar, Eugene Vorontsov, Samira Ebrahimi Kahou, and Yoshua Bengio. 2019. Towards non-saturating recurrent units for modelling long-term dependencies. In AAAI Conference on Artificial Intelligence.

Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy Colwell, and Adrian Weller. 2020. Rethinking attention with performers.

Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. 2014. Empirical evaluation of gated recurrent neural networks on sequence modeling. In NIPS 2014 Deep Learning and Representation Learning Workshop.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai 2 reasoning challenge. In arXiv:1803.05457.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. In arXiv, volume $\mathrm{abs} / 2110.14168$.

Tri Dao, Daniel Y Fu, Stefano Ermon, Atri Rudra, and Christopher Re. 2022a. Flashattention: Fast and memory-efficient exact attention with IO-awareness In Advances in Neural Information Processing Systems.

Tri Dao, Daniel Y Fu, Khaled K Saab, Armin W Thomas, Atri Rudra, and Christopher Ré. 2022b. Hungry hungry hippos: Towards language modeling with state space models. arXiv preprint $\operatorname{arXiv}: 2212.14052$.

Dorottya Demszky, Dana Movshovitz-Attias, Jeongwoo Ko, Alan S. Cowen, Gaurav Nemade, and Sujith Ravi. 2020. Goemotions: A dataset of finegrained emotions. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020, pages 4040-4054. Association for Computational Linguistics.

Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. 2020. The pile: An $800 \mathrm{gb}$ dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027.

Albert Gu, Karan Goel, and Christopher Ré. 2022. Efficiently modeling long sequences with structured state spaces. In The International Conference on Learning Representations (ICLR).

Albert Gu, Çaglar Gülçehre, Tom Le Paine, Matthew W. Hoffman, and Razvan Pascanu. 2019. Improving the gating mechanism of recurrent neural networks. ArXiv, abs/1910.09890. Mandy Guo, Joshua Ainslie, David C Uthus, Santiago Ontanon, Jianmo Ni, Yun-Hsuan Sung, and Yinfei Yang. 2022. Longt5: Efficient text-to-text transformer for long sequences. In Findings of the Association for Computational Linguistics: NAACL 2022, pages 724-736.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Identity mappings in deep residual networks.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. In International Conference on Learning Representations.

Sepp Hochreiter. 1998. The vanishing gradient problem during learning recurrent neural nets and problem solutions. International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems, $6(02): 107-116$.

Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long short-term memory. Neural Computation, $9(8): 1735-1780$.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent Sifre. 2022. Training compute-optimal large language models.

Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Hassan Ismail Fawaz, Germain Forestier, Jonathan Weber, Lhassane Idoumghar, and Pierre-Alain Muller. 2019. Deep learning for time series classification: a review. Data mining and knowledge discovery, 33(4):917-963.

Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. 2021. Perceiver: General perception with iterative attention. In International conference on machine learning, pages 4651-4664. PMLR.

Hanhwi Jang, Joonsung Kim, Jae-Eon Jo, Jaewon Lee, and Jangwoo Kim. 2019. Mnnfast: A fast and scalable system architecture for memory-augmented neural networks. In Proceedings of the 46th International Symposium on Computer Architecture, pages 250-263.

Matt Gardner Johannes Welbl Nelson F. Liu. 2017. Crowdsourcing multiple choice science questions. In DOI:10.18653/v1/W17-4413. Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. 2017. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. In $A C L$.

John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Žídek, Anna Potapenko, and et al. 2021. Highly accurate protein structure prediction with alphafold. Nature, 596(7873):583-589.

Sekitoshi Kanai, Yasuhiro Fujiwara, and Sotetsu Iwamura. 2017. Preventing gradient explosions in gated recurrent units. In NIPS.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361.

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. 2020. Transformers are rnns: Fast autoregressive transformers with linear attention. In International Conference on Machine Learning, pages 5156-5165. PMLR.

Nikita Kitaev, L. Kaiser, and Anselm Levskaya. 2020. Reformer: The efficient transformer. ArXiv, $\mathrm{abs} / 2001.04451$.

Jan Kocoń, Igor Cichecki, Oliwier Kaszyca, Mateusz Kochanek, Dominika Szydło, Joanna Baran, Julita Bielaniewicz, Marcin Gruza, Arkadiusz Janz, Kamil Kanclerz, Anna Kocoń, Bartłomiej Koptyra, Wiktoria Mieleszczenko-Kowszewicz, Piotr Miłkowski, Marcin Oleksy, Maciej Piasecki, Łukasz Radliński, Konrad Wojtasik, Stanisław Woźniak, and Przemysław Kazienko. 2023. Chatgpt: Jack of all trades, master of none.

Jan Kocoń, Piotr Miłkowski, and Monika ZaśkoZielińska. 2019. Multi-level sentiment analysis of polemo 2.0: Extended corpus of multi-domain consumer reviews. In Proceedings of the $23 \mathrm{rd}$ Conference on Computational Natural Language Learning (CoNLL), pages 980-991.

Phong Le and Willem Zuidema. 2016. Quantifying the vanishing gradient and long distance dependency problem in recursive neural networks and recursive lstms. In Proceedings of the 1st Workshop on Representation Learning for NLP, pages 87-93.

Tao Lei, Yu Zhang, Sida I. Wang, Hui Dai, and Yoav Artzi. 2018. Simple recurrent units for highly parallelizable recurrence. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 4470-4481, Brussels, Belgium. Association for Computational Linguistics.

Hanxiao Liu, Zihang Dai, David R. So, and Quoc V. Le. 2021. Pay attention to mlps. Xuezhe Ma, Xiang Kong, Sinong Wang, Chunting Zhou, Jonathan May, Hao Ma, and Luke Zettlemoyer. 2021. Luna: Linear unified nested attention. Advances in Neural Information Processing Systems, $34: 2441-2453$.

Xuezhe Ma, Chunting Zhou, Xiang Kong, Junxian He, Liangke Gui, Graham Neubig, Jonathan May, and Luke Zettlemoyer. 2023. Mega: Moving average equipped gated attention. In ICLR.

Eric Martin and Chris Cundy. 2017. Parallelizing linear recurrent neural nets over sequence length. ArXiv, abs/1709.04057.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. 2022. Locating and editing factual associations in GPT. Advances in Neural Information Processing Systems, 36.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. Can a suit of armor conduct electricity? a new dataset for open book question answering. In $E M N L P$.

John Miller and Moritz Hardt. 2018. Stable recurrent models. arXiv: Learning.

Nasrin Mostafazadeh, Nathanael Chambers, Xiaodong He, Devi Parikh, Dhruv Batra, Lucy Vanderwende, Pushmeet Kohli, and James Allen. 2016. A corpus and cloze evaluation for deeper understanding of commonsense stories. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 839-849.

OpenAI. 2022. Introducing chatgpt. https://openai . com/blog/chatgpt.

OpenAI. 2023. Gpt-4 technical report.

Antonio Orvieto, Samuel L Smith, Albert Gu, Anushan Fernando, Caglar Gulcehre, Razvan Pascanu, and Soham De. 2023. Resurrecting recurrent neural networks for long sequences. arXiv preprint $\operatorname{arXiv}: 2303.06349$.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow in structions with human feedback.

Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc Quan Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernandez. 2016. The LAMBADA dataset: Word prediction requiring a broad discourse context. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1525-1534, Berlin, Germany. Association for Computational Linguistics. Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio. 2012. On the difficulty of training recurrent neural networks. In International Conference on Machine Learning.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. 2019. Pytorch: An imperative style, high-performance deep learning library.

Michael Poli, Stefano Massaroli, Eric Nguyen, Daniel Y Fu, Tri Dao, Stephen Baccus, Yoshua Bengio, Stefano Ermon, and Christopher Ré. 2023. Hyena hierarchy: Towards larger convolutional language models. arXiv preprint arXiv:2302.10866.

Ofir Press, Noah A. Smith, and Mike Lewis. 2022. Train short, test long: Attention with linear biases enables input length extrapolation. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022.

Ilan Price, Jordan Gifford-Moore, Jory Flemming, Saul Musker, Maayan Roichman, Guillaume Sylvain, Nithum Thain, Lucas Dixon, and Jeffrey Sorensen. 2020. Six attributes of unhealthy conversations. In Proceedings of the Fourth Workshop on Online Abuse and Harms, pages 114-124, Online. Association for Computational Linguistics.

Markus N. Rabe and Charles Staats. 2022. Selfattention does not need $o\left(n^{2}\right)$ memory.

Melissa Roemmele, Cosmin Adrian Bejan, , and Andrew S. Gordon. 2018. Choice of plausible alternatives: An evaluation of commonsense causal reasoning. In $A A A I$.

Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. 2022. Bloom: A 176bparameter open-access multilingual language model. arXiv preprint arXiv:2211.05100.

Ramsha Siddiqui. 2019. SARCASMANIA: Sarcasm Exposed! http://www.kaggle.com/rmsharks4/ sarcasmania-dataset. [Online; accessed 02February-2023].

David R. So, Wojciech Manke, Hanxiao Liu, Zihang Dai, Noam Shazeer, and Quoc V. Le. 2021. Primer: Searching for efficient transformers for language modeling. CoRR, abs/2109.08668.

Yi Tay, Dara Bahri, Donald Metzler, Da-Cheng Juan, Zhe Zhao, and Che Zheng. 2020. Synthesizer: Rethinking self-attention in transformer models.

Yi Tay, Mostafa Dehghani, Dara Bahri, and Donald Metzler. 2022. Efficient transformers: A survey. ACM Computing Surveys, 55(6):1-28. Ilya O. Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner, Daniel Keysers, Jakob Uszkoreit, Mario Lucic, and Alexey Dosovitskiy. 2021. Mlp-mixer: An all-mlp architecture for vision. CoRR, abs/2105.01601.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023 Llama: Open and efficient foundation language models.

Aäron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew W. Senior, and Koray Kavukcuoglu. 2016. Wavenet: A generative model for raw audio. ArXiv, abs/1609.03499.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.

David Vilares and Carlos Gómez-Rodríguez. 2019. Head-qa: A healthcare dataset for complex reasoning. In $A C L$.

Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. 2019. Superglue: A stickier benchmark for general-purpose language understanding systems. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. 2018. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, pages 353-355, Brussels, Belgium Association for Computational Linguistics.

Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, and Hao Ma. 2020. Linformer: Self-attention with linear complexity.

Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, and S Yu Philip. 2020. A comprehensive survey on graph neural networks. IEEE transactions on neural networks and learning systems, 32(1):4-24.

Ellery Wulczyn, Nithum Thain, and Lucas Dixon. 2017. Ex machina: Personal attacks seen at scale. In Proceedings of the 26th International Conference on World Wide Web, WWW 2017, Perth, Australia, April 3-7, 2017, pages 1391-1399. ACM. Weihao Yu, Mi Luo, Pan Zhou, Chenyang Si, Yichen Zhou, Xinchao Wang, Jiashi Feng, and Shuicheng Yan. 2022. Metaformer is actually what you need for vision.

Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. 2020. Big bird: Transformers for longer sequences. Advances in Neural Information Processing Systems, 33.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. Hellaswag: Can a machine really finish your sentence? In $A C L$.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2020. Winogrande: An adversarial winograd schema challenge at scale. In $A C L$.

Shuangfei Zhai, Walter Talbott, Nitish Srivastava, Chen Huang, Hanlin Goh, Ruixiang Zhang, and Josh Susskind. 2021. An attention free transformer.

Sheng Zhang, Xiaodong Liu, Jingjing Liu, Jianfeng Gao, Kevin Duh, and Benjamin Van Durme. 2018 Record: Bridging the gap between human and machine commonsense reading comprehension. In arXiv: 1810.12885

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068.

\section{A Author Contributions}

Bo Peng Original RWKV idea, original code, performance optimizations, original experiments, and trained RWKV models from $0.1 \mathrm{~B}$ to $14 \mathrm{~B}$.

Eric Alcaide Manuscript (initial draft sections 1, 2; sections 4, 7 and 8; revision and proofreading; final version ). Figures $(2,3,4,7)$. Experiments section 6. Appendices D, I. Contributions to Appendix K.

Quentin Anthony Led writing the paper. Manuscript (initial draft sections 1, 2, 3; revision and proofreading; final version).

Zhenyuan Zhang Manuscript (revision and proofreading) Figure 3. Experiments Appendix G. Contributions to Appendices B and K.

Kranthi Kiran GV Manuscript (sections 2 and 5; contributions to section 3; revision and proofreading). Tables 3 and 4. Appendix C. Xiangru Tang Manuscript (sections 2 and 3; contributions to abstract; revision and proofreading). Contributions to Appendix K.

Matteo Grella Manuscript (sections 4.5, 4.6, 8; contributions to sections 1,7 and 9 ; proofreading and revision). Contributions to Appendix B.

Ferdinand Mom Manuscript (contributions to section 1, 2, 4.3, 4.6; proofreading and revision). Contributions to Appendix B.

Atsushi Saito Manuscript (sections 3 and 5; contributions to section 2). Figures 1a, 1b, 1c. Contributions to Appendix $\mathrm{H}$

Krishna Sri Ipsit Mantri Figure 4

Rui-Jie Zhu Tables 1 and 5. Experiments for table 5

Peng Zhou Contributions to Table 5 .

Qihang Zhao Manuscript (proofreading and revision). Contributions to Table 5.

Xuzheng He Manuscript (contributions to section 3; proofreading and revision). Contributions to Figures 1, 7. Appendix G. Contributions to appendix $F$.

Hayden Lau Manuscript (contributions to section 1; proofreading and revision). Contributions to Appendix K.

Michael Chung Manuscript (contributions to section 4.6; proofreading and revision).

Haowen Hou Figure 8. Appendix E

Jiaming Kong Manuscript (revision and proofreading). Appendix $F$.

Johan S. Wind RWKV performance optimizations (CUDA), Contributions to Appendix C.

Jian Zhu Manuscript (section 2; proofreading and revision). Figures 3 and 5 .

Huanqi Cao Manuscript (contributions to 4.2 and 4.3; proofreading and revision). Experiments for Appendix G.

Samuel Arcadinho Contributions to Figures 6, 10, and 11. Contributions to Appendix I.

Xin Cheng Manuscript (proofreading and revision). Contributions to Appendix K, H. Alon Albalak Manuscript (abstract and sections 1,$9 ;$ proofreading and revision).

Jan Kocon Manuscript (sections 1; proofreading and revision). Contributions to Appendix J.

Przemysław Kazienko Manuscript (section 6; proofreading and revision). Contributions Appendix J.

Ruichong Zhang Manuscript (proofreading and revision); Contributions to Figure 5 and Appendix $\mathrm{K}$.

Stanisław Woźniak Appendix J.

Bartlomiej Koptyra Contributions to Appendix J.

\section{B Time-Mixing Block as an RNN Cell}

As stated in 4.3, the RWKV time-mixing block can be formulated as an RNN, as the $W K V$ computation can be written in such a recursive form:

$$
\begin{aligned}
a_{0}, b_{0} & =0, \\
w k v_{t} & =\frac{a_{t-1}+e^{u+k_{t}} v_{t}}{b_{t-1}+e^{u+k_{t}}}, \\
a_{t} & =e^{-w} a_{t-1}+e^{k_{t}} v_{t}, \\
b_{t} & =e^{-w} b_{t-1}+e^{k_{t}} .
\end{aligned}
$$

The dataflow of the RNN-like time-mixing is shown in Fig. 7, where the hidden states $h$ is the numerator-denominator tuple $(a, b)$.

![](https://cdn.mathpix.com/cropped/2023_06_04_9aef8213db464f91daabg-14.jpg?height=431&width=605&top_left_y=1829&top_left_x=294)

Figure 7: RWKV time-mixing block formulated as an RNN cell. Color codes: yellow $(\mu)$ denotes the token shift, red (1) denotes the denominator, blue (2) denotes the numerator, pink (3) denotes the fraction computations in 14. $h$ denotes the numerator-denominator tuple $(a, b)$.

To avoid overflow in calculating $e^{k_{t}}$, a numerical trick is used in the official implementation. Note that

$$
\begin{aligned}
a_{1} & =e^{-w} a_{0}+e^{k_{0}} v_{0}=e^{k_{0}} v_{0}, \\
b_{1} & =e^{-w} b_{0}+e^{k_{0}}=e^{k_{0}},
\end{aligned}
$$

and we set $a_{1}^{\prime}=v_{0}, b_{1}^{\prime}=1, p_{0}=k_{0}$, where $p_{t-1}$ stores the shared exponents of $a_{t}$ and $b_{t}$. Now the above recursion can be converted into a numerical safe version, for each time step $t>1$ :

$$
\begin{aligned}
q & :=\max \left(p_{t-1}, u+k_{t}\right), \\
a_{t}^{*} & =e^{p_{t-1}-q} a_{t-1}^{\prime}+e^{u+k_{t}-q} v_{t}, \\
b_{t}^{*} & =e^{p_{t-1}-q} b_{t-1}^{\prime}+e^{u+k_{t}-q}, \\
w k v_{t} & =\frac{a_{t}^{*}}{b_{t}^{*}} .
\end{aligned}
$$

The update to $a_{t}^{\prime}, b_{t}^{\prime}$ and their shared exponent are also carried out in similar fashion:

$$
\begin{aligned}
q & :=\max \left(p_{t-1}-w, k_{t}\right), \\
a_{t}^{\prime} & =e^{p_{t-1}-w-q} a_{t-1}^{\prime}+e^{k_{t}-q} v_{t}, \\
b_{t}^{\prime} & =e^{p_{t-1}-w-q} b_{t-1}^{\prime}+e^{k_{t-q}}, \\
p_{t} & =q .
\end{aligned}
$$

\section{Parameter and FLOP Count for the RWKV Models}

The following section provides an overview of the different RWKV model architectures along with their respective parameter and FLOP counts in Table 2 .

\begin{tabular}{ccccc}
\hline Name & Layers & Model Dimension & Parameters & FLOPs per token \\
\hline $169 \mathrm{M}$ & 12 & 768 & $1.693 \times 10^{8}$ & $2.613 \times 10^{8}$ \\
$430 \mathrm{M}$ & 24 & 1024 & $4.304 \times 10^{8}$ & $7.573 \times 10^{8}$ \\
$1.5 \mathrm{~B}$ & 24 & 2048 & $1.515 \times 10^{9}$ & $2.823 \times 10^{9}$ \\
3B & 32 & 2560 & $2.985 \times 10^{9}$ & $5.710 \times 10^{9}$ \\
7B & 32 & 4096 & $7.393 \times 10^{9}$ & $1.437 \times 10^{10}$ \\
$14 \mathrm{~B}$ & 40 & 5120 & $1.415 \times 10^{10}$ & $2.778 \times 10^{10}$ \\
\hline
\end{tabular}

Table 2: RWKV model architectures and associated FLOP counts

The number of parameters for each model is computed using the formula: \#parameters $=$ $2 V D+13 D^{2} L+D(11 L+4)$ where $V=50277$ is the vocabulary size, $D$ represents the Model Dimension and $L$ corresponds to the number of layers.

FLOPs is for a forward pass for one token. It was calculated as $6\left(V D+13 D^{2} L\right)$, which is the twice (add and multiply) the number of parameters in linear layers. The backwards pass FLOPs can be approximated as twice that of the forward pass. So the total is $6\left(V D+13 D^{2} L\right)$ per token for training ( $3 \mathrm{x}$ fw FLOPs). It is noteworthy that FLOPs are independent of the context length, unlike regular transformers. The FLOP approximations in this paper are in line with the methodology used by Kaplan et al. (2020).

Alternative approximations for FLOPs include doubling the parameters which yields similar results within $2 \%$ for $14 \mathrm{~B}$ and a $30 \%$ discrepancy for $169 \mathrm{M}$ variant. Another approximation is based on the number of non-embedding parameters multiplied by 2 . This gives $2\left(V D+13 D^{2} L+D(11 L+\right.$ 4)) resulting in $1.6 \%$ more FLOPs for $14 \mathrm{~B}$ model and $8 \%$ more FLOPs for $169 \mathrm{M}$ model.

\section{Parameter initializations}

We describe the specific parameter initializations below and motivate the design choices. Parameters belonging to residual blocks are often adjusted by layer depth and total number of layers. Let \# denote the vocabulary size, $s$ denote the embedding dimension, $d$ denote the hidden size (we use $d=4 s), L$ the number of layers, $l$ the layer index (from 0 to L-1), we use the following initializations:

- Embeddings are initialized to $\mathcal{U}( \pm 1 e-4)$ as explained in 4.7

- For the channel-mixing blocks (11), $\mu_{k_{i}}$ and $\mu_{r_{i}}$ are initialized to $\left(\frac{i}{s}\right)^{1-\frac{l}{L}}$

- For the time-mixing blocks (16), initializations are $\mu_{k_{i}}=\left(\frac{i}{s}\right)^{1-\frac{l}{L}}, \mu_{v_{i}}=\left(\frac{i}{s}\right)^{1-\frac{l}{L}}+\frac{0.3 l}{L-1}$ and $\mu_{r_{i}}=0.5\left(\frac{i}{s}\right)^{1-\frac{l}{L}}$

- $w_{i}$ (14), also known as "time decay", is initialized to $-5+8 \cdot\left(\frac{i}{d-1}\right)^{0.7+\frac{1.3 l}{L-1}}$. Intuitively, it is the discount factor applied to previous tokens over time.

- $u_{i}$ (14), also known as "bonus", is set to $0.5(((i+1) \bmod 3)-1)+\log 0.3$. It is the special weighting applied to the current token in equation 14. The alternating zigzag pattern initially creates subtle variations in the tensor elements, which are intended to help the model treat different dimensions of the embedding distinctively.

- $W_{o}$ (15) (time-mixing) and $W_{v}$ (channelmixing) are initialized to $\mathcal{N}\left(0, \sqrt{\frac{d}{s}}=2\right)$

- All $W_{r}, W_{k}, W_{v}$ weights are initialized to 0 so the model can start learning from the beginning without noisy signals. - All LayerNorm weights start from 1 and biases from 0.

\section{E Small Init Embedding}

This section presents experimental validation of small initialization embedding. The experimental setup is as follows. In the baseline configuration, the parameters are initialized using a normal distribution with a mean of 0.0 and a standard deviation of 0.02 , which is a commonly used initialization method in models like BERT and GPT. On the other hand, in the small initialization of the embedding (small init emb) experiment, the parameters are initialized using a uniform distribution with a range of 1e-4, which is slightly different from RWKV where a normal distribution with a standard deviation of 1e-4 is used. However, this difference is negligible and does not affect our conclusions. The experiments were conducted with a batch size of 400 . As depicted in the figure 8 , the loss curve for the small init emb exhibits a faster rate of decrease and convergence compared to the traditional initialization using a normal distribution.

![](https://cdn.mathpix.com/cropped/2023_06_04_9aef8213db464f91daabg-15.jpg?height=551&width=716&top_left_y=1438&top_left_x=1087)

Figure 8: Effect of small initialization embedding.

\section{F Gradient Stability in RWKV}

In this section, we present a mathematical description of the gradient stability property in RWKV, focusing specifically on the time-mixing block. By gradient stability we mean that if the inputs $x_{t}$ are bounded and the model parameters are fixed, then the gradients with respect to $W_{k}$ and $W_{v}$ are uniformly bounded for all $T$ (thus not exploding). Consequently, we can control the amount each $x_{t}$ contributes to the gradient at $T$ in a naturally decaying fashion by the weight decay mechanism $w$ (thus not vanishing unless desired).

First, we make the simplification that there are no token shifts, this will not affect the final conclusion. In this scenario, $w k v_{T}$ can be written as

$$
w k v_{T}=\frac{\sum_{t=1}^{T} K_{t}^{e} v_{t}}{\sum_{t=1}^{T} K_{t}^{e}}=\mathrm{E}\left(v_{t}\right)=\frac{\mathrm{S}\left(v_{t}\right)}{\mathrm{S}(1)}
$$

where

$$
\begin{gathered}
v_{t}=W_{v} x_{t}, \quad \frac{\partial\left(v_{t}\right)_{i}}{\partial\left(W_{v}\right)_{i, j}}=\left(x_{t}\right)_{j}, \\
K_{t}^{e}=e^{W_{k} x_{t}+w_{T, t}}, \quad \frac{\partial\left(K_{t}^{e}\right)_{i}}{\partial\left(W_{k}\right)_{i, j}}=\left(x_{t}\right)_{j}\left(K_{t}^{e}\right)_{i},
\end{gathered}
$$

and $\mathrm{S}(\cdot)$ and $\mathrm{E}(\cdot)$ are shorthand for denoting sums and averages over weights $K_{t}^{e}$.

The loss function at position $T$ can be written as

$$
L_{T}=l\left(f\left(w k v_{T}\right), y_{T}\right)
$$

Because $w k v_{T}$ relates to $\left(W_{k}\right)_{i, j}$ and $\left(W_{v}\right)_{i, j}$ only through the $i$-th channel $\left(w k v_{T}\right)_{i}$, we have

$$
\frac{\partial L_{T}}{\partial\left(W_{v}\right)_{i, j}}=\frac{\partial L_{T}}{\partial\left(w k v_{T}\right)_{i}} \frac{\partial\left(w k v_{T}\right)_{i}}{\partial\left(W_{v}\right)_{i, j}}
$$

The first part of above equation contains trivial operations like output layers, and other layers of time-mixing, which can be proven inductively. The second part of above equation can be bounded as

$$
\begin{aligned}
\left|\frac{\partial\left(w k v_{T}\right)_{i}}{\partial\left(W_{v}\right)_{i, j}}\right| & =\left|\frac{\partial \mathrm{E}_{i}\left[\left(v_{t}\right)_{i}\right]}{\partial\left(W_{v}\right)_{i, j}}\right| \\
& =\left|\mathrm{E}_{i}\left[\left(x_{t}\right)_{j}\right]\right| \leq \max _{t}\left|\left(x_{t}\right)_{j}\right|,
\end{aligned}
$$

which is irrelevant to $T$. Similarly,

$$
\begin{aligned}
\frac{\partial\left(w k v_{T}\right)_{i}}{\partial\left(W_{k}\right)_{i, j}} & =\partial \frac{\mathrm{S}_{i}\left[\left(v_{t}\right)_{i}\right]}{\mathrm{S}_{i}(1)} / \partial\left(W_{k}\right)_{i, j} \\
& =\frac{\mathrm{S}_{i}\left[\left(x_{t}\right)_{j}\left(v_{t}\right)_{i}\right]}{\mathrm{S}_{i}(1)}-\frac{\mathrm{S}_{i}\left[\left(x_{t}\right)_{j}\right] \mathrm{S}_{i}\left[\left(v_{t}\right)_{i}\right]}{\mathrm{S}_{i}(1)^{2}} \\
& =\mathrm{E}_{i}\left[\left(x_{t}\right)_{j}\left(v_{t}\right)_{i}\right]-\mathrm{E}_{i}\left[\left(x_{t}\right)_{j}\right] \mathrm{E}_{i}\left[\left(v_{t}\right)_{i}\right] \\
& =\operatorname{cov}_{i}\left(\left(x_{t}\right)_{j},\left(v_{t}\right)_{i}\right)
\end{aligned}
$$

can also be bounded. Note that $w k v$ 's softmax operation contains at least two non-zero terms ( $u$ and $w$ ), so the above "covariance" will not degenerate into 0 .
![](https://cdn.mathpix.com/cropped/2023_06_04_9aef8213db464f91daabg-16.jpg?height=934&width=760&top_left_y=248&top_left_x=1065)

Figure 9: Model behavior visualizations of the RWKV model.

\section{G Model Behavior Visualization}

In Figure 9, we present visualizations of some behavior of the RWKV model.

The top plot illustrates the time decays $\left(e^{-w}\right)$ in each layer of the RWKV-169M model, sorted along the channel axis. Notably, several decays in the last layers are very close or equal to one, implying that certain information is preserved and propagated throughout the model's temporal context. Meanwhile, many decays in the initial layer are close to zero, which corresponds to local operations in $w k v$ (14), likely to be associated with tasks such as text parsing or lexical analysis. (Note that the local operations in $w k v$ is due to the extra parameter $u$, when $e^{-w}$ is degenerated into 0 .) These patterns of time decays are partly learned, but also come from parameter initialization as it speeds up training.

The bottom plot shows the information retrieval and propagation path in the RWKV-430M model. The experiment follows the causal trace method introduced by Meng et al. (2022), where we

1. Run the model once, and record all states and activation of each layer during the computation;

2. Corrupt the input embeddings of the subject using noise ("The Eiffel Tower" in this example); 3. Restore the states and activation of a certain layer at a certain token during the computation, and record the log-probability of the model outputting the correct answer ("Paris").

Unlike transformers, RWKV relies on recursive propagation of information in the time dimension. In this case, the fact that "the Eiffel Tower is located in Paris" is retrieved in layer 4. It is then passed down to the subsequent layers. In layer 20, mostly, the information is propagated through time until reaching where it is needed. Finally, it is passed down to the last layer for outputting the answer.

\section{H Evaluation Details}

The results for following tasks are in Table 3 and 4 .

Tasks:

- LAMBADA (Paperno et al., 2016). A benchmark dataset that evaluates the model's contextual reasoning and language comprehension abilities by presenting context-target pairs, where the objective is to predict the most probable target token.

- PIQA (Bisk et al., 2020). A benchmark for the task of physical common sense reasoning, which consists of a binary choice task that can be better understood as a set of two pairs, namely (Goal, Solution).

- HellaSwag (Zellers et al., 2019) A novel benchmark for commonsense Natural Language Inference (NLI) which is build by adversarial filtering against transformer models.

- Winogrande (Zellers et al., 2020) A dataset designed to evaluate the acquisition of common sense reasoning by neural language models, aiming to determine whether we are accurately assessing the true capabilities of machine common sense.

- StoryCloze (Mostafazadeh et al., 2016) A benchmark to present a novel approach to assess comprehension of narratives, narrative generation, and script acquisition, focusing on commonsense reasoning.

- ARC Challenge (Clark et al., 2018) A dataset designed for multiple-choice question answering, encompassing science exam questions ranging from third grade to ninth grade.

- ARC Easy An easy subset of ARC.

- HeadQA (Vilares and Gómez-Rodríguez, 2019) A benchmark consisting of graduatelevel questions encompassing various fields such as medicine, nursing, biology, chemistry, psychology, and pharmacology.

- OpenBookQA (Mihaylov et al., 2018) A QA dataset to evaluate human comprehension of a subject by incorporating open book facts, scientific knowledge, and perceptual common sense, drawing inspiration from open book exams.

- SciQ (Johannes Welbl Nelson F. Liu, 2017) A multiple-choice QA dataset which was created using an innovative approach to gather well-crafted multiple-choice questions that are focused on a specific domain.

- TriviaQA (Joshi et al., 2017) A QA-IR dataset which is constituted of triples of questions, answers, supporting evidence, and independently collected evidence documents, with an average of six documents per question for reliable sources.

- ReCoRD (Zhang et al., 2018) A benchmark for evaluating commonsense reasoning in reading comprehension by generating queries from CNN/Daily Mail news articles and requiring text span answers from corresponding summarizing passages.

- COPA (Roemmele et al., 2018) A dataset to evaluate achievement in open-domain commonsense causal reasoning.

- MMMLU (Hendrycks et al., 2021) A multitask dataset for 57 tasks containing elementary mathematics, US history, computer science, law, etc.

\section{Inference results}

Figures 10 and 11 illustrate, respectively, the results on time (s) and memory (RAM, VRAM) requirements for LLM inference in float32 precision. We benchmark the following model families and sizes:

- RWKV: 169m, 430m, 1.4b, 3b, 7b, 14b

- Bloom (Scao et al., 2022): 560m, 1b, 3b

- OPT (Zhang et al., 2022): 125m, 350m, 1.3b, $2.7 b, 6.7 b, 13 b$

- GPT-Neo (Black et al., 2022): 125m, 1.3b, $2.7 \mathrm{~b}$

- Pythia (Biderman et al., 2023): 160m, 410m, $1.4 b, 2.8 b, 6.7 b, 12 b$

Missing models in are due to Out Of Memory (OOM) errors. A comparison at 512 tokens is shown in Figure 11 as some large transformer models produced an OOM when inferencing longer se- 

\begin{tabular}{|c|c|c|c|c|c|c|c|c|}
\hline Model & $\begin{array}{l}\text { Params } \\
\text { B }\end{array}$ & $\begin{array}{l}\text { PIQA } \\
\text { acc }\end{array}$ & $\begin{array}{l}\text { StoryCloze } \\
\text { acc }\end{array}$ & $\begin{array}{l}\text { HellaSwag } \\
\text { acc_norm }\end{array}$ & $\begin{array}{l}\text { WinoGrande } \\
\text { acc }\end{array}$ & $\begin{array}{l}\text { ARC-e } \\
\text { acc }\end{array}$ & $\begin{array}{l}\text { ARC-c } \\
\text { acc_norm }\end{array}$ & $\begin{array}{l}\text { OBQA } \\
\text { acc_norm }\end{array}$ \\
\hline RWKV-4 & 0.17 & 65.07 & 58.79 & 32.26 & 50.83 & 47.47 & 24.15 & 29.60 \\
\hline Pythia & 0.16 & 62.68 & 58.47 & 31.63 & 52.01 & 45.12 & 23.81 & 29.20 \\
\hline GPT-Neo & 0.16 & 63.06 & 58.26 & 30.42 & 50.43 & 43.73 & 23.12 & 26.20 \\
\hline RWKV-4 & 0.43 & 67.52 & 63.87 & 40.90 & 51.14 & 52.86 & 25.17 & 32.40 \\
\hline Pythia & 0.40 & 66.70 & 62.64 & 39.10 & 53.35 & 50.38 & 25.77 & 30.00 \\
\hline GPT-Neo & 0.40 & 65.07 & 61.04 & 37.64 & 51.14 & 48.91 & 25.34 & 30.60 \\
\hline RWKV-4 & 1.5 & 72.36 & 68.73 & 52.48 & 54.62 & 60.48 & 29.44 & 34.00 \\
\hline Pythia & 1.4 & 71.11 & 67.66 & 50.82 & 56.51 & 57.74 & 28.58 & 30.80 \\
\hline GPT-Neo & 1.4 & 71.16 & 67.72 & 48.94 & 54.93 & 56.19 & 25.85 & 33.60 \\
\hline RWKV-4 & 3.0 & 74.16 & $\mathbf{7 0 . 7 1}$ & 59.89 & 59.59 & 65.19 & 33.11 & 37.00 \\
\hline Pythia & 2.8 & 73.83 & 70.71 & 59.46 & 61.25 & 62.84 & 32.25 & 35.20 \\
\hline GPT-Neo & 2.8 & 72.14 & 69.54 & 55.82 & 57.62 & 61.07 & 30.20 & 33.20 \\
\hline RWKV-4 & 7.4 & 76.06 & 73.44 & 65.51 & 61.01 & 67.80 & 37.46 & 40.20 \\
\hline Pythia & 6.9 & 74.54 & 72.96 & 63.92 & 61.01 & 66.79 & 35.07 & 38.00 \\
\hline GPT-J & 6.1 & 75.41 & 74.02 & 66.25 & 64.09 & 66.92 & 36.60 & 38.20 \\
\hline RWKV-4 & 14.2 & $\mathbf{7 7 . 4 8}$ & 76.06 & 70.65 & 63.85 & 70.24 & 38.99 & 41.80 \\
\hline GPT-level $^{*}$ & 14.2 & 76.49 & 74.97 & 68.72 & 65.14 & 70.77 & 37.99 & 39.27 \\
\hline Pythia (c.f.) & 11.8 & 75.90 & 74.40 & 67.38 & 64.72 & 69.82 & 36.77 & 38.80 \\
\hline GPT-NeoX (c.f.) & 20.6 & 77.69 & 76.11 & 71.42 & 65.98 & 72.69 & 40.44 & 40.20 \\
\hline
\end{tabular}

Table 3: Zero-Shot Performance of the model on Common Sense Reasoning Tasks. ${ }^{*}$ Interpolation of Pythia and GPT-Neo models

\begin{tabular}{lllllllll}
\hline Model & $\begin{array}{l}\text { Params } \\
\text { B }\end{array}$ & $\begin{array}{l}\text { LAMBADA } \\
\text { ppl }\end{array}$ & $\begin{array}{l}\text { LAMBADA } \\
\text { acc }\end{array}$ & $\begin{array}{l}\text { headQA } \\
\text { acc_norm }\end{array}$ & $\begin{array}{l}\text { sciq } \\
\text { acc }\end{array}$ & $\begin{array}{l}\text { triviaQA } \\
\text { acc }\end{array}$ & $\begin{array}{l}\text { ReCoRD } \\
\text { em }\end{array}$ & $\begin{array}{l}\text { COPA } \\
\text { acc }\end{array}$ \\
\hline RWKV-4 & 0.17 & $\mathbf{2 9 . 3 3}$ & 32.99 & 25.78 & $\mathbf{7 7 . 5 0}$ & 1.26 & 62.03 & $\mathbf{6 6 . 0 0}$ \\
Pythia & 0.16 & 24.38 & $\mathbf{3 8 . 9 7}$ & $\mathbf{2 5 . 8 2}$ & 76.50 & $\mathbf{1 . 3 1}$ & $\mathbf{6 6 . 3 2}$ & 62.00 \\
GPT-Neo & 0.16 & 30.27 & 37.36 & 25.16 & 76.60 & 1.18 & 64.92 & 64.00 \\
\hline RWKV-4 & 0.43 & 13.04 & 45.16 & $\mathbf{2 7 . 3 2}$ & 80.30 & $\mathbf{2 . 3 5}$ & 70.48 & 65.00 \\
Pythia & 0.40 & $\mathbf{1 1 . 5 8}$ & $\mathbf{5 0 . 4 4}$ & 25.09 & $\mathbf{8 1 . 5 0}$ & 2.03 & $\mathbf{7 5 . 0 5}$ & $\mathbf{6 7 . 0 0}$ \\
GPT-Neo & 0.40 & 13.88 & 47.29 & 26.00 & 81.10 & 1.38 & 73.79 & 65.00 \\
\hline RWKV-4 & 1.5 & 7.04 & 56.43 & $\mathbf{2 7 . 6 4}$ & 85.00 & $\mathbf{5 . 6 5}$ & 76.97 & $\mathbf{7 7 . 0 0}$ \\
Pythia & 1.4 & $\mathbf{6 . 5 8}$ & $\mathbf{6 0 . 4 3}$ & 27.02 & $\mathbf{8 5 . 5 0}$ & 5.52 & $\mathbf{8 1 . 4 3}$ & 73.00 \\
GPT-Neo & 1.4 & 7.5 & 57.25 & 27.86 & 86.00 & 5.24 & 80.62 & 69.00 \\
\hline RWKV-4 & 3.0 & 5.25 & 63.96 & 28.45 & 86.50 & $\mathbf{1 1 . 6 8}$ & 80.87 & $\mathbf{8 2 . 0 0}$ \\
Pythia & 2.8 & $\mathbf{4 . 9 3}$ & $\mathbf{6 5 . 3 6}$ & $\mathbf{2 8 . 9 6}$ & $\mathbf{8 7 . 7 0}$ & 9.63 & 85.10 & 77.00 \\
GPT-Neo & 2.8 & 5.63 & 62.22 & 27.17 & 89.30 & 4.82 & $\mathbf{8 3 . 8 0}$ & 80.00 \\
\hline RWKV-4 & 7.4 & 4.38 & 67.18 & $\mathbf{3 1 . 2 2}$ & 88.80 & $\mathbf{1 8 . 3 0}$ & 83.68 & $\mathbf{8 5 . 0 0}$ \\
Pythia & 6.9 & 4.3 & $\mathbf{6 7 . 9 8}$ & 28.59 & 90.00 & 15.42 & 86.44 & $\mathbf{8 5 . 0 0}$ \\
GPT-J & 6.1 & $\mathbf{4 . 1}$ & 68.31 & 28.67 & $\mathbf{9 1 . 5 0}$ & 16.74 & $\mathbf{8 7 . 7 1}$ & 83.00 \\
\hline RWKV-4 & 14.2 & $\mathbf{3 . 8 6}$ & 70.83 & $\mathbf{3 2 . 6 4}$ & 90.40 & $\mathbf{2 4 . 5 8}$ & 85.67 & $\mathbf{8 5 . 0 0}$ \\
GPT-level & 14.2 & 3.81 & $\mathbf{7 0 . 9 4}$ & 31.03 & $\mathbf{9 2 . 2 0}$ & 22.37 & 87.89 & 82.66 \\
\hline Pythia (c.f.) & 11.8 & 3.89 & 70.44 & 30.74 & 91.80 & 20.57 & 87.58 \\
GPT-NeoX (c.f.) & 20.6 & 3.64 & 71.94 & 31.62 & 93.00 & 25.99 & 88.52 & 82.00 \\
\hline
\end{tabular}

Table 4: Zero-Shot Performance of various models on different tasks. * Interpolation of Pythia and GPT-Neo models 

\begin{tabular}{llllllll}
\hline Method & $\mathrm{L}$ & $\mathrm{d}$ & $\mathrm{T}$ & Train bpc & Test bpc & Time Complexity & Space Complexity \\
\hline Transformer & 12 & 512 & 1024 & 0.977 & 1.137 & $O\left(T^{2} d\right)$ & $O\left(T^{2}+T d\right)$ \\
Transformer & 24 & 256 & 1024 & 1.039 & 1.130 & $O\left(T^{2} d\right)$ & $O\left(T^{2}+T d\right)$ \\
\hline Reformer & 12 & 512 & 1024 & 1.040 & 1.195 & $O(T \log T d)$ & $O(T \log T+T d)$ \\
Synthesizer & 12 & 512 & 1024 & 0.994 & 1.298 & $O\left(T^{2} d\right)$ & $O\left(T^{2}+T d\right)$ \\
Linear Transformer & 12 & 512 & 1024 & 0.981 & 1.207 & $O\left(T d^{2}\right)$ & $O\left(T d+d^{2}\right)$ \\
Performer & 12 & 512 & 1024 & 1.002 & 1.199 & $O\left(T d^{2} \log d\right)$ & $O\left(T d \log d+d^{2} \log d\right)$ \\
AFT-simple & 12 & 512 & 1024 & 0.854 & 1.180 & $O(T d)$ & $O(T d)$ \\
\hline RWKV-RNN & 6 & 512 & 1024 & 0.720 & - & $O(\mathbf{T d})$ & $O(\mathbf{d})$ \\
\hline
\end{tabular}

Table 5: Enwik8 results, measured in bits per character (bpc): the lower the better. Baseline comparisons are made with Reformer (Kitaev et al., 2020), Synthesizer (Tay et al., 2020) (the best performing dense version), Linear Transformer (Katharopoulos et al., 2020), Performer (Choromanski et al., 2020). $L, d$, and $T$ denote the number of blocks (network depth), dimension of features, and sequence length, respectively. Both Linear Transformer and Performer are implemented with customized CUDA kernels (github.com/idiap/fast-transformers), and all other models are implemented in native Pytorch.

quences. For GPU experiments, we use an NVIDIA A100 with 80GB of VRAM. For CPU experiments, we use an AMD EPYC processor with $30 \mathrm{CPU}$ cores and $200 \mathrm{GiB}$ RAM.
![](https://cdn.mathpix.com/cropped/2023_06_04_9aef8213db464f91daabg-19.jpg?height=1054&width=784&top_left_y=1353&top_left_x=224)

Figure 10: Text generation inference memory (CPU RAM, GPU VRAM) for LLMs. Model parameters are not accounted.
![](https://cdn.mathpix.com/cropped/2023_06_04_9aef8213db464f91daabg-19.jpg?height=1120&width=766&top_left_y=1249&top_left_x=1062)

Figure 11: Text generation inference time for LLMs. 

\begin{tabular}{llrrrrr}
\hline Task Name & $\begin{array}{l}\text { Measure } \\
\text { type }\end{array}$ & $\begin{array}{r}\text { ChatGPT } \\
\text { [\%] }\end{array}$ & $\begin{array}{r}\text { GPT-4 } \\
\text { [\%] }\end{array}$ & $\begin{array}{r}\text { RWKV-4 } \\
\text { GPT [\%] }\end{array}$ & $\begin{array}{r}\text { RWKV-4 } \\
\text { changed [\%] }\end{array}$ & $\begin{array}{r}\text { SOTA } \\
{[\%]}\end{array}$ \\
\hline RTE & F1 Macro & 88.1 & $\mathbf{9 1 . 3}$ & 44.2 & 74.8 & 92.1 \\
WNLI & Accuracy & 81.7 & $\mathbf{9 1 . 6}$ & 47.9 & 49.3 & 97.9 \\
GoEmotions & F1 Macro & $\mathbf{2 5 . 6}$ & 23.1 & 7.9 & 7.9 & 52.8 \\
PolEmo2 & F1 Macro & $\mathbf{4 4 . 1}$ & 41.0 & 38.2 & 40.9 & 76.4 \\
\hline
\end{tabular}

Table 6: ChatGPT, GPT-4 and RWKV-4-Raven-14B reasoning performance comparison in RTE (Wang et al., 2019), WNLI (Wang et al., 2018), GoEmotions (Demszky et al., 2020), and PolEmo2 (Kocoń et al., 2019) benchmarks. SOTA is provided as a supplementary reference.

\section{J Importance of prompt construction and comparison to GPT models}

Inspired by article (Kocoń et al., 2023), we compared the zero-shot performance of the RWKV4-Raven-14B with ChatGPT (access in February 2023) and GPT-4 using several known NLP tasks, i.e., recognizing textual entailment (RTE), Winograd Natural Language Inference (WNLI), and recognizing emotions elicited in readers (GoEmotions and PolEmo2). Each model got the same prompts manually chosen to receive proper responses from the ChatGPT model. As shown in Tab. 6, RWKV performs significantly worse than ChatGPT and GPT-4 in specific task performance. We suspect that this disparity is likely caused by the choice of prompts used to generate the answers. Given that prompts are in natural language and do not consider that RWKV is an RNN, so it can not look back inside an instruction.

When the instruction style was adapted to respect that RNNs is not capable for retrospective processing, quality on some datasets increased significantly (ex. for RTE (Wang et al., 2019) F1 Macro increased from $44.2 \%$ to $74.8 \%$ ). We hypothesize that RWKV models are more sensitive to the position of the components in the context, as RNN-based architectures cannot look back and readjust the weight of previous information. For better performance, desired information should be after the question. Example of ChatGPT prompt to RTE:

Having premise <here is a premise $>$ judge if the following hypothesis <here is a hypothesis $>$ are logically connected with the premise? Answer "entailment" if yes, or "not_entailment" if no.

RWKV prompt taking into account the characteristics of the RNN:

Can you tell me if the hypothesis is entailment or is not entailment to the premise?

\begin{tabular}{llrrr}
\hline Task Name & $\begin{array}{l}\text { Measure } \\
\text { type }\end{array}$ & $\begin{array}{r}\text { ChatGPT } \\
{[\%]}\end{array}$ & $\begin{array}{r}\text { RWKV-4 } \\
\text { adapted [\%] }\end{array}$ & $\begin{array}{r}\text { SOTA } \\
{[\%]}\end{array}$ \\
\hline Aggression & F1 Macro & $\mathbf{6 9 . 1 0}$ & 56.66 & 74.45 \\
MathQA & Accuracy & 71.40 & $\mathbf{8 0 . 6 9}$ & 83.20 \\
Sarcasm & F1 Macro & 49.88 & $\mathbf{5 0 . 9 6}$ & 53.57 \\
TweetSent & F1 Macro & $\mathbf{6 3 . 3 2}$ & 52.50 & 72.07 \\
Unhealthy & F1 Macro & $\mathbf{4 5 . 2 1}$ & 43.30 & 50.96 \\
\hline
\end{tabular}

Table 7: ChatGPT and RWKV-4-Raven-14B performance comparison in Aggresion (Wulczyn et al., 2017), Sarcasm (Siddiqui, 2019), Unhealthy (Price et al., 2020), MathQA (Cobbe et al., 2021), and TweetSent (Barbieri et al., 2020) benchmarks. SOTA is provided as a supplementary reference.

premise: <here is a premise $>$ hypothesis: <here is a hypothesis>

While separating the instruction from the input is relatively easy to do, other aspects of prompt engineering are harder to quantify. Testing the approach of stating the input after the question on multiple other tasks, shown in tab. 7, suggests that better prompts might reduce the disparity between models. Raven achieves comparable result to ChatGPT on unhealthy conversation detection and even surpasses it on the sarcasm detection dataset. While this approach in prompting looks necessary, it alone is not enough to replace the capability of having free access to the whole context. Therefore, prompt engineering seems to be of significantly more importance to the RNN models compared to standard transformers. It is entirely possible that good prompts to $\mathrm{RNN}$ models do not mean additional restrictions, but should simply be constructed using completely different guidelines. While authors of a forementioned paper (Kocoń et al., 2023) perform chain-of-thought to improve results on MathQA dataset, trying the same approach on the Raven model achieved a very low accuracy of $48.11 \%$. Despite that, using a very similar prompt without the chain-of-thought outperformed ChatGPT on this dataset. It would not be surprising if, thanks to the hidden state, RNN models do not need additional steps during inference, and adding those only makes the model forget the information it analyzed and deemed important. This further emphasizes the importance of the order of information the model receives.

The template used to prompt the Raven model in MathQA with chain-of-thought prompting:

Write the reasoning and highlight the answer to the question at the end in format: 'Answer: '. The question is: <here is question> The template used to prompt the Raven model in MathQA without chain-of-thought prompting:

Write the answer to the math question in the format:

'Answer: '.

The question is: <here is question> 

\section{K Cases}

In this part, we present a few instances of outputs produced by RWKV model using a Chat interface ${ }^{56}$.

\section{RWKV-4-Raven-14B}

\section{Tell me about ravens.}

Ravens are large, black birds with a distinctive white head and neck. They are found in most parts of the world, including North America, Europe, Asia, and Australia. Ravens are known for their intelligence and problem-solving abilities. They are also considered to be symbols of death and transformation in many cultures. Ravens are often associated with the afterlife or death because they have been known to eat carrion or even other birds. In some cultures, ravens are also believed to be messengers of the gods or guardians of treasure.

\section{RWKV-4-Raven-14B}

\section{Explain the concept of machine learning to a 10-year-old.}

Machine learning is like a super-smart computer that can learn on its own. It can recognize patterns and make decisions based on what it has learned. Just like how you might remember something from the past, the computer can remember what it's learned and use it to make predictions or decisions in the future. It's like having a smart friend who always knows what to do!

\section{RWKV-4-Raven-14B}

\section{Write a short poem describing nature's beauty in springtime.}

Nature's beauty in springtime is rare, A symphony of colors, a gentle surprise. The flowers bloom, the birds sing and sing, As the sun shines down, warming our skin.

The trees sway in the gentle breeze, As we take in their beauty with ease. The sky is blue, the clouds are white, A sight to behold, it's truly divine.

${ }^{5} \mathrm{https}: / /$ github.com/BlinkDL/ChatRWKV

${ }^{6}$ https://huggingface.co/spaces/BlinkDL/ChatRWKV-gradio RWKV-4-Raven-14B

Below is the numerical stable formula for updating the state variable in RWKV:

$$
\begin{aligned}
a_{t}^{*} & =e^{p-q} a_{t-1}+e^{u+k_{t}-q} v_{t} \\
b_{t}^{*} & =e^{p-q} b_{t-1}+e^{u+k_{t}-q} \\
w k v_{t} & =\frac{a_{t}^{*}}{b_{t}^{*}}
\end{aligned}
$$

Please write out the corresponding pytorch code to do so, where both $a_{t}$ and $b_{t}$ are torch.Tensor $([n])$

Here is the PyTorch code to update the state variable in RWKV using the given formula:

![](https://cdn.mathpix.com/cropped/2023_06_04_9aef8213db464f91daabg-23.jpg?height=574&width=1321&top_left_y=887&top_left_x=336)

This code defines a function update_state that takes in the state variables a and $\mathrm{b}$, the control variables $\mathrm{u}$ and $\mathrm{v}$, and the parameters wkv and torch.randn. It then uses the PyTorch torch.exp function to compute the exponential of the control variables and the log of the normalization factor. Finally, it uses these values to update the state.

\section{RWKV-4-Raven-14B}

\section{Explain the following metaphor: Life is like cats.}

The metaphor "Life is like cats" suggests that life can be unpredictable and full of unexpected twists and turns, just like a cat's behavior. It implies that we cannot fully control or predict the outcome of our lives, just as we cannot control the movements of a cat. The metaphor also implies that life is complex and multifaceted, with many different variables and factors at play. Just as a cat can have multiple personalities or moods, life can have many twists and turns that are difficult to predict or understand. Overall, the metaphor suggests that life is complex and unpredictable, much like a cat's behavior. RWKV-4-Raven-14B

Test the convergence of $\sum_{i=0}^{\infty} 2^{-i}$.

The given series is a geometric series with the first term $a_{0}=2$ and common ratio $r=\frac{1}{2}$. We can find the sum of this series using the formula for a geometric series:

$$
\sum_{i=0}^{\infty} a_{i}=\frac{a_{0}}{1-r}=\frac{\frac{1}{2}}{1-\frac{1}{2}}=\frac{1}{1-\frac{1}{2}}=\frac{2}{1}=2
$$

So, the sum of the given series is 2 .

\section{RWKV-4-Raven-14B}

I would like to cook some Chinese food at home. Do you have any suggestions on the meal I could choose?

Yes, of course! If you are not very experienced in cooking Chinese food, you could start with making noodles or rice, then make a dish using one of those. I also think making stir-fry with some meat is an excellent idea.

I want to make something more special. What about red braised pork belly?

Yes, red braised pork belly is also very popular and tasty.

Tell me the steps to cook.

You will need a few ingredients such as pork belly, star anise, sugar, ginger, scallions and wine. Put all of these in a pot, cook on medium heat and simmer until the meat is tender. After that, remove the meat from the pot and reserve the sauce. Slice the meat and add some ginger slices into it, then boil until it's almost done.

\section{What do I need to be careful with the most during the cooking process?}

To cook pork belly properly, it is very important to monitor the heat of the stove and be patient with the cooking process. Overcooking it can cause the pork to be tough, dry and difficult to chew. You can add a few ingredients like soy sauce or black beans to give the pork more flavor.

Tell me more about the proper heat and timing.

If the temperature of the water is too high, it can damage the collagen and connective tissues, resulting in tough pork belly. It is essential to control the temperature carefully, especially when braising the pork. Additionally, keep the water simmering but not boiling to maintain a low simmer, as high heat will dry out the pork belly and result in tough and unpalatable meat. Lastly, monitor the heat during cooking and make sure the heat is medium, as this is what is most appropriate for slow cooking pork belly.

\section{I see. Thank you for your advice.}

You're welcome! It was a pleasure chatting with you. Let me know if you need any further assistance in the future. RWKV-4-Raven-14B

If all apples are fruits and all fruits are sweet, are all apples sweet? Please explain your answer in detail.

Yes, all apples are fruits, and all fruits are sweet. Therefore, it is safe to assume that all apples are sweet.

If birds can fly. Penguins are birds. Can penguins fly? Please explain your answer in detail.

No, penguins cannot fly. Penguins are birds, but they are not capable of flight like other birds. They have wings and can glide for short distances, but they cannot take off or fly long distances like other birds.

The day before two days after the day before tomorrow is Saturday. What day is it today? Please explain your answer in detail.

Today is Friday. Two days after tomorrow is Sunday. Tomorrow is Saturday.

If five cats can catch five mice in five minutes, how long will it take one cat to catch one mouse? Please explain your answer in detail.

If five cats can catch five mice in five minutes, it means that each cat can catch one mouse in one minute. To find out how long it will take one cat to catch one mouse, we can use the concept of proportionality.

Let's assume that it takes $\mathrm{x}$ minutes for a cat to catch a mouse. Then, the number of mice caught by a cat in one minute is equal to the number of cats multiplied by the number of mice caught by each cat: $x=5 * 1=5$

So, it will take 5 minutes for one cat to catch one mouse.

\section{RWKV-4-Raven-14B}

Summarize the following paragraph.

<The Abstract of this paper goes here>

The paper presents a novel architecture, Receptance Weighted Key Value (RWKV), that combines the efficient parallelizable training of Transformers with the efficient inference of RNNs. The model is able to scale to tens of billions of parameters and exhibits linear computational complexity during training and inference, making it a promising alternative to Transformers for sequence processing tasks.

\section{Please further summarize it in one sentence.}

The researchers have developed a new architecture that combines the efficient parallelizable training of Transformers with the efficient inference of RNNs, which allows for linear scaling in memory and computational requirements."""


observation_of_gravitational_waves = r"""\title{
Observation of Gravitational Waves from a Binary Black Hole Merger
}

\author{
B. P. Abbott et al. \\ (LIGO Scientific Collaboration and Virgo Collaboration) \\ (Received 21 January 2016; published 11 February 2016)
}

\begin{abstract}
On September 14, 2015 at 09:50:45 UTC the two detectors of the Laser Interferometer Gravitational-Wave Observatory simultaneously observed a transient gravitational-wave signal. The signal sweeps upwards in frequency from 35 to $250 \mathrm{~Hz}$ with a peak gravitational-wave strain of $1.0 \times 10^{-21}$. It matches the waveform predicted by general relativity for the inspiral and merger of a pair of black holes and the ringdown of the resulting single black hole. The signal was observed with a matched-filter signal-to-noise ratio of 24 and a false alarm rate estimated to be less than 1 event per 203000 years, equivalent to a significance greater than $5.1 \sigma$. The source lies at a luminosity distance of $410_{-180}^{+160} \mathrm{Mpc}$ corresponding to a redshift $z=0.09_{-0.04}^{+0.03}$. In the source frame, the initial black hole masses are $36_{-4}^{+5} M_{\odot}$ and $29_{-4}^{+4} M_{\odot}$, and the final black hole mass is $62_{-4}^{+4} M_{\odot}$, with $3.0_{-0.5}^{+0.5} M_{\odot} c^{2}$ radiated in gravitational waves. All uncertainties define $90 \%$ credible intervals. These observations demonstrate the existence of binary stellar-mass black hole systems. This is the first direct detection of gravitational waves and the first observation of a binary black hole merger.
\end{abstract}

DOI: 10.1103/PhysRevLett.116.061102

\section*{INTRODUCTION}

In 1916, the year after the final formulation of the field equations of general relativity, Albert Einstein predicted the existence of gravitational waves. He found that the linearized weak-field equations had wave solutions: transverse waves of spatial strain that travel at the speed of light, generated by time variations of the mass quadrupole moment of the source [1,2]. Einstein understood that gravitational-wave amplitudes would be remarkably small; moreover, until the Chapel Hill conference in 1957 there was significant debate about the physical reality of gravitational waves [3].

Also in 1916, Schwarzschild published a solution for the field equations [4] that was later understood to describe a black hole [5,6], and in 1963 Kerr generalized the solution to rotating black holes [7]. Starting in the 1970s theoretical work led to the understanding of black hole quasinormal modes [8-10], and in the 1990s higher-order postNewtonian calculations [11] preceded extensive analytical studies of relativistic two-body dynamics [12,13]. These advances, together with numerical relativity breakthroughs in the past decade [14-16], have enabled modeling of binary black hole mergers and accurate predictions of their gravitational waveforms. While numerous black hole candidates have now been identified through electromagnetic observations [17-19], black hole mergers have not previously been observed.

*Full author list given at the end of the article.

Published by the American Physical Society under the terms of the Creative Commons Attribution 3.0 License. Further distribution of this work must maintain attribution to the author $(s)$ and the published article's title, journal citation, and DOI. The discovery of the binary pulsar system PSR B1913+16 by Hulse and Taylor [20] and subsequent observations of its energy loss by Taylor and Weisberg [21] demonstrated the existence of gravitational waves. This discovery, along with emerging astrophysical understanding [22], led to the recognition that direct observations of the amplitude and phase of gravitational waves would enable studies of additional relativistic systems and provide new tests of general relativity, especially in the dynamic strong-field regime.

Experiments to detect gravitational waves began with Weber and his resonant mass detectors in the 1960s [23], followed by an international network of cryogenic resonant detectors [24]. Interferometric detectors were first suggested in the early 1960s [25] and the 1970s [26]. A study of the noise and performance of such detectors [27], and further concepts to improve them [28], led to proposals for long-baseline broadband laser interferometers with the potential for significantly increased sensitivity [29-32]. By the early 2000s, a set of initial detectors was completed, including TAMA 300 in Japan, GEO 600 in Germany, the Laser Interferometer Gravitational-Wave Observatory (LIGO) in the United States, and Virgo in Italy. Combinations of these detectors made joint observations from 2002 through 2011, setting upper limits on a variety of gravitational-wave sources while evolving into a global network. In 2015, Advanced LIGO became the first of a significantly more sensitive network of advanced detectors to begin observations [33-36].

A century after the fundamental predictions of Einstein and Schwarzschild, we report the first direct detection of gravitational waves and the first direct observation of a binary black hole system merging to form a single black hole. Our observations provide unique access to the properties of space-time in the strong-field, high-velocity regime and confirm predictions of general relativity for the nonlinear dynamics of highly disturbed black holes.

\section*{OBSERVATION}

On September 14, 2015 at 09:50:45 UTC, the LIGO Hanford, WA, and Livingston, LA, observatories detected the coincident signal GW150914 shown in Fig. 1. The initial detection was made by low-latency searches for generic gravitational-wave transients [41] and was reported within three minutes of data acquisition [43]. Subsequently, matched-filter analyses that use relativistic models of compact binary waveforms [44] recovered GW150914 as the most significant event from each detector for the observations reported here. Occurring within the 10-ms intersite

![](https://cdn.mathpix.com/cropped/2023_06_04_7f30e21414823ce83335g-02.jpg?height=1409&width=1742&top_left_y=545&top_left_x=186)

FIG. 1. The gravitational-wave event GW150914 observed by the LIGO Hanford (H1, left column panels) and Livingston (L1, right column panels) detectors. Times are shown relative to September 14, 2015 at 09:50:45 UTC. For visualization, all time series are filtered with a 35-350 Hz bandpass filter to suppress large fluctuations outside the detectors' most sensitive frequency band, and band-reject filters to remove the strong instrumental spectral lines seen in the Fig. 3 spectra. Top row, left: H1 strain. Top row, right: L1 strain. GW150914 arrived first at L1 and $6.9_{-0.4}^{+0.5} \mathrm{~ms}$ later at $\mathrm{H} 1$; for a visual comparison, the $\mathrm{H} 1$ data are also shown, shifted in time by this amount and inverted (to account for the detectors' relative orientations). Second row: Gravitational-wave strain projected onto each detector in the $35-350 \mathrm{~Hz}$ band. Solid lines show a numerical relativity waveform for a system with parameters consistent with those recovered from GW150914 [37,38] confirmed to $99.9 \%$ by an independent calculation based on [15]. Shaded areas show $90 \%$ credible regions for two independent waveform reconstructions. One (dark gray) models the signal using binary black hole template waveforms [39]. The other (light gray) does not use an astrophysical model, but instead calculates the strain signal as a linear combination of sine-Gaussian wavelets [40,41]. These reconstructions have a 94\% overlap, as shown in [39]. Third row: Residuals after subtracting the filtered numerical relativity waveform from the filtered detector time series. Bottom row:A time-frequency representation [42] of the strain data, showing the signal frequency increasing over time. propagation time, the events have a combined signal-tonoise ratio (SNR) of 24 [45].

Only the LIGO detectors were observing at the time of GW150914. The Virgo detector was being upgraded, and GEO 600, though not sufficiently sensitive to detect this event, was operating but not in observational mode. With only two detectors the source position is primarily determined by the relative arrival time and localized to an area of approximately $600 \operatorname{deg}^{2}(90 \%$ credible region) $[39,46]$.

The basic features of GW150914 point to it being produced by the coalescence of two black holes-i.e., their orbital inspiral and merger, and subsequent final black hole ringdown. Over $0.2 \mathrm{~s}$, the signal increases in frequency and amplitude in about 8 cycles from 35 to $150 \mathrm{~Hz}$, where the amplitude reaches a maximum. The most plausible explanation for this evolution is the inspiral of two orbiting masses, $m_{1}$ and $m_{2}$, due to gravitational-wave emission. At the lower frequencies, such evolution is characterized by the chirp mass [11]

$$
\mathcal{M}=\frac{\left(m_{1} m_{2}\right)^{3 / 5}}{\left(m_{1}+m_{2}\right)^{1 / 5}}=\frac{c^{3}}{G}\left[\frac{5}{96} \pi^{-8 / 3} f^{-11 / 3} \dot{f}\right]^{3 / 5},
$$

where $f$ and $\dot{f}$ are the observed frequency and its time derivative and $G$ and $c$ are the gravitational constant and speed of light. Estimating $f$ and $\dot{f}$ from the data in Fig. 1, we obtain a chirp mass of $\mathcal{M} \simeq 30 M_{\odot}$, implying that the total mass $M=m_{1}+m_{2}$ is $\gtrsim 70 M_{\odot}$ in the detector frame. This bounds the sum of the Schwarzschild radii of the binary components to $2 G M / \mathrm{c}^{2} \gtrsim 210 \mathrm{~km}$. To reach an orbital frequency of $75 \mathrm{~Hz}$ (half the gravitational-wave frequency) the objects must have been very close and very compact; equal Newtonian point masses orbiting at this frequency would be only $\simeq 350 \mathrm{~km}$ apart. A pair of neutron stars, while compact, would not have the required mass, while a black hole neutron star binary with the deduced chirp mass would have a very large total mass, and would thus merge at much lower frequency. This leaves black holes as the only known objects compact enough to reach an orbital frequency of $75 \mathrm{~Hz}$ without contact. Furthermore, the decay of the waveform after it peaks is consistent with the damped oscillations of a black hole relaxing to a final stationary Kerr configuration. Below, we present a general-relativistic analysis of GW150914; Fig. 2 shows the calculated waveform using the resulting source parameters.

\section*{DETECTORS}

Gravitational-wave astronomy exploits multiple, widely separated detectors to distinguish gravitational waves from local instrumental and environmental noise, to provide source sky localization, and to measure wave polarizations. The LIGO sites each operate a single Advanced LIGO
![](https://cdn.mathpix.com/cropped/2023_06_04_7f30e21414823ce83335g-03.jpg?height=862&width=870&top_left_y=153&top_left_x=1080)

FIG. 2. Top: Estimated gravitational-wave strain amplitude from GW150914 projected onto H1. This shows the full bandwidth of the waveforms, without the filtering used for Fig. 1. The inset images show numerical relativity models of the black hole horizons as the black holes coalesce. Bottom: The Keplerian effective black hole separation in units of Schwarzschild radii $\left(R_{S}=2 G M / c^{2}\right)$ and the effective relative velocity given by the post-Newtonian parameter $v / c=\left(G M \pi f / c^{3}\right)^{1 / 3}$, where $f$ is the gravitational-wave frequency calculated with numerical relativity and $M$ is the total mass (value from Table I).

detector [33], a modified Michelson interferometer (see Fig. 3) that measures gravitational-wave strain as a difference in length of its orthogonal arms. Each arm is formed by two mirrors, acting as test masses, separated by $L_{x}=L_{y}=L=4 \mathrm{~km}$. A passing gravitational wave effectively alters the arm lengths such that the measured difference is $\Delta L(t)=\delta L_{x}-\delta L_{y}=h(t) L$, where $h$ is the gravitational-wave strain amplitude projected onto the detector. This differential length variation alters the phase difference between the two light fields returning to the beam splitter, transmitting an optical signal proportional to the gravitational-wave strain to the output photodetector.

To achieve sufficient sensitivity to measure gravitational waves, the detectors include several enhancements to the basic Michelson interferometer. First, each arm contains a resonant optical cavity, formed by its two test mass mirrors, that multiplies the effect of a gravitational wave on the light phase by a factor of 300 [48]. Second, a partially transmissive power-recycling mirror at the input provides additional resonant buildup of the laser light in the interferometer as a whole [49,50]: $20 \mathrm{~W}$ of laser input is increased to $700 \mathrm{~W}$ incident on the beam splitter, which is further increased to $100 \mathrm{~kW}$ circulating in each arm cavity. Third, a partially transmissive signal-recycling mirror at the output optimizes 
![](https://cdn.mathpix.com/cropped/2023_06_04_7f30e21414823ce83335g-04.jpg?height=1028&width=1548&top_left_y=152&top_left_x=296)

(a)

FIG. 3. Simplified diagram of an Advanced LIGO detector (not to scale). A gravitational wave propagating orthogonally to the detector plane and linearly polarized parallel to the 4-km optical cavities will have the effect of lengthening one 4-km arm and shortening the other during one half-cycle of the wave; these length changes are reversed during the other half-cycle. The output photodetector records these differential cavity length variations. While a detector's directional response is maximal for this case, it is still significant for most other angles of incidence or polarizations (gravitational waves propagate freely through the Earth). Inset (a): Location and orientation of the LIGO detectors at Hanford, WA (H1) and Livingston, LA (L1). Inset (b): The instrument noise for each detector near the time of the signal detection; this is an amplitude spectral density, expressed in terms of equivalent gravitational-wave strain amplitude. The sensitivity is limited by photon shot noise at frequencies above $150 \mathrm{~Hz}$, and by a superposition of other noise sources at lower frequencies [47]. Narrow-band features include calibration lines (33-38, 330, and $1080 \mathrm{~Hz}$ ), vibrational modes of suspension fibers (500 Hz and harmonics), and $60 \mathrm{~Hz}$ electric power grid harmonics.

the gravitational-wave signal extraction by broadening the bandwidth of the arm cavities [51,52]. The interferometer is illuminated with a 1064-nm wavelength Nd:YAG laser, stabilized in amplitude, frequency, and beam geometry $[53,54]$. The gravitational-wave signal is extracted at the output port using a homodyne readout [55].

These interferometry techniques are designed to maximize the conversion of strain to optical signal, thereby minimizing the impact of photon shot noise (the principal noise at high frequencies). High strain sensitivity also requires that the test masses have low displacement noise, which is achieved by isolating them from seismic noise (low frequencies) and designing them to have low thermal noise (intermediate frequencies). Each test mass is suspended as the final stage of a quadruple-pendulum system [56], supported by an active seismic isolation platform [57]. These systems collectively provide more than 10 orders of magnitude of isolation from ground motion for frequencies above $10 \mathrm{~Hz}$. Thermal noise is minimized by using low-mechanical-loss materials in the test masses and their suspensions: the test masses are $40-\mathrm{kg}$ fused silica substrates with low-loss dielectric optical coatings [58,59], and are suspended with fused silica fibers from the stage above [60].

To minimize additional noise sources, all components other than the laser source are mounted on vibration isolation stages in ultrahigh vacuum. To reduce optical phase fluctuations caused by Rayleigh scattering, the pressure in the $1.2-\mathrm{m}$ diameter tubes containing the armcavity beams is maintained below $1 \mu \mathrm{Pa}$.

Servo controls are used to hold the arm cavities on resonance [61] and maintain proper alignment of the optical components [62]. The detector output is calibrated in strain by measuring its response to test mass motion induced by photon pressure from a modulated calibration laser beam [63]. The calibration is established to an uncertainty $(1 \sigma)$ of less than $10 \%$ in amplitude and 10 degrees in phase, and is continuously monitored with calibration laser excitations at selected frequencies. Two alternative methods are used to validate the absolute calibration, one referenced to the main laser wavelength and the other to a radio-frequency oscillator [64]. Additionally, the detector response to gravitational waves is tested by injecting simulated waveforms with the calibration laser.

To monitor environmental disturbances and their influence on the detectors, each observatory site is equipped with an array of sensors: seismometers, accelerometers, microphones, magnetometers, radio receivers, weather sensors, ac-power line monitors, and a cosmic-ray detector [65]. Another $\sim 10^{5}$ channels record the interferometer's operating point and the state of the control systems. Data collection is synchronized to Global Positioning System (GPS) time to better than $10 \mu \mathrm{s}$ [66]. Timing accuracy is verified with an atomic clock and a secondary GPS receiver at each observatory site.

In their most sensitive band, $100-300 \mathrm{~Hz}$, the current LIGO detectors are 3 to 5 times more sensitive to strain than initial LIGO [67]; at lower frequencies, the improvement is even greater, with more than ten times better sensitivity below $60 \mathrm{~Hz}$. Because the detectors respond proportionally to gravitational-wave amplitude, at low redshift the volume of space to which they are sensitive increases as the cube of strain sensitivity. For binary black holes with masses similar to GW150914, the space-time volume surveyed by the observations reported here surpasses previous observations by an order of magnitude [68].

\section*{DETECTOR VALIDATION}

Both detectors were in steady state operation for several hours around GW150914. All performance measures, in particular their average sensitivity and transient noise behavior, were typical of the full analysis period $[69,70]$.

Exhaustive investigations of instrumental and environmental disturbances were performed, giving no evidence to suggest that GW150914 could be an instrumental artifact [69]. The detectors' susceptibility to environmental disturbances was quantified by measuring their response to specially generated magnetic, radio-frequency, acoustic, and vibration excitations. These tests indicated that any external disturbance large enough to have caused the observed signal would have been clearly recorded by the array of environmental sensors. None of the environmental sensors recorded any disturbances that evolved in time and frequency like GW150914, and all environmental fluctuations during the second that contained GW150914 were too small to account for more than $6 \%$ of its strain amplitude. Special care was taken to search for long-range correlated disturbances that might produce nearly simultaneous signals at the two sites. No significant disturbances were found.

The detector strain data exhibit non-Gaussian noise transients that arise from a variety of instrumental mechanisms. Many have distinct signatures, visible in auxiliary data channels that are not sensitive to gravitational waves; such instrumental transients are removed from our analyses [69]. Any instrumental transients that remain in the data are accounted for in the estimated detector backgrounds described below. There is no evidence for instrumental transients that are temporally correlated between the two detectors.

\section*{SEARCHES}

We present the analysis of 16 days of coincident observations between the two LIGO detectors from September 12 to October 20, 2015. This is a subset of the data from Advanced LIGO's first observational period that ended on January 12, 2016.

GW150914 is confidently detected by two different types of searches. One aims to recover signals from the coalescence of compact objects, using optimal matched filtering with waveforms predicted by general relativity. The other search targets a broad range of generic transient signals, with minimal assumptions about waveforms. These searches use independent methods, and their response to detector noise consists of different, uncorrelated, events. However, strong signals from binary black hole mergers are expected to be detected by both searches.

Each search identifies candidate events that are detected at both observatories consistent with the intersite propagation time. Events are assigned a detection-statistic value that ranks their likelihood of being a gravitational-wave signal. The significance of a candidate event is determined by the search background-the rate at which detector noise produces events with a detection-statistic value equal to or higher than the candidate event. Estimating this background is challenging for two reasons: the detector noise is nonstationary and non-Gaussian, so its properties must be empirically determined; and it is not possible to shield the detector from gravitational waves to directly measure a signal-free background. The specific procedure used to estimate the background is slightly different for the two searches, but both use a time-shift technique: the time stamps of one detector's data are artificially shifted by an offset that is large compared to the intersite propagation time, and a new set of events is produced based on this time-shifted data set. For instrumental noise that is uncorrelated between detectors this is an effective way to estimate the background. In this process a gravitationalwave signal in one detector may coincide with time-shifted noise transients in the other detector, thereby contributing to the background estimate. This leads to an overestimate of the noise background and therefore to a more conservative assessment of the significance of candidate events.

The characteristics of non-Gaussian noise vary between different time-frequency regions. This means that the search backgrounds are not uniform across the space of signals being searched. To maximize sensitivity and provide a better estimate of event significance, the searches sort both their background estimates and their event candidates into different classes according to their time-frequency morphology. The significance of a candidate event is measured against the background of its class. To account for having searched multiple classes, this significance is decreased by a trials factor equal to the number of classes [71].

\section*{A. Generic transient search}

Designed to operate without a specific waveform model, this search identifies coincident excess power in timefrequency representations of the detector strain data [43,72], for signal frequencies up to $1 \mathrm{kHz}$ and durations up to a few seconds.

The search reconstructs signal waveforms consistent with a common gravitational-wave signal in both detectors using a multidetector maximum likelihood method. Each event is ranked according to the detection statistic $\eta_{c}=\sqrt{2 E_{c} /\left(1+E_{n} / E_{c}\right)}$, where $E_{c}$ is the dimensionless coherent signal energy obtained by cross-correlating the two reconstructed waveforms, and $E_{n}$ is the dimensionless residual noise energy after the reconstructed signal is subtracted from the data. The statistic $\eta_{c}$ thus quantifies the SNR of the event and the consistency of the data between the two detectors.

Based on their time-frequency morphology, the events are divided into three mutually exclusive search classes, as described in [41]: events with time-frequency morphology of known populations of noise transients (class C1), events with frequency that increases with time (class C3), and all remaining events (class $\mathrm{C} 2$ ). Detected with $\eta_{c}=20.0$, GW150914 is the strongest event of the entire search. Consistent with its coalescence signal signature, it is found in the search class $\mathrm{C} 3$ of events with increasing time-frequency evolution. Measured on a background equivalent to over 67400 years of data and including a trials factor of 3 to account for the search classes, its false alarm rate is lower than 1 in 22500 years. This corresponds to a probability $<2 \times 10^{-6}$ of observing one or more noise events as strong as GW150914 during the analysis time, equivalent to $4.6 \sigma$. The left panel of Fig. 4 shows the $\mathrm{C} 3$ class results and background.

The selection criteria that define the search class C3 reduce the background by introducing a constraint on the signal morphology. In order to illustrate the significance of GW150914 against a background of events with arbitrary shapes, we also show the results of a search that uses the same set of events as the one described above but without this constraint. Specifically, we use only two search classes: the $\mathrm{C} 1$ class and the union of $\mathrm{C} 2$ and $\mathrm{C} 3$ classes $(\mathrm{C} 2+\mathrm{C} 3)$. In this two-class search the GW150914 event is found in the $\mathrm{C} 2+\mathrm{C} 3$ class. The left panel of Fig. 4 shows the C2 + C3 class results and background. In the background of this class there are four events with $\eta_{c} \geq 32.1$, yielding a false alarm rate for GW150914 of 1 in 8400 years. This corresponds to a false alarm probability of $5 \times 10^{-6}$ equivalent to $4.4 \sigma$

![](https://cdn.mathpix.com/cropped/2023_06_04_7f30e21414823ce83335g-06.jpg?height=677&width=841&top_left_y=1361&top_left_x=1073)

FIG. 4. Search results from the generic transient search (left) and the binary coalescence search (right). These histograms show the number of candidate events (orange markers) and the mean number of background events (black lines) in the search class where GW150914 was found as a function of the search detection statistic and with a bin width of 0.2. The scales on the top give the significance of an event in Gaussian standard deviations based on the corresponding noise background. The significance of GW150914 is greater than $5.1 \sigma$ and $4.6 \sigma$ for the binary coalescence and the generic transient searches, respectively. Left: Along with the primary search (C3) we also show the results (blue markers) and background (green curve) for an alternative search that treats events independently of their frequency evolution ( $\mathrm{C} 2+\mathrm{C} 3)$. The classes $\mathrm{C} 2$ and $\mathrm{C} 3$ are defined in the text. Right: The tail in the black-line background of the binary coalescence search is due to random coincidences of GW150914 in one detector with noise in the other detector. (This type of event is practically absent in the generic transient search background because they do not pass the time-frequency consistency requirements used in that search.) The purple curve is the background excluding those coincidences, which is used to assess the significance of the second strongest event. For robustness and validation, we also use other generic transient search algorithms [41]. A different search [73] and a parameter estimation follow-up [74] detected GW150914 with consistent significance and signal parameters.

\section*{B. Binary coalescence search}

This search targets gravitational-wave emission from binary systems with individual masses from 1 to $99 M_{\odot}$, total mass less than $100 M_{\odot}$, and dimensionless spins up to 0.99 [44]. To model systems with total mass larger than $4 M_{\odot}$, we use the effective-one-body formalism [75], which combines results from the post-Newtonian approach $[11,76]$ with results from black hole perturbation theory and numerical relativity. The waveform model $[77,78]$ assumes that the spins of the merging objects are aligned with the orbital angular momentum, but the resulting templates can, nonetheless, effectively recover systems with misaligned spins in the parameter region of GW150914 [44]. Approximately 250000 template waveforms are used to cover this parameter space.

The search calculates the matched-filter signal-to-noise ratio $\rho(t)$ for each template in each detector and identifies maxima of $\rho(t)$ with respect to the time of arrival of the signal [79-81]. For each maximum we calculate a chi-squared statistic $\chi_{r}^{2}$ to test whether the data in several different frequency bands are consistent with the matching template [82]. Values of $\chi_{r}^{2}$ near unity indicate that the signal is consistent with a coalescence. If $\chi_{r}^{2}$ is greater than unity, $\rho(t)$ is reweighted as $\hat{\rho}=\rho /\left\{\left[1+\left(\chi_{r}^{2}\right)^{3}\right] / 2\right\}^{1 / 6}[83,84]$. The final step enforces coincidence between detectors by selecting event pairs that occur within a 15 -ms window and come from the same template. The $15-\mathrm{ms}$ window is determined by the 10 -ms intersite propagation time plus $5 \mathrm{~ms}$ for uncertainty in arrival time of weak signals. We rank coincident events based on the quadrature sum $\hat{\rho}_{c}$ of the $\hat{\rho}$ from both detectors [45].

To produce background data for this search the SNR maxima of one detector are time shifted and a new set of coincident events is computed. Repeating this procedure $\sim 10^{7}$ times produces a noise background analysis time equivalent to 608000 years.

To account for the search background noise varying across the target signal space, candidate and background events are divided into three search classes based on template length. The right panel of Fig. 4 shows the background for the search class of GW150914. The GW150914 detectionstatistic value of $\hat{\rho}_{c}=23.6$ is larger than any background event, so only an upper bound can be placed on its false alarm rate. Across the three search classes this bound is 1 in 203000 years. This translates to a false alarm probability $<2 \times 10^{-7}$, corresponding to $5.1 \sigma$.

A second, independent matched-filter analysis that uses a different method for estimating the significance of its events $[85,86]$, also detected GW150914 with identical signal parameters and consistent significance. TABLE I. Source parameters for GW150914. We report median values with $90 \%$ credible intervals that include statistical errors, and systematic errors from averaging the results of different waveform models. Masses are given in the source frame; to convert to the detector frame multiply by $(1+z)$ [90]. The source redshift assumes standard cosmology [91].

\begin{tabular}{lc}
\hline \hline Primary black hole mass & $36_{-4}^{+5} M_{\odot}$ \\
Secondary black hole mass & $29_{-4}^{+4} M_{\odot}$ \\
Final black hole mass & $62_{-4}^{+4} M_{\odot}$ \\
Final black hole spin & $0.67_{-0.07}^{+0.05}$ \\
Luminosity distance & $410_{-180}^{+160} \mathrm{Mpc}$ \\
Source redshift $z$ & $0.09_{-0.04}^{+0.03}$ \\
\hline \hline
\end{tabular}

When an event is confidently identified as a real gravitational-wave signal, as for GW150914, the background used to determine the significance of other events is reestimated without the contribution of this event. This is the background distribution shown as a purple line in the right panel of Fig. 4. Based on this, the second most significant event has a false alarm rate of 1 per 2.3 years and corresponding Poissonian false alarm probability of 0.02 . Waveform analysis of this event indicates that if it is astrophysical in origin it is also a binary black hole merger [44].

\section*{SOURCE DISCUSSION}

The matched-filter search is optimized for detecting signals, but it provides only approximate estimates of the source parameters. To refine them we use general relativity-based models $[77,78,87,88]$, some of which include spin precession, and for each model perform a coherent Bayesian analysis to derive posterior distributions of the source parameters [89]. The initial and final masses, final spin, distance, and redshift of the source are shown in Table I. The spin of the primary black hole is constrained to be $<0.7$ ( $90 \%$ credible interval) indicating it is not maximally spinning, while the spin of the secondary is only weakly constrained. These source parameters are discussed in detail in [39]. The parameter uncertainties include statistical errors and systematic errors from averaging the results of different waveform models.

Using the fits to numerical simulations of binary black hole mergers in $[92,93]$, we provide estimates of the mass and spin of the final black hole, the total energy radiated in gravitational waves, and the peak gravitational-wave luminosity [39]. The estimated total energy radiated in gravitational waves is $3.0_{-0.5}^{+0.5} M_{\odot} c^{2}$. The system reached a peak gravitational-wave luminosity of $3.6_{-0.4}^{+0.5} \times 10^{56} \mathrm{erg} / \mathrm{s}$, equivalent to $200_{-20}^{+30} M_{\odot} c^{2} / \mathrm{s}$.

Several analyses have been performed to determine whether or not GW150914 is consistent with a binary black hole system in general relativity [94]. A first consistency check involves the mass and spin of the final black hole. In general relativity, the end product of a black hole binary coalescence is a Kerr black hole, which is fully described by its mass and spin. For quasicircular inspirals, these are predicted uniquely by Einstein's equations as a function of the masses and spins of the two progenitor black holes. Using fitting formulas calibrated to numerical relativity simulations [92], we verified that the remnant mass and spin deduced from the early stage of the coalescence and those inferred independently from the late stage are consistent with each other, with no evidence for disagreement from general relativity.

Within the post-Newtonian formalism, the phase of the gravitational waveform during the inspiral can be expressed as a power series in $f^{1 / 3}$. The coefficients of this expansion can be computed in general relativity. Thus, we can test for consistency with general relativity $[95,96]$ by allowing the coefficients to deviate from the nominal values, and seeing if the resulting waveform is consistent with the data. In this second check [94] we place constraints on these deviations, finding no evidence for violations of general relativity.

Finally, assuming a modified dispersion relation for gravitational waves [97], our observations constrain the Compton wavelength of the graviton to be $\lambda_{g}>10^{13} \mathrm{~km}$, which could be interpreted as a bound on the graviton mass $m_{g}<1.2 \times 10^{-22} \mathrm{eV} / c^{2}$. This improves on Solar System and binary pulsar bounds $[98,99]$ by factors of a few and a thousand, respectively, but does not improve on the modeldependent bounds derived from the dynamics of Galaxy clusters [100] and weak lensing observations [101]. In summary, all three tests are consistent with the predictions of general relativity in the strong-field regime of gravity.

GW150914 demonstrates the existence of stellar-mass black holes more massive than $\simeq 25 M_{\odot}$, and establishes that binary black holes can form in nature and merge within a Hubble time. Binary black holes have been predicted to form both in isolated binaries [102-104] and in dense environments by dynamical interactions [105-107]. The formation of such massive black holes from stellar evolution requires weak massive-star winds, which are possible in stellar environments with metallicity lower than $\simeq 1 / 2$ the solar value [108,109]. Further astrophysical implications of this binary black hole discovery are discussed in [110].

These observational results constrain the rate of stellarmass binary black hole mergers in the local universe. Using several different models of the underlying binary black hole mass distribution, we obtain rate estimates ranging from 2-400 $\mathrm{Gpc}^{-3} \mathrm{yr}^{-1}$ in the comoving frame [111-113]. This is consistent with a broad range of rate predictions as reviewed in [114], with only the lowest event rates being excluded.

Binary black hole systems at larger distances contribute to a stochastic background of gravitational waves from the superposition of unresolved systems. Predictions for such a background are presented in [115]. If the signal from such a population were detected, it would provide information about the evolution of such binary systems over the history of the universe.

\section*{OUTLOOK}

Further details about these results and associated data releases are available at [116]. Analysis results for the entire first observational period will be reported in future publications. Efforts are under way to enhance significantly the global gravitational-wave detector network [117]. These include further commissioning of the Advanced LIGO detectors to reach design sensitivity, which will allow detection of binaries like GW150914 with 3 times higher SNR. Additionally, Advanced Virgo, KAGRA, and a possible third LIGO detector in India [118] will extend the network and significantly improve the position reconstruction and parameter estimation of sources.

\section*{CONCLUSION}

The LIGO detectors have observed gravitational waves from the merger of two stellar-mass black holes. The detected waveform matches the predictions of general relativity for the inspiral and merger of a pair of black holes and the ringdown of the resulting single black hole. These observations demonstrate the existence of binary stellar-mass black hole systems. This is the first direct detection of gravitational waves and the first observation of a binary black hole merger.

\section*{ACKNOWLEDGMENTS}

The authors gratefully acknowledge the support of the United States National Science Foundation (NSF) for the construction and operation of the LIGO Laboratory and Advanced LIGO as well as the Science and Technology Facilities Council (STFC) of the United Kingdom, the MaxPlanck Society (MPS), and the State of Niedersachsen, Germany, for support of the construction of Advanced LIGO and construction and operation of the GEO600 detector. Additional support for Advanced LIGO was provided by the Australian Research Council. The authors gratefully acknowledge the Italian Istituto Nazionale di Fisica Nucleare (INFN), the French Centre National de la Recherche Scientifique (CNRS), and the Foundation for Fundamental Research on Matter supported by the Netherlands Organisation for Scientific Research, for the construction and operation of the Virgo detector, and for the creation and support of the EGO consortium. The authors also gratefully acknowledge research support from these agencies as well as by the Council of Scientific and Industrial Research of India, Department of Science and Technology, India, Science \& Engineering Research Board (SERB), India, Ministry of Human Resource Development, India, the Spanish Ministerio de Economía y Competitividad, the Conselleria d'Economia i Competitivitat and Conselleria d'Educació, Cultura i Universitats of the Govern de les Illes Balears, the National Science Centre of Poland, the European Commission, the Royal Society, the Scottish Funding Council, the Scottish Universities Physics Alliance, the Hungarian Scientific Research Fund (OTKA), the Lyon Institute of Origins (LIO), the National Research Foundation of Korea, Industry Canada and the Province of Ontario through the Ministry of Economic Development and Innovation, the Natural Sciences and Engineering Research Council of Canada, Canadian Institute for Advanced Research, the Brazilian Ministry of Science, Technology, and Innovation, Russian Foundation for Basic Research, the Leverhulme Trust, the Research Corporation, Ministry of Science and Technology (MOST), Taiwan, and the Kavli Foundation. The authors gratefully acknowledge the support of the NSF, STFC, MPS, INFN, CNRS and the State of Niedersachsen, Germany, for provision of computational resources. This article has been assigned the document numbers LIGOP150914 and VIR-0015A-16.

[1] A. Einstein, Sitzungsber. K. Preuss. Akad. Wiss. 1, 688 (1916).

[2] A. Einstein, Sitzungsber. K. Preuss. Akad. Wiss. 1, 154 (1918).

[3] P. R. Saulson, Gen. Relativ. Gravit. 43, 3289 (2011).

[4] K. Schwarzschild, Sitzungsber. K. Preuss. Akad. Wiss. 1, 189 (1916).

[5] D. Finkelstein, Phys. Rev. 110, 965 (1958).

[6] M. D. Kruskal, Phys. Rev. 119, 1743 (1960).

[7] R. P. Kerr, Phys. Rev. Lett. 11, 237 (1963).

[8] C. V. Vishveshwara, Nature (London) 227, 936 (1970).

[9] W. H. Press, Astrophys. J. 170, L105 (1971).

[10] S. Chandrasekhar and S. L. Detweiler, Proc. R. Soc. A 344, $441(1975)$.

[11] L. Blanchet, T. Damour, B. R. Iyer, C. M. Will, and A. G. Wiseman, Phys. Rev. Lett. 74, 3515 (1995).

[12] L. Blanchet, Living Rev. Relativity 17, 2 (2014).

[13] A. Buonanno and T. Damour, Phys. Rev. D 59, 084006 (1999).

[14] F. Pretorius, Phys. Rev. Lett. 95, 121101 (2005).

[15] M. Campanelli, C. O. Lousto, P. Marronetti, and Y. Zlochower, Phys. Rev. Lett. 96, 111101 (2006).

[16] J. G. Baker, J. Centrella, D.-I. Choi, M. Koppitz, and J. van Meter, Phys. Rev. Lett. 96, 111102 (2006).

[17] B. L. Webster and P. Murdin, Nature (London) 235, 37 (1972).

[18] C. T. Bolton, Nature (London) 240, 124 (1972).

[19] J. Casares and P. G. Jonker, Space Sci. Rev. 183, 223 $(2014)$

[20] R. A. Hulse and J. H. Taylor, Astrophys. J. 195, L51 (1975).

[21] J. H. Taylor and J. M. Weisberg, Astrophys. J. 253, 908 (1982).

[22] W. Press and K. Thorne, Annu. Rev. Astron. Astrophys. 10, 335 (1972).

[23] J. Weber, Phys. Rev. 117, 306 (1960).

[24] P. Astone et al., Phys. Rev. D 82, 022003 (2010). [25] M. E. Gertsenshtein and V. I. Pustovoit, Sov. Phys. JETP 16, 433 (1962).

[26] G. E. Moss, L. R. Miller, and R. L. Forward, Appl. Opt. 10, $2495(1971)$

[27] R. Weiss, Electromagnetically coupled broadband gravitational antenna, Quarterly Report of the Research Laboratory for Electronics, MIT Report No. 105, 1972, https:// dcc.ligo.org/LIGO-P720002/public/main.

[28] R. W. P. Drever, in Gravitational Radiation, edited by N. Deruelle and T. Piran (North-Holland, Amsterdam, 1983), p. 321.

[29] R. W. P. Drever, F. J. Raab, K. S. Thorne, R. Vogt, and R. Weiss, Laser Interferometer Gravitational-wave Observatory (LIGO) Technical Report, 1989, https://dcc.ligo.org/ LIGO-M890001/public/main.

[30] A. Abramovici et al., Science 256, 325 (1992).

[31] A. Brillet, A. Giazotto et al., Virgo Project Technical Report No. VIR-0517A-15, 1989, https://tds.ego-gw.it/ql/? $\mathrm{c}=11247$

[32] J. Hough et al., Proposal for a joint German-British interferometric gravitational wave detector, MPQ Technical Report 147, No. GWD/137/JH(89), 1989, http://eprints .gla.ac.uk/114852.

[33] J. Aasi et al., Classical Quantum Gravity 32, 074001 (2015).

[34] F. Acernese et al., Classical Quantum Gravity 32, 024001 $(2015)$.

[35] C. Affeldt et al., Classical Quantum Gravity 31, 224002 (2014).

[36] Y. Aso, Y. Michimura, K. Somiya, M. Ando, O. Miyakawa, T. Sekiguchi, D. Tatsumi, and H. Yamamoto, Phys. Rev. D 88, 043007 (2013).

[37] The waveform shown is SXS:BBH:0305, available for download at http://www.black-holes.org/waveforms.

[38] A. H. Mroué et al., Phys. Rev. Lett. 111, 241104 (2013).

[39] B. Abbott et al., https://dcc.ligo.org/LIGO-P1500218/ public/main.

[40] N. J. Cornish and T. B. Littenberg, Classical Quantum Gravity 32, $135012(2015)$

[41] B. Abbott et al., https://dcc.ligo.org/LIGO-P1500229/ public/main.

[42] S. Chatterji, L. Blackburn, G. Martin, and E. Katsavounidis, Classical Quantum Gravity 21, S1809 (2004)

[43] S. Klimenko et al., arXiv:1511.05999 [Phys. Rev. D (to be published)].

[44] B. Abbott et al., https://dcc.ligo.org/LIGO-P1500269/ public/main.

[45] S. A. Usman et al., arXiv:1508.02357.

[46] B. Abbott et al., https://dcc.ligo.org/LIGO-P1500227/ public/main.

[47] B. Abbott et al., https://dcc.ligo.org/LIGO-P1500237/ public/main.

[48] R. W. P. Drever, The Detection of Gravitational Waves, edited by D. G. Blair (Cambridge University Press, Cambridge, England, 1991).

[49] R. W. P. Drever et al., in Quantum Optics, Experimental Gravity, and Measurement Theory, edited by P. Meystre and M. O. Scully, NATO ASI, Ser. B, Vol. 94 (Plenum Press, New York, 1983), pp. 503-514.

[50] R. Schilling (unpublished).

[51] B. J. Meers, Phys. Rev. D 38, 2317 (1988). [52] J. Mizuno, K. A. Strain, P. G. Nelson, J. M. Chen, R. Schilling, A. Rüdiger, W. Winkler, and K. Danzmann, Phys. Lett. A 175, 273 (1993).

[53] P. Kwee et al., Opt. Express 20, 10617 (2012).

[54] C. L. Mueller et al., Rev. Sci. Instrum. 87, 014502 (2016).

[55] T. T. Fricke et al., Classical Quantum Gravity 29, 065005 (2012).

[56] S. M. Aston et al., Classical Quantum Gravity 29, 235004 (2012).

[57] F. Matichard et al., Classical Quantum Gravity 32, 185003 $(2015)$

[58] G. M. Harry et al., Classical Quantum Gravity 24, 405 (2007).

[59] M. Granata et al., Phys. Rev. D 93, 012007 (2016).

[60] A. V. Cumming et al., Classical Quantum Gravity 29, $035003(2012)$.

[61] A. Staley et al., Classical Quantum Gravity 31, 245010 (2014).

[62] L. Barsotti, M. Evans, and P. Fritschel, Classical Quantum Gravity 27, $084026(2010)$

[63] B. Abbott et al., https://dcc.ligo.org/LIGO-P1500248/ public/main.

[64] E. Goetz et al., in Gravitational Waves: Proceedings, of the 8th Edoardo Amaldi Conference, Amaldi, New York, 2009; E. Goetz and R. L. Savage Jr., Classical Quantum Gravity 27, $084024(2010)$.

[65] A. Effler, R. M. S. Schofield, V. V. Frolov, G. González, K. Kawabe, J. R. Smith, J. Birch, and R. McCarthy, Classical Quantum Gravity 32, 035017 (2015).

[66] I. Bartos, R. Bork, M. Factourovich, J. Heefner, S. Márka, Z. Márka, Z. Raics, P. Schwinberg, and D. Sigg, Classical Quantum Gravity 27, 084025 (2010).

[67] J. Aasi et al., Classical Quantum Gravity 32, 115012 (2015).

[68] J. Aasi et al., Phys. Rev. D 87, 022002 (2013).

[69] B. Abbott et al., https://dcc.ligo.org/LIGO-P1500238/ public/main.

[70] L. Nuttall et al., Classical Quantum Gravity 32, 245005 $(2015)$.

[71] L. Lyons, Ann. Appl. Stat. 2, 887 (2008).

[72] S. Klimenko, I. Yakushin, A. Mercer, and G. Mitselmakher, Classical Quantum Gravity 25, 114029 (2008).

[73] R. Lynch, S. Vitale, R. Essick, E. Katsavounidis, and F. Robinet, arXiv:1511.05955.

[74] J. Kanner, T. B. Littenberg, N. Cornish, M. Millhouse, E. Xhakaj, F. Salemi, M. Drago, G. Vedovato, and S. Klimenko, Phys. Rev. D 93, 022002 (2016).

[75] A. Buonanno and T. Damour, Phys. Rev. D 62, 064015 (2000).

[76] L. Blanchet, T. Damour, G. Esposito-Farèse, and B. R. Iyer, Phys. Rev. Lett. 93, 091101 (2004).

[77] A. Taracchini et al., Phys. Rev. D 89, 061502 (2014).

[78] M. Pürrer, Classical Quantum Gravity 31, 195010 (2014).

[79] B. Allen, W. G. Anderson, P. R. Brady, D. A. Brown, and J. D. E. Creighton, Phys. Rev. D 85, 122006 (2012).

[80] B. S. Sathyaprakash and S. V. Dhurandhar, Phys. Rev. D 44, 3819 (1991).

[81] B.J. Owen and B. S. Sathyaprakash, Phys. Rev. D 60, 022002 (1999).

[82] B. Allen, Phys. Rev. D 71, 062001 (2005). [83] J. Abadie et al., Phys. Rev. D 85, 082002 (2012).

[84] S. Babak et al., Phys. Rev. D 87, 024033 (2013).

[85] K. Cannon et al., Astrophys. J. 748, 136 (2012).

[86] S. Privitera, S. R. P. Mohapatra, P. Ajith, K. Cannon, N. Fotopoulos, M. A. Frei, C. Hanna, A. J. Weinstein, and J. T. Whelan, Phys. Rev. D 89, 024003 (2014),

[87] M. Hannam, P. Schmidt, A. Bohé, L. Haegel, S. Husa, F. Ohme, G. Pratten, and M. Pürrer, Phys. Rev. Lett. 113, 151101 (2014).

[88] S. Khan, S. Husa, M. Hannam, F. Ohme, M. Pürrer, X. Jiménez Forteza, and A. Bohé, Phys. Rev. D 93, 044007 (2016).

[89] J. Veitch et al., Phys. Rev. D 91, 042003 (2015).

[90] A. Krolak and B. F. Schutz, Gen. Relativ. Gravit. 19, 1163 (1987).

[91] P. A. R. Ade et al., arXiv:1502.01589.

[92] J. Healy, C. O. Lousto, and Y. Zlochower, Phys. Rev. D 90, 104004 (2014).

[93] S. Husa, S. Khan, M. Hannam, M. Pürrer, F. Ohme, X. Jiménez Forteza, and A. Bohé, Phys. Rev. D 93, 044006 (2016).

[94] B. Abbott et al., https://dcc.ligo.org/LIGO-P1500213/ public/main.

[95] C. K. Mishra, K. G. Arun, B. R. Iyer, and B. S. Sathyaprakash, Phys. Rev. D 82, 064010 (2010).

[96] T. G. F. Li, W. Del Pozzo, S. Vitale, C. Van Den Broeck, M. Agathos, J. Veitch, K. Grover, T. Sidery, R. Sturani, and A. Vecchio, Phys. Rev. D 85, 082003 (2012),

[97] C. M. Will, Phys. Rev. D 57, 2061 (1998).

[98] C. Talmadge, J. P. Berthias, R. W. Hellings, and E. M. Standish, Phys. Rev. Lett. 61, 1159 (1988).

[99] L. S. Finn and P. J. Sutton, Phys. Rev. D 65, 044022 (2002).

[100] A. S. Goldhaber and M. M. Nieto, Phys. Rev. D 9, 1119 (1974).

[101] S. Choudhury and S. SenGupta, Eur. Phys. J. C 74, 3159 (2014).

[102] A. Tutukov and L. Yungelson, Nauchnye Informatsii 27, 70 (1973).

[103] V. M. Lipunov, K. A. Postnov, and M. E. Prokhorov, Mon. Not. R. Astron. Soc. 288, 245 (1997).

[104] K. Belczynski, S. Repetto, D. Holz, R. O'Shaughnessy, T. Bulik, E. Berti, C. Fryer, M. Dominik, arXiv:1510.04615 [Astrophys. J. (to be published)].

[105] S. Sigurdsson and L. Hernquist, Nature (London) 364, 423 (1993).

[106] S. F. Portegies Zwart and S. L. W. McMillan, Astrophys. J. Lett. 528, L17 (2000).

[107] C. L. Rodriguez, M. Morscher, B. Pattabiraman, S. Chatterjee, C.-J. Haster, and F. A. Rasio, Phys. Rev. Lett. 115, 051101 (2015),

[108] K. Belczynski, T. Bulik, C. L. Fryer, A. Ruiter, F. Valsecchi, J. S. Vink, and J. R. Hurley, Astrophys. J. 714, 1217 (2010).

[109] M. Spera, M. Mapelli, and A. Bressan, Mon. Not. R. Astron. Soc. 451, 4086 (2015).

[110] B. Abbott et al., https://dcc.ligo.org/LIGO-P1500262/ public/main [Astrophys. J. Lett. (to be published)].

[111] B. Abbott et al., https://dcc.ligo.org/LIGO-P1500217/ public/main. [112] C. Kim, V. Kalogera, and D. R. Lorimer, Astrophys. J. 584, 985 (2003).

[113] W. M. Farr, J. R. Gair, I. Mandel, and C. Cutler, Phys. Rev. D 91, 023005 (2015).

[114] J. Abadie et al., Classical Quantum Gravity 27, 173001 (2010).

[115] B. Abbott et al., https://dcc.ligo.org/LIGO-P1500222/ public/main. [116] LIGO Open Science Center (LOSC), https://losc.ligo.org/ events/GW150914/.

[117] B. P. Abbott et al. (LIGO Scientific Collaboration and Virgo Collaboration), Living Rev. Relativity 19, 1 (2016).

[118] B. Iyer et al., LIGO-India Technical Report No. LIGOM1100296, 2011, https://dcc.ligo.org/LIGO-M1100296/ public/main.

B. P. Abbott, ${ }_{1}^{1}$ R. Abbott, ${ }_{1}^{1}$ T. D. Abbott, ${ }^{2}$ M. R. Abernathy, ${ }_{1}^{1}$ F. Acernese, ${ }^{3,4}$ K. Ackley, ${ }^{5}$ C. Adams, ${ }^{6}$ T. Adams, ${ }^{7}$ P. Addesso, ${ }^{3}$ R. X. Adhikari, ${ }_{1}$ V. B. Adya, ${ }_{8}^{8}$ C. Affeldt, ${ }^{8}$ M. Agathos, ${ }^{9}$ K. Agatsuma, ${ }^{9}$ N. Aggarwal, ${ }^{10}$ O. D. Aguiar, ${ }^{11}$ L. Aiello, ${ }^{12,13}$ A. Ain, ${ }^{14}$ P. Ajith, ${ }^{15}$ B. Allen, ${ }^{8,16,17}$ A. Allocca, ${ }^{18,19}$ P. A. Altin, ${ }^{20}$ S. B. Anderson, ${ }^{1}$ W. G. Anderson, ${ }^{16}$ K. Arai, ${ }^{1}$ M. A. Arain, ${ }^{5}$ M. C. Araya, ${ }^{1}$ C. C. Arceneaux, ${ }^{21}$ J. S. Areeda, ${ }^{22}$ N. Arnaud, ${ }^{23}$ K. G. Arun, ${ }^{24}$ S. Ascenzi, ${ }^{25,13}$ G. Ashton, $^{26}$ M. Ast, ${ }^{27}$ S. M. Aston, ${ }^{6}$ P. Astone, ${ }^{28}$ P. Aufmuth, ${ }^{8}$ C. Aulbert, ${ }^{8}$ S. Babak, ${ }^{29}$ P. Bacon, ${ }^{30}$ M. K. M. Bader, ${ }^{9}$ P. T. Baker, ${ }^{31}$ F. Baldaccini, ${ }^{32,33}$ G. Ballardin, ${ }^{34}$ S. W. Ballmer, ${ }^{35}$ J. C. Barayoga, ${ }^{1}$ S. E. Barclay, ${ }^{36}$ B. C. Barish, ${ }^{1}$ D. Barker, ${ }^{37}$ F. Barone, ${ }^{3,4}$ B. Barr, ${ }^{36}$ L. Barsotti, ${ }^{10}$ M. Barsuglia, ${ }^{30}$ D. Barta, ${ }^{38}$ J. Bartlett, ${ }^{37}$ M. A. Barton, ${ }^{37}$ I. Bartos, ${ }^{39}$ R. Bassiri, ${ }^{40}$ A. Basti, ${ }^{18,19}$ J. C. Batch, ${ }^{37}$ C. Baune, ${ }^{8}$ V. Bavigadda, ${ }^{34}$ M. Bazzan, ${ }^{41,42}$ B. Behnke, ${ }^{29}$ M. Bejger, ${ }^{43}$ C. Belczynski, ${ }^{44}$ A. S. Bell, ${ }^{36}$ C. J. Bell, ${ }^{36}$ B. K. Berger, ${ }^{1}$ J. Bergman, ${ }^{37}$ G. Bergmann, ${ }^{8}$ C. P. L. Berry, ${ }^{45}$ D. Bersanetti, ${ }^{46,47}$ A. Bertolini, ${ }^{9}$ J. Betzwieser, ${ }^{6}$ S. Bhagwat, ${ }^{35}$ R. Bhandare, ${ }^{48}$ I. A. Bilenko, ${ }^{49}$ G. Billingsley, ${ }_{1}$ J. Birch, ${ }^{6}$ R. Birney, ${ }^{50}$ O. Birnholtz, ${ }^{8}$ S. Biscans, ${ }^{10}$ A. Bisht, ${ }^{8,17}$ M. Bitossi, ${ }^{34}$ C. Biwer, ${ }^{35}$ M. A. Bizouard, ${ }^{23}$ J. K. Blackburn, ${ }^{1}$ C. D. Blair, ${ }^{51}$ D. G. Blair, ${ }^{51}$ R. M. Blair, ${ }^{37}$ S. Bloemen, ${ }^{52}$ O. Bock, ${ }^{8}$ T. P. Bodiya, ${ }_{10}^{10}$ M. Boer, ${ }^{53}$ G. Bogaert, ${ }^{53}$ C. Bogan, ${ }^{8}$ A. Bohe, ${ }^{29}$ P. Bojtos, ${ }^{54}$ C. Bond, ${ }^{45}$ F. Bondu, ${ }^{55}$ R. Bonnand, B. A. Boom, ${ }^{9}$ R. Bork, ${ }^{1}$ V. Boschi, ${ }^{18,19}$ S. Bose, ${ }^{56,14}$ Y. Bouffanais, ${ }^{30}$ A. Bozzi, ${ }^{34}$ C. Bradaschia, ${ }^{19}$ P. R. Brady, ${ }^{16}$ V. B. Braginsky, ${ }^{49}$ M. Branchesi, ${ }^{57,58}$ J. E. Brau, ${ }^{59}$ T. Briant, ${ }^{60}$ A. Brillet, ${ }^{53}$ M. Brinkmann, ${ }^{8}$ V. Brisson, ${ }^{23}$ P. Brockill, ${ }^{16}$ A. F. Brooks, ${ }_{1}^{1}$ D. A. Brown, ${ }^{35}$ D. D. Brown, ${ }^{45}$ N. M. Brown, ${ }^{10}$ C. C. Buchanan, ${ }^{2}$ A. Buikema, ${ }^{10}$ T. Bulik, ${ }^{44}$ H. J. Bulten, ${ }^{61,9}$ A. Buonanno, ${ }^{29,62}$ D. Buskulic, ${ }^{7}$ C. Buy, ${ }^{30}$ R. L. Byer, ${ }^{40}$ M. Cabero, ${ }^{8}$ L. Cadonati, ${ }^{63}$ G. Cagnoli, ${ }^{64,65}$ C. Cahillane, ${ }^{1}$ J. Calderón Bustillo, ${ }^{66,63}$ T. Callister, ${ }^{1}$ E. Calloni, ${ }^{67,4}$ J. B. Camp, ${ }^{68}$ K. C. Cannon, ${ }^{69}$ J. Cao, ${ }^{70}$ C. D. Capano, ${ }^{8}$ E. Capocasa, ${ }^{30}$ F. Carbognani, ${ }^{34}$ S. Caride, ${ }^{71}$ J. Casanueva Diaz, ${ }^{23}$ C. Casentini, ${ }^{25,13}$ S. Caudill, ${ }^{16}$ M. Cavaglià, ${ }^{21}$ F. Cavalier, ${ }^{23}$ R. Cavalieri, ${ }^{34}$ G. Cella, ${ }^{19}$ C. B. Cepeda, ${ }^{1}$ L. Cerboni Baiardi, ${ }^{57,58}$ G. Cerretani, ${ }^{18,19}$ E. Cesarini, ${ }^{25,13}$ R. Chakraborty, ${ }^{1}$ T. Chalermsongsak, ${ }^{1}$ S. J. Chamberlin, ${ }^{72}$ M. Chan, ${ }^{36}$ S. Chao, ${ }^{73}$ P. Charlton, ${ }^{74}$ E. Chassande-Mottin, ${ }^{30}$ H. Y. Chen, ${ }^{75}$ Y. Chen, ${ }^{76}$ C. Cheng, ${ }^{73}$ A. Chincarini, ${ }^{47}$ A. Chiummo, ${ }^{34}$ H. S. Cho, ${ }^{77}$ M. Cho, ${ }^{62}$ J. H. Chow, ${ }^{20}$ N. Christensen, ${ }^{78}$ Q. Chu, ${ }^{51}$ S. Chua, ${ }^{60}$ S. Chung, ${ }^{51}$ G. Ciani, ${ }^{5}$ F. Clara, ${ }^{37}$ J. A. Clark, ${ }^{63}$ F. Cleva, ${ }^{53}$ E. Coccia, ${ }^{25,12,13}$ P.-F. Cohadon, ${ }^{60}$ A. Colla, ${ }^{79,28}$ C. G. Collette, ${ }^{80}$ L. Cominsky, ${ }^{81}$ M. Constancio Jr., ${ }^{11}$ A. Conte, ${ }^{79,28}$ L. Conti, ${ }^{42}$ D. Cook, ${ }^{37}$ T. R. Corbitt, ${ }^{2}$ N. Cornish, ${ }^{31}$ A. Corsi, ${ }^{71}$ S. Cortese, ${ }^{34}$ C. A. Costa, ${ }^{11}$ M. W. Coughlin, ${ }^{78}$ S. B. Coughlin, ${ }^{82}$ J.-P. Coulon, ${ }^{53}$ S. T. Countryman, ${ }^{39}$ P. Couvares, ${ }^{1}$ E. E. Cowan, ${ }^{63}$ D. M. Coward, ${ }^{51}$ M. J. Cowart, ${ }^{6}$ D. C. Coyne, ${ }^{1}$ R. Coyne, ${ }^{71}$ K. Craig, ${ }^{36}$ J. D. E. Creighton, ${ }^{16}$ T. D. Creighton, ${ }^{83}$ J. Cripe, ${ }^{2}$ S. G. Crowder, ${ }^{84}$ A. M. Cruise, ${ }^{45}$ A. Cumming, ${ }^{36}$ L. Cunningham, ${ }^{36}$ E. Cuoco, ${ }^{34}$ T. Dal Canton, ${ }^{8}$ S. L. Danilishin, ${ }^{36}$ S. D'Antonio, ${ }^{13}$ K. Danzmann, ${ }^{17,8}$ N. S. Darman, ${ }^{85}$ C. F. Da Silva Costa, ${ }^{5}$ V. Dattilo, ${ }^{34}$ I. Dave, $^{48}$ H. P. Daveloza, ${ }^{83}$ M. Davier, ${ }^{23}$ G. S. Davies, ${ }^{36}$ E. J. Daw, ${ }^{86}$ R. Day, ${ }^{34}$ S. De $^{35}$ D. DeBra, ${ }^{40}$ G. Debreczeni, ${ }^{38}$ J. Degallaix, $^{65}$ M. De Laurentis, ${ }^{67,4}$ S. Deléglise, ${ }^{60}$ W. Del Pozzo, ${ }^{45}$ T. Denker, ${ }^{8,17}$ T. Dent, ${ }^{8}$ H. Dereli, ${ }^{53}$ V. Dergachev, ${ }^{1}$ R. T. DeRosa, ${ }^{6}$ R. De Rosa, ${ }^{67,4}$ R. DeSalvo, ${ }^{87}$ S. Dhurandhar, ${ }^{14}$ M. C. Díaz, ${ }^{83}$ L. Di Fiore, ${ }^{4}$ M. Di Giovanni, ${ }^{79,28}$ A. Di Lieto, ${ }^{18,19}$ S. Di Pace, ${ }^{79,28}$ I. Di Palma, ${ }^{29,8}$ A. Di Virgilio, ${ }^{19}$ G. Dojcinoski, ${ }^{88}$ V. Dolique, ${ }^{65}$ F. Donovan, ${ }^{10}$ K. L. Dooley, ${ }^{21}$ S. Doravari, ${ }^{6,8}$ R. Douglas, ${ }^{36}$ T. P. Downes, ${ }^{16}$ M. Drago, ${ }^{8,89,90}$ R. W. P. Drever, ${ }^{1}$ J. C. Driggers, ${ }^{37}$ Z. Du, ${ }^{70}$ M. Ducrot, ${ }^{7}$ S. E. Dwyer, ${ }^{37}$ T. B. Edo, ${ }^{86}$ M. C. Edwards, ${ }^{78}$ A. Effler, ${ }^{6}$ H.-B. Eggenstein, ${ }^{8}$ P. Ehrens, ${ }^{1}$ J. Eichholz, ${ }^{5}$ S. S. Eikenberry, ${ }^{5}$ W. Engels, ${ }^{76}$ R. C. Essick, ${ }_{10}^{10}$ T. Etzel, ${ }^{1}$ M. Evans, ${ }^{10}$ T. M. Evans, ${ }^{6}$ R. Everett, ${ }^{72}$ M. Factourovich, ${ }^{39}$ V. Fafone, ${ }^{25,13,12}$ H. Fair, ${ }^{35}$ S. Fairhurst, ${ }^{91}$ X. Fan, ${ }^{70}$ Q. Fang, ${ }^{51}$ S. Farinon, ${ }^{47}$ B. Farr, ${ }^{75}$ W. M. Farr, ${ }^{45}$ M. Favata, ${ }^{88}$ M. Fays, ${ }^{91}$ H. Fehrmann, ${ }^{8}$ M. M. Fejer, ${ }^{40}$ D. Feldbaum, ${ }^{5}$ I. Ferrante, ${ }^{18,19}$ E. C. Ferreira, ${ }^{11}$ F. Ferrini, ${ }^{34}$ F. Fidecaro, ${ }^{18,19}$ L. S. Finn, ${ }^{72}$ I. Fiori, $^{34}$ D. Fiorucci, ${ }^{30}$ R. P. Fisher, ${ }^{35}$ R. Flaminio, ${ }^{65,92}$ M. Fletcher, ${ }^{36}$ H. Fong, ${ }^{69}$ J.-D. Fournier, ${ }^{53}$ S. Franco, ${ }^{23}$ S. Frasca, ${ }^{79,28}$ F. Frasconi, ${ }^{19}$ M. Frede, ${ }^{8}$ Z. Frei, ${ }^{54}$ A. Freise, ${ }^{45}$ R. Frey, ${ }^{59}$ V. Frey, ${ }^{23}$ T. T. Fricke, ${ }^{8}$ P. Fritschel, ${ }^{10}$ V. V. Frolov, ${ }^{6}$ P. Fulda, M. Fyffe, ${ }^{6}$ H. A. G. Gabbard, ${ }^{21}$ J. R. Gair, ${ }^{93}$ L. Gammaitoni, ${ }^{32,33}$ S. G. Gaonkar, ${ }^{14}$ F. Garufi, ${ }^{67,4}$ A. Gatto, ${ }^{30}$ G. Gaur, ${ }^{94,95}$ N. Gehrels, ${ }^{68}$ G. Gemme, ${ }^{47}$ B. Gendre, ${ }^{53}$ E. Genin,${ }^{34}$ A. Gennai, ${ }^{19}$ J. George, ${ }^{48}$ L. Gergely, ${ }^{96}$ V. Germain, ${ }^{7}$ Abhirup Ghosh, ${ }^{15}$ Archisman Ghosh, ${ }^{15}$ S. Ghosh,${ }^{52,9}$ J. A. Giaime,${ }^{2,6}$ K. D. Giardina, ${ }^{6}$ A. Giazotto, ${ }^{19}$ K. Gill,,${ }^{97}$ A. Glaefke, ${ }^{36}$ J. R. Gleason, ${ }^{5}$ E. Goetz, ${ }^{98}$ R. Goetz, ${ }^{5}$ L. Gondan, ${ }^{54}$ G. González, ${ }^{2}$ J. M. Gonzalez Castro, ${ }^{18,19}$ A. Gopakumar, ${ }^{99}$ N. A. Gordon, ${ }^{36}$ M. L. Gorodetsky, ${ }^{49}$ S. E. Gossan, ${ }^{1}$ M. Gosselin, ${ }^{34}$ R. Gouaty, ${ }^{7}$ C. Graef,${ }^{36}$ P. B. Graff, ${ }^{62}$ M. Granata, ${ }^{65}$ A. Grant, ${ }^{36}$ S. Gras, ${ }^{10}$ C. Gray, ${ }^{37}$ G. Greco, ${ }^{57,58}$ A. C. Green, ${ }^{45}$ R. J. S. Greenhalgh, ${ }^{100}$ P. Groot, ${ }^{52}$ H. Grote, ${ }^{8}$ S. Grunewald, ${ }^{29}$ G. M. Guidi, ${ }^{57,58}$ X. Guo, ${ }^{70}$ A. Gupta, ${ }^{14}$ M. K. Gupta, ${ }^{95}$ K. E. Gushwa, ${ }^{1}$ E. K. Gustafson, ${ }^{1}$ R. Gustafson, ${ }^{98}$ J. J. Hacker, ${ }^{22}$ B. R. Hall, ${ }^{56}$ E. D. Hall, ${ }^{1}$ G. Hammond, ${ }^{36}$ M. Haney, ${ }_{9}^{99}$ M. M. Hanke, ${ }^{8}$ J. Hanks,${ }^{37}$ C. Hanna ${ }^{72}$ M. D. Hannam, ${ }^{91}$ J. Hanson, ${ }^{6}$ T. Hardwick, ${ }^{2}$ J. Harms,${ }^{57,58}$ G. M. Harry, ${ }_{101}^{1}$ I. W. Harry, ${ }^{29}$ M. J. Hart, ${ }^{36}$ M. T. Hartman, ${ }^{5}$ C.-J. Haster, ${ }^{45}$ K. Haughian, ${ }^{36}$ J. Healy, ${ }^{102}$ J. Heefner, ${ }^{1, a}$ A. Heidmann, ${ }^{60}$ M. C. Heintze, ${ }^{5,6}$ G. Heinzel, ${ }^{8}$ H. Heitmann, ${ }^{53}$ P. Hello, ${ }^{23}$ G. Hemming, ${ }^{34}$ M. Hendry, ${ }^{36}$ I. S. Heng, ${ }^{36}$ J. Hennig, ${ }^{36}$ A. W. Heptonstall, ${ }^{1}$ M. Heurs, ${ }^{8,17}$ S. Hild,${ }^{36}$ D. Hoak, ${ }^{103}$ K. A. Hodge, ${ }^{1}$ D. Hofman, ${ }^{65}$ S. E. Hollitt, ${ }^{104}$ K. Holt, ${ }^{6}$ D. E. Holz,${ }^{75}$ P. Hopkins,${ }^{91}$ D. J. Hosken, ${ }^{104}$ J. Hough, ${ }^{36}$ E. A. Houston, ${ }^{36}$ E. J. Howell, ${ }^{51}$ Y. M. Hu,${ }^{36}$ S. Huang, ${ }^{73}$ E. A. Huerta, ${ }^{105,82}$ D. Huet,${ }^{23}$ B. Hughey, ${ }^{97}$ S. Husa, ${ }^{66}$ S. H. Huttner, ${ }^{36}$ T. Huynh-Dinh, ${ }^{6}$ A. Idrisy, ${ }^{72}$ N. Indik, ${ }^{8}$ D. R. Ingram, ${ }^{37}$ R. Inta, ${ }^{71}$ H. N. Isa, ${ }^{36}$ J.-M. Isac, ${ }^{60}$ M. Isi, ${ }^{1}$ G. Islas, ${ }^{22}$ T. Isogai, ${ }^{10}$ B. R. Iyer, ${ }^{15}$ K. Izumi, ${ }^{37}$ M. B. Jacobson, ${ }^{1}$ T. Jacqmin, ${ }^{60}$ H. Jang, ${ }^{77}$ K. Jani, ${ }^{63}$ P. Jaranowski, ${ }^{106}$ S. Jawahar, ${ }^{107}$ F. Jiménez-Forteza, ${ }^{66}$ W. W. Johnson, ${ }^{2}$ N. K. Johnson-McDaniel, ${ }^{15}$ D. I. Jones,$^{26}$ R. Jones, ${ }^{36}$ R. J. G. Jonker, ${ }^{9}$ L. Ju, ${ }^{51}$ K. Haris, ${ }^{108}$ C. V. Kalaghatgi, ${ }^{24,91}$ V. Kalogera, ${ }^{82}$ S. Kandhasamy, ${ }^{21}$ G. Kang, ${ }^{77}$ J. B. Kanner, ${ }^{1}$ S. Karki, ${ }^{59}$ M. Kasprzack, ${ }^{2,23,34}$ E. Katsavounidis, ${ }^{10}$ W. Katzman, ${ }^{6}$ S. Kaufer, ${ }^{17}$ T. Kaur, ${ }^{51}$ K. Kawabe ${ }^{37}$ F. Kawazoe,${ }^{8,17}$ F. Kéfélian ${ }^{53}$ M. S. Kehl, ${ }^{69}$ D. Keitel,,${ }^{8,66}$ D. B. Kelley, ${ }^{35}$ W. Kells, ${ }^{1}$ R. Kennedy, ${ }^{86}$ D. G. Keppel, ${ }^{8}$ J. S. Key, ${ }^{83}$ A. Khalaidovski, ${ }^{8}$ F. Y. Khalili, ${ }^{49}$ I. Khan, ${ }^{12}$ S. Khan, ${ }^{91}$ Z. Khan, ${ }^{95}$ E. A. Khazanov, ${ }^{109}$ N. Kijbunchoo, ${ }^{37}$ C. Kim, ${ }^{77}$ J. Kim, ${ }^{110}$ K. Kim, ${ }^{111}$ Nam-Gyu Kim, ${ }^{77}$ Namjun Kim, ${ }^{40}$ Y.-M. Kim, ${ }^{110}$ E. J. King, ${ }^{104}$ P. J. King, ${ }^{37}$ D. L. Kinzel,,${ }^{6}$ J. S. Kissel,,${ }^{37}$ L. Kleybolte, ${ }^{27}$ S. Klimenko, ${ }^{5}$ S. M. Koehlenbeck, ${ }^{8}$ K. Kokeyama, ${ }^{2}$ S. Koley, ${ }^{9}$ V. Kondrashov, ${ }^{1}$ A. Kontos, ${ }^{10}$ S. Koranda, ${ }^{16}$ M. Korobko, ${ }^{27}$ W. Z. Korth, ${ }^{1}$ I. Kowalska, ${ }^{44}$ D. B. Kozak, ${ }^{1}$ V. Kringel, ${ }^{8}$ B. Krishnan, ${ }_{8}^{8}$ A. Królak,,${ }^{112,113}$ C. Krueger, ${ }^{17}$ G. Kuehn, ${ }^{8}$ P. Kumar, ${ }^{69}$ R. Kumar, ${ }^{36}$ L. Kuo, ${ }^{73}$ A. Kutynia, ${ }^{112}$ P. Kwee, ${ }^{8}$ B. D. Lackey, ${ }^{35}$ M. Landry, ${ }^{37}$ J. Lange, ${ }^{102}$ B. Lantz ${ }^{40}$ P. D. Lasky, ${ }^{114}$ A. Lazzarini, ${ }^{1}$ C. Lazzaro, ${ }^{63,42}$ P. Leaci, ${ }^{29,79,28}$ S. Leavey, ${ }^{36}$ E. O. Lebigot, ${ }^{30,70}$ C. H. Lee, ${ }^{110}$ H. K. Lee, ${ }^{111}$ H. M. Lee, ${ }^{115}$ K. Lee, ${ }^{36}$ A. Lenon,${ }^{35}$ M. Leonardi, ${ }^{89,90}$ J. R. Leong, ${ }^{8}$ N. Leroy, ${ }^{23}$ N. Letendre, ${ }^{7}$ Y. Levin, ${ }^{114}$ B. M. Levine, ${ }^{37}$ T. G. F. Li, ${ }^{1}$ A. Libson, ${ }^{10}$ T. B. Littenberg, ${ }^{116}$ N. A. Lockerbie, ${ }^{107}$ J. Logue, ${ }^{36}$ A. L. Lombardi, ${ }^{103}$ L. T. London, ${ }^{91}$ J. E. Lord, ${ }^{35}$ M. Lorenzini, ${ }^{12,13}$ V. Loriette, ${ }^{117}$ M. Lormand, ${ }^{6}$ G. Losurdo, ${ }^{58}$ J. D. Lough,${ }^{8,17}$ C. O. Lousto, ${ }^{102}$ G. Lovelace, ${ }^{22}$ H. Lück, ${ }^{17,8}$ A. P. Lundgren, ${ }^{8}$ J. Luo, ${ }^{78}$ R. Lynch, ${ }^{10}$ Y. Ma, ${ }^{51}$ T. MacDonald, ${ }^{40}$ B. Machenschalk, ${ }^{8}$ M. MacInnis, ${ }^{10}$ D. M. Macleod, ${ }_{2}^{2}$ F. Magaña-Sandoval, ${ }^{35}$ R. M. Magee, ${ }^{56}$ M. Mageswaran, ${ }^{1}$ E. Majorana, ${ }^{28}$ I. Maksimovic ${ }^{117}$ V. Malvezzi, ${ }^{25,13}$ N. Man, ${ }^{53}$ I. Mandel, ${ }^{45}$ V. Mandic ${ }^{84}$ V. Mangano, ${ }^{36}$ G. L. Mansell, ${ }^{20}$ M. Manske, ${ }^{16}$ M. Mantovani, ${ }^{34}$ F. Marchesoni,,${ }^{188,33}$ F. Marion, ${ }^{7} \mathrm{~S}$. Márka, ${ }^{39}$ Z. Márka, ${ }^{39}$ A. S. Markosyan ${ }^{40}$ E. Maros, ${ }^{1}$ F. Martelli, ${ }^{57,58}$ L. Martellini, ${ }^{53}$ I. W. Martin, ${ }^{36}$ R. M. Martin, ${ }^{5}$ D. V. Martynov, ${ }^{1}$ J. N. Marx, ${ }^{1}$ K. Mason, ${ }^{10}$ A. Masserot, ${ }^{7}$ T. J. Massinger, ${ }^{35}$ M. Masso-Reid,${ }^{36}$ F. Matichard, ${ }^{10}$ L. Matone ${ }^{39}$ N. Mavalvala, ${ }^{10}$ N. Mazumder, ${ }^{56}$ G. Mazzolo, ${ }^{8}$ R. McCarthy, ${ }^{37}$ D. E. McClelland,${ }^{20}$ S. McCormick, ${ }^{6}$ S. C. McGuire, ${ }^{119}$ G. McIntyre, ${ }^{1}$ J. McIver, ${ }^{1}$ D. J. McManus, ${ }^{20}$ S. T. McWilliams, ${ }^{105}$ D. Meacher, ${ }^{72}$ G. D. Meadors, ${ }^{29,8}$ J. Meidam,,${ }^{9}$ A. Melatos, ${ }^{85}$ G. Mendell, ${ }^{37}$ D. Mendoza-Gandara, ${ }^{8}$ R. A. Mercer, ${ }^{16}$ E. Merilh, ${ }^{37}$ M. Merzougui, ${ }^{53}$ S. Meshkov, ${ }^{1} \mathrm{C}$. Messenger,${ }^{36}$ C. Messick, ${ }^{72}$ P. M. Meyers,${ }^{84}$ F. Mezzani, ${ }^{28,79}$ H. Miao, ${ }^{45}$ C. Michel, ${ }^{65}$ H. Middleton, ${ }^{45}$ E. E. Mikhailov, ${ }^{120}$ L. Milano, ${ }^{67,4}$ J. Miller, ${ }^{10}$ M. Millhouse, ${ }^{31}$ Y. Minenkov, ${ }^{13}$ J. Ming, ${ }^{29,8}$ S. Mirshekari, ${ }^{121}$ C. Mishra, ${ }^{15}$ S. Mitra, ${ }^{14}$ V. P. Mitrofanov, ${ }^{49}$ G. Mitselmakher, ${ }^{5}$ R. Mittleman, ${ }^{10}$ A. Moggi, ${ }^{19}$ M. Mohan, ${ }^{34}$ S. R. P. Mohapatra, ${ }^{10}$ M. Montani, ${ }^{57,58}$ B. C. Moore, ${ }^{88}$ C. J. Moore, ${ }^{122}$ D. Moraru, ${ }^{37}$ G. Moreno, ${ }^{37}$ S. R. Morriss, ${ }^{83}$ K. Mossavi, ${ }^{8}$ B. Mours, ${ }^{7}$ C. M. Mow-Lowry, ${ }^{45}$ C. L. Mueller, ${ }^{5}$ G. Mueller, ${ }^{5}$ A. W. Muir, ${ }^{91}$ Arunava Mukherjee,$^{15}$ D. Mukherjee, ${ }^{16}$ S. Mukherjee ${ }^{83}$ N. Mukund, ${ }^{14}$ A. Mullavey, ${ }^{6}$ J. Munch, ${ }^{104}$ D. J. Murphy, ${ }^{39}$ P. G. Murray, ${ }^{36}$ A. Mytidis, ${ }^{5}$ I. Nardecchia, ${ }^{25,13}$ L. Naticchioni ${ }^{79,28}$ R. K. Nayak, ${ }^{123}$ V. Neculaa ${ }^{5}$ K. Nedkova, ${ }^{103}$ G. Nelemans, ${ }^{52,9}$ M. Neri, ${ }^{4,47}$ A. Neunzert, ${ }^{98}$ G. Newton, ${ }^{36}$ T. T. Nguyen, ${ }^{20}$ A. B. Nielsen, ${ }^{8}$ S. Nissanke, ${ }^{52,9}$ A. Nitz, ${ }^{8}$ F. Nocera, ${ }^{34}$ D. Nolting, ${ }^{6}$ M. E. N. Normandin,${ }^{83}$ L. K. Nuttall, ${ }^{35}$ J. Oberling, ${ }^{37}$ E. Ochsner, ${ }^{16}$ J. O'Dell, ${ }^{100}$ E. Oelker, ${ }^{10}$ G. H. Ogin, ${ }^{124}$ J. J. Oh ${ }_{1}^{125}$ S. H. Oh,${ }^{125}$ F. Ohme, ${ }^{91}$ M. Oliver, ${ }^{66}$ P. Oppermann, ${ }^{8}$ Richard J. Oram, ${ }^{6}$ B. O'Reilly, ${ }^{6}$ R. O'Shaughnessy, ${ }^{102}$ C. D. Ott, ${ }^{76}$ D. J. Ottaway, ${ }^{104}$ R. S. Ottens, ${ }^{5}$ H. Overmier, ${ }^{6}$ B. J. Owen, ${ }^{71}$ A. Pai, ${ }^{108}$ S. A. Pai, ${ }^{48}$ J. R. Palamos, ${ }^{59}$ O. Palashov, ${ }^{109}$ C. Palomba, ${ }^{28}$ A. Pal-Singh, ${ }^{27}$ H. Pan, ${ }^{73}$ Y. Pan, ${ }^{62}$ C. Pankow, ${ }^{82}$ F. Pannarale, ${ }^{91}$ B. C. Pant, ${ }^{48}$ F. Paoletti, ${ }^{34,19}$ A. Paoli, ${ }^{34}$ M. A. Papa ${ }^{29,16,8}$ H. R. Paris, ${ }^{40}$ W. Parker, ${ }^{6}$ D. Pascucci, ${ }^{36}$ A. Pasqualetti, ${ }^{34}$ R. Passaquieti, ${ }^{18,19}$ D. Passuello, ${ }^{19}$ B. Patricelli, ${ }^{18,19}$ Z. Patrick, ${ }^{40}$ B. L. Pearlstone, ${ }^{36}$ M. Pedraza,,${ }^{1}$ R. Pedurand, ${ }^{65}$ L. Pekowsky, ${ }^{35}$ A. Pele, ${ }^{6}$ S. Penn, ${ }^{126}$ A. Perreca, ${ }^{1}$ H. P. Pfeiffer, ${ }^{69,29}$ M. Phelps, ${ }^{36}$ O. Piccinni, ${ }^{79,28}$ M. Pichot, ${ }^{53}$ M. Pickenpack, ${ }^{8}$ F. Piergiovanni, ${ }^{57,58}$ V. Pierro, ${ }^{87}$ G. Pillant, ${ }^{34}$ L. Pinard, ${ }^{65}$ I. M. Pinto, ${ }^{87}$ M. Pitkin, ${ }^{36}$ J. H. Poeld, ${ }^{8}$ R. Poggiani, ${ }^{18,19}{ }^{19}$. Popolizio, ${ }^{34}$ A. Post $^{8}$ J. Powell, ${ }^{36}$ J. Prasad, ${ }^{14}$ V. Predoi, ${ }^{91}$ S. S. Premachandra, ${ }_{114}$ T. Prestegard, ${ }^{84}$ L. R. Price, ${ }^{1}$ M. Prijatelj, ${ }^{34}$ M. Principe, ${ }^{87}$ S. Privitera, ${ }^{29}$ R. Prix, ${ }^{8}$ G. A. Prodi, ${ }^{89,90}$ L. Prokhorov, ${ }^{49}$ O. Puncken, ${ }^{8}$ M. Punturo, ${ }^{33}$ P. Puppo, ${ }^{28}$ M. Pürrer, ${ }^{29} \mathrm{H}_{\text {Q }}$ Qi $^{16}{ }^{16}$ J. Qin, ${ }^{51}$ V. Quetschke, ${ }^{83}$ E. A. Quintero, ${ }^{1}$ R. Quitzow-James, ${ }^{59}$ F. J. Raab, ${ }^{37}$ D. S. Rabeling ${ }^{20}$ H. Radkins, ${ }^{37}$ P. Raffai, ${ }^{54}$ S. Raja, ${ }^{48}$ M. Rakhmanov, ${ }^{83}$ C. R. Ramet, ${ }^{6}$ P. Rapagnani, ${ }^{79,28}$ V. Raymond ${ }^{29}$ M. Razzano, ${ }^{18,19}$ V. Re, ${ }^{25}$ J. Read, ${ }^{22}$ C. M. Reed, ${ }^{37}$ T. Regimbau ${ }^{53}$ L. Rei,${ }^{47}$ S. Reid, ${ }^{50}$ D. H. Reitze, ${ }^{1,5}$ H. Rew, ${ }^{120}$ S. D. Reyes,${ }^{35}$ F. Ricci, ${ }^{79,28}$ K. Riles, ${ }^{98}$ N. A. Robertson, ${ }^{1,36}$ R. Robie, ${ }^{36}$ F. Robinet, ${ }^{23}$ A. Rocchi, ${ }^{13}$ L. Rolland, ${ }^{7}$ J. G. Rollins, ${ }^{1}$ V. J. Roma, ${ }^{59}$ J. D. Romano, ${ }^{83}$ R. Romano, ${ }^{3,4}$ G. Romanov, ${ }^{120}$ J. H. Romie, ${ }^{6}$ D. Rosińska, ${ }^{127,43}$ S. Rowan, ${ }^{36}$ A. Rüdiger, ${ }^{8}$ P. Ruggi,,${ }^{34}$ K. Ryan,$^{37}$ S. Sachdev, ${ }^{1}$ T. Sadecki, ${ }^{37}$ L. Sadeghian, ${ }^{16}$ L. Salconi, ${ }^{34}$ M. Saleem, ${ }^{108}$ F. Salemi, ${ }^{8}$ A. Samajdar, ${ }^{123}$ L. Sammut, ${ }^{85,114}$ L. M. Sampson, ${ }^{82}$ E. J. Sanchez, ${ }^{1}$ V. Sandberg, ${ }^{37}$ B. Sandeen, ${ }^{82}$ G. H. Sanders, ${ }^{1}$ J. R. Sanders ${ }^{98,35}$ B. Sassolas, ${ }^{65}$ B. S. Sathyaprakash, ${ }^{91}$ P. R. Saulson, ${ }^{35}$ O. Sauter, ${ }^{98}$ R. L. Savage, ${ }^{37}$ A. Sawadsky, ${ }^{17}$ P. Schale, ${ }^{59}$ R. Schilling,${ }^{8, \mathrm{~b}}$ J. Schmidt, ${ }^{8}$ P. Schmidt, ${ }^{1,76}$ R. Schnabel, ${ }^{27}$ R. M. S. Schofield, ${ }^{59}$ A. Schönbeck, ${ }^{27}$ E. Schreiber, ${ }^{8}$ D. Schuette, ${ }^{8,17}$ B. F. Schutz, ${ }^{91,29}$ J. Scott, ${ }^{36}$ S. M. Scott, ${ }^{20}$ D. Sellers, ${ }^{6}$ A. S. Sengupta, ${ }^{94}$ D. Sentenac, ${ }^{34}$ V. Sequino, ${ }^{25,13}$ A. Sergeev, ${ }^{109}$ G. Serna, ${ }^{22}$ Y. Setyawati, ${ }^{52,9}$ A. Sevigny, ${ }^{37}$ D. A. Shaddock, ${ }^{20}$ T. Shaffer, ${ }^{37}$ S. Shah, ${ }^{52,9}$ M. S. Shahriar, ${ }^{82}$ M. Shaltev, ${ }^{8}$ Z. Shao, ${ }^{1}$ B. Shapiro, ${ }^{40}$ P. Shawhan, ${ }^{62}$ A. Sheperd, ${ }^{16}$ D. H. Shoemaker, ${ }^{10}$ D. M. Shoemaker, ${ }^{63}$ K. Siellez, ${ }^{53,63}$ X. Siemens, ${ }^{16}$ D. Sigg ${ }^{37}$ A. D. Silva, ${ }^{11}$ D. Simakov, ${ }^{8}$ A. Singer, ${ }^{1}$ L. P. Singer, ${ }^{68}$ A. Singh, ${ }^{29,8}$ R. Singh, ${ }^{2}$ A. Singhal, ${ }^{12}$ A. M. Sintes, ${ }^{66}$ B. J. J. Slagmolen, ${ }^{20}$ J. R. Smith, ${ }^{22}$ M. R. Smith, ${ }^{1}$ N. D. Smith, ${ }^{1}$ R. J. E. Smith, ${ }^{1}$ E. J. Son, ${ }^{125}$ B. Sorazu, ${ }^{36}$ F. Sorrentino, ${ }^{47}$ T. Souradeep ${ }^{14}$ A. K. Srivastava, ${ }^{95}$ A. Staley, ${ }^{39}$ M. Steinke, ${ }^{8}$ J. Steinlechner,${ }^{36}$ S. Steinlechner, ${ }^{36}$ D. Steinmeyer,${ }^{8,17}$ B. C. Stephens, ${ }^{16}$ S. P. Stevenson, ${ }^{45}$ R. Stone, ${ }^{83}$ K. A. Strain,${ }^{36}$ N. Straniero, ${ }^{65}$ G. Stratta, ${ }^{57,58}$ N. A. Strauss, ${ }^{78}$ S. Strigin,,$^{49}$ R. Sturani, ${ }^{121}$ A. L. Stuver, ${ }^{6}$ T. Z. Summerscales, ${ }^{128}$ L. Sun ${ }^{85}$ P. J. Sutton, ${ }^{91}$ B. L. Swinkels, ${ }^{34}$ M. J. Szczepańczyk, ${ }^{97}$ M. Tacca, ${ }^{30}$ D. Talukder, ${ }^{59}$ D. B. Tanner, ${ }^{5}$ M. Tápai, ${ }^{96}$ S. P. Tarabrin, ${ }^{8}$ A. Taracchini, ${ }^{29}$ R. Taylor, ${ }^{1}$ T. Theeg, ${ }^{8}$ M. P. Thirugnanasambandam, ${ }^{1}$ E. G. Thomas, ${ }^{45}$ M. Thomas, ${ }^{6}$ P. Thomas, ${ }^{37} \mathrm{~K}$. A. Thorne, ${ }^{6} \mathrm{~K}$. S. Thorne ${ }^{76}$ E. Thrane, ${ }^{114}$ S. Tiwari, ${ }^{12}$ V. Tiwari, ${ }^{91}$ K. V. Tokmakov ${ }_{107}$ C. Tomlinson, ${ }^{86}$ M. Tonelli, ${ }^{18,19}$ C. V. Torres, ${ }^{83, \mathrm{c}}$ C. I. Torrie, ${ }^{1}$ D. Töyrä, ${ }^{45}$ F. Travasso, ${ }^{32,33}$ G. Traylor, ${ }^{6}$ D. Trifirò, ${ }^{21}$ M. C. Tringali ${ }^{89,90}$ L. Trozzo, ${ }^{129,19}$ M. Tse ${ }^{10}$ M. Turconi, ${ }^{53}$ D. Tuyenbayev, ${ }^{83}$ D. Ugolini, ${ }^{130}$ C. S. Unnikrishnan, ${ }^{99}$ A. L. Urban, ${ }^{16}$ S. A. Usman, ${ }^{35}$ H. Vahlbruch, ${ }^{17}$ G. Vajente, ${ }^{1}$ G. Valdes, ${ }^{83}$ M. Vallisneri, ${ }^{76}$ N. van Bakel, ${ }^{9}$ M. van Beuzekom, ${ }^{9}$ J. F. J. van den Brand, ${ }^{61,9}$ C. Van Den Broeck,,${ }^{9}$ D. C. Vander-Hyde, ${ }^{35,22}$ L. van der Schaaf, ${ }^{9}$ J. V. van Heijningen, ${ }^{9}$ A. A. van Veggel, ${ }^{36}$ M. Vardaro, ${ }^{41,42}$ S. Vass, ${ }^{1}$ M. Vasúth,,$^{38}$ R. Vaulin, ${ }^{10}$ A. Vecchio, ${ }^{45}$ G. Vedovato, ${ }^{42}$ J. Veitch, ${ }^{45}$ P. J. Veitch, ${ }^{104}$ K. Venkateswara, ${ }^{131}$ D. Verkindt, ${ }^{7}$ F. Vetrano, ${ }^{57,58}$ A. Viceré,,${ }^{57,58}$ S. Vinciguerra, ${ }^{45}$ D. J. Vine,${ }^{50}$ J.-Y. Vinet, ${ }^{53}$ S. Vitale, ${ }^{10}$ T. Vo, ${ }^{35}$ H. Vocca, ${ }^{32,33}$ C. Vorvick, ${ }^{37}$ D. Voss, ${ }^{5}$ W. D. Vousden, ${ }^{45}$ S. P. Vyatchanin, ${ }^{49}$ A. R. Wade ${ }^{20}$ L. E. Wade,${ }^{132}$ M. Wade, ${ }^{132}$ S. J. Waldman, ${ }^{10}$ M. Walker, ${ }^{2}$ L. Wallace, ${ }^{1}$ S. Walsh, ${ }^{16,8,29}$ G. Wang, ${ }^{12}$ H. Wang, ${ }^{45}$ M. Wang, ${ }^{45}$ X. Wang, ${ }^{70}$ Y. Wang ${ }_{5}{ }^{51}$ H. Ward, ${ }^{36}$ R. L. Ward,${ }^{20}$ J. Warner, ${ }^{37}$ M. Was, ${ }^{7}$ B. Weaver, ${ }^{37}$ L.-W. Wei, ${ }^{53}$ M. Weinert, ${ }^{8}$ A. J. Weinstein, ${ }^{1}$ R. Weiss ${ }_{10}^{10}$ T. Welborn, ${ }_{6}^{6}$ L. Wen,${ }^{51}$ P. Weßels, ${ }^{8}$ T. Westphal, ${ }^{8}$ K. Wette, ${ }^{8}$ J. T. Whelan, ${ }^{102,8}$ S. E. Whitcomb, ${ }^{1}$ D. J. White, ${ }^{86}$ B. F. Whiting, ${ }^{5}$ K. Wiesner, ${ }^{8}$ C. Wilkinson, ${ }^{37}$ P. A. Willems, ${ }^{1}$ L. Williams, ${ }^{5}$ R. D. Williams, ${ }^{1}$ A. R. Williamson, ${ }^{91}$ J. L. Willis, ${ }^{133}$ B. Willke, ${ }^{17,8}$ M. H. Wimmer, ${ }^{8,17}$ L. Winkelmann, ${ }^{8}$ W. Winkler, ${ }^{8}$ C. C. Wipf, ${ }^{1}$ A. G. Wiseman, ${ }^{16}$ H. Wittel, ${ }^{8,17}$ G. Woan, ${ }^{36}$ J. Worden,${ }^{37}$ J. L. Wright ${ }^{36}$ G. Wu, ${ }^{6}$ J. Yablon,${ }^{82}$ I. Yakushin, ${ }^{6}$ W. Yam, ${ }^{10}$ H. Yamamoto, ${ }_{1}$ C. C. Yancey, ${ }^{62}$ M. J. Yap ${ }^{20}$ H. Yu,${ }^{10}$ M. Yvert, ${ }^{7}$ A. Zadrożny, ${ }^{112}$ L. Zangrando, ${ }^{42}$ M. Zanolin,${ }^{97}$ J.-P. Zendri, ${ }^{42}$ M. Zevin, ${ }^{82}$ F. Zhang, ${ }^{10}$ L. Zhang, ${ }_{1}$ M. Zhang, ${ }^{120}$ Y. Zhang, ${ }^{102}$ C. Zhao, ${ }^{51}$ M. Zhou, ${ }^{82}$ Z. Zhou, ${ }^{82}$ X. J. Zhu, ${ }^{51}$ M. E. Zucker, ${ }^{1,10}$ S. E. Zuraw, ${ }^{103}$ and J. Zweizig ${ }^{1}$

(LIGO Scientific Collaboration and Virgo Collaboration)

${ }^{1}$ LIGO, California Institute of Technology, Pasadena, California 91125, USA

${ }^{2}$ Louisiana State University, Baton Rouge, Louisiana 70803, USA

${ }^{3}$ Università di Salerno, Fisciano, I-84084 Salerno, Italy

${ }^{4}$ INFN, Sezione di Napoli, Complesso Universitario di Monte S. Angelo, I-80126 Napoli, Italy

${ }^{5}$ University of Florida, Gainesville, Florida 32611, USA

${ }^{6}$ LIGO Livingston Observatory, Livingston, Louisiana 70754, USA

${ }^{7}$ Laboratoire d'Annecy-le-Vieux de Physique des Particules (LAPP), Université Savoie Mont Blanc, CNRS/IN2P3, F-74941 Annecy-le-Vieux, France

${ }^{8}$ Albert-Einstein-Institut, Max-Planck-Institut für Gravitationsphysik, D-30167 Hannover, Germany

${ }^{9}$ Nikhef, Science Park, 1098 XG Amsterdam, Netherlands ${ }^{10}$ LIGO, Massachusetts Institute of Technology, Cambridge, Massachusetts 02139, USA

${ }^{11}$ Instituto Nacional de Pesquisas Espaciais, 12227-010 São José dos Campos, São Paulo, Brazil

${ }^{12}$ INFN, Gran Sasso Science Institute, I-67100 L'Aquila, Italy

${ }^{13}$ INFN, Sezione di Roma Tor Vergata, I-00133 Roma, Italy

${ }^{14}$ Inter-University Centre for Astronomy and Astrophysics, Pune 411007, India

${ }^{15}$ International Centre for Theoretical Sciences, Tata Institute of Fundamental Research, Bangalore 560012, India

${ }^{16}$ University of Wisconsin-Milwaukee, Milwaukee, Wisconsin 53201, USA

${ }^{17}$ Leibniz Universität Hannover, D-30167 Hannover, Germany

${ }^{18}$ Università di Pisa, I-56127 Pisa, Italy

${ }^{19}$ INFN, Sezione di Pisa, I-56127 Pisa, Italy

${ }^{20}$ Australian National University, Canberra, Australian Capital Territory 0200, Australia

${ }^{21}$ The University of Mississippi, University, Mississippi 38677, USA

${ }^{22}$ California State University Fullerton, Fullerton, California 92831, USA

${ }^{23} L A L$, Université Paris-Sud, CNRS/IN2P3, Université Paris-Saclay, Orsay, France

${ }^{24}$ Chennai Mathematical Institute, Chennai, India 603103

${ }^{25}$ Università di Roma Tor Vergata, I-00133 Roma, Italy

${ }^{26}$ University of Southampton, Southampton SO17 1BJ, United Kingdom

${ }^{27}$ Universität Hamburg, D-22761 Hamburg, Germany

${ }^{28}$ INFN, Sezione di Roma, I-00185 Roma, Italy

${ }^{29}$ Albert-Einstein-Institut, Max-Planck-Institut für Gravitationsphysik, D-14476 Potsdam-Golm, Germany

${ }^{30}$ APC, AstroParticule et Cosmologie, Université Paris Diderot, CNRS/IN2P3, CEA/Irfu, Observatoire de Paris,

Sorbonne Paris Cité, F-75205 Paris Cedex 13, France

${ }^{31}$ Montana State University, Bozeman, Montana 59717, USA

${ }^{32}$ Università di Perugia, I-06123 Perugia, Italy

${ }^{33}$ INFN, Sezione di Perugia, I-06123 Perugia, Italy

${ }^{34}$ European Gravitational Observatory (EGO), I-56021 Cascina, Pisa, Italy

${ }^{35}$ Syracuse University, Syracuse, New York 13244, USA

${ }^{36} S U P A$, University of Glasgow, Glasgow G12 8QQ, United Kingdom

${ }^{37}$ LIGO Hanford Observatory, Richland, Washington 99352, USA

${ }^{38}$ Wigner RCP, RMKI, H-1121 Budapest, Konkoly Thege Miklós út 29-33, Hungary

${ }^{39}$ Columbia University, New York, New York 10027, USA

${ }^{40}$ Stanford University, Stanford, California 94305, USA

${ }^{41}$ Università di Padova, Dipartimento di Fisica e Astronomia, I-35131 Padova, Italy

${ }^{42}$ INFN, Sezione di Padova, I-35131 Padova, Italy

${ }^{43} C A M K-P A N, 00-716$ Warsaw, Poland

${ }^{44}$ Astronomical Observatory Warsaw University, 00-478 Warsaw, Poland

${ }^{45}$ University of Birmingham, Birmingham B15 2TT, United Kingdom

${ }^{46}$ Università degli Studi di Genova, I-16146 Genova, Italy

${ }^{47}$ INFN, Sezione di Genova, I-16146 Genova, Italy

${ }^{48} R R C A T$, Indore MP 452013, India

${ }^{49}$ Faculty of Physics, Lomonosov Moscow State University, Moscow 119991, Russia

${ }^{50}$ SUPA, University of the West of Scotland, Paisley PA1 2BE, United Kingdom

${ }^{51}$ University of Western Australia, Crawley, Western Australia 6009, Australia

${ }^{52}$ Department of Astrophysics/IMAPP, Radboud University Nijmegen, P.O. Box 9010, 6500 GL Nijmegen, Netherlands

${ }^{53}$ Artemis, Université Côte d'Azur, CNRS, Observatoire Côte d'Azur, CS 34229, Nice cedex 4, France

${ }^{54}$ MTA Eötvös University, "Lendulet" Astrophysics Research Group, Budapest 1117, Hungary

${ }^{55}$ Institut de Physique de Rennes, CNRS, Université de Rennes 1, F-35042 Rennes, France

${ }^{56}$ Washington State University, Pullman, Washington 99164, USA

${ }^{57}$ Università degli Studi di Urbino "Carlo Bo," I-61029 Urbino, Italy

${ }^{58}$ INFN, Sezione di Firenze, I-50019 Sesto Fiorentino, Firenze, Italy

${ }^{59}$ University of Oregon, Eugene, Oregon 97403, USA

${ }^{60}$ Laboratoire Kastler Brossel, UPMC-Sorbonne Universités, CNRS, ENS-PSL Research University, Collège de France, F-75005 Paris, France

${ }^{61} V U$ University Amsterdam, $1081 \mathrm{HV}$ Amsterdam, Netherlands

${ }^{62}$ University of Maryland, College Park, Maryland 20742, USA

${ }^{63}$ Center for Relativistic Astrophysics and School of Physics, Georgia Institute of Technology, Atlanta, Georgia 30332, USA

${ }^{64}$ Institut Lumière Matière, Université de Lyon, Université Claude Bernard Lyon 1, UMR CNRS 5306, 69622 Villeurbanne, France

${ }^{65}$ Laboratoire des Matériaux Avancés (LMA), IN2P3/CNRS, Université de Lyon, F-69622 Villeurbanne, Lyon, France

${ }^{66}$ Universitat de les Illes Balears, IAC3_IEEC, E-07122 Palma de Mallorca, Spain

${ }^{67}$ Università di Napoli "Federico II," Complesso Universitario di Monte S. Angelo, I-80126 Napoli, Italy ${ }^{68}$ NASA/Goddard Space Flight Center, Greenbelt, Maryland 20771, USA

${ }^{69}$ Canadian Institute for Theoretical Astrophysics, University of Toronto, Toronto, Ontario M5S 3H8, Canada

${ }^{70}$ Tsinghua University, Beijing 100084, China

${ }^{71}$ Texas Tech University, Lubbock, Texas 79409, USA

${ }^{72}$ The Pennsylvania State University, University Park, Pennsylvania 16802, USA

${ }^{73}$ National Tsing Hua University, Hsinchu City, 30013 Taiwan, Republic of China

${ }^{74}$ Charles Sturt University, Wagga Wagga, New South Wales 2678, Australia

${ }^{75}$ University of Chicago, Chicago, Illinois 60637, USA

${ }^{76}$ Caltech CaRT, Pasadena, California 91125, USA

${ }^{77}$ Korea Institute of Science and Technology Information, Daejeon 305-806, Korea

${ }^{78}$ Carleton College, Northfield, Minnesota 55057, USA

${ }^{79}$ Università di Roma "La Sapienza," I-00185 Roma, Italy

${ }^{80}$ University of Brussels, Brussels 1050, Belgium

${ }^{81}$ Sonoma State University, Rohnert Park, California 94928, USA

${ }^{82}$ Northwestern University, Evanston, Illinois 60208, USA

${ }^{83}$ The University of Texas Rio Grande Valley, Brownsville, Texas 78520, USA

${ }^{84}$ University of Minnesota, Minneapolis, Minnesota 55455, USA

${ }^{85}$ The University of Melbourne, Parkville, Victoria 3010, Australia

${ }^{86}$ The University of Sheffield, Sheffield S10 2TN, United Kingdom

${ }^{87}$ University of Sannio at Benevento, I-82100 Benevento, Italy and INFN, Sezione di Napoli, I-80100 Napoli, Italy

${ }^{88}$ Montclair State University, Montclair, New Jersey 07043, USA

${ }^{89}$ Università di Trento, Dipartimento di Fisica, I-38123 Povo, Trento, Italy

${ }^{90}$ INFN, Trento Institute for Fundamental Physics and Applications, I-38123 Povo, Trento, Italy

${ }^{91}$ Cardiff University, Cardiff CF24 3AA, United Kingdom

${ }^{92}$ National Astronomical Observatory of Japan, 2-21-1 Osawa, Mitaka, Tokyo 181-8588, Japan

${ }^{93}$ School of Mathematics, University of Edinburgh, Edinburgh EH9 3FD, United Kingdom

${ }^{94}$ Indian Institute of Technology, Gandhinagar Ahmedabad Gujarat 382424, India

${ }^{95}$ Institute for Plasma Research, Bhat, Gandhinagar 382428, India

${ }^{96}$ University of Szeged, Dóm tér 9, Szeged 6720, Hungary

${ }^{97}$ Embry-Riddle Aeronautical University, Prescott, Arizona 86301, USA

${ }^{98}$ University of Michigan, Ann Arbor, Michigan 48109, USA

${ }^{99}$ Tata Institute of Fundamental Research, Mumbai 400005, India

${ }^{100}$ Rutherford Appleton Laboratory, HSIC, Chilton, Didcot, Oxon OX11 OQX, United Kingdom

${ }^{101}$ American University, Washington, D.C. 20016, USA

${ }^{102}$ Rochester Institute of Technology, Rochester, New York 14623, USA

${ }^{103}$ University of Massachusetts-Amherst, Amherst, Massachusetts 01003, USA

${ }^{104}$ University of Adelaide, Adelaide, South Australia 5005, Australia

${ }^{105}$ West Virginia University, Morgantown, West Virginia 26506, USA

${ }^{106}$ University of Biat ystok, 15-424 Biat ystok, Poland

${ }^{107}$ SUPA, University of Strathclyde, Glasgow G1 $1 X Q$, United Kingdom

${ }^{108}$ IISER-TVM, CET Campus, Trivandrum Kerala 695016, India

${ }^{109}$ Institute of Applied Physics, Nizhny Novgorod, 603950, Russia

${ }^{110}$ Pusan National University, Busan 609-735, Korea

${ }^{111}$ Hanyang University, Seoul 133-791, Korea

${ }^{112}$ NCBJ, 05-400 Świerk-Otwock, Poland

${ }^{113}$ IM-PAN, 00-956 Warsaw, Poland

${ }^{114}$ Monash University, Victoria 3800, Australia

${ }^{115}$ Seoul National University, Seoul 151-742, Korea

${ }^{116}$ University of Alabama in Huntsville, Huntsville, Alabama 35899, USA

${ }^{117}$ ESPCI, CNRS, F-75005 Paris, France

${ }^{118}$ Università di Camerino, Dipartimento di Fisica, I-62032 Camerino, Italy

${ }^{119}$ Southern University and A\&M College, Baton Rouge, Louisiana 70813, USA

${ }^{120}$ College of William and Mary, Williamsburg, Virginia 23187, USA

${ }^{121}$ Instituto de Física Teórica, University Estadual Paulista/ICTP South American Institute for Fundamental Research, São Paulo SP 01140-070, Brazil

${ }^{122}$ University of Cambridge, Cambridge CB2 $1 T N$, United Kingdom

${ }^{123}$ IISER-Kolkata, Mohanpur, West Bengal 741252, India

${ }^{124}$ Whitman College, 345 Boyer Avenue, Walla Walla, Washington 99362 USA

${ }^{125}$ National Institute for Mathematical Sciences, Daejeon 305-390, Korea

${ }^{126}$ Hobart and William Smith Colleges, Geneva, New York 14456, USA ${ }^{127}$ Janusz Gil Institute of Astronomy, University of Zielona Góra, 65-265 Zielona Góra, Poland

${ }^{128}$ Andrews University, Berrien Springs, Michigan 49104, USA

${ }^{129}$ Università di Siena, I-53100 Siena, Italy

${ }^{130}$ Trinity University, San Antonio, Texas 78212, USA

${ }^{131}$ University of Washington, Seattle, Washington 98195, USA

${ }^{132}$ Kenyon College, Gambier, Ohio 43022, USA

${ }^{133}$ Abilene Christian University, Abilene, Texas 79699, USA

${ }^{\mathrm{a}}$ Deceased, April 2012.

${ }^{b}$ Deceased, May 2015.

${ }^{c}$ Deceased, March 2015."""

summary_prompt_3 = r"""
Hello, Claude. Your task is to provide a comprehensive overview of the academic research paper provided within the <paper> tags. The paper is formatted in LaTeX, but your overview should be written primarily in Markdown, with the exception of mathematical equations which should be written in LaTeX between $$ signs so it can be rendered in markdown. Aim for an analysis of approximately 3000 words or more, to ensure a thorough explanation of not only the paper's content, but also the fundamental principles and mechanics of the subject matter.

    Abstract: Start by delivering a detailed understanding of the research question and its significance. Interpret the methods and results in depth. Discuss the real-world implications and applications of the findings, and decode any specialized terms used in the abstract.

    Introduction: Present a full exploration of the central topic or question. Discuss the motivations, objectives, and hypotheses of the research. Illuminate the key concepts and background information upon which the research is based. How does this research contribute to the existing body of knowledge?

    Methodology: Immerse yourself deeply into the methods, experimental design, and analysis methods used. Discuss the reasoning behind the choices made during the study. Highlight the strengths and potential limitations of the methodology. When explaining mathematical or statistical methods, use LaTeX enclosed by $$ to illustrate the equations, and explain why these methods are appropriate for the research question.

    Results: Interpret the key findings, outcomes, and patterns in detail. Discuss how these results answer the research questions or hypotheses. When referencing figures, use the "mathpix" link provided above the figure declaration in the paper. For instance, when discussing a figure in the results, phrase it as: "As shown in Fig.~X: ![]([Insert corresponding mathpix link here]), [insert explanation of figure here]". Elucidate any statistical significance and its relevance in an easy-to-understand manner.

    Discussion & Conclusion: Discuss the findings in relation to prior work, and explain the theoretical and practical applications of the results. Summarize the key takeaways and contributions of the research. Discuss the implications of these findings for the broader field of study.

    Subject Matter Analysis: This part is critical. You should detail the principles, theories, and workings of the core subject matter of the research. Discuss how the research findings apply to or change our understanding of these principles.

Remember, your tone should be academic yet accessible. Technical terms are expected, but make sure you provide clear and concise explanations. Utilize analogies and metaphors to help clarify complex ideas, and explain figures, tables, and any mathematical concepts (using LaTeX for equations) whenever they are important for understanding. Discuss any limitations or potential areas for future research mentioned in the paper. Your goal is to make this research accessible to someone who has a solid understanding of the subject but is not a specialist in the field."""


summary_prompt_2 = r"""Hello Claude, I have an academic research paper here written in LaTeX that I need your assistance with. It's nestled between the <paper> tags, and your task is to transform it into a more accessible version using Markdown for the main text, but remember to maintain mathematical equations in LaTeX format, placed between $$ signs.

Your analysis should be reasonably thorough, targeting around 1000 to 1500 words. You should focus on conveying the main ideas, methodologies, results, and implications of the paper in a way that someone with a basic understanding of the subject can comprehend. Here's a suggested structure for your analysis:

Overview: Begin with a succinct summary of the research paper's key question or idea, its significance, and what the researchers hoped to achieve. Make this engaging and steer clear of complex jargon.

Problem Statement: Break down the problem or question the research attempts to address. Highlight its relevance and introduce any key concepts in a straightforward manner.

Methodology: Describe the methods the researchers employed. If any mathematical or statistical methods were used, display these in LaTeX format between $$$$ signs, and provide a basic explanation of why these methods were appropriate.

Results: Offer a clear explanation of the study's findings. When referencing figures, use the "mathpix" link provided above the figure declaration in the paper. For example, when discussing a figure in the results, phrase it as: "As shown in Fig.~X: ![]([Insert corresponding mathpix link here]), [insert plain English explanation of figure here]". Make sure to explain the importance of these results.

Discussion & Conclusion: Summarize the main discoveries and their potential practical implications. Compare these findings with previous work and discuss the new understanding this research contributes to the field.

Subject Matter Analysis: This is a key component. Help the reader understand the fundamental principles and theories that the research builds upon. Describe how these findings modify or add to our comprehension of these principles.

Your tone should be educational and approachable. Although technical terms may be necessary, always offer plain language explanations or comparisons to help illustrate complex concepts. Don't forget to discuss any limitations or suggestions for future research mentioned in the paper. The aim is to make the research accessible to those with a basic understanding of the field, but not necessarily an expertise."""


summary_prompt_1 = r"""Hello Claude! You know how a translator turns one language into another? Well, I need you to do something similar, but with a twist. I want you to transform a detailed, technical research paper into plain English that's so simple and understandable, even my grandma could get it.

Academic papers can often feel like they're written in another language, with all their jargon and complex concepts. What I need you to do is to untangle these elements, and present the main ideas in a way that's as easy as pie. Here's a handy guide to help you out:

Simple Summary: Start by telling us what the paper is all about, like you're telling a story. Make it interesting and show us why the research matters. Use simple, everyday words that everyone knows.

Problem Simplified: Break down the problem the researchers were looking at. Try to compare it to everyday problems or experiences we all have, so it's relatable and easy to understand.

Method Unpacked: Explain how the researchers went about solving the problem. Use comparisons and everyday examples to make their process clear.

Results Revealed: Tell us what the researchers found out, but do it in a way that anyone could understand. Help us see why their findings are important.

Implications Explained: Show us why these results matter in real life. Talk about how they might change everyday things, and use examples to make it clear.

Remember, while you're doing this, your tone should be respectful and helpful. Your goal is to make this interesting and engaging for everyone. If you have to use any fancy words or technical terms, make sure you explain them in a way that's as clear as day. For example, if you have to talk about something like an attention mechanism in a neural network, compare it to how someone reading a book can pay attention to one sentence, but still understand the whole story.


"""


general_quiz_prompt = r"""{
  Claude, let's construct an enlightening quiz that tests a reader's understanding of the core concepts and takeaways from an academic research paper. This quiz should consist of TEN questions which focus on the research's main themes, methodologies, results, and implications.

Each question should:

    Target fundamental aspects of the research. Avoid questions about specific, detailed facts such as the datasets used or minor procedural steps. Instead, aim for questions that reflect an understanding of the research's overarching methodology, significant discoveries, or potential implications.

    Present four potential answers.

    Clearly identify the correct answer among these options.

    Provide a comprehensive explanation for the correct answer. This explanation should not just justify the right answer, but also clarify the reasoning behind it, how it connects to the information in the summary, and why the other answers are incorrect.

Structure each question, its answers, the identified correct answer, and the detailed explanation into a single JSON object. This object should have the keys 'question', 'options', 'answer', and 'explanation'.

Compile these objects into a JSON array to complete the quiz. Ensure that the 'answer' is always one of the 'options' and matches it exactly.

Your output should be the JSON array alone, without any additional text or data. Aim to create TEN questions. Ensure your task is complete when you have ten questions that probe the reader's understanding of the research's central themes, not its minor details.
"""

client = anthropic.Client(st.secrets['anthropic_api_key'])


def convert_to_latex_with_mathpix(uploaded_pdf_path):
    url = "https://api.mathpix.com/v3/pdf"

    headers = {
        "app_id": st.secrets['mathpix_id'],
        "app_key": st.secrets['mathpix_api_key'],
        "Content-type": "application/json"
    }

    data = {
        "options_json": json.dumps({
            "conversion_formats": {
                "tex.zip": True
            },
            "math_inline_delimiters": ["$", "$"],
            "rm_spaces": True
        })
    }

    with open(uploaded_pdf_path, 'rb') as f:
        files = {'file': f}
        r = requests.post(url, headers=headers, data=data, files=files)
    
    if r.status_code != 200:
        print(f"Error: {r.status_code}, {r.text}")
        return None

    latex_content = r.text

    return latex_content



def ask__claude(combined_prompt, temp=0.3):
    no_tokens = anthropic.count_tokens(combined_prompt)
    response = ""
    print(f"ASKING CLAUDE {no_tokens}: {combined_prompt}")
    if no_tokens > 100000:
        raise ValueError(f"Text is too long {no_tokens}.")
    response = client.completion(
        prompt=f"{anthropic.HUMAN_PROMPT} {combined_prompt} {anthropic.AI_PROMPT}",
        model="claude-v1.3-100k",
        max_tokens_to_sample=100000,
        temperature=temp,
    )
    return response

def get_question_context(question):
    print("Getting question context")
    sections_dict = st.session_state['sections']
    vectorizer = TfidfVectorizer()
    sections_texts = list(sections_dict.values())
    X = vectorizer.fit_transform(sections_texts + [question])
    cosine_similarities = cosine_similarity(X[-1:], X[:-1])[0]
    sorted_indices = cosine_similarities.argsort()[::-1]
    most_similar_section_index = sorted_indices[0]
    second_most_similar_section_index = sorted_indices[1]
    most_relevant_section_text = sections_texts[most_similar_section_index]
    second_most_relevant_section_text = sections_texts[second_most_similar_section_index]
    abstract_section_text = sections_texts[0]
    most_relevant_section_key = list(sections_dict.keys())[most_similar_section_index]
    second_most_relevant_section_key = list(sections_dict.keys())[second_most_similar_section_index]
    abstract_section_key = list(sections_dict.keys())[0]

    print(f"MOST RELEVANT SECTION: {most_relevant_section_key} : {most_relevant_section_text}")
    print(f"SECOND MOST RELEVANT SECTION: {second_most_relevant_section_key} : {second_most_relevant_section_text}")
    print(f"ABSTRACT: {abstract_section_key} : {abstract_section_text}")

    relevant_sections = {
        most_relevant_section_key: most_relevant_section_text,
        second_most_relevant_section_key: second_most_relevant_section_text,
        abstract_section_key: abstract_section_text,
    }
    combined_text = "\n".join(relevant_sections.values())

    return combined_text





def ask_claude_question(question,concise):
    print("ASKING CLAUDE A QUESTION")
    question_context = get_question_context(question)
    question_prompt = ""
    question_prompt_1 = fr"""Hello, Claude. I'm seeking your help to respond to a user's question about the following section from an academic research paper. Ensure your answer is accessible and easy to grasp for anyone, 
    even if they have no prior knowledge of the subject matter. Express your response in Markdown format, but for any mathematical equations, please use LaTeX enclosed within $$. When referencing figures, use the 'mathpix' 
    links given above the figure declaration in the LaTeX document. The user's question is: {question}. The relevant context from the paper is: {question_context}."""

    questions_prompt_2 = fr""" Greetings, Claude. A user is curious about a particular portion of an academic research paper. Your job is to deliver a clear and concise answer, suitable for someone with a basic understanding of the field. 
    Please formulate your response in Markdown, but make sure to use LaTeX for any mathematical equations. If there are figures involved, use the 'mathpix' links provided above the figure declaration in the LaTeX document. Also, provide a 
    brief explanation of any math involved using LaTeX. The question posed by the user is: {question}. The context from the paper is: {question_context}."""

    questions_prompt_3 = fr"""Hello, Claude. We have an inquiry from a user about an excerpt from an academic research paper. Your response should be in-depth and detailed, appropriate for someone with a good grasp of the subject. 
    The response should be written in Markdown, except for any mathematical equations which should be written in LaTeX. If there are figures to refer to, use the 'mathpix' links given above the figure declaration in the LaTeX document, 
    and elaborate on any mathematical concepts using LaTeX. The question from the user is: {question}. The relevant context from the paper is: {question_context}. """

    if st.session_state['slider_value'] == 1:
        question_prompt = question_prompt_1
    elif st.session_state['slider_value'] == 2:
        question_prompt = questions_prompt_2
    elif st.session_state['slider_value'] == 3:
        question_prompt = questions_prompt_3

    if concise:
        question_prompt += "\n **Please keep your response as concise as possible, while still answering the question fully.**"
        
    response = ask__claude(question_prompt)
    response = response['completion']
    return response


prompt_repo = [{"summary":summary_prompt_1 },{"summary":summary_prompt_2},{"summary": summary_prompt_3}]



PAGES = ["Summary", "Questions", "Quiz"]

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", PAGES)

if selection == "Questions":
    concise_responses = st.checkbox('Concise responses')
else:
    concise_responses = False

if selection == "Questions":
    if st.button('Clear Chat'):
        st.session_state['chat_history'] = []
        st.experimental_rerun()


if selection == "Summary":
    if 'slider_value' not in st.session_state:
        st.session_state['slider_value'] = 1

    st.session_state['slider_value'] = st.slider('Comprehensiveness, 1 is a more shallow level understanding, while 3 is a deeper understanding', 1, 3, st.session_state['slider_value'])
    st.write('You selected:', st.session_state['slider_value'])

    PRE_CONVERTED_PAPERS = {
        "Attention is All you Need": attention_is_all_you_need,
        "RWKV": rwkv,
        "Genome engineering using the CRISPR-Cas9 system": crispr_cas9_systems,
        "Observation of Gravitational Waves from a Binary Black Hole Merger": observation_of_gravitational_waves,
    }

    paper_selection = st.selectbox("Choose a pre-converted paper:", list(PRE_CONVERTED_PAPERS.keys()))

    uploaded_pdf = st.file_uploader("Or upload your own PDF:", type=["pdf"])

    if st.button('Submit'):
        depth = st.session_state['slider_value']
        st.session_state['chat_history'] = []
        if uploaded_pdf is not None:
            with st.spinner("Converting PDF to LaTeX..."):
                with tempfile.NamedTemporaryFile(delete=False) as temp_pdf_file:
                    temp_pdf_file.write(uploaded_pdf.getvalue())
                    temp_pdf_file.flush()  
                    st.session_state['raw_latex'] = convert_to_latex_with_mathpix(temp_pdf_file.name)
        else:
            st.session_state['raw_latex'] = PRE_CONVERTED_PAPERS[paper_selection]
        sections = re.split(r"(\\section\{.*?\})", st.session_state['raw_latex'])
        sections = sections[1:]
        section_dict = {}
        for i in range(0, len(sections), 2):
            key = f"Section {i//2 + 1}"
            value = sections[i] + sections[i+1]
            section_dict[key] = value
        st.session_state['sections'] = section_dict
        prepend = fr""" <paper> {st.session_state['raw_latex']} </paper> """
        st.session_state['explanations'] = ask__claude(f"{prepend} {prompt_repo[depth-1]['summary']}")
        st.session_state['quiz_questions'] = ask__claude(f" <summary> {st.session_state['explanations']} </summary> {general_quiz_prompt}",0.3)
        st.session_state['quiz_questions'] = json.loads(st.session_state['quiz_questions']['completion'])
        

    if 'explanations' in st.session_state:
        st.markdown(st.session_state['explanations']['completion'])



    

elif selection == "Questions":
    question = st.text_input("Ask something about the paper")
    if st.button('Ask'):
        if question:
            st.session_state['chat_history'].append(f"##### You: {question}")
            response = ask_claude_question(question, concise_responses)
            st.session_state['chat_history'].append(f"##### Claude: \n {response}")
            st.experimental_rerun()

    for chat in st.session_state['chat_history']:
        st.markdown(chat)


elif selection == "Quiz":
    quiz_questions = st.session_state['quiz_questions']
    for i, question in enumerate(quiz_questions):
        st.markdown(f"### Question {i+1}: {question['question']}")
        options = question['options']
        answers = [st.button(option, key=f"question_{i}_{j}") for j, option in enumerate(options)]
        if True in answers:
            selected_option = options[answers.index(True)]
            if selected_option == question['answer']:
                st.write("Correct! 🎉")
                st.write("Explanation: ", question['explanation'])
            else:
                st.write("Sorry, that's incorrect. Try again!")






   




























