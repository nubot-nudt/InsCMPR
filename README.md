# InsCMPR
This repository contains the implementation of our paper submission: InsCMPR: Efficient Cross-Modal Place Recognition via Instance-Aware Hybrid Mamba-Transformer
* We propose a novel instance-aware modality alignment strategy for the CMPR task. By leveraging a pre-trained vision foundation model, our method aligns multi-modal data at both the pixel and instance levels, effectively mitigating domain shifts and generating superior global descriptors.
* We introduce a novel dual-branch hybrid Mamba-Transformer network for the CMPR task, capable of efficiently processing multi-modal data aligned at different levels in parallel, enhancing the robustness and accuracy of CMPR.
* Extensive experimental results on KITTI, NCLT and HAOMO datasets show that our proposed method can achieve state-of-the-art performance while running in real-time at about 30Hz.
