# Resnets for Binary Classification

Model is trained on the same [colored dataset](../cnn/cnn.md) as previously used in cnns, to classify the input images.

Model architecture is as shown below.
<p align="center"> <table> <tr> <td align="center"><img src="resblock_corrected.png" width="128"><br> <sub>Residual Block</sub></td> <td align="center"> <img src="resnet4.png" width="128"><br> <sub>ResNet4</sub> </td> </tr> </table> </p>

<p align="center"><strong>resnet4</strong></p>
<p align="center"> <table> <tr> <td align="center"> <img src="resnet4bc_TL.png" width="220"><br> <sub>Train Loss</sub> </td> <td align="center"> <img src="resnet4bc_VL.png" width="220"><br> <sub>Validation Loss</sub> </td> </tr> <tr> <td align="center"> <img src="resnet4bc_TA.png" width="220"><br> <sub>Train Accuracy</sub> </td> <td align="center"> <img src="resnet4bc_VA.png" width="220"><br> <sub>Validation Accuracy</sub> </td> </tr> </table> </p>

<p align="center"><strong>cnn v resnet4</strong></p>
<p align="center"><img src="cnn_v_resnet.png" width="480"></p>

**Observation**: The learning rate in both the trainings was set to `5e-4`. The primary difference appears to be the number of convolutional operations: ResNet-4 has more convolutions, and consequently more activations. In principle, for shallow networks, ResNets and vanilla CNNs should not differ significantly. However, we observe a substantial performance gap between the two.

**Next Steps:** Retrain CNN with the same number of convoution operations as that of the resnet4 keeping the architecture pretty much the same except the skip connection.

## Resnet4 v CNN4
As expected, the performance difference between the two shallow network architectures is not significant.
<p align="center"><img src="resnet4vcnn4.png" width="480"></p>