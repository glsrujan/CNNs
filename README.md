# Welcome to cnn_1

Introduction to Convolutional Neural Networks

# Findings

***Training Images looked something like this***
<p align="center"> <table> <tr> <td align="center"><img src="docs/resnet8/BW/Circle_0.png" width="64"> <img src="docs/resnet8/BW/Circle_1.png" width="64"> <img src="docs/resnet8/BW/Circle_2.png" width="64"> <img src="docs/resnet8/BW/Circle_3.png" width="64"> <img src="docs/resnet8/BW/Circle_4.png" width="64"> <br> <sub>Class 0 - Circle</sub> </td> </tr> <tr> <td align="center"> <img src="docs/resnet8/BW/Rectangle_0.png" width="64"> <img src="docs/resnet8/BW/Rectangle_1.png" width="64"> <img src="docs/resnet8/BW/Rectangle_2.png" width="64"> <img src="docs/resnet8/BW/Rectangle_3.png" width="64">  <img src="docs/resnet8/BW/Rectangle_3.png" width="64"> <br> <sub>Class 1 - Rectangle</sub></td> </tr> </table> </p>

Binary Cross entropy with logits loss with a CNN Resnet4 architecture, while the train Accuracy reached 90%, validation accuracy was bad. Model did not learn to classify simple Black and white images with circles or rectangles.

<p align="center"> <table> <tr> <td align="center"> <img src="docs/resnet8/resnet8_back_white_images_loss_train.png" width="220"><br> <sub>Train Loss</sub> </td> <td align="center"> <img src="docs/resnet8/resnet8_back_white_images_loss_val.png" width="220"><br> <sub>Validation Loss</sub> </td> </tr> <tr> <td align="center"> <img src="docs/resnet8/resnet8_back_white_images_acc_train.png" width="220"><br> <sub>Train Accuracy</sub> </td> <td align="center"> <img src="docs/resnet8/resnet8_back_white_images_acc_val.png" width="220"><br> <sub>Validation Accuracy</sub> </td> </tr> </table> </p>

`Possible Issues`: Model too complex, Less training data, poor regularisation stratergy. Above all, maybe the way I have designed the architecture itself is flawed. 

<p align="center"> <table> <tr> <td align="center"><img src="docs/resnet8/ResBlock.png" width="128"><br> <sub>Residual Block</sub></td> <td align="center"> <img src="docs/resnet8/Resnet4.png" width="128"><br> <sub>ResNet4</sub> </td> </tr> </table> </p>

**Debugging**
- 
* Played with lr `1e-3` -> `1e-5`
* Played with # of ResBocks `2` -> `4`
* Played with Dropouts `0.2` - `0.5`
* Filter from `in 3` - `out 64` (Kinda knew this was an overkill)

## Fallback to Vanilla CNN

At this point, figured, either there is a bug in my code or the model is too complex, so decided to tank resnet and go with vanilla CNN. Hopefully, will pick up resnet in the future and figure out what really happened here.

<p align="center"> <table> <tr> <td align="center"><img src="docs/resnet8/CNN8.png" width="128"><br> <sub>CNN8</sub></td> </tr> </table> </p>


Got a pretty decent performance, reached Validation accuracy of 98% - 99% 

<p align="center"> <table> <tr> <td align="center"> <img src="docs/resnet8/CNN2_BW_CR_LT.png" width="220"><br> <sub>Train Loss</sub> </td> <td align="center"> <img src="docs/resnet8/CNN2_BW_CR_LV.png" width="220"><br> <sub>Validation Loss</sub> </td> </tr> <tr> <td align="center"> <img src="docs/resnet8/CNN2_BW_CR_AT.png" width="220"><br> <sub>Train Accuracy</sub> </td> <td align="center"> <img src="docs/resnet8/CNN2_BW_CR_AV.png" width="220"><br> <sub>Validation Accuracy</sub> </td> </tr> </table> </p>

**Bumped it up a notch**

Generated colored synthetic images, circles had varying radius

<p align="center"> <table> <tr> <td align="center"><img src="docs/resnet8/colored/Circle_0.png" width="64"> <img src="docs/resnet8/colored/Circle_1.png" width="64"> <img src="docs/resnet8/colored/Circle_2.png" width="64"> <img src="docs/resnet8/colored/Circle_3.png" width="64"> <img src="docs/resnet8/colored/Circle_4.png" width="64"> <br> <sub>Class 0 - Circle</sub> </td> </tr> <tr> <td align="center"> <img src="docs/resnet8/colored/Rectangle_0.png" width="64"> <img src="docs/resnet8/colored/Rectangle_1.png" width="64"> <img src="docs/resnet8/colored/Rectangle_2.png" width="64"> <img src="docs/resnet8/colored/Rectangle_3.png" width="64">  <img src="docs/resnet8/colored/Rectangle_3.png" width="64"> <br> <sub>Class 1 - Rectangle</sub></td> </tr> </table> </p>

<p align="center"> <table> <tr> <td align="center"> <img src="docs/resnet8/CNN2_COL_CR_LT.png" width="220"><br> <sub>Train Loss</sub> </td> <td align="center"> <img src="docs/resnet8/CNN2_COL_CR_LV.png" width="220"><br> <sub>Validation Loss</sub> </td> </tr> <tr> <td align="center"> <img src="docs/resnet8/CNN2_COL_CR_AT.png" width="220"><br> <sub>Train Accuracy</sub> </td> <td align="center"> <img src="docs/resnet8/CNN2_COL_CR_AV.png" width="220"><br> <sub>Validation Accuracy</sub> </td> </tr> </table> </p>

Decent performance, reached Validation accuracy of 98% - 99%.

**Observation**
* Images are 64x64, so circles with 1 px radius looks almost similar to a rectangle due to quantisation, or the ones that are closer to the image boundary. The softmax values for these are very close to the decision boundry or sometimes wrongly classified.

<p align="center"> <table> <tr> <td align="center"> <img src="docs/resnet8/colored/wrongPred_C.png" width="220"><br> <sub>Wrong Prediction</sub> </td> <td align="center"> <img src="docs/resnet8/colored/LowConfidencePred_R.png" width="212"><br> <sub>Low Confidence Prediction</sub> </td> </tr> </table> </p>

