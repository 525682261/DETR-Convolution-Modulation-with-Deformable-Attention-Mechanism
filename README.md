# End-to-End Target Detection Enhanced by Hierarchical Convolution Modulation and Sparse Attention Transmission <img src="figs/dinosaur.png" width="30">

This is the official implementation of the paper "[End-to-End Target Detection Enhanced by Hierarchical Convolution Modulation and Sparse Attention Transmission](https://github.com/gzyao/detr-project)". 

Authors: [Lixin he]\*, [Guangzhuang Yao]\*,[Zhi Cheng]\*,[Xiaofeng Wang]\*,[Zhi Hu]\*,[Luqing Ge]\*,[Jie Li]\*.

## Data
<details>
  <summary>Data</summary>

Please download [COCO 2017](https://cocodataset.org/) dataset and organize them as following:
```
COCODIR/
  ├── train2017/
  ├── val2017/
  └── annotations/
  	├── instances_train2017.json
  	└── instances_val2017.json
```
</details>

# Abstract
This study proposes an end-to-end target detection model that significantly improves accuracy, particularly for both large and small targets. The key innovations lie in the hierarchical convolution modulation of deformable attention and a sparse feature transmission strategy. By modulating deformable attention with convolution, our approach enables effective extraction of both global and local features. Additionally, a deep-shallow layered convolutional module is employed to enhance coarse- and fine-grained feature extraction. The sparse transmission strategy ensures optimal feature output from the encoder layers. Experimental results on the MS-COCO dataset demonstrate the effectiveness of our approach, achieving state-of-the-art performance with notable gains in detection accuracy for both small and large targets.

# Highlights
1. Multi-scale features are convolved in deep and shallow layers.
2. The traditional deformable attention mechanism is modulated by the layered convolution mentioned above.
3. The sparse feature transmission ensures the optimal output of the last layer encoder.



# Methods
![method](figs/f1.png "model arch")


# Experimental
![Experimental](figs/table.png)

# Detection effect comparison with DINO
## small targets
<img src="figs/small-targets.png" alt="small targets" style="width: 50%; height: auto;" />

## big targets
<img src="figs/big-targets.png" alt="big targets" style="width: 50%; height: auto;" />


## Run
<details>
  <summary>1. Inference and Visualizations</summary>

For inference and visualizations, we provide a inference.py as an example.

</details>

<details>
  <summary>2. Train a 4-scale model for 12 epochs</summary>
  
We use the 4-scale model trained for 12 epochs as an example to demonstrate how to evaluate and train our model.
Replace your corresponding path in the main. py file and execute：python main. py.
</details>
