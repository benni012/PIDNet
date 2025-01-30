# Project 4: Real Time Domain Adaptation in Semantic Segmentation
#### Advisor: Claudia Cuttano

## Additions
- added support for loveda dataset
- added dacs / adaptnet
- added visualization scripts

## Requirements

```pip install tensorboardX yacs torchprofile```

## Data Preparation
See the [example notebook](https://colab.research.google.com/drive/1XeZl2EOh4jY5AOS-Nyy3LTs-LKU39HRB?usp=sharing) on Colab.

## Training
```python tools/train.py --cfg configs/loveda/pidnet_small_loveda.yaml```  
```python tools/train_dacs.py --cfg configs/loveda/pidnet_small_loveda_dacs.yaml```  
```python tools/train_adapt.py --cfg configs/loveda/pidnet_small_loveda_adapt.yaml```

## Misc.

- To visualize the results (will load best.pt or provided), run the following command:  
```python tools/visualize.py --cfg configs/loveda/pidnet_small_loveda.yaml```
- To generate a full visualisation, create `vis_models/models`, put all the `.pt` files inside and run:  
```python tools/visualize_export.py --cfg configs/loveda/pidnet_small_loveda.yaml```
- To run a speed test for latency, macs and parameters, run the following command:  
```python tools/speed.py --cfg configs/loveda/pidnet_small_loveda.yaml```
