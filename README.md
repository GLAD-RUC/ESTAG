# Equivariant Spatio-Temporal Attentive Graph Networks for Physical Dynamics Simulation


[**[Paper]**]([https://openreview.net/forum?id=35nFSbEBks&noteId=BwW3NXOyXx](https://openreview.net/pdf?id=35nFSbEBks)) 

## Initialize Environment
It is suggested to use `conda` to manage the python environment, and you can install the required packages from the provided `environment.yml` directly.
```python
conda env create --file environment.yml
```


## Data Preparation
We provide three datasets in [Google Drive](https://drive.google.com/drive/folders/1CN7HSH4Wz0dLekWDuOZKVyPSoK7-6Bxa?usp=drive_link)


**1. MD17**

The MD17 dataset can be downloaded from [MD17](http://quantum-machine.org/gdml/#datasets). 


**2. Motion Capture**

The raw data were obtained via [CMU Motion Capture Database](http://mocap.cs.cmu.edu/search.php?subjectnumber=35). The preprocessed dataset as well as the splits are provided in  `motion` folder.

**3. Protein MD**


We provide the data preprocessing code in `mdanalysis/preprocess.py`. One can simply run

```python
python mdanalysis/preprocess.py
```


## Training and Evaluation

**1. MD17**

```bash
python main_md.py --exp_name='exp_1' --model='estag' --mol='aspirin' --n_layers=2 --fft=True --eat=True --with_mask
```

**2. Protein MD**

```bash
python main_mdanalysis.py --exp_name='exp_2' --model='estag' --n_layers=2 --fft=True --eat=True --with_mask
```

**3. Motion Capture**

```bash
python main_motion.py --exp_name='exp_3' --model='estag' --n_layers=2 --fft=True --eat=True --with_mask
```



## Rollout
```bash
bash rollout/rollout.sh
```



## Visualization

Here we demonstrate with MD17 as an instance, and the same procedure can be employed for Protein and Motion.

1. Predict the states (coordinates) of the next frame according to a selected trajectory
    ```python
    python rollout/md17_pred.py
    ```

2. Based on the predicted coordinates, launch the file `visualization/vis_md.ipynb` to display the molecule.






