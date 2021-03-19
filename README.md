<h1>Other Repos repos used</h1>

https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch

https://gist.github.com/akTwelve/dc79fc8b9ae66828e7c7f648049bc42d

https://github.com/mnslarcher/kmeans-anchors-ratios

https://github.com/l3p-cv/lost


<h1>General Notes</h1>

The project was intended to create an interface for multiple object detectors for fish detection. Due to time issues we just implemented one detector (EfficientDet). Also the utilities are so far just hardcoded for EfficientDet use.

<h1>Initialize Repository</h1>

1) python setup_ds.py
2) conda activate FishDet

<h1>Get pretrained weights (COCO pretrained EfficientDet Weights, plus Deepfish trained FishDet)</h1>

python get_pretrained_weights.py

<h1> Setup a COCO Style Dataset for EfficientDet</h1>

python setup_ds.py --mode coco --c {compound_coefficient} --ds_name {name of dataset/project}  --path {path/to/COCO/Dataset/Folder}

<h2>Note</h2>

Folder structure must be like this

    /ds_name
    
        /annotations
        
            /instances_train.json
            
            /instances_val.sjon
            
        /train
        
            /img1
            
            /img2
            
            ..
            
            /imgn
            
        /val
        
            /img1
            
            /img2
            
            ..
            
            /imgn
            

<h1>Setup a known dataset for EfficientDet: (So far just Deepfish supported)</h1>

python setup_ds.py --mode known --ds_name {ds_name}

<h2>Note</h2>

This will use the project config I generated with --c 4. Also this will use bbox annotations I generated from the original per pixel annotations.

<h1>Setup a folder with images (for inference purposes)</h1>

python setup_ds.py --mode any --ds_name {ds_name} --path {path/to/imagefolder}

<h1>Combine two COCO Style Datasets</h1>

python combine_ds.py --ds1 {name of first dataset/project} --ds2 {name of second dataset/project} --ds_name {name of new dataset} --c {compound_coefficient}

<h1>Train EfficientDet</h1>

python Interface.py --do train --project {name of project/dataset} --c {compound_coefficient} --load_weights {weights in Yet-another-EfficientDet/weights. If ommited, training will be conducted on randomly initialized weights} --detector EfficientDet --batch_size {batch_size} --lr {learnrate} --num_epochs {number of epochs} --head_only (if True will just train the regression / classification layers. If false will train whole network) 

<h2>Note</h2>

Best weights will be the newest .pth file in Yet-another-EfficientDet/logs/{ds_name}

<h1>Predict using EfficientDet</h1>

python Interface.py --do infer --project {name of project/dataset} --c {compound_coefficient} --load_weights {weights in Yet-another-EfficientDet/weights. If ommited will throw error} --detector EfficientDet --infer_mode { Must be in ["lost", "coco", "viz", "all"]. Lost creates lost style annotations, coco creates coco style annotations, viz saves the images plus draws bboxes in them, all does everything} --path {path/to/images/ to do inference on} --conf_threshold {confidence threshold. Will be used to filter bboxes by confidence}

<h2>Note</h2>

Inference results can be found in Yet-another-EfficientDet/inference/{timestamp of session}

<h1>Using the lost pipeline</h1>

python lost_pipeline.py --path {path/to/lost} --c 4

<h2>Note</h2>

Helpful tips and instructions should be printed by the script. If you have by any chance problem running this, contact me or run the used os commands in the script one after another by hand. You have to set the dataset names and paths probably by yourself though.
 


