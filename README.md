# basic CNN application
we are learning how to package a CNN application


## Directory structure -
```
├── [data]
│   └── [PetImages]
│       ├── [catt]
│       └── [dogg]
├── [utils]
│   ├── config.py
│   ├── data_management.py
│   └── model.py
├── [VGGmodel]
│    ├── [checkpoints]
│    │   └── vgg_16model_checkpoint.h5
│    └── [models]
│        ├── VGG16_model_at_20201205_120019.h5
│        └── VGG16_model_at_20201205_133403.h5
├── predict.py
├── README.md
├── training.py
├── original_vgg_base.h5
├── [base_log_dir]
│   └── [tensorboard_log_dir]
│       ├── [log_at_20201205_132300]
│       │   ├── train
│       │   └── validation
│       └── [log_at_20201205_132819]
│           ├── train
│           └── validation
├── dog.jpg   ## testing image for prediction
└── cat.jpeg  ## testing image for prediction
```