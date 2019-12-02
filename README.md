# Semantic Transformation

> This is the code for paper **Transformation of Dense and Sparse Text Representations**. If you are interested to our work, or have any questions or find problems in our code please tell us. Thanks very much.

REQUIRMENTS:

python2.7.13

pytorch0.4.1

INSTRUCTIONS:

python2 runner.py 

--save [save path] 

--dictionary [dictionary file] 

--train-data [train file] 

--val-data [valid file] 

--test-data [test file] 

--class-number [class number] 

--cuda


We attach the SST1 data here and you can run the following instruction directly to get the results on this dataset.

python2 runner.py --save ./save/ve --dictionary ./SST1/dict.json --train-data ./SST1/train.json --val-data ./SST1/dev.json --test-data ./SST1/test.json --class-number 5 --cuda

Note that if our code benefits you, please cite our paper.

@article{hu2019transformation,
  title={Transformation of Dense and Sparse Text Representations},
  author={Hu, Wenpeng and Wang, Mengyu and Liu, Bing and Ji, Feng and Chen, Haiqing and Zhao, Dongyan and Ma, Jinwen and Yan, Rui},
  journal={arXiv preprint arXiv:1911.02914},
  year={2019}
}
