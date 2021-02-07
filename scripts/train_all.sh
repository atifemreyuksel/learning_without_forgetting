echo "LWF started"
python train.py --name imagenet_mnist_lwf_ep100_alexnet --gpu_ids 0 --batch_size 32 --pretrained --dataset_dir data/MNIST --num_classes 10 --imsize 128 --eval_old_task --train_method lwf
echo "LWF eq_prob started"
python train.py --name imagenet_mnist_lwf_eq_prob_ep100_alexnet --gpu_ids 0 --batch_size 32 --pretrained --dataset_dir data/MNIST --num_classes 10 --imsize 128 --eval_old_task --train_method lwf_eq_prob
echo "feature extraction started"
python train.py --name imagenet_mnist_featext_ep100_alexnet --gpu_ids 0 --batch_size 32 --pretrained --dataset_dir data/MNIST --num_classes 10 --imsize 128 --eval_old_task --train_method featext
echo "finetune started"
python train.py --name imagenet_mnist_finetune_ep100_alexnet --gpu_ids 0 --batch_size 32 --pretrained --dataset_dir data/MNIST --num_classes 10 --imsize 128 --eval_old_task --train_method finetune
echo "finetune_fc started"
python train.py --name imagenet_mnist_finetune_fc_ep100_alexnet --gpu_ids 0 --batch_size 32 --pretrained --dataset_dir data/MNIST --num_classes 10 --imsize 128 --eval_old_task --train_method finetune_fc