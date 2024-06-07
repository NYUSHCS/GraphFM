#!/bin/bash
#SBATCH -p sfscai
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --gres=gpu:1


module purge
module load miniconda
source activate GraphFM


for dataset in 'cora' 'pubmed' 'citeseer' 'Flickr' 'Reddit' 'ogbn-arxiv'; do
   for type_model in 'GraphMAE' 'GraphMAE2' 'S2GAE' 'GraphCL' 'GraphECL' 'CCA-SSG' 'GBT' 'BGRL'; do
      for batch_type in 'full_batch' 'node_sampling' 'subgraph_sampling'; do
          python -u main.py --dataset $dataset --type_model $type_model --cuda True --cuda_num 0 > ${batch_type}_${dataset}_${type_model}.txt 2>&1
      done
   done
done
