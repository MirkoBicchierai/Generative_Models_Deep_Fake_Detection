# python3 train_mlp.py --classes ORIGINAL NT --dataset CLIP
# python3 train_mlp.py --classes ORIGINAL DF --dataset CLIP
# python3 train_mlp.py --classes ORIGINAL F2F --dataset CLIP
# python3 train_mlp.py --classes ORIGINAL FSH --dataset CLIP
# python3 train_mlp.py --classes ORIGINAL FS --dataset CLIP
# python3 train_mlp.py --classes ORIGINAL F2F DF FSH FS NT --dataset CLIP
python3 train_mlp.py --classes ORIGINAL DF FSH FS NT --dataset CLIP --one_vs_rest
python3 train_mlp.py --classes ORIGINAL F2F FSH FS NT --dataset CLIP --one_vs_rest
python3 train_mlp.py --classes ORIGINAL F2F DF FS NT --dataset CLIP --one_vs_rest
python3 train_mlp.py --classes ORIGINAL F2F DF FSH NT --dataset CLIP --one_vs_rest
python3 train_mlp.py --classes ORIGINAL F2F DF FSH FS --dataset CLIP --one_vs_rest

# python3 train_mlp.py --classes ORIGINAL NT --dataset DINO
# python3 train_mlp.py --classes ORIGINAL DF --dataset DINO 
# python3 train_mlp.py --classes ORIGINAL F2F --dataset DINO 
# python3 train_mlp.py --classes ORIGINAL FSH --dataset DINO 
# python3 train_mlp.py --classes ORIGINAL FS --dataset DINO
# python3 train_mlp.py --classes ORIGINAL F2F DF FSH FS NT --dataset DINO
# python3 train_mlp.py --classes ORIGINAL F2F DF FSH FS NT   --dataset DINO --one_vs_rest

