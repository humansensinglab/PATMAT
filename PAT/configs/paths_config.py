## Pretrained models paths
e4e = './pretrained_models/e4e_ffhq_encode.pt'
stylegan2_ada_ffhq = './pretrained_models/CelebA-HQ_512.pkl'
style_clip_pretrained_mappers = ''
ir_se50 = './pretrained_models/model_ir_se50.pth'
dlib = './pretrained_models/align.dat'

## Dirs for output files
checkpoints_dir = './checkpoints'
embedding_base_dir = './embeddings'
styleclip_output_dir = './StyleCLIP_results'
experiments_output_dir = './output'

## Input info
### Input dir, where the images reside
###Currently the data separation step (see section 3.2.1 + 3.2.2)
#### and regularization is hardcoded via the filenames in this directory.
#### see how it is done in MultiIDCoach function. We hope to automate this step
input_data_path = 'INPUT_PATH'


### Inversion identifier, used to keeping track of the inversion results. Both the latent code and the generator
input_data_id = 'INPUT-WORD'

## Keywords
pti_results_keyword = 'RESULT-KEYWORD-FOR-HOUSEKEEPING'
e4e_results_keyword = 'e4e'
sg2_results_keyword = 'SG2'
sg2_plus_results_keyword = 'SG2_plus'
multi_id_model_type = 'multi_id'

## Edit directions
interfacegan_age = 'editings/interfacegan_directions/age.pt'
interfacegan_smile = 'editings/interfacegan_directions/smile.pt'
interfacegan_rotation = 'editings/interfacegan_directions/rotation.pt'
ffhq_pca = 'editings/ganspace_pca/ffhq_pca.pt'
