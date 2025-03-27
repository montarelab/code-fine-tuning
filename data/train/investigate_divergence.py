import pickle
import torch

"""
feature_maps = ['s5.pathway0_res0.branch1_bn',
                's5.pathway0_res0.branch2',
                's5.pathway0_res1.branch2',
                's5.pathway0_res2.branch2',
                's5.pathway1_res0.branch1_bn',
                's5.pathway1_res0.branch2',
                's5.pathway1_res1.branch2',
                's5.pathway1_res2.branch2',
                'head.act'
                ]

print('not equal:')

# iterate through all batches
for batch in range(9):
    with open('/srv/beegfs02/scratch/da_action/data/output/randomness/run1/' + str(batch) + 'features.pkl', 'rb') as f:
        feature_maps_1 = pickle.load(f)

    with open('/srv/beegfs02/scratch/da_action/data/output/randomness/run2/' + str(batch) + 'features.pkl', 'rb') as f:
        feature_maps_2 = pickle.load(f)

    # iterate through all feature maps
    for j in feature_maps:
        if not torch.eq(feature_maps_1[j], feature_maps_2[j]).all():
            print('batch: ', str(batch), ', feature map: ', j)
            if j == 'head.act':
                print('run1: ', feature_maps_1[j])
                print('run2: ', feature_maps_2[j])
                for h in range(feature_maps_1[j].shape[0]):
                    for k in range(feature_maps_1[j].shape[1]):
                        if feature_maps_1[j][h,k].item() != feature_maps_2[j][h,k].item():
                            print(h, k, ':', feature_maps_1[j][h,k].item(), feature_maps_2[j][h,k].item())
                            print(type(feature_maps_1[j][h,k].item()))



"""
#print(dest_object_name['head.act'])

# loss before bw
print('loss before bw:')
with open('/srv/beegfs02/scratch/da_action/data/output/first_it_loss/run1/loss_before_bw.pkl', 'rb') as f:
    loss1 = pickle.load(f)
with open('/srv/beegfs02/scratch/da_action/data/output/first_it_loss/run2/loss_before_bw.pkl', 'rb') as f:
    loss2 = pickle.load(f)
print('run1: ', loss1.item())
print('run2: ', loss2.item())

# loss after bw
print('loss after bw:')
with open('/srv/beegfs02/scratch/da_action/data/output/first_it_loss/run1/loss_after_bw.pkl', 'rb') as f:
    loss1 = pickle.load(f)
with open('/srv/beegfs02/scratch/da_action/data/output/first_it_loss/run2/loss_after_bw.pkl', 'rb') as f:
    loss2 = pickle.load(f)
print('run1: ', loss1.item())
print('run2: ', loss2.item())

# model divergence before bw
print('model divergence before bw:')
with open('/srv/beegfs02/scratch/da_action/data/output/first_it_loss/run1/model_before_bw.pkl', 'rb') as f:
    model1 = pickle.load(f)

with open('/srv/beegfs02/scratch/da_action/data/output/first_it_loss/run2/model_before_bw.pkl', 'rb') as f:
    model2 = pickle.load(f)

for (name1, p1), (name2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
    if not torch.eq(p1, p2).all():
        print(name1, name2)

# model divergence after bw
print('model divergence after bw:')
with open('/srv/beegfs02/scratch/da_action/data/output/first_it_loss/run1/model_after_bw.pkl', 'rb') as f:
    model1 = pickle.load(f)

with open('/srv/beegfs02/scratch/da_action/data/output/first_it_loss/run2/model_after_bw.pkl', 'rb') as f:
    model2 = pickle.load(f)

for (name1, p1), (name2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
    if not torch.eq(p1, p2).all():
        print(name1, name2)

# model divergence optimizer
print('model divergence after optimizer step:')
with open('/srv/beegfs02/scratch/da_action/data/output/first_it_loss/run1/model_optimizer.pkl', 'rb') as f:
    model1 = pickle.load(f)

with open('/srv/beegfs02/scratch/da_action/data/output/first_it_loss/run2/model_optimizer.pkl', 'rb') as f:
    model2 = pickle.load(f)

for (name1, p1), (name2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
    if not torch.eq(p1, p2).all():
        print(name1, name2)