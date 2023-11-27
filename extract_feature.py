import os
import h5py

def extract_feature(model, data_loader, dataset):
    model.eval()
    outfile = 'Features/' + str(dataset) + '/global'
    if not os.path.exists(outfile):
        os.makedirs(outfile)
    f = h5py.File(outfile + '/train.hdf5', 'w')
    max_count = len(data_loader)*data_loader.batch_size
    all_labels = f.create_dataset('all_labels',(max_count,), dtype='i')
    all_feats=None
    count=0
    
    for i, (x,y) in enumerate(data_loader):
        if i%100 == 0:
            print('{:d}/{:d}'.format(i, len(data_loader)))
        x = x.cuda()
        feature = model(x)
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', (max_count, feature.size(1),), dtype='f')
        all_feats[count:count+feature.size(0),:] = feature.data.cpu().numpy()
        all_labels[count:count+feature.size(0)] = y.cpu().numpy()
        count = count + feature.size(0)
    
    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count
    
    f.close()